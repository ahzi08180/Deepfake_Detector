import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import os
import numpy as np
import cv2
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# --- 1. FFT 處理函式 ---
def get_fft_spectrum(img_pil):
    """提取圖片的 FFT 頻譜特徵並轉為 Tensor"""
    # 轉灰階並轉為 numpy
    img_gray = np.array(img_pil.convert('L'))
    # 執行 FFT
    f = np.fft.fft2(img_gray)
    fshift = np.fft.fftshift(f)
    # 取振幅譜並進行對數變換
    magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1)
    # 標準化到 0-1
    magnitude_spectrum = cv2.normalize(magnitude_spectrum, None, 0, 1, cv2.NORM_MINMAX)
    # 轉回 Tensor 並調整尺寸
    fft_tensor = torch.from_numpy(magnitude_spectrum).float().unsqueeze(0) # [1, H, W]
    return fft_tensor

# --- 2. 自定義 Dataset (雙流版) ---
class RVFDatasetFFT(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_rel_path = str(self.data.loc[idx, 'path'])
        img_path = os.path.join(self.root_dir, img_rel_path)
        label = int(self.data.loc[idx, 'label'])

        # 1. 讀圖
        image_pil = Image.open(img_path).convert("RGB")

        # 2. 先 Resize（關鍵）
        image_pil = transforms.Resize((224, 224))(image_pil)

        # 3. FFT（從 resize 後、未 normalize 的圖）
        fft_feature = get_fft_spectrum(image_pil)   # [1,224,224]

        # 4. RGB transform（ToTensor + Normalize）
        if self.transform:
            image = self.transform(image_pil)       # [3,224,224]
        else:
            image = transforms.ToTensor()(image_pil)

        # 5. 拼接 RGB + FFT
        combined_input = torch.cat((image, fft_feature), dim=0)  # [4,224,224]

        return combined_input, label


# --- 3. 設定與準備 ---
DATA_ROOT = "./rvf10k"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(0.2, 0.2, 0.2),
    transforms.GaussianBlur(3, sigma=(0.1, 2.0)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

valid_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

train_loader = DataLoader(RVFDatasetFFT(os.path.join(DATA_ROOT, "train.csv"), DATA_ROOT, train_transform), batch_size=BATCH_SIZE, shuffle=True)
valid_loader = DataLoader(RVFDatasetFFT(os.path.join(DATA_ROOT, "valid.csv"), DATA_ROOT, valid_transform), batch_size=BATCH_SIZE)

# --- 4. 修改模型以接收 4 通道 (RGB + FFT) ---
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
# 修改第一層卷積
old_conv = model.conv1
model.conv1 = nn.Conv2d(4, old_conv.out_channels, kernel_size=old_conv.kernel_size, 
                        stride=old_conv.stride, padding=old_conv.padding, bias=False)
# 將原有的權重複製給前 3 個通道，第 4 通道初始化
with torch.no_grad():
    model.conv1.weight[:, :3, :, :] = old_conv.weight
    model.conv1.weight[:, 3, :, :] = torch.mean(old_conv.weight, dim=1)

model.fc = nn.Linear(model.fc.in_features, 2)
model = model.to(DEVICE)

optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
criterion = nn.CrossEntropyLoss()

# --- 5. 訓練迴圈 (同前，略縮) ---
EPOCHS = 1
print(f"🚀 開始 FFT+RGB 雙流訓練...")
print(f"🚀 訓練設備: {DEVICE}")
print(f"🚀 訓練樣本數: {len(train_loader.dataset)}")
print(f"🚀 驗證樣本數: {len(valid_loader.dataset)}")
print(f"🚀 批次大小: {BATCH_SIZE}")
print(f"🚀 總訓練輪數: {EPOCHS}")

for epoch in range(EPOCHS):
    model.train()
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
    for imgs, lbls in pbar:
        imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)
        optimizer.zero_grad()
        loss = criterion(model(imgs), lbls)
        loss.backward()
        optimizer.step()
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

# --- 6. 最終評估 (Classification Report & Confusion Matrix) ---

print("\n📊 正在生成最終評估報告...")
model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in tqdm(valid_loader, desc="Validating"):
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# 印出 Classification Report
# 根據你的 CSV: 0 是 Fake, 1 是 Real
target_names = ['Fake (0)', 'Real (1)']
print("\n\n📝 Classification Report:")
print(classification_report(all_labels, all_preds, target_names=target_names))

# 繪製 Confusion Matrix
cm = confusion_matrix(all_labels, all_preds)
print("\n\n混淆矩陣:")
print(cm)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.savefig('confusion_matrix.png')
print("✅ 混淆矩陣已儲存為 confusion_matrix.png")

torch.save(model.state_dict(), "rvf10k_model.pth")
print("✅ 模型已儲存。")