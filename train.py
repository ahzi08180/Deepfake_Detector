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
    img_gray = np.array(img_pil.convert('L'))
    f = np.fft.fft2(img_gray)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1)
    magnitude_spectrum = cv2.normalize(magnitude_spectrum, None, 0, 1, cv2.NORM_MINMAX)
    return torch.from_numpy(magnitude_spectrum).float().unsqueeze(0)  # [1,H,W]

# --- 2. 自定義 Dataset ---
class RVFDatasetFFT(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_rel_path = str(self.data.loc[idx, 'path'])
        img_path = os.path.join(self.root_dir, img_rel_path)
        label = int(self.data.loc[idx, 'label'])

        # 讀圖
        image_pil = Image.open(img_path).convert("RGB")
        image_pil = transforms.Resize((224,224))(image_pil)

        # FFT
        fft_feature = get_fft_spectrum(image_pil)

        # RGB transform
        if self.transform:
            image = self.transform(image_pil)
        else:
            image = transforms.ToTensor()(image_pil)

        # 拼接 RGB + FFT
        combined_input = torch.cat((image, fft_feature), dim=0)  # [4,224,224]

        return combined_input, label

# --- 3. 設定與準備 ---
DATA_ROOT = "./rvf10k"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32

train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(0.2,0.2,0.2),
    transforms.GaussianBlur(3, sigma=(0.1,2.0)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])
valid_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

train_loader = DataLoader(RVFDatasetFFT(os.path.join(DATA_ROOT,"train.csv"),DATA_ROOT,train_transform),
                          batch_size=BATCH_SIZE, shuffle=True)
valid_loader = DataLoader(RVFDatasetFFT(os.path.join(DATA_ROOT,"valid.csv"),DATA_ROOT,valid_transform),
                          batch_size=BATCH_SIZE)

# --- 4. 初始化 EfficientNet-B0 (預訓練) ---
model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)

# 修改第一層卷積以支援 4 通道
old_conv = model.features[0][0]  # EfficientNet-B0 第一層卷積
model.features[0][0] = nn.Conv2d(4, old_conv.out_channels, kernel_size=old_conv.kernel_size,
                                 stride=old_conv.stride, padding=old_conv.padding, bias=False)
with torch.no_grad():
    model.features[0][0].weight[:, :3, :, :] = old_conv.weight
    model.features[0][0].weight[:, 3, :, :] = torch.mean(old_conv.weight, dim=1)

# 修改最後全連接層
model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
model = model.to(DEVICE)

# --- 5. 優化器與損失 ---
optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
criterion = nn.CrossEntropyLoss()

# --- 6. 訓練迴圈 ---
EPOCHS = 3
print(f"🚀 開始 EfficientNet-B0 + FFT 雙流訓練 on {DEVICE} ...")
print(f"訓練資料數量: {len(train_loader.dataset)}，驗證資料數量: {len(valid_loader.dataset)}")
print(f'每批次大小: {BATCH_SIZE}，總訓練回合數: {EPOCHS}')
print("-----------------------------------------------------")

for epoch in range(EPOCHS):
    model.train()
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
    for imgs, lbls in pbar:
        imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, lbls)
        loss.backward()
        optimizer.step()

        # 計算準確率
        _, predicted = torch.max(outputs.data, 1)
        total += lbls.size(0)
        correct += (predicted == lbls).sum().item()
        current_acc = 100 * correct / total

        pbar.set_postfix({"loss": f"{loss.item():.4f}", "acc": f"{current_acc:.2f}%"})

# --- 7. 驗證 ---
model.eval()
all_preds, all_labels = [], []

with torch.no_grad():
    for imgs, lbls in tqdm(valid_loader, desc="Validating"):
        imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)
        outputs = model(imgs)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(lbls.cpu().numpy())

target_names = ['Fake (0)','Real (1)']
print(classification_report(all_labels, all_preds, target_names=target_names))

cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.savefig('confusion_matrix.png')
print("✅ 混淆矩陣已儲存為 confusion_matrix.png")

torch.save(model.state_dict(), "rvf10k_efficientnetb0.pth")
print("✅ EfficientNet-B0 模型已儲存。")