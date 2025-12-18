import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix

# --- 1. 設定 ---
DATA_ROOT = "./rvf10k" 
BATCH_SIZE = 32
EPOCHS = 5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 2. 自定義 Dataset ---
class RVFDataset(Dataset):
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
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label

# --- 3. 數據準備 ---
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

train_loader = DataLoader(RVFDataset(os.path.join(DATA_ROOT, "train.csv"), DATA_ROOT, transform), batch_size=BATCH_SIZE, shuffle=True)
valid_loader = DataLoader(RVFDataset(os.path.join(DATA_ROOT, "valid.csv"), DATA_ROOT, transform), batch_size=BATCH_SIZE)

# --- 4. 模型與優化器 ---
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, 2) 
model = model.to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# --- 5. 訓練與評估迴圈 ---
print(f"🚀 開始訓練，使用設備: {DEVICE}")

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
    
    for images, labels in train_pbar:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        # 計算準確率
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        current_acc = 100 * correct / total
        
        train_pbar.set_postfix({"loss": f"{loss.item():.4f}", "acc": f"{current_acc:.2f}%"})

    avg_loss = running_loss / len(train_loader)
    print(f"✨ Epoch {epoch+1} 完成! 平均 Loss: {avg_loss:.4f}")

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