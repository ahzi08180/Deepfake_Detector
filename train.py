import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, roc_auc_score
from tqdm import tqdm

# =====================
# Config
# =====================
DATA_DIR = "RVF10K"
BATCH_SIZE = 16
IMG_SIZE = 300
EPOCHS_STAGE1 = 5
EPOCHS_STAGE2 = 10
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =====================
# Data Augmentation
# (專為 deepfake 設計)
# =====================
train_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomApply([
        transforms.GaussianBlur(kernel_size=5)
    ], p=0.3),
    transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
    transforms.RandomGrayscale(p=0.1),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

val_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# =====================
# Dataset
# =====================
train_set = datasets.ImageFolder(
    os.path.join(DATA_DIR, "train"), transform=train_tf
)
val_set = datasets.ImageFolder(
    os.path.join(DATA_DIR, "val"), transform=val_tf
)

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)

print("Class mapping:", train_set.class_to_idx)

# =====================
# Model (EfficientNet)
# =====================
model = models.efficientnet_b3(weights="IMAGENET1K_V1")

in_features = model.classifier[1].in_features
model.classifier[1] = nn.Sequential(
    nn.Linear(in_features, 256),
    nn.BatchNorm1d(256),
    nn.ReLU(),
    nn.Dropout(0.4),
    nn.Linear(256, 1)
)

model = model.to(DEVICE)

# =====================
# Loss & Optim
# =====================
criterion = nn.BCEWithLogitsLoss()

def evaluate(model):
    model.eval()
    preds, labels = [], []

    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            logits = model(x).squeeze()
            prob = torch.sigmoid(logits)

            preds.extend(prob.cpu().numpy())
            labels.extend(y.cpu().numpy())

    acc = accuracy_score(labels, [p > 0.5 for p in preds])
    auc = roc_auc_score(labels, preds)
    return acc, auc

# =====================
# Stage 1: Train Head
# =====================
print("🔒 Stage 1: Freeze Backbone")

for param in model.features.parameters():
    param.requires_grad = False

optimizer = optim.Adam(model.classifier.parameters(), lr=1e-3)

for epoch in range(EPOCHS_STAGE1):
    model.train()
    loop = tqdm(train_loader)

    for x, y in loop:
        x, y = x.to(DEVICE), y.float().to(DEVICE)
        optimizer.zero_grad()

        logits = model(x).squeeze()
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        loop.set_description(f"Stage1 Epoch [{epoch+1}/{EPOCHS_STAGE1}]")
        loop.set_postfix(loss=loss.item())

    acc, auc = evaluate(model)
    print(f"Val Acc: {acc:.4f} | Val AUC: {auc:.4f}")

# =====================
# Stage 2: Fine-tuning
# =====================
print("🔓 Stage 2: Unfreeze Backbone")

for param in model.parameters():
    param.requires_grad = True

optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)

for epoch in range(EPOCHS_STAGE2):
    model.train()
    loop = tqdm(train_loader)

    for x, y in loop:
        x, y = x.to(DEVICE), y.float().to(DEVICE)
        optimizer.zero_grad()

        logits = model(x).squeeze()
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        loop.set_description(f"Stage2 Epoch [{epoch+1}/{EPOCHS_STAGE2}]")
        loop.set_postfix(loss=loss.item())

    acc, auc = evaluate(model)
    print(f"Val Acc: {acc:.4f} | Val AUC: {auc:.4f}")

# =====================
# Save
# =====================
torch.save(model.state_dict(), "deepfake_rvf10k.pth")
print("✅ Model saved as deepfake_rvf10k.pth")
