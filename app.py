import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os

# --- 頁面設定 ---
st.set_page_config(page_title="Deepfake Detector", layout="centered")
st.title("🛡️ Deepfake 影像辨識系統")

# --- 載入模型函式 ---
@st.cache_resource
def load_trained_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.resnet18(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, 2)
    
    # 載入你訓練好的權重
    if os.path.exists("rvf10k_model.pth"):
        model.load_state_dict(torch.load("rvf10k_model.pth", map_location=device))
        st.sidebar.success("✅ 成功載入自定義訓練權重")
    else:
        st.sidebar.warning("⚠️ 找不到權重檔，將使用隨機初始權重 (僅供測試介面用)")
        
    model.to(device)
    model.eval()
    return model, device

model, device = load_trained_model()

# --- 預處理 ---
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# --- UI 介面 ---
uploaded_file = st.file_uploader("請上傳一張人臉照片...", type=["jpg", "png", "jpeg"])

if uploaded_file:
    img = Image.open(uploaded_file).convert('RGB')
    st.image(img, caption='待測圖片', use_container_width=True)
    
    if st.button("執行偵測"):
        img_tensor = transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            outputs = model(img_tensor)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            # 根據你的 CSV: Index 0=Fake, Index 1=Real
            fake_prob = probs[0][0].item()
            real_prob = probs[0][1].item()
        
        st.divider()
        if real_prob > fake_prob:
            st.success(f"結果：這是一張【真實】照片")
            st.progress(real_prob)
            st.write(f"真實度信心：{real_prob*100:.2f}%")
        else:
            st.error(f"結果：🚨 疑似為【Deepfake】偽造照片")
            st.progress(fake_prob)
            st.write(f"偽造度信心：{fake_prob*100:.2f}%")