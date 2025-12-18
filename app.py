import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image, ImageDraw
from facenet_pytorch import MTCNN
import numpy as np
import cv2
import os

st.set_page_config(page_title="FFT Deepfake Detector", layout="wide")
st.title("🛡️ EfficientNet-B0 Deepfake 偵測系統 (FFT + RGB)")

@st.cache_resource
def load_all():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mtcnn = MTCNN(image_size=224, margin=20, device=device)
    
    # 初始化 EfficientNet-B0
    model = models.efficientnet_b0(weights=None)

    # === 對齊 train.py：4 channel input ===
    old_conv = model.features[0][0]
    model.features[0][0] = nn.Conv2d(
        4,
        old_conv.out_channels,
        kernel_size=old_conv.kernel_size,
        stride=old_conv.stride,
        padding=old_conv.padding,
        bias=False
    )

    # === ⭐ 關鍵：對齊 train.py 的 classifier ===
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)

    # 載入訓練好的權重
    model_path = "rvf10k_efficientnetb0.pth"
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device).eval()
    
    return mtcnn, model, device

mtcnn, model, device = load_all()

def process_fft(face_pil):
    img_gray = np.array(face_pil.convert('L'))
    f = np.fft.fft2(img_gray)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20 * np.log(np.abs(fshift)+1)
    magnitude_spectrum = cv2.normalize(magnitude_spectrum, None, 0.0, 1.0, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    magnitude_spectrum = np.clip(magnitude_spectrum, 0.0, 1.0)
    return torch.from_numpy(magnitude_spectrum).float().unsqueeze(0)

uploaded_file = st.file_uploader("上傳照片進行頻域分析...", type=["jpg", "png", "jpeg"])

if uploaded_file:
    img = Image.open(uploaded_file).convert('RGB')
    img_draw = img.copy()
    draw = ImageDraw.Draw(img_draw)

    # 使用 MTCNN 偵測人臉
    boxes, _ = mtcnn.detect(img)

    if boxes is None or len(boxes) == 0:
        st.error("❌ 未偵測到人臉，Deepfake 分析需包含清楚人臉。")
        st.info("👉 建議：正面、單人、臉部佔畫面 30% 以上")
        st.stop()
    
    # 畫出 bounding box
    for box in boxes:
        draw.rectangle(box.tolist(), outline="red", width=3)

    # 取第一張偵測到的人臉
    x1, y1, x2, y2 = [int(b) for b in boxes[0]]
    face = img.crop((x1, y1, x2, y2))
    face = face.resize((224,224))

    # 空域 Tensor + Normalize
    normalize = transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    rgb_tensor = normalize(transforms.ToTensor()(face)).to(device)

    # 頻域 Tensor
    fft_tensor = process_fft(face).to(device)

    # 合併 4 通道
    input_tensor = torch.cat((rgb_tensor, fft_tensor), dim=0).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)
        prob = torch.softmax(output, dim=1)[0]

    fake_prob = prob[0].item()
    real_prob = prob[1].item()

    # --- UI 顯示 ---
    col1, col2, col3 = st.columns(3)
    with col1:
        st.image(img_draw, caption="偵測到的人臉 & Bounding Box", width='stretch')
    with col2:
        fft_viz = fft_tensor.squeeze().cpu().numpy()
        st.image(fft_viz, caption="FFT 頻譜 (AI 偽影偵測)", clamp=True, width='stretch')
    with col3:
        st.metric("🟥 偽造機率", f"{fake_prob*100:.2f}%")
        st.metric("🟩 真實機率", f"{real_prob*100:.2f}%")
        if fake_prob > real_prob:
            st.error("🚨 判定為 AI 生成 (Deepfake)")
        else:
            st.success("✅ 判定為真實人臉")
