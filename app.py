import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from facenet_pytorch import MTCNN
import numpy as np
import cv2

st.set_page_config(page_title="FFT Deepfake Detector", layout="wide")
st.title("🛡️ 頻域分析 Deepfake 偵測系統 (FFT + RGB)")

@st.cache_resource
def load_all():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mtcnn = MTCNN(image_size=224, margin=20, device=device)
    
    # 初始化 4 通道 ResNet18
    model = models.resnet18(weights=None)
    model.conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.fc = nn.Linear(model.fc.in_features, 2)
    
    if torch.cuda.is_available():
        model.load_state_dict(torch.load("rvf10k_fft_model.pth"))
    else:
        model.load_state_dict(torch.load("rvf10k_fft_model.pth", map_location='cpu'))
    
    model.to(device).eval()
    return mtcnn, model, device

mtcnn, model, device = load_all()

def process_fft(face_pil):
    img_gray = np.array(face_pil.convert('L'))
    f = np.fft.fft2(img_gray)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1)
    magnitude_spectrum = cv2.normalize(magnitude_spectrum, None, 0, 1, cv2.NORM_MINMAX)
    return torch.from_numpy(magnitude_spectrum).float().unsqueeze(0)

uploaded_file = st.file_uploader("上傳照片進行頻域分析...", type=["jpg", "png", "jpeg"])

if uploaded_file:
    img = Image.open(uploaded_file).convert('RGB')
    
    if st.button("開始深度檢測"):
        face = mtcnn(img) # 這裡得到的是 [3, 224, 224] 的 Tensor
        face = torch.clamp(face, 0, 1)
        
        if face is not None:
            # 1. 準備空域 Tensor (需要 Normalize)
            normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            rgb_tensor = normalize(face)
            
            # 2. 準備頻域 Tensor
            face_pil = transforms.ToPILImage()(face)
            fft_tensor = process_fft(face_pil).to(device)
            
            # 3. 合併為 4 通道
            input_tensor = torch.cat((rgb_tensor, fft_tensor), dim=0).unsqueeze(0).to(device)
            
            with torch.no_grad():
                output = model(input_tensor)
                prob = torch.softmax(output, dim=1)[0]
            
            # UI 顯示
            col1, col2, col3 = st.columns(3)
            with col1:
                st.image(img, caption="原始圖")
            with col2:
                # 顯示 FFT 頻譜圖供視覺化參考
                fft_viz = fft_tensor.squeeze().cpu().numpy()
                st.image(fft_viz, caption="FFT 頻譜 (AI 偽影偵測)")
            with col3:
                fake_prob = prob[0].item()
                real_prob = prob[1].item()

                st.metric("🟥 偽造機率", f"{fake_prob*100:.2f}%")
                st.metric("🟩 真實機率", f"{real_prob*100:.2f}%")

                if fake_prob > real_prob:
                    st.error("🚨 判定為 AI 生成 (Deepfake)")
                else:
                    st.success("✅ 判定為真實人臉")

        else:
            st.warning("偵測不到人臉。")