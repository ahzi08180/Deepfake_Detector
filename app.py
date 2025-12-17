import streamlit as st
import torch
import torch.nn as nn
import timm
from PIL import Image
from facenet_pytorch import MTCNN
import torchvision.transforms as transforms

# --- 頁面設定 ---
st.set_page_config(page_title="Deepfake 圖片偵測器", layout="centered")
st.title("🛡️ Deepfake 圖片偵測系統")
st.write("上傳一張人臉照片，系統將分析其是否為 AI 生成。")

# --- 1. 定義模型架構 ---
class DeepfakeDetector(nn.Module):
    def __init__(self):
        super(DeepfakeDetector, self).__init__()
        # 使用高效的 EfficientNet-B0
        self.backbone = timm.create_model('efficientnet_b0', pretrained=True)
        num_ftrs = self.backbone.classifier.in_features
        self.backbone.classifier = nn.Sequential(
            nn.Linear(num_ftrs, 1) # 輸出單一數值，用於二分類
        )

    def forward(self, x):
        return torch.sigmoid(self.backbone(x))

# --- 2. 載入模型與工具 ---
@st.cache_resource
def load_tools():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 人臉偵測器
    mtcnn = MTCNN(image_size=224, margin=20, device=device)
    # 預測模型
    model = DeepfakeDetector().to(device)
    model.eval()
    return mtcnn, model, device

mtcnn, model, device = load_tools()

# 影像預處理
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# --- 3. 網頁介面實作 ---
uploaded_file = st.file_uploader("選擇一張圖片...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='原始圖片', use_column_width=True)
    
    with st.spinner('正在分析中...'):
        # 步驟 A: 偵測人臉
        face = mtcnn(image)
        
        if face is not None:
            # 步驟 B: 預處理並預測
            # MTCNN 輸出的 face 已經是 Tensor
            face_input = face.unsqueeze(0).to(device)
            
            with torch.no_grad():
                output = model(face_input).item()
            
            # 顯示結果 (假設 > 0.5 為 Fake)
            st.divider()
            if output > 0.5:
                st.error(f"判定結果：🚨 疑似為 Deepfake 偽造圖片")
                st.progress(output)
                st.write(f"偽造可能性: {output*100:.2f}%")
            else:
                st.success(f"判定結果：✅ 這看起來像是 真實照片")
                st.progress(output)
                st.write(f"真實可能性: {(1-output)*100:.2f}%")
                
            # 顯示模型看到的人臉部分
            st.image(face.permute(1, 2, 0).numpy() / 2 + 0.5, caption="偵測到的人臉區域", width=150)
        else:
            st.warning("無法在圖片中偵測到清晰的人臉，請換一張試試看。")

st.info("註：此模型目前使用基礎預訓練權重，若要達到商用精準度，需載入針對 DFDC 資料集訓練後的 .pth 權重檔案。")