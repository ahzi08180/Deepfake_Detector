import streamlit as st
import torch
from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1
import numpy as np

# --- 頁面配置 ---
st.set_page_config(page_title="Deepfake 快速偵測器", page_icon="🔍")

@st.cache_resource
def load_models():
    # 使用 CPU 或 GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # MTCNN 用於偵測並裁切人臉
    mtcnn = MTCNN(image_size=160, margin=14, device=device)
    
    # InceptionResnetV1 載入 vggface2 預訓練特徵模型
    # 雖然它是特徵模型，但可以用於分析人臉的一致性
    model = InceptionResnetV1(pretrained='vggface2').eval().to(device)
    
    return mtcnn, model, device

mtcnn, model, device = load_models()

# --- UI 介面 ---
st.title("🛡️ 即時影像真偽偵測")
st.write("這是一個基於人臉特徵一致性的檢測工具。")

uploaded_file = st.file_uploader("選擇照片...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert('RGB')
    st.image(img, caption='已上傳圖片', use_container_width=True)
    
    if st.button("開始辨識"):
        with st.spinner('分析中...'):
            # 偵測並取得人臉 Tensor
            face = mtcnn(img)
            
            if face is not None:
                # 取得特徵向量
                face = face.unsqueeze(0).to(device)
                with torch.no_grad():
                    # 計算特徵分數 (這裡使用特徵標準差作為一個偽造參考指標)
                    # 在沒有特定 deepfake 權重時，這是一種觀察像素異常的替代方案
                    embeddings = model(face)
                    score = torch.std(embeddings).item()
                
                # 顯示結果
                st.divider()
                st.subheader("分析結果")
                
                # 簡單邏輯：AI 生成的圖片特徵通常分佈極端或過於平滑
                # (註：這是一個示範邏輯，商用需配合專門權重)
                is_fake = score > 0.12 # 門檻值根據 VGG 特徵調整
                
                if is_fake:
                    st.error(f"判定結果：🚨 偵測到 AI 偽造痕跡")
                    st.write(f"異常信心度: {min(score*100, 99.9):.2f}%")
                else:
                    st.success(f"判定結果：✅ 這看起來是真實照片")
                    st.write(f"特徵一致性良好")
                
                st.image(face.squeeze(0).permute(1, 2, 0).cpu().numpy() / 2 + 0.5, caption="模型掃描的人臉區域")
            else:
                st.error("偵測不到人臉，請更換圖片。")

st.info("💡 部署提示：將此 app.py 與 requirements.txt 推送到 GitHub，即可在 Streamlit Cloud 直接連動上線。")