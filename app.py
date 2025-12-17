import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import numpy as np

# =====================
# Config
# =====================
IMG_SIZE = 300
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = "deepfake_rvf10k.pth"

st.set_page_config(page_title="Deepfake Image Detector")
st.title("🧠 Deepfake vs Real Image Detector")

# =====================
# Transform (一定要和 training 一樣)
# =====================
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# =====================
# Load Model
# =====================
@st.cache_resource
def load_model():
    model = models.efficientnet_b3(weights=None)

    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Sequential(
        nn.Linear(in_features, 256),
        nn.BatchNorm1d(256),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(256, 1)
    )

    state = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state)
    model.to(DEVICE)
    model.eval()
    return model

model = load_model()

# =====================
# UI
# =====================
uploaded_file = st.file_uploader(
    "📤 Upload a face image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    x = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logit = model(x).squeeze()
        prob_fake = torch.sigmoid(logit).item()
        prob_real = 1 - prob_fake

    # =====================
    # Result
    # =====================
    st.subheader("🔍 Prediction Result")

    if prob_fake >= 0.5:
        st.error(f"🟥 FAKE ({prob_fake*100:.2f}%)")
    else:
        st.success(f"🟩 REAL ({prob_real*100:.2f}%)")

    st.markdown("### 📊 Confidence")
    st.progress(int(prob_fake * 100))
    st.write(f"Fake Probability: **{prob_fake:.4f}**")
    st.write(f"Real Probability: **{prob_real:.4f}**")
