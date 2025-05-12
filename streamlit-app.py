import streamlit as st
from PIL import Image
import torch
import torchvision.models.efficientnet as efficientnet
from torchvision.ops.misc import Conv2dNormActivation
import torchvision.transforms as transforms
from torchvision import models
import torch.nn.modules.container as container
from torch.nn.modules.conv import Conv2d
import numpy as np
import cv2

# --- allowlist EfficientNet for safe unpickling under PyTorch ≥2.6 ---
torch.serialization.add_safe_globals([efficientnet.EfficientNet, container.Sequential, Conv2dNormActivation, torch.nn.modules.conv.Conv2d])


# device setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# load your full-model checkpoint
PATH = "EfficientNet_B4NO2Model.pt"
my_model = torch.load(PATH, map_location=device)  # weights_only=True by default, but EfficientNet is now allow-listed
my_model.to(device)
my_model.eval()

# prediction helper
def predict(model, opencv_image):
    transform = transforms.Compose([
        transforms.Resize((100, 100)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    pil_image = Image.fromarray(opencv_image)
    x = transform(pil_image).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(x)
        idx = torch.argmax(logits, dim=1).item()

    class_names = [
        'No₂ Deficiency − Class 1: Apply N-Fertilizer immediately',
        'N₂O Deficiency − Class 2: Apply N-Fertilizer soon',
        'Ideal − Class 3: Do not apply N-Fertilizer; monitor closely',
        'Ideal − Class 4: Do not apply N-Fertilizer; monitor'
    ]
    return class_names[idx]

# styling
st.markdown("""
    <style>
      body { background-color: #000; color: #fff; }
    </style>
""", unsafe_allow_html=True)

# app layout
st.title("Nitrogen Deficiency for Rice Crop Prediction App")
st.write("Upload or take a photo of a rice leaf to detect nitrogen deficiency.")

tab1, tab2 = st.tabs(["Upload Image", "Capture Image"])

with tab1:
    upload = st.file_uploader("Choose an image...", type=['jpg','png','jpeg','jfif'])
    if upload:
        data = np.frombuffer(upload.read(), np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        col1, col2 = st.columns(2)
        col1.subheader("Input")
        col1.image(img, channels="BGR")
        pred = predict(my_model, img)
        col2.subheader("Prediction")
        col2.write(pred)

with tab2:
    cam_img = st.camera_input("Take a picture")
    if cam_img:
        data = np.frombuffer(cam_img.read(), np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        st.subheader("Prediction")
        st.write(predict(my_model, img))
