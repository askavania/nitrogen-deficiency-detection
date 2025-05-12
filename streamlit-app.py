import streamlit as st
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as T
from torchvision import models
import numpy as np
import cv2

# 1) Determine device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 2) Define and cache model loader
@st.cache_resource
def load_model(path="EfficientNet_B4NO2Model.pt"):
    # rebuild architecture
    model = models.efficientnet_b4(pretrained=False)
    model.classifier = nn.Sequential(
        nn.Dropout(0.5, inplace=True),
        nn.Linear(1792, 512, bias=False),
        nn.BatchNorm1d(512),
        nn.ReLU(inplace=True),
        nn.Linear(512, 4)
    )
    # load only the state dict
    sd = torch.load(path, map_location=device)
    model.load_state_dict(sd)
    return model.to(device).eval()

model = load_model()

# 3) Prediction helper
def predict(img: np.ndarray):
    tf = T.Compose([
        T.Resize((224, 224)),       # recommended size for B4
        T.ToTensor(),
        T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])
    img = Image.fromarray(img[..., ::-1])  # BGR→RGB
    x = tf(img).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(x)
        idx = torch.argmax(out, dim=1).item()
    names = [
        'No₂ Deficiency – Class 1',
        'N₂O Deficiency – Class 2',
        'Ideal – Class 3',
        'Ideal – Class 4'
    ]
    return names[idx]

# 4) App layout
st.title("Rice Leaf Nitrogen Deficiency Detector")
st.write("Upload or capture a photo of a leaf to classify its nitrogen status.")

for tab_name, widget in [("Upload Image", st.file_uploader), ("Capture Image", st.camera_input)]:
    with st.tab(tab_name):
        file = widget("Your image", type=['jpg','png','jpeg','jfif'])
        if file:
            arr = np.frombuffer(file.read(), np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            st.image(img, channels="BGR", caption="Input image")
            with st.spinner("Predicting..."):
                label = predict(img)
            st.success(f"Prediction: **{label}**")
