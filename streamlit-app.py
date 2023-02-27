import streamlit as st
from PIL import Image
import torch
import torch.nn as nn
import pickle
import torchvision.transforms as transforms

# Load the saved PyTorch model
# model_str = open('NO2_model.pkl', 'rb').read()
# model = nn.Sequential(*pickle.loads(model_str))
model = torch.hub.load('EfficientNet_B4NO2Model.pth',source=local, force_reload=True)

# Define the transformations for the input image
transform = transforms.Compose([
    transforms.Resize((100, 100)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Define the list of class names
class_names = ['class1', 'class2', 'class3', 'class4']

# Define the Streamlit app
def app():
    st.title("Nitrogen Deficiency for Rice Crop Prediction App")
    st.write("Upload a photo of a rice leaf to see if it has nitrogen deficiency or not!")

    # Allow the user to upload an image
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    # If an image is uploaded, make a prediction with the model
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        image_tensor = transform(image).unsqueeze(0)
        
        model.eval()
        with torch.no_grad():
            outputs = model(image_tensor)
            class_index = torch.argmax(output, dim=1).item()
            class_name = class_names[class_index]

        st.image(image, caption='Uploaded Image', use_column_width=True)
        st.write("The predicted class is", class_name)

# Run the Streamlit app
if __name__ == '__main__':
    app()
