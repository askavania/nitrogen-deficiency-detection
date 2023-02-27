import streamlit as st
from PIL import Image
import torch
import torchvision.transforms as transforms

# Load the saved PyTorch model
model = torch.hub.load('askavania/nitrogen-deficiency-detection', 'NO2_model.pkl', force_reload=True)

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

        with torch.no_grad():
            outputs = model(image_tensor)
            _, predicted = torch.max(outputs.data, 1)
            predicted_class = class_names[predicted]

        st.image(image, caption='Uploaded Image', use_column_width=True)
        st.write("The predicted class is", predicted_class)

# Run the Streamlit app
if __name__ == '__main__':
    app()
