import streamlit as st
from PIL import Image
import torch
import pickle
import torchvision.transforms as transforms
from torchvision import models
import numpy as np
import cv2

device = 'cuda' if torch.cuda.is_available() else 'cpu'
PATH = "EfficientNet_B4NO2Model.pt"
my_model = torch.load(PATH, map_location='cpu')
my_model.eval()

# Define a function to make predictions with the trained model
def predict(model, opencv_Image):
    # Load the image and transform it to the appropriate format
    transform = transforms.Compose([
        transforms.Resize((100, 100)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    #image = Image.open(image_path)
    pil_image = Image.fromarray(opencv_Image)
    image = transform(pil_image).unsqueeze(0).to(device)
    image = image

    # Make a prediction with the trained model
    #model.eval()
    with torch.no_grad():
        output = model(image)
        class_index = torch.argmax(output, dim=1).item()

    # Map the predicted index to the class name
    class_names = ['No2 Deficiency observed - Class 1:  \n Apply N-Fertilizer immediately', 
                   'N02 Deficiency observed - Class 2: \n Apply N-Fertilizer soon', 
                   'Ideal range - Class 3: \n Do not apply N-Fertilizer and continue to monitor closely', 
                   'Ideal range - Class 4: \n Do not apply N-Fertilizer and continue to monitor']
    class_name = class_names[class_index]

    return class_name

# Set the background color of the Streamlit app to black using CSS
st.markdown("""
    <style>
    body {
        background-color: #000054;
    }
    </style>
""", unsafe_allow_html=True)

# Define the rest of your Streamlit app code here


# Defining the rest of Streamlit app code
st.title("Nitrogen Deficiency for Rice Crop Prediction App")
st.write("Upload or take a photo of a rice leaf to see if it has nitrogen deficiency or not!")

tab1, tab2, tab3 = st.tabs(["Upload Image", "Capture Image", "Use Case Test Images"])

with tab1:
    test_image = st.file_uploader('Image', type=['jpg', 'png','jpeg', 'jfif'] )
    col1, col2 = st.columns(2)
    if test_image is not None:
        # Convert the file read to the bytes array.
        file_bytes = np.asarray(bytearray(test_image.read()), dtype=np.uint8)
        # Converting the byte array into opencv image. 0 for grayscale and 1 for bgr
        test_image_decoded = cv2.imdecode(file_bytes,1) 
        col1.subheader('Uploaded Image')
        col1.image(test_image_decoded, channels = "BGR")
        prediction = predict(my_model, test_image_decoded)
        col2.subheader('Predicted Class')
        col2.write(prediction)

with tab2:
    img_camera = st.camera_input("Capture Image")
    
    if img_camera is not None:
        # Convert the file read to the bytes array.
        file_bytes = np.asarray(bytearray(img_camera.read()), dtype=np.uint8)
        # Converting the byte array into opencv image. 0 for grayscale and 1 for bgr
        test_image_decoded = cv2.imdecode(file_bytes,1) 
        prediction = predict(my_model, test_image_decoded)
        st.subheader('Predicted Class')
        st.write(prediction)


        # Define a dictionary of preset images and their corresponding labels
preset_images = {
    'Class 1 sample 1': 'https://github.com/askavania/nitrogen-deficiency-detection/blob/main/Data/NitrogenDeficiencyImage/RealWorldTest/googleimage1-class1.jpg',
    'Class 1 sample 2': 'https://github.com/askavania/nitrogen-deficiency-detection/blob/main/Data/NitrogenDeficiencyImage/RealWorldTest/googleimage2.jpg',
    'Class 1 sample 3': 'https://github.com/askavania/nitrogen-deficiency-detection/blob/main/Data/NitrogenDeficiencyImage/RealWorldTest/googleimage4.jpg',
    'Class 1 sample 4': 'https://github.com/askavania/nitrogen-deficiency-detection/blob/main/Data/NitrogenDeficiencyImage/RealWorldTest/googleimage6.jpg',
    'Class 1 sample 3': 'https://github.com/askavania/nitrogen-deficiency-detection/blob/main/Data/NitrogenDeficiencyImage/RealWorldTest/googleimage4.jpg',
    'Class 3 sample 1': 'https://github.com/askavania/nitrogen-deficiency-detection/blob/main/Data/NitrogenDeficiencyImage/RealWorldTest/googleimage7.jpg',
    'Class 3 sample 2': 'https://github.com/askavania/nitrogen-deficiency-detection/blob/main/Data/NitrogenDeficiencyImage/RealWorldTest/googleimage8.jpg',
    'Class 4 sample 1': 'https://github.com/askavania/nitrogen-deficiency-detection/blob/main/Data/NitrogenDeficiencyImage/RealWorldTest/unknown4.jpg',
}

# Define a new tab for the preset images
# with tab3:
#     st.title('Preset Images')
#     for label, image_url in preset_images.items():
#         col1, col2 = st.columns(2)
#         with col1:
#             st.subheader(label.title())
#             st.image(image_url, use_column_width=True)
#         with col2:
#             st.write('Lorem ipsum dolor sit amet, consectetur adipiscing elit. Duis euismod ut nulla ac efficitur. Aenean rhoncus dolor quis erat congue bibendum.')

            
# Define the images and their corresponding labels
image_paths = preset_images.items()
labels = preset_images.keys()

# Define a function to make predictions for preset images
def predict_preset_images(model, image_paths):
    predictions = []
    for path in image_paths:
        image = cv2.imread(path)
        prediction = predict(model, image)
        predictions.append(prediction)
    return predictions

# Create a new tab for preset images
with tab3:
    st.markdown('## Preset Images')
    selected_image = st.selectbox('Select an image:', options=image_paths)
    if st.button('Predict'):
        image = cv2.imread(selected_image)
        prediction = predict(my_model, image)
        st.write('Prediction:', prediction)

# Create another tab for multiple preset images
#with st.sidebar:
#     st.markdown('## Multiple Preset Images')
#     st.write('This tab will predict the labels of multiple preset images.')
#     if st.button('Predict All'):
#         predictions = predict_preset_images(my_model, image_paths)
#         st.write('Predictions:')
#         for i, label in enumerate(labels):
#             st.write(f'{label}: {predictions[i]}')
