import streamlit as st
from PIL import Image
import torch
import torch.nn as nn
import pickle
import torchvision.transforms as transforms

# Load the saved PyTorch model
# model_str = open('NO2_model.pkl', 'rb').read()
# model = nn.Sequential(*pickle.loads(model_str))
model = pickle.load(open('EfficientNet_B4NO2Model.pth', 'rb'))

# Define a function to make predictions with the trained model
def predict(model, image_path):
    # Load the image and transform it to the appropriate format
    transform = transforms.Compose([
        transforms.Resize((100, 100)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0)
    image = image.to(device)

    # Make a prediction with the trained model
    model.eval()
    with torch.no_grad():
        output = model(image)
        class_index = torch.argmax(output, dim=1).item()

    # Map the predicted index to the class name
    class_names = ['class1', 'class2', 'class3', 'class4']
    class_name = class_names[class_index]

    return class_name

# # Example usage: predict the class of a new image
# image_path = "/kaggle/input/rice-leaf-test/googleimage1.jpg"
# class_name = predict(net, image_path)
# image_show = Image.open(image_path)
# plt.imshow(image_show)
# plt.title('The predicted class is :' + class_name)
# plt.axis('off')
# plt.show()

# # Define the transformations for the input image
# transform = transforms.Compose([
#     transforms.Resize((100, 100)),
#     transforms.ToTensor(),
#     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
# ])

# # Define the list of class names
# class_names = ['class1', 'class2', 'class3', 'class4']

# Define the Streamlit app
def app():
    st.title("Nitrogen Deficiency for Rice Crop Prediction App")
    st.write("Upload a photo of a rice leaf to see if it has nitrogen deficiency or not!")

    # Allow the user to upload an image
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    # If an image is uploaded, make a prediction with the model
    if uploaded_file is not None:
        class_name = predict(model, uploaded_file)      
#         image = Image.open(uploaded_file)
#         image_tensor = transform(image).unsqueeze(0)
        
#         model.eval()
#         with torch.no_grad():
#             outputs = model(image_tensor)
#             class_index = torch.argmax(output, dim=1).item()
#             class_name = class_names[class_index]

        st.image(image, caption='Uploaded Image', use_column_width=True)
        st.write("The predicted class is", class_name)

# Run the Streamlit app
if __name__ == '__main__':
    app()
