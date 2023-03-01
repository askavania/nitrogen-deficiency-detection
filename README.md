# Nitrogen Deficiency Detection for Rice Crops
---

# Executive Summary: Rice Leaf Disease Classification Model

## Overview:
The objective of this project is to build an image classifier that can accurately detect nitrogen deficiency in rice crops from images, to assist farmers in identifying and treating the disease in a timely and efficient manner. The model was built using PyTorch, an open-source machine learning framework, and was trained on a dataset of over 4,000 images of healthy and diseased rice leaves. The model achieved an F1-score of 0.973 for the training set and 0.955 for the holdout set, indicating its high accuracy in classifying rice leaf diseases.

The dataset used in this study comprises images of nitrogen deficient rice crop leaves that are matched with a leaf color chart, and was obtained from Sambalpur University. The LCC method was developed by the International Rice Research Institute (IRRI) and has been adopted by several countries, including Bangladesh, China, India, Indonesia, the Philippines, Thailand, and Vietnam. This method is a simple and low-cost tool for farmers to assess the nitrogen status of rice plants, which can help them to make better decisions regarding the application of fertilizers. However, one potential disadvantage of the Leaf Color Chart (LCC) method is that it relies on human perception and judgment, which can lead to variability in results between different observers. Using the LCC method requires training and practice to achieve consistent and accurate results, which inexperienced/new farmers may lack, hence we are exploring the usage of computer vision to help bridge this gap.

The images in this dataset are categorized into subgroups based on their LCC value.

The dataset is organized into four subfolders, each representing a different level of nitrogen deficiency according to the leaf color chart. The objective is to accurately classify the test data into one of the four nitrogen deficiency labels.

## Data Preprocessing:
Before training the model, the dataset was preprocessed to ensure that it was properly formatted for use in the model. This included resizing the images to a uniform size of 100x100 pixels, normalizing the pixel values, and dividing the dataset into training and validation sets.

## Model Training and Evaluation:
We used the EfficientNet_B4 pre-trained model as the base model and fine-tuned it using transfer learning. The dataset was split into 80% training and 20% holdout sets, and we trained the model for 20 epochs with a batch size of 16, using the Adam optimizer and a learning rate of 0.001. The model was evaluated on the holdout set, achieving an accuracy of a F1 score of 95.5%.

## Model Deployment:
To make predictions with the model, a predictor function was defined, which takes as input the test dataset and outputs the F1-score, confusion matrix, and classification report in the notebook. Additionally, a predict function was defined to allow users to input a new image and receive a classification of the image's disease status, connected to the streamlit cloud app.

## Conclusion:
The model and streamlit app developed in this project has the potential to assist farmers in identifying and treating rice leaf nitrogen deficiency in a timely and efficient manner, without the judgement error that humans may make. Its high accuracy, as demonstrated by an F1-score of 95.5% on the holdout set, makes it a valuable tool in the agricultural industry. Future work could involve a model for object detection to recognise the plants in the entire field, expanding the dataset to include more diverse types of rice leaf diseases, refining the model architecture, or exploring transfer learning techniques to improve the model's performance. 

<br><br>
---
The links to to the app and research sources are below;<br>
App : https://rice-crop-plant-nitrogen-deficiency-detection.streamlit.app/
<br>
About the Leaf Color Chart: https://pdf.usaid.gov/pdf_docs/PA00K938.pdf
<br>
Dataset : https://github.com/askavania/nitrogen-deficiency-detection/tree/main/Data/NitrogenDeficiencyImage
<br>


