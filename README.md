# Nitrogen Deficiency Detection for Rice Crops
---

# Executive Summary: Rice Leaf Disease Classification Model

## Overview:
The objective of this project is to build an image classifier that can accurately detect nitrogen deficiency in rice crops from images, to assist farmers in identifying and treating the disease in a timely and efficient manner. The model was built using PyTorch, an open-source machine learning framework, and was trained on a dataset of over 4,000 images of healthy and diseased rice leaves. The model achieved an F1-score of 0.933 for the training set and 0.986 for the holdout set, indicating its high accuracy in classifying rice leaf diseases.

The dataset used in this study comprises images of nitrogen deficient rice crop leaves that are matched with a leaf color chart, and was obtained from Sambalpur University. The LCC method was developed by the International Rice Research Institute (IRRI) and has been adopted by several countries, including Bangladesh, China, India, Indonesia, the Philippines, Thailand, and Vietnam. This method is a simple and low-cost tool for farmers to assess the nitrogen status of rice plants, which can help them to make better decisions regarding the application of fertilizers. However, one potential disadvantage of the Leaf Color Chart (LCC) method is that it relies on human perception and judgment, which can lead to variability in results between different observers. Using the LCC method requires training and practice to achieve consistent and accurate results, which inexperienced/new farmers may lack, hence we are exploring the usage of computer vision to help bridge this gap.

The images in this dataset are categorized into subgroups based on their LCC value.

The dataset is organized into four subfolders, each representing a different level of nitrogen deficiency according to the leaf color chart. The objective is to accurately classify the test data into one of the four nitrogen deficiency labels.

## Data Preprocessing:
Before training the model, the dataset was preprocessed to ensure that it was properly formatted for use in the model. This included resizing the images to a uniform size of 100x100 pixels, normalizing the pixel values, and dividing the dataset into training and validation sets.

## Model Training and Evaluation:
In order to classify the images, a convolutional neural network (CNN) was used. Specifically, a pre-trained EfficientNet model was used as a feature extractor, and a fully connected layer was added to the output to map the features to the number of classes. The model was trained for 20 epochs using the Adam optimizer, and achieved an accuracy of 93.8% on the training set.

To further evaluate the performance of the model, a holdout set was used to test the model's ability to generalize to unseen data. The model achieved an accuracy of 98.6% on the holdout set, which is slightly higher than the training set accuracy. This could be due to the smaller size of the holdout set (1,000 images) compared to the training set (4,000+ images), which has higher variations after transformation.

The F1-score of the model was calculated to be 93.3% on the training set, and 98.6% on the holdout set. Although at one glance this might traditionally suggest an overfit, but further real world testing was done outside the experiment that suggests that the model is effective at classifying rice leaf nitrogen deficiency properly.

## Model Deployment:
To make predictions with the model, a predictor function was defined, which takes as input the test dataset and outputs the F1-score, confusion matrix, and classification report in the notebook. Additionally, a predict function was defined to allow users to input a new image and receive a classification of the image's disease status, connected to the streamlit cloud app.

## Conclusion:
The model and app developed in this project has the potential to assist farmers in identifying and treating rice leaf nitrogen deficiency in a timely and efficient manner, without the judgement error that humans may make. Its high accuracy, as demonstrated by an F1-score of 98.6% on the holdout set, makes it a valuable tool in the agricultural industry. Future work could involve a model for object detection to recognise the plants in the entire field, expanding the dataset to include more diverse types of rice leaf diseases, refining the model architecture, or exploring transfer learning techniques to improve the model's performance. 
