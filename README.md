# T-Shirt or Pants Classifier

## Overview
The "T-Shirt or Pants Classifier" is a machine learning project aimed at classifying images of clothing into two categories: t-shirts and pants. The project utilizes image processing techniques and Support Vector Machine (SVM) classification to achieve accurate predictions.

## Project Structure
- model.py: This script contains the code for data loading, preprocessing, model training, and evaluation.
- test.py: This script provides functions to load the trained model and make predictions on new images.
- capture.py: This script allows the user to take images through the webcam of the computer, in order to be added to the dataset for training or testing. 
- data: This directory stores the training images categorized into t-shirts and pants.
- Test_data: This directory contains images for testing the trained model.

## Technologies Used
1. Python: The primary programming language for implementing the machine learning algorithms and image processing techniques.
2. scikit-learn: Utilized for implementing the SVM classifier and grid search for hyperparameter tuning.
3. NumPy: Used for numerical computations and handling of arrays.
4. OpenCV: Used to simplify the interaction with the computer webcam and the capturing of images.
5. PIL (Python Imaging Library): Employed for image processing tasks such as resizing and converting images to grayscale.
6. Joblib: Used for saving and loading the trained SVM model.
7. Anaconda Distribution: The project environment is managed using Anaconda, ensuring easy package installation and dependency management.
   
   
## Project Workflow

1. Data Loading and Preprocessing: Images are loaded from the specified directory, resized to a consistent size, converted to grayscale, and flattened into feature vectors.

2. Label Encoding: The class labels (t-shirt, pants) are encoded into numerical values using scikit-learn's LabelEncoder.

3. Model Training: The SVM model is trained using the training data. Grid search is performed to find the optimal hyperparameters for the SVM classifier.

4. Model Evaluation: The trained model is evaluated on the test set to assess its performance in terms of accuracy.

5. Prediction: The trained model is saved, and functions are provided to load the model and make predictions on new images.

## Instructions for Usage
1. Training: To train the model, run the model.py script. Ensure that the image data directory (input_data) is correctly specified.

2. Testing: Use the test.py script to load the trained model and make predictions on new images. Provide the path to the image file as input to the predict function.

## Results and Performance
The model achieves a high accuracy on the test set, indicating its effectiveness in classifying t-shirts and pants.
Hyperparameter tuning using grid search helps in optimizing the SVM model for better performance.
The model is capable of making accurate predictions on unseen images, demonstrating its generalization ability.



