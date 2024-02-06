import os
import numpy as np
from skimage.io import imread
from skimage.transform import resize
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn import preprocessing
import joblib
from PIL import Image

cat = ['t-shirt', 'pants']
input_data = r'C:\Users\scorp\Desktop\Programmin\Machine Learning\T_Shirt_OR_Pants\data'


# Load images and extract features
def load_images(folder_path, category):
    image_data = []
    labels = []
    for filename in os.listdir(os.path.join(folder_path, category)):
        if filename.endswith(".jpg"):  # assuming all images are in jpg format
            img = imread(os.path.join(folder_path, category, filename))

            img_resized = resize(img, (100, 100))  # Resize images to a consistent size

            gray_scaled = Image.fromarray((img_resized*255).astype(np.uint8)).convert("L")

            features = np.array(gray_scaled).flatten()  # Flatten the image into a 1D array (feature vector)

            image_data.append(features)
            labels.append(category)
    return image_data, labels

# Load data and labels
data = []
labels = []

for category in cat:
    category_data, category_labels = load_images(input_data, category)
    data.extend(category_data)
    labels.extend(category_labels)
print("Data successfully loaded!!")
# Convert data and labels to numpy arrays
data = np.array(data)
labels = np.array(labels)


label_encoder = preprocessing.LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels)

print("Original Labels:", np.unique(labels))  # 0 maps to pants and 1 maps to T_shirt
print("Encoded Labels:", np.unique(encoded_labels))

print("Training Data...")
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data, encoded_labels, test_size=0.1, random_state=42)

# Perform grid search to find the best hyperparameters for the SVM
param_grid = {'C': [1, 10, 100], 'gamma': [0.1, 0.01, 0.001], 'kernel': ['rbf', 'linear']}
grid_search = GridSearchCV(SVC(), param_grid, cv=5)
grid_search.fit(X_train, y_train)

# Print the best hyperparameters found by grid search
print("Best Hyperparameters:", grid_search.best_params_)

# Train the SVM model with the best hyperparameters
best_svm = SVC(C=1, gamma=grid_search.best_params_['gamma'], kernel=grid_search.best_params_['kernel'])
best_svm.fit(X_train, y_train)

joblib.dump(best_svm, 'model1%')

# Evaluate the model on the test set
accuracy = best_svm.score(X_test, y_test)
print("Accuracy on the test set:", accuracy)
