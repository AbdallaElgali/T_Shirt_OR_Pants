import joblib
from skimage.io import imread
from skimage.transform import resize
import numpy as np
from PIL import Image
import os


path = r'C:\Users\scorp\Desktop\Programmin\Machine Learning\T_Shirt_OR_Pants\MyModel82%'
model = joblib.load(path)

def predict(img_path):

    img = imread(img_path)
    img_resized = resize(img, (100,100))

    gray_img = Image.fromarray((img_resized*255).astype(np.uint8)).convert("L")
    features = np.array(gray_img).flatten()

    prediction = model.predict([features])

    return prediction

prediction = predict(r'C:\Users\scorp\Desktop\Programmin\Machine Learning\T_Shirt_OR_Pants\unused_data\1f2ba16d-c9a7-4ab9-bbb1-d13114078685.jpg')

if prediction == 0:
    actual_prediction = 'pants'
else:
    actual_prediction = 't_shirt'

print(actual_prediction)
'''
predictions = []
for c in range(42):
    prediction = predict(c)
    predictions.append(prediction[0])

count_0s = 0
count_1s = 0
for i, p in enumerate(predictions):
    if predictions[i] == 0:
        count_0s += 1
    elif predictions[i] == 1:
        count_1s += 1

percentage_of_0s = count_0s/len(predictions)
percentage_of_1s = count_1s/len(predictions)

print(percentage_of_0s, percentage_of_1s)
'''