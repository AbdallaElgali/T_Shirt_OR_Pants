import cv2
import os
import time
import keyboard

""" Module to collect images to be able to add to the train and test dataset for more personalization, and to check if the model is overfitting or not """

cam = cv2.VideoCapture(0)
image_format = '.png'
path = r'C:\Users\scorp\Desktop\Programmin\Machine Learning\T_Shirt_OR_Pants\data'  # Change the path to your local directory

os.makedirs(path, exist_ok=True)
count = 0
while True:
    ret, frame = cam.read()

    if not ret:
        print("Failed to capture image")
        break


    file_name = os.path.join(path, f'{count}.png')
    cv2.imwrite(file_name, frame)
    cv2.imshow('Webcam', frame)
    count = count + 1
    time.sleep(0.2)

cam.release()
cv2.destroyAllWindows()
