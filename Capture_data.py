import cv2
import os
import time
import keyboard

""" Goal is to collect personal pictures in 3 different types of pictures of Hoodies and TShirts as the dataset to use for the ML module """

cam = cv2.VideoCapture(0)

image_format = '.png'

os.makedirs(r'C:\Users\scorp\Desktop\Programmin\Machine Learning\T_Shirt_OR_Pants\data', exist_ok=True)
count = 0
while True:
    ret, frame = cam.read()

    if not ret:
        print("Failed to capture image")
        break


    file_name = os.path.join(r'C:\Users\scorp\Desktop\Programmin\Machine Learning\T_Shirt_OR_Pants\data', f'{count}.png')
    cv2.imwrite(file_name, frame)
    cv2.imshow('Webcam', frame)
    count = count + 1
    time.sleep(0.2)

cam.release()
cv2.destroyAllWindows()