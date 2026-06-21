import os
import matplotlib.pyplot as plt
import random
import numpy as np
from PIL import Image
import time

parent_folder = r"Pneumonia-Detection\Dataset-of-Normal-and-Pneumo"

"""No. of files in side the dataset"""
train_images_pneumonia = os.listdir(os.path.join(parent_folder,"Training","pneumonia"))
train_images_normal = os.listdir(os.path.join(parent_folder,"Training","normal"))
test_images_pneumonia = os.listdir(os.path.join(parent_folder,"Testing","pneumonia"))
test_images_normal = os.listdir(os.path.join(parent_folder,"Testing","normal"))

plt.bar(x=["Train Normal","Train Pneumonia","Test Normal","Test Pneumonia"],
        height=[len(train_images_normal),len(train_images_pneumonia),len(test_images_normal),len(test_images_pneumonia)])
plt.title("Traning v/s Testing")
plt.show()

print(f"Training Data Size: {len(train_images_normal)}")
print(f"Testing Data Size: {len(test_images_normal)}")
time.sleep(5)

print("Train images Pneumonia: ")
for idx in train_images_pneumonia:
    with Image.open(os.path.join(parent_folder,"Training","pneumonia",idx)) as img:
        print(img.size)
time.sleep(5)
print("Train images Normal: ")
for idx in train_images_normal:
    with Image.open(os.path.join(parent_folder,"Training","normal",idx)) as img:
        print(img.size)
time.sleep(5)
print("Test images Pneumonia: ")
for idx in test_images_pneumonia:
    with Image.open(os.path.join(parent_folder,"Testing","pneumonia",idx)) as img:
        print(img.size)
time.sleep(5)
print("Test images Normal: ")
for idx in test_images_normal:
    with Image.open(os.path.join(parent_folder,"Testing","normal",idx)) as img:
        print(img.size)

"""found all images are of different sizes"""