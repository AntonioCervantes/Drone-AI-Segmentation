import cv2
import numpy as np
import os
import pandas as pd

from tqdm import tqdm
from glob import glob
from albumentations import RandomCrop, HorizontalFlip, VerticalFlip

from sklearn.model_selection import train_test_split
from PIL import Image

from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Activation, MaxPool2D, UpSampling2D, Concatenate
from tensorflow.keras.models import Model

import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, Callback
from tensorflow.keras.utils import plot_model
import os

from utils.data_IO import (create_dir, create_dataframe)
from utils.data_preprocess import (read_image, read_mask, 
                                   tf_dataset, preprocess,
                                   augment_data)
from utils.models import mobileunet

# Show GPUs devices
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())


###############
### Data IO ###
###############

path = "../data/dataset/semantic_drone_dataset"
#images = sorted(glob(os.path.join(path, "original_images/*")))
#masks = sorted(glob(os.path.join(path, "label_images_semantic/*")))
#print(f"Original images:  {len(images)} - Original masks: {len(masks)}")

#create_dir("../data/dataset/semantic_drone_dataset/new_data/images/")
#create_dir("../data/dataset/semantic_drone_dataset/new_data/masks/")

save_path = "../data/dataset/semantic_drone_dataset/new_data/"

#augment_data(images, masks, save_path, augment=True)

images = sorted(glob(os.path.join(save_path, "images/*")))
masks = sorted(glob(os.path.join(save_path, "masks/*")))
print(f"Augmented images:  {len(images)} - Augmented masks: {len(masks)}")

# Create dataframe
image_path =  os.path.join(save_path, "images/")
label_path = os.path.join(save_path, "masks/")
df_images = create_dataframe(image_path)
df_masks = create_dataframe(label_path)
print('Total Images: ', len(df_images))
#print(df_images)

# Split data
X_trainval, X_test = train_test_split(df_images['id'], test_size=0.1, random_state=19)
X_trainval=df_images['id']
X_train, X_val = train_test_split(X_trainval, test_size=0.2, random_state=19)

print(f"Train Size : {len(X_train)} images")
print(f"Val Size   :  {len(X_val)} images")
print(f"Test Size  :  {len(X_test)} images")

y_train = X_train #the same values for images (X) and labels (y)
y_test = X_test
y_val = X_val

img_train = [os.path.join(image_path, f"{name}.jpg") for name in X_train]
mask_train = [os.path.join(label_path, f"{name}.png") for name in y_train]
img_val = [os.path.join(image_path, f"{name}.jpg") for name in X_val]
mask_val = [os.path.join(label_path, f"{name}.png") for name in y_val]
img_test = [os.path.join(image_path, f"{name}.jpg") for name in X_test]
mask_test = [os.path.join(label_path, f"{name}.png") for name in y_test]

####################
### Define Model ###
####################

# Define the resolution of the images and 
H = 320   # to keep the original ratio 
W = 480 
shape = (H, W, 3)

# Define the number of classes
num_classes = 23

# Define Unet model
model = mobileunet(shape, num_classes)

# Display model summary
model.summary()
#plot_model(model,to_file='model.png')


###################
### Train Model ###
###################

# Seeding
np.random.seed(42)
tf.random.set_seed(42)

# Hyperparameters
lr = 1e-4
batch_size = 4
epochs = 35

# Compile Model
model.compile(loss="categorical_crossentropy", optimizer=tf.keras.optimizers.Adam(lr), metrics=['accuracy'])

# Get dataset
train_dataset = tf_dataset(img_train, mask_train, batch_size)
valid_dataset = tf_dataset(img_val, mask_val, batch_size)

# Specify step size
train_steps = len(img_train)//batch_size
valid_steps = len(img_val)//batch_size

# Check points
checkpointer = [
    ModelCheckpoint(filepath="./model.h5",monitor='val_loss',verbose=2,save_best_only=True),
    ReduceLROnPlateau(monitor='val_loss', patience=3, factor=0.1, verbose=2, min_lr=1e-6),
    EarlyStopping(monitor='val_loss', patience=10, verbose=2)
]

# Train model
model.fit(train_dataset,
          steps_per_epoch=train_steps,
          validation_data=valid_dataset,
          validation_steps=valid_steps,
          epochs=epochs,
          callbacks=checkpointer
         )













