import tensorflow as tf
# import tensorflow_datasets as tfds
import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
import os
import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
import glob
import matplotlib.image as mpimg
import streamlit as st
from tensorflow.keras.utils import load_img
from tensorflow.keras.utils import img_to_array
from PIL import Image
# print(tf.__version__)

import splitfolders
splitfolders.ratio('data', output="output", seed=1337, ratio=(0.65, 0.35)) 

img_width, img_height = 400,400

train_data_dir = 'output/train'
validation_data_dir = 'output/val'
nb_train_samples = 64
nb_validation_samples = 32
epochs = 500
batch_size = 8
if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)
    
model = Sequential()
model.add(Conv2D(16, (3, 3), input_shape=input_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.4))
model.add(Dense(3))
model.add(Activation('softmax'))
model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')
test_datagen = ImageDataGenerator(rescale=1. / 255)
train_datagen = ImageDataGenerator(rescale = 1. /255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical')

import time
class TimeHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.times = []

    def on_epoch_begin(self, batch, logs={}):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, batch, logs={}):
        self.times.append(time.time() - self.epoch_time_start)
time_callback = TimeHistory()
import pandas as pd
def historyfile():
    history = model.fit_generator(
        train_generator,
        steps_per_epoch=nb_train_samples // batch_size,
        epochs=15,
        validation_data=validation_generator,
        validation_steps=nb_validation_samples // batch_size,
        callbacks=[time_callback])
    
    hist_df = pd.DataFrame(history.history) 
    # or save to csv: 
    hist_df['elaped time'] = time_callback.times
    hist_df['epochs'] = hist_df.index+1
    hist_df
    return hist_df

def training_withepoch(epocs):
    history = model.fit_generator(
        train_generator,
        steps_per_epoch=nb_train_samples // batch_size,
        epochs=epocs,
        validation_data=validation_generator,
        validation_steps=nb_validation_samples // batch_size,
        callbacks=[time_callback])

def load_image(filename):
    	# load the image
	img = load_img(filename, target_size=(400,400))
	# convert to array
	img = img_to_array(img)
	# reshape into a single sample with 3 channels
	img = img.reshape( 1,400, 400, 3)
	# center pixel data
	img = img.astype('float32')
	img = img /255.0
	return img

def print_per_cluster(folder_name):
    count = 0

    for img_path in glob.glob(folder_name):
        img = load_image(img_path)
        result = model.predict(img)
        st.write(result[0])
        image_pt = Image.open(img_path)
        st.image(image_pt, caption='First head 5 images')
        count +=1
        if count == 5:
            break