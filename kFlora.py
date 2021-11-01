import matplotlib.pyplot as plt
import seaborn as sns

import keras
from keras.models import Sequential
from keras.applications import mobilenet_v2
from keras.layers import Dense, Conv2D , MaxPool2D , Flatten , Dropout
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import adam_v2
from sklearn.metrics import classification_report,confusion_matrix
import cv2
import os
import numpy as np
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
from keras import layers
from keras.models import load_model
from keras.applications import *

epochs = 5
img_size = 224

def get_root_drive():
    root_path = "flowers"
    if 'COLAB_GPU' in os.environ:
        root_path = '/content/drive/MyDrive/Colab Notebooks/flowers'
    return root_path


def isgpu2():
    from tensorflow.python.client import device_lib
    print(device_lib.list_local_devices())


isgpu2()

train_path = get_root_drive()

datagenTraining = ImageDataGenerator(
        validation_split = 0.2,
        rescale=1./255
)

datagenValidation = ImageDataGenerator(
        validation_split = 0.2,
        rescale=1.0/255)

train_generator = datagenTraining.flow_from_directory(
        train_path,
        shuffle=True,
        subset ='training',
        target_size = (img_size, img_size),
        class_mode = 'categorical')
val_generator = datagenValidation.flow_from_directory(
        train_path,
        shuffle=True,
        subset ='validation',
        target_size=(img_size, img_size),
        class_mode='categorical')


def save_model(mdl, filename):
    mdl.save(filename)
    print("Saved model to disk")

def load_saved_model(name):
    model = load_model(name)
    return model


def get_pretrained_mobilenet(outputs, image_size):
    base_model = keras.applications.mobilenet_v2.MobileNetV2(input_shape=(image_size, image_size, 3), include_top=False,
                                                             weights="imagenet")
    base_model.trainable = False
    model = keras.Sequential([base_model,
                              keras.layers.GlobalAveragePooling2D(),
                              keras.layers.Dropout(0.2),
                              keras.layers.Dense(outputs, activation="softmax")
                              ])
    return model


def get_model_1():
    model = Sequential()
    model.add(Conv2D(32,3,padding="same", activation="relu", input_shape=(img_size,img_size,3)))
    model.add(MaxPool2D())
    model.add(Conv2D(32, 3, padding="same", activation="relu"))
    model.add(MaxPool2D())
    model.add(Conv2D(64, 3, padding="same", activation="relu"))
    model.add(MaxPool2D())
    model.add(Dropout(0.4))
    model.add(Flatten())
    model.add(Dense(128,activation="relu"))
    model.add(Dense(5, activation="softmax"))
    model.summary()
    return model


def run_loaded_model(model):
    Scores = model.evaluate(val_generator, verbose=2)
    print('Validation loss:', Scores[0])
    print('Validation accuracy:', Scores[1])


def plot_history(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs_range = range ( epochs )

    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.suptitle('Training History')

    ax1.plot ( epochs_range, acc, label='Training Accuracy' )
    ax1.plot ( epochs_range, val_acc, label='Validation Accuracy' )
    ax1.legend ( loc='lower right' )
    ax1.set_title ( 'Training and Validation Accuracy' )

    ax2.plot ( epochs_range, loss, label='Training Loss' )
    ax2.plot ( epochs_range, val_loss, label='Validation Loss' )
    ax2.legend ( loc='upper right' )
    ax2.set_title ( 'Training and Validation Loss' )
    plt.show ()


def print_quick_stats(model):
    Scores = model.evaluate(val_generator, verbose=2)
    print('Validation loss:', Scores[0])
    print('Validation accuracy:', Scores[1])

def run_model_1():
    model = get_model_1()
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(train_generator, epochs=epochs, validation_data=val_generator, verbose=2)

    print_quick_stats(model)
    plot_history(history)

    return model

def run_model_pretrained_xception(img_size, outputs):
    base_model = keras.applications.xception.Xception(weights='imagenet', include_top=False, input_shape=(img_size, img_size, 3))
    base_model.trainable = False
    model = keras.Sequential([base_model,
                              keras.layers.GlobalAveragePooling2D(),
                              keras.layers.Dropout(0.2),
                              keras.layers.Dense(outputs, activation="softmax")
                              ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(train_generator, epochs=epochs, validation_data=val_generator, verbose=2)

    print_quick_stats(model)

    return model

def run_model_pretrained_mobilenetv2(img_size, outputs):
    base_model = keras.applications.mobilenet_v2.MobileNetV2(input_shape=(img_size, img_size, 3), include_top=False,
                                                             weights="imagenet")
    base_model.trainable = False
    model = keras.Sequential([base_model,
                              keras.layers.GlobalAveragePooling2D(),
                              keras.layers.Dropout(0.2),
                              keras.layers.Dense(outputs, activation="softmax")
                              ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(train_generator, epochs=epochs, validation_data=val_generator, verbose=2)

    print_quick_stats(model)

    return model

#m = run_model_pretrained_mobilenetv2(img_size, 5)
#m = run_model_pretrained_xception(img_size, 5)
#save_model(m, "xceptionnet.h5")
m = run_model_1()
#save_model(m)

#m = load_saved_model("xceptionnet.h5")
#run_loaded_model()



