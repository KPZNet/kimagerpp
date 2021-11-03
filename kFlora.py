import os
from os import listdir
from PIL import Image

import keras
import matplotlib.pyplot as plt
from keras import layers
from keras.utils.vis_utils import plot_model
from keras.applications import mobilenet_v2
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Dropout
from keras.models import Sequential
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import pandas as pd

def get_root_drive():
    root_path = "flowers"
    if 'COLAB_GPU' in os.environ:
        root_path = '/content/drive/MyDrive/Colab Notebooks/flowers'
    return root_path


def clean_files(folderToCheck, verbose = 0):
    print("-------------- FILE Checks --------------");
    def check_file(filename):
        try:
            if verbose >= 2:
                print("Checking: ", filename)
            img = Image.open(filename) 
            tst = img.verify() 
        except (IOError, SyntaxError) as e:
            if verbose >= 1:
                print('\tBad file removed:', filename) 
            os.remove(filename)

    for directory, subdirectories, files, in os.walk(folderToCheck):
      if verbose >= 1:
          print("Checking {0} with {1} files".format(directory, len(files) ))
      for file in files:
          filePath = os.path.join(directory, file)
          check_file(filePath)

    print("-------------- DONE FILE Checks --------------");


  

def get_root_drive_predict():
    root_path = "test_flowers"
    if 'COLAB_GPU' in os.environ:
        root_path = '/content/drive/MyDrive/Colab Notebooks/test_flowers'
    return root_path

def isgpu2():
    from tensorflow.python.client import device_lib
    print(device_lib.list_local_devices())

def get_data_generators(images_path, img_size):
    datagenTraining = ImageDataGenerator(
        validation_split=0.2,
        rescale=1. / 255
    )
    datagenValidation = ImageDataGenerator(
        validation_split=0.2,
        rescale=1.0 / 255)
    train_generator = datagenTraining.flow_from_directory(
        images_path,
        shuffle=True,
        subset='training',
        target_size=(img_size, img_size),
        class_mode='categorical')
    val_generator = datagenValidation.flow_from_directory(
        images_path,
        shuffle=True,
        subset='validation',
        target_size=(img_size, img_size),
        class_mode='categorical')
    return train_generator, val_generator

def get_data_generators_randomized(images_path, img_size):

    datagenTraining = ImageDataGenerator(
        validation_split=0.2,
        rescale=1. / 255,
        
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range = 30,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range = 0.2, # Randomly zoom image
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip = True,  # randomly flip images
        vertical_flip=False
    )
    datagenValidation = ImageDataGenerator(
        validation_split=0.2,
        rescale=1.0 / 255,
        
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range = 30,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range = 0.2, # Randomly zoom image
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip = True,  # randomly flip images
        vertical_flip=False
        )
    train_generator = datagenTraining.flow_from_directory(
        images_path,
        shuffle=True,
        subset='training',
        target_size=(img_size, img_size),
        class_mode='categorical')
    val_generator = datagenValidation.flow_from_directory(
        images_path,
        shuffle=True,
        subset='validation',
        target_size=(img_size, img_size),
        class_mode='categorical')
    return train_generator, val_generator


def get_predict_images(predict_path, img_size):

    imPredict = ImageDataGenerator(rescale=1.0 / 255)
    predict_generator = imPredict.flow_from_directory(
        predict_path,
        shuffle=False,
        subset='training',
        target_size=(img_size, img_size),
        class_mode='categorical')
    return predict_generator


def save_model(mdl, filename):
    mdl.save(filename)
    print("Saved model to disk")

def load_saved_model(name):
    model = load_model(name)
    return model

def model_predict(model, data, model_name="Model"):
    Scores = model.evaluate(data, verbose=2)
    print("")
    print("Model: ", model_name)
    print("Summary")
    #print('Prediction loss:', Scores[0])
    print('Identification accuracy:', Scores[1])
    print("")

    dct = {v: k for k, v in data.class_indices.items()}
    # Generate predictions for samples
    predictions = model.predict(data)
    preds = np.argmax(predictions, axis=1)
    df = pd.DataFrame()

    for i,k in enumerate(preds):
        r = [ (dct[k] == dct[data.classes[i]] , dct[data.classes[i]] , dct[k] , data.filenames[i] ) ]
        df = df.append( pd.DataFrame(r, columns=['Found','Expected','Identified','File']) )

    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1200)
    pd.set_option('display.colheader_justify', 'right')
    pd.set_option('display.precision', 3)
    print(df)
    print("")


def plot_history(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs_range = range ( epochs )

    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.suptitle('Epoch Training History')

    ax1.plot ( epochs_range, acc, label='Training Accuracy' )
    ax1.plot ( epochs_range, val_acc, label='Validation Accuracy' )
    ax1.legend ( loc='lower right' )
    ax1.set_title ( 'Training and Validation Accuracy' )

    ax2.plot ( epochs_range, loss, label='Training Loss' )
    ax2.plot ( epochs_range, val_loss, label='Validation Loss' )
    ax2.legend ( loc='upper right' )
    ax2.set_title ( 'Training and Validation Loss' )
    plt.show ()

def print_quick_stats(model, v_data, verb = 1):
    Scores = model.evaluate(v_data, verbose=verb)
    print("")
    print("Model Quick Stats")
    print('Model Test Loss:', Scores[0])
    print('Model Test Accuracy:', Scores[1])
    print("")

def run_model_1(epochs, outputs, img_size, train_generator, val_generator, verb = 1):
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
    model.add(Dense(outputs, activation="softmax"))
    model.summary()

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(train_generator, epochs=epochs, validation_data=val_generator, verbose=verb)

    return model, history

def run_model_pretrained_xception(epochs, outputs, img_size, train_generator, val_generator, verb = 1):
    base_model = keras.applications.xception.Xception(weights='imagenet', include_top=False, input_shape=(img_size, img_size, 3))
    base_model.trainable = False
    model = keras.Sequential([base_model,
                              keras.layers.GlobalAveragePooling2D(),
                              keras.layers.Dropout(0.2),
                              keras.layers.Dense(outputs, activation="softmax")
                              ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(train_generator, epochs=epochs, validation_data=val_generator, verbose=verb)

    return model, history

def run_model_pretrained_mobilenetv2(epochs, outputs, img_size,train_generator, val_generator, verb = 1):
    base_model = keras.applications.mobilenet_v2.MobileNetV2(input_shape=(img_size, img_size, 3), include_top=False,
                                                             weights="imagenet")
    base_model.trainable = False
    model = keras.Sequential([base_model,
                              keras.layers.GlobalAveragePooling2D(),
                              keras.layers.Dropout(0.2),
                              keras.layers.Dense(outputs, activation="softmax")
                              ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(train_generator, epochs=epochs, validation_data=val_generator, verbose = verb)

    return model, history


# "mobilenet.h5"
# "Kmodel.h5"
# "xceptionnet.h5"

epochs = 100
outputs = 5
img_size = 224
train_path = get_root_drive()
pred_path = get_root_drive_predict()
train_gen, val_gen = get_data_generators_randomized(img_size=img_size, images_path=train_path)

clean_files(train_path, 1)
clean_files(pred_path, 1)

#m,h = run_model_pretrained_mobilenetv2(epochs, outputs, img_size, train_gen, val_gen, 2)
#save_model(m, "mobilenet224.h5")
#print_quick_stats(m, val_gen)
#plot_history(h)

#m,h = run_model_pretrained_xception(epochs,outputs, img_size, train_gen, val_gen, 2)
#save_model(m, "xceptionnet224-Randomized.h5")
#print_quick_stats(m, val_gen)
#plot_history(h)

#m,h = run_model_1(epochs,outputs, img_size, train_gen, val_gen, 2)
#save_model(m, "Kmodel.h5")
#print_quick_stats(m, val_gen)
#plot_history(h)


#m = load_saved_model("xceptionnet224.h5")
#pimages = get_predict_images(pred_path, img_size)
#model_predict(m, pimages, "XCeptionNet-224")

#m = load_saved_model("xceptionnet.h5")
#pimages = get_predict_images(pred_path, img_size)
#model_predict(m, pimages, "XCeptionNet")

#m = load_saved_model("mobilenet.h5")
#pimages = get_predict_images(pred_path, img_size)
#model_predict(m, pimages, "mobilenet")

m = load_saved_model("xceptionnet224-Randomized.h5")
#plot_model(m, to_file='xceptionnet.png')
pimages = get_predict_images(pred_path, img_size)
model_predict(m, pimages, "xceptionnet224-Randomized.h5")




