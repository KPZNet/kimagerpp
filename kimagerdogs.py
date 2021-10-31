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


def get_root_drive():
    root_path = "petimages"
    if 'COLAB_GPU' in os.environ:
        root_path = '/content/drive/MyDrive/Colab Notebooks/petimages'
    return root_path


epochs = 50
labels = ['cat', 'dog']
#img_size = 224
img_size = 180
image_size = (180, 180)


def isgpu2():
    from tensorflow.python.client import device_lib
    print(device_lib.list_local_devices())


def get_data(data_dir, num=500):
    data = []
    for label in labels:
        path = os.path.join(data_dir, label)
        class_num = labels.index(label)
        lstims = os.listdir ( path )
        lstims = lstims[:num]
        for img in lstims:
            try:
                img_arr = cv2.imread(os.path.join(path, img))[...,::-1] #convert BGR to RGB format
                resized_arr = cv2.resize(img_arr, (img_size, img_size)) # Reshaping images to preferred size
                data.append([resized_arr, class_num])
            except Exception as e:
                print(e)
    return np.array(data)

isgpu2()

root_path = get_root_drive()

data = get_data(root_path,2000)

def split_data(d, indx, num=2000):
    a = d[np.in1d(d[:, 1], indx)]
    a = a[:num,:]
    a_train, a_val = train_test_split(a,test_size=0.20, random_state=42)
    return a_train, a_val

a_train, a_val = split_data(data,0)
b_trian, b_val = split_data(data,1)

train = np.concatenate( (a_train, b_trian) )
val = np.concatenate( (a_val, b_val) )

print("Training Samples: ", len(train))
print("Validation Samples: ",len(val))

x_train = []
y_train = []
x_val = []
y_val = []

for feature, label in train:
  x_train.append(feature)
  y_train.append(label)

for feature, label in val:
  x_val.append(feature)
  y_val.append(label)

# Normalize the data
x_train = np.array(x_train) / 255
x_val = np.array(x_val) / 255

x_train.reshape(-1, img_size, img_size, 1)
y_train = np.array(y_train)
y_trainc = np_utils.to_categorical(y_train, 2)

x_val.reshape(-1, img_size, img_size, 1)
y_valb = np.array(y_val)
y_valc = np_utils.to_categorical(y_valb, 2)


def make_model_2(input_shape, num_classes):

    data_augmentation = keras.Sequential(
    [layers.RandomFlip("horizontal"), layers.RandomRotation(0.1),])

    inputs = keras.Input(shape=input_shape)
    # Image augmentation block
    x = data_augmentation(inputs)

    # Entry block
    x = layers.Rescaling(1.0 / 255)(x)
    x = layers.Conv2D(32, 3, strides=2, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.Conv2D(64, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    previous_block_activation = x  # Set aside residual

    for size in [128, 256, 512, 728]:
        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(size, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(size, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

        # Project residual
        residual = layers.Conv2D(size, 1, strides=2, padding="same")(
            previous_block_activation
        )
        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    x = layers.SeparableConv2D(1024, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.GlobalAveragePooling2D()(x)
    if num_classes == 2:
        activation = "sigmoid"
        units = 1
    else:
        activation = "softmax"
        units = num_classes

    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(units, activation=activation)(x)
    return keras.Model(inputs, outputs)



datagen = ImageDataGenerator(
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
        vertical_flip=False)  # randomly flip images


datagen.fit(x_train)

def make_model_1():
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
    model.add(Dense(2, activation="softmax"))
    
    model.summary()
    return model

model = make_model_1()


def run_model_1():
    learning_rate = 0.000001
    opti = adam_v2.Adam(learning_rate=learning_rate, decay=learning_rate/epochs)
    #model.compile(optimizer = opti , loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True) , metrics = ['accuracy'])
    model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
    
    #history = model.fit(x_train,y_train,epochs = epochs, validation_data = (x_val, y_valb), verbose=2)
    history = model.fit(x_train,y_trainc,epochs = epochs, validation_data = (x_val, y_valc), verbose=2)
    
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    
    epochs_range = range(epochs)
    
    plt.figure(figsize=(15, 15))
    plt.subplot(2, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')
    
    plt.subplot(2, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    #plt.show()
    
    
    predict_x=model.predict(x_val) 
    classes_x=np.argmax(predict_x,axis=1)
    predictions = classes_x.reshape(1,-1)[0]
    print(classification_report(y_val, predictions, target_names = ['INITIAL cat (Class 0)','dog (Class 1)' ]))

def run_model_2():
    model = make_model_2(input_shape=image_size + (3,), num_classes=2)
    #keras.utils.plot_model(model, show_shapes=True)

    callbacks = [
    keras.callbacks.ModelCheckpoint("save_at_{epoch}.h5"),
    ]
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    model.fit(
        x_train,y_train,epochs = epochs, validation_data = (x_val, y_valb), verbose=2,
    )
    predict_x=model.predict(x_val) 
    classes_x=np.argmax(predict_x,axis=1)
    predictions = classes_x.reshape(1,-1)[0]
    print(classification_report(y_val, predictions, target_names = ['INITIAL cat (Class 0)','dog (Class 1)' ]))


run_model_1()
#run_model_2()


