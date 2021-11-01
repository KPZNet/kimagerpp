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
    root_path = "flowers"
    if 'COLAB_GPU' in os.environ:
        root_path = '/content/drive/MyDrive/Colab Notebooks/flowers'
    return root_path


epochs = 25
labels = ['daisy', 'rose', 'tulip', 'sunflower']
img_size = 224


def isgpu2():
    from tensorflow.python.client import device_lib
    print(device_lib.list_local_devices())


def get_data(data_dir):
    data = []
    for label in labels:
        path = os.path.join(data_dir, label)
        class_num = labels.index(label)
        lstims = os.listdir ( path )
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

data = get_data(root_path)

def split_data(d, indx):
    a = d[np.in1d(d[:, 1], indx)]
    a = a[:500,:]
    a_train, a_val = train_test_split(a,test_size=0.20, random_state=42)
    return a_train, a_val

a_train, a_val = split_data(data,0)
b_trian, b_val = split_data(data,1)
c_trian, c_val = split_data(data,2)
d_trian, d_val = split_data(data,3)


train = np.concatenate( (a_train, b_trian, c_trian, d_trian) )
val = np.concatenate( (a_val, b_val, c_val, d_val) )

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
y_trainc = np_utils.to_categorical(y_train, 4)

x_val.reshape(-1, img_size, img_size, 1)
y_valb = np.array(y_val)
y_valc = np_utils.to_categorical(y_valb, 4)

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

def get_model_a():
    model = Sequential()
    model.add(Conv2D(32,3,padding="same", activation="relu", input_shape=(224,224,3)))
    model.add(MaxPool2D())
    model.add(Conv2D(32, 3, padding="same", activation="relu"))
    model.add(MaxPool2D())
    model.add(Conv2D(64, 3, padding="same", activation="relu"))
    model.add(MaxPool2D())
    model.add(Dropout(0.4))
    model.add(Flatten())
    model.add(Dense(128,activation="relu"))
    model.add(Dense(4, activation="softmax"))
    model.summary()
    return model

def get_model_b():
    model = keras.Sequential([
            layers.Conv2D(32, (5,5), activation = 'relu', input_shape = (224, 224, 3)),
            layers.MaxPooling2D(2),
            layers.Conv2D(64, (3,3), activation = 'relu'),
            layers.MaxPooling2D(2),
            layers.Conv2D(128, (3,3), activation = 'relu'),
            layers.Conv2D(128, (3,3), activation = 'relu'),
            layers.MaxPooling2D(2),
            layers.Conv2D(256, (3,3), activation = 'relu'),
            layers.Conv2D(256, (3,3), activation = 'relu', padding = "SAME"),
            layers.Flatten(),
            layers.Dense(64, activation = 'relu'),
            layers.Dropout(0.4),
            layers.Dense(128, activation = 'relu'),
            layers.Dropout(0.4),
            layers.Dense(4, activation = 'softmax')
            ])
    return model

model = get_model_b()

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
print(classification_report(y_val, predictions, target_names = ['INITIAL Daisy (Class 0)','Rose (Class 1)','Tulip (Class 2)','Sunflower (Class 3)' ]))

target_names = ['Daisy', 'Rose', 'Tulip', 'Sunflower']
#print(classification_report(y_valb, classes_x, target_names=target_names))


#Scores = model.evaluate(x_val, y_valb, verbose=2)
#print('Validation loss:', Scores[0])
#print('Validation accuracy:', Scores[1])

#Scores = model.evaluate(x_train, y_train, verbose=2)
#print('Training loss:', Scores[0])
#print('Training accuracy:', Scores[1])