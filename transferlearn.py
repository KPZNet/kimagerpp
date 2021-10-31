
def transfer_learner():
    global model, learning_rate, opti, history, predict_x, classes_x, predictions, acc, val_acc, loss, val_loss, epochs_range
    base_model = keras.applications.mobilenet_v2.MobileNetV2(input_shape=(224, 224, 3), include_top=False,
                                                             weights="imagenet")
    base_model.trainable = False
    model = keras.Sequential([base_model,
                              keras.layers.GlobalAveragePooling2D(),
                              keras.layers.Dropout(0.2),
                              keras.layers.Dense(2, activation="softmax")
                              ])
    learning_rate = 0.000001
    opti = adam_v2.Adam(learning_rate=learning_rate, decay=learning_rate / epochs)
    model.compile(optimizer=opti,
                  loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    history = model.fit(x_train, y_train, epochs=epochs, validation_data=(x_val, y_val))
    predict_x = model.predict(x_val)
    classes_x = np.argmax(predict_x, axis=1)
    predictions = classes_x.reshape(1, -1)[0]
    print(classification_report(y_val, predictions, target_names=['PRETRAINED Daisy (Class 0)', 'Rose (Class 1)']))
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
    plt.show()
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
    plt.show()

