

# !pip install tensorflowjs
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.utils import image_dataset_from_directory
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers.experimental.preprocessing import RandomFlip , RandomRotation , RandomZoom
from tqdm import tqdm

from google.colab import drive
path= drive.mount('/content/gdrive')
train_datayol='/content/gdrive/MyDrive/bitirme/train'
test_datayol= '/content/gdrive/MyDrive/bitirme/test'
val_datayol= '/content/gdrive/MyDrive/bitirme/val'

BATCH_SIZE = 32
IMG_HEIGHT = 224
IMG_WIDTH = 224
epochs = 10

train_dataset = image_dataset_from_directory(train_datayol,
                                             shuffle=True,
                                             batch_size = BATCH_SIZE,
                                             label_mode= "categorical",
                                             color_mode="rgb",
                                             image_size=(IMG_HEIGHT,IMG_WIDTH))
validation_dataset = image_dataset_from_directory(val_datayol,
                                                  shuffle=True,
                                                  batch_size= BATCH_SIZE,
                                                  label_mode="categorical",
                                                  color_mode ="rgb",
                                                  image_size=(IMG_HEIGHT,IMG_WIDTH))
test_dataset = image_dataset_from_directory(test_datayol,
                                             shuffle=True,
                                             batch_size = BATCH_SIZE,
                                             label_mode="categorical",
                                             color_mode="rgb",
                                             image_size=(IMG_HEIGHT,IMG_WIDTH))
class_names = train_dataset.class_names
print(class_names)

plt.figure(figsize=(10, 10))
for images, labels in train_dataset.take(1):
  for i in range(9):
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(images[i].numpy().astype("uint8"))
    plt.title(class_names[np.argmax(labels[i])])
    plt.axis("off")

from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import ResNet50

# Define model
model = Sequential()

# Download ResNet50
pretrained_model= ResNet50(include_top=False,
                           input_shape=(224, 224, 3), # Input shape dimension
                           pooling='avg', # AvgPool
                           classes=3, # Total output
                           weights='imagenet')

# Disable train on ResNet50
for layer in pretrained_model.layers:
    layer.trainable=False

# ResNet50 Layers
model.add(pretrained_model)
# Flatten all data
model.add(Flatten())
# Fully connected layers with 512 neurons
model.add(Dense(512, activation='relu'))
# Output layers
model.add(Dense(3, activation='softmax'))

# Give summary
model.summary()

model.compile(optimizer=Adam(learning_rate=0.001),
                     loss='mse',
                     metrics=['accuracy'])

# Finally train this neural net. Ten epochs seems ok to me
history = model.fit(train_dataset, validation_data=validation_dataset, epochs=epochs,verbose=1)

labels = []
predictions = []

for X, y in tqdm(validation_dataset):
    for i in range(len(X)):
        result = model(np.expand_dims(X[i], axis=0))

        predictions.append(class_names[np.argmax(result)])
        labels.append(class_names[np.argmax(y[i])])

"""YAPAY SİNİR AĞI MODELİNİN KARMAŞIKLIK MATRİSİ

"""

from sklearn.metrics import ConfusionMatrixDisplay

ConfusionMatrixDisplay.from_predictions(labels, predictions, cmap='Blues')
plt.show()

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

print('Accuracy score : ', accuracy_score(labels, predictions))
print('Precision score : ', precision_score(labels, predictions, average='weighted'))
print('Recall score : ', recall_score(labels, predictions, average='weighted'))
print('F1 score : ', f1_score(labels, predictions, average='weighted'))

from sklearn.metrics import classification_report

print(classification_report(labels, predictions))

test_dataset.class_names
labels = []
predictions = []

for X, y in tqdm(test_dataset):
    for i in range(len(X)):
        result = model(np.expand_dims(X[i], axis=0))

        predictions.append(class_names[np.argmax(result)] == "Monkeypox")
        labels.append(test_dataset.class_names[np.argmax(y[i])] == "Monkeypox_augmented")
        predictions.append(class_names[np.argmax(result)] == "Rosacea")
        labels.append(test_dataset.class_names[np.argmax(y[i])] == "Rosacea_augmented")

from sklearn.metrics import ConfusionMatrixDisplay

ConfusionMatrixDisplay.from_predictions(labels, predictions, cmap='Blues')
plt.show()

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

print('Accuracy score : ', accuracy_score(labels, predictions))
print('Precision score : ', precision_score(labels, predictions, average='weighted'))
print('Recall score : ', recall_score(labels, predictions, average='weighted'))
print('F1 score : ', f1_score(labels, predictions, average='weighted'))

from sklearn.metrics import classification_report

print(classification_report(labels, predictions))

from sklearn.metrics import RocCurveDisplay

RocCurveDisplay.from_predictions(labels, predictions, pos_label=True)
plt.show()

test_accu = model.evaluate(test_dataset)
print("The testing accuracy is: ",test_accu[1]*100,"%")

pretrained_model.trainable = True
print("Number of layers in the base model: ", len(pretrained_model.layers))
# Fine-tune from this layer onwards
fine_tune_at = 100

# Freeze all the layers before the `fine_tune_at` layer
for layer in pretrained_model.layers[:fine_tune_at]:
  layer.trainable = False

model.compile(optimizer=Adam(learning_rate=0.001*3),
                              loss='mse',

                              metrics=["accuracy"]
                              )
model.summary()
len(model.trainable_variables)

fine_tune_epochs = 10
total_epochs = epochs + fine_tune_epochs

history_fine = model.fit( train_dataset,
                                        #initial_epochs, # history.epoch[-1],
                                         epochs= 5,
                                         validation_data= validation_dataset)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.ylabel('Accuracy')
plt.ylim([min(plt.ylim()),1])
plt.title('Training and Validation Accuracy')

plt.subplot(2, 1, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.ylabel('Cross Entropy')
plt.ylim([0,1.0])
plt.title('Training and Validation Loss')
plt.xlabel('epoch')
plt.show()

test_accu = model.evaluate(test_dataset)
print("The testing accuracy is: ",test_accu[1]*100,"%")

import tensorflowjs as tfjs
tfjs.converters.save_keras_model(model,'models')