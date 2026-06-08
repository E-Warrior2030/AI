

#Getting the Data from Google Drive
from google.colab import drive
drive.mount('/content/drive')

#Just making sure that the mounting worked
import os
data_dir = '/content/drive/MyDrive/ML Capstone Project/Data/Raw/chest_xray'
print(os.listdir(data_dir))

#Here we make directories easier to access
train_dir = os.path.join(data_dir, 'train')
val_dir = os.path.join(data_dir, 'val')
test_dir = os.path.join(data_dir, 'test')

print(os.listdir(train_dir))

#Knowing the numbers of photos in every directory to make sure there isnt any imbalances


#Training
norm_train_dir = os.path.join(train_dir, 'NORMAL')
print('total training normal images:', len(os.listdir(norm_train_dir)))
pneumonia_train_dir = os.path.join(train_dir, 'PNEUMONIA')
print('total training pneumonia images:', len(os.listdir(pneumonia_train_dir)))
print("\n")

#Val
norm_val_dir = os.path.join(val_dir, 'NORMAL')
print('total val normal images:', len(os.listdir(norm_val_dir)))
pneumonia_val_dir = os.path.join(val_dir, 'PNEUMONIA')
print('total val pneumonia images:', len(os.listdir(pneumonia_val_dir)))
print("\n")

#Test
norm_test_dir = os.path.join(test_dir, 'NORMAL')
print('total test normal images:', len(os.listdir(norm_test_dir)))
pneumonia_test_dir = os.path.join(test_dir, 'PNEUMONIA')
print('total test pneumonia images:', len(os.listdir(pneumonia_test_dir)))

#Now we are building a CNN

#All what we are doing we have already done in 01_ml_models. But with few changes due to this being a cnn not ml
from PIL import Image
import numpy as np

IMG_SIZE = 64

def load_images_cnn(folder, label):
    images = []
    labels = []
    for filename in os.listdir(folder):
        if filename.endswith('.jpeg'):
            img = Image.open(os.path.join(folder, filename)).convert('L')
            img = img.resize((IMG_SIZE, IMG_SIZE))
            img = np.array(img).reshape(64, 64, 1)
            images.append(img)
            labels.append(label)
    return images, labels

X_normal, y_normal = load_images_cnn(norm_train_dir, 0)
X_pneumonia, y_pneumonia = load_images_cnn(pneumonia_train_dir, 1)

X = np.array(X_normal + X_pneumonia)
y = np.array(y_normal + y_pneumonia)

print(X.shape)
print(y.shape)

X = X / 255.0 #Normalization

from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

print(X_train.shape)
print(y_train.shape)
print(X_val.shape)
print(y_val.shape)

X_normal_test, y_normal_test = load_images_cnn(norm_test_dir, 0)
X_pneumonia_test, y_pneumonia_test = load_images_cnn(pneumonia_test_dir, 1)

X_test = np.array(X_normal_test + X_pneumonia_test) / 255.0
y_test = np.array(y_normal_test + y_pneumonia_test)

print(X_test.shape)
print(y_test.shape)

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

model = keras.Sequential([
    # Block 1
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(64, 64, 1)),
    layers.MaxPooling2D(2, 2),

    # Block 2
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D(2, 2),

    # Flatten + Dense
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(1, activation='sigmoid')  # binary output
])

model.summary()

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

history1 = model.fit(X_train, y_train,
                    epochs=10,
                    validation_data=(X_val, y_val))

import matplotlib.pyplot as plt

# Plot training and validation accuracy

plt.figure(figsize=(12, 4))

# Accuracy
plt.subplot(1, 2, 1)
plt.plot(history1.history['accuracy'], label='Train')
plt.plot(history1.history['val_accuracy'], label='Val')
plt.title('Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# Loss
plt.subplot(1, 2, 2)
plt.plot(history1.history['loss'], label='Train')
plt.plot(history1.history['val_loss'], label='Val')
plt.title('Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.show()

model_improved = keras.Sequential([
    # Block 1
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(64, 64, 1)),
    layers.BatchNormalization(),
    layers.MaxPooling2D(2, 2),

    # Block 2
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.BatchNormalization(  ),
    layers.MaxPooling2D(2, 2),

    # Flatten + Dense
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(1, activation='sigmoid')
])

model_improved.summary()

model_improved.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

history2 = model_improved.fit(X_train, y_train,
                    epochs=10,
                    validation_data=(X_val, y_val))

plt.figure(figsize=(12, 4))

# Accuracy
plt.subplot(1, 2, 1)
plt.plot(history2.history['accuracy'], label='Train')
plt.plot(history2.history['val_accuracy'], label='Val')
plt.title('Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# Loss
plt.subplot(1, 2, 2)
plt.plot(history2.history['loss'], label='Train')
plt.plot(history2.history['val_loss'], label='Val')
plt.title('Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.show()

model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0001),
              loss='binary_crossentropy',
              metrics=['accuracy'])

history3 = model.fit(X_train, y_train,
                    epochs=10,
                    validation_data=(X_val, y_val))

import matplotlib.pyplot as plt

# Plot training and validation accuracy

plt.figure(figsize=(12, 4))

# Accuracy
plt.subplot(1, 2, 1)
plt.plot(history3.history['accuracy'], label='Train')
plt.plot(history3.history['val_accuracy'], label='Val')
plt.title('Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# Loss
plt.subplot(1, 2, 2)
plt.plot(history3.history['loss'], label='Train')
plt.plot(history3.history['val_loss'], label='Val')
plt.title('Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.show()

model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.01),
              loss='binary_crossentropy',
              metrics=['accuracy'])

history4 = model.fit(X_train, y_train,
                    epochs=10,
                    validation_data=(X_val, y_val))

import matplotlib.pyplot as plt

# Plot training and validation accuracy

plt.figure(figsize=(12, 4))

# Accuracy
plt.subplot(1, 2, 1)
plt.plot(history4.history['accuracy'], label='Train')
plt.plot(history4.history['val_accuracy'], label='Val')
plt.title('Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# Loss
plt.subplot(1, 2, 2)
plt.plot(history4.history['loss'], label='Train')
plt.plot(history4.history['val_loss'], label='Val')
plt.title('Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.show()

def load_images_rgb(folder, label):
    images = []
    labels = []
    for filename in os.listdir(folder):
        if filename.endswith('.jpeg'):
            img = Image.open(os.path.join(folder, filename)).convert('RGB')
            img = img.resize((IMG_SIZE, IMG_SIZE))
            img = np.array(img).reshape(64, 64, 3)
            images.append(img)
            labels.append(label)
    return images, labels

X_normal_rgb, y_normal_rgb = load_images_rgb(norm_train_dir, 0)
X_pneumonia_rgb, y_pneumonia_rgb = load_images_rgb(pneumonia_train_dir, 1)

X_rgb = np.array(X_normal_rgb + X_pneumonia_rgb) / 255.0
y_rgb = np.array(y_normal_rgb + y_pneumonia_rgb)

X_train_rgb, X_val_rgb, y_train_rgb, y_val_rgb = train_test_split(X_rgb, y_rgb, test_size=0.2, random_state=42)

X_normal_test_rgb, y_normal_test_rgb = load_images_rgb(norm_test_dir, 0)
X_pneumonia_test_rgb, y_pneumonia_test_rgb = load_images_rgb(pneumonia_test_dir, 1)
X_test_rgb = np.array(X_normal_test_rgb + X_pneumonia_test_rgb) / 255.0
y_test_rgb = np.array(y_normal_test_rgb + y_pneumonia_test_rgb)

print(X_train_rgb.shape)

from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models

# Load pretrained MobileNetV2 without top layer
base_model = MobileNetV2(input_shape=(64, 64, 3), include_top=False, weights='imagenet')
base_model.trainable = False  # Freeze base layers

# Build new head
inputs = layers.Input(shape=(64, 64, 3))
x = base_model(inputs, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(128, activation='relu')(x)
x = layers.Dropout(0.3)(x)
outputs = layers.Dense(1, activation='sigmoid')(x)

transfer_model = models.Model(inputs, outputs)
transfer_model.summary()

transfer_model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0001),
                       loss='binary_crossentropy',
                       metrics=['accuracy'])

history_transfer = transfer_model.fit(X_train_rgb, y_train_rgb,
                                      epochs=10,
                                      validation_data=(X_val_rgb, y_val_rgb))

# Unfreeze top layers of base model
base_model.trainable = True

# Freeze all layers except last 20
for layer in base_model.layers[:-20]:
    layer.trainable = False

# Recompile with lower learning rate
transfer_model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.00001),
                       loss='binary_crossentropy',
                       metrics=['accuracy'])

history_finetune = transfer_model.fit(X_train_rgb, y_train_rgb,
                                      epochs=10,
                                      validation_data=(X_val_rgb, y_val_rgb))

# Best CNN (baseline)
loss, acc = model.evaluate(X_test, y_test)
print(f"CNN Test Accuracy: {acc:.4f}")

# Best Phase 1 model was Logistic Regression — but that's in notebook 1
# Just note the Phase 1 test accuracy was 0.7532

#Testing
from PIL import Image
import numpy as np

def predict_image(img_path):
    img = Image.open(img_path).convert('L')
    img = img.resize((64, 64))
    img = np.array(img).reshape(1, 64, 64, 1) / 255.0

    prediction = model.predict(img)[0][0]

    if prediction > 0.5:
        print(f"PNEUMONIA (confidence: {prediction:.2%})")
    else:
        print(f"NORMAL (confidence: {1-prediction:.2%})")

# Test on a random image
test_img_path = '/content/drive/MyDrive/ml_test_image.jpg'
predict_image(test_img_path)

test_img_path1 = '/content/drive/MyDrive/ML_R_Image.jpg'
predict_image(test_img_path1)
test_img_path2 = '/content/drive/MyDrive/ML_P_Image.jpg'
predict_image(test_img_path2)