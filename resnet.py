import os
import numpy as np
import cv2
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# 1. Data preparation
def load_images_labels(data_path, subfolders):
    images, labels = [], []
    for subfolder in subfolders:
        label = subfolders[subfolder]
        folder_path = os.path.join(data_path, subfolder)
        for filename in os.listdir(folder_path):
            img_path = os.path.join(folder_path, filename)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (128, 128))
            img = img / 255.0
            images.append(img)
            labels.append(label)
    return np.array(images), np.array(labels)

train_path = './data/Gambo/Train'
test_path = './data/Gambo/Test'
subfolders = {"corrected": 0, "normal": 1, "reversal": 2}

train_images, train_labels = load_images_labels(train_path, subfolders)
test_images, test_labels = load_images_labels(test_path, subfolders)

# 2. Divide the dataset
train_images, val_images, train_labels, val_labels = train_test_split(train_images, train_labels, test_size=0.1, random_state=42)

# 3. Choose a model architecture
def create_classification_model(input_shape):
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
    for layer in base_model.layers:
        layer.trainable = False
    
    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(3, activation='softmax')
    ])
    
    return model

input_shape = (128, 128, 1)
classification_model = create_classification_model(input_shape)

# 4. Train the models (classification model)
# ImageDataGenerator for data augmentation
data_gen = ImageDataGenerator(rotation_range=15, width_shift_range=0.1, height_shift_range=0.1, zoom_range=0.1)

train_data_gen = data_gen.flow(train_images, train_labels)
val_data_gen = data_gen.flow(val_images, val_labels)

classification_model.compile(optimizer=Adam(lr=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

callbacks = [EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)]

history = classification_model.fit(train_data_gen, epochs=50, validation_data=val_data_gen, callbacks=callbacks)

# 5. Evaluation and fine-tuning
# Evaluate the classification model on the validation set
val_loss, val_acc = classification_model.evaluate(val_data_gen)
print(f"Validation loss: {val_loss}, Validation accuracy: {val_acc}")

# 6. Testing and deployment
# Evaluate the classification model on the test set
test_loss, test_acc = classification_model.evaluate(test_images, test_labels)
print(f"Test loss: {test_loss}, Test accuracy: {test_acc}")
