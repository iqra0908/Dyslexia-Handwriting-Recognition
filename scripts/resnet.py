import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf
import zipfile
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict

def preprocess_image(image):
    img = cv2.imdecode(np.frombuffer(image, np.uint8), cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (128, 128)) 
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    img = img / 255.0
    return img

def load_images_labels(zip_file, subfolders, num_samples=10):
    images, labels = [], []
    count = defaultdict(lambda: defaultdict(int))
    with zipfile.ZipFile(zip_file) as archive:
        for subfolder in subfolders:
            for filename in archive.namelist():
                if 'png' in filename and not filename.endswith('/'):
                    if '-' in filename:
                        label_name = filename.split('/')[-1].split('-')[0]
                    elif '_' in filename:
                        label_name = filename.split('/')[-1].split('_')[0]
                    else:
                        break
                    if count[label_name][subfolder] < num_samples:
                        img_data = archive.read(filename)
                        img = preprocess_image(img_data)
                        images.append(img)
                        labels.append(label_name)
                        count[label_name][subfolder] += 1
                    if all(count[label_name][sf] >= num_samples for sf in subfolders for label_name in count):
                        break
                    
    print(labels)
    return np.array(images), np.array(labels)




# Modify the num_samples parameter to 50

zip_file = './data/Gambo.zip'
subfolders = {"Train/Normal": 1, "Train/Reversal": 2}
train_images, train_labels = load_images_labels(zip_file, subfolders, num_samples=2)

subfolders = {"Test/Normal": 1, "Test/Reversal": 2}
test_images, test_labels = load_images_labels(zip_file, subfolders, num_samples=1)

train_images, val_images, train_labels, val_labels = train_test_split(train_images, train_labels, test_size=0.3, random_state=42, stratify=train_labels)

# Convert string labels to numerical labels
le = LabelEncoder()
train_labels = le.fit_transform(train_labels)
val_labels = le.transform(val_labels)
test_labels = le.transform(test_labels)

unique_labels = np.unique(train_labels)
num_classes = len(unique_labels)

# 3. Choose a model architecture
def create_classification_model(input_shape, base_model_type='ResNet50'):
    if base_model_type == 'ResNet50':
        base_model = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
    elif base_model_type == 'EfficientNetB0':
        base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=input_shape)
    else:
        raise ValueError('Invalid base_model_type. Choose from ["ResNet50", "EfficientNetB0"].')
    
    for layer in base_model.layers:
        layer.trainable = False
    
    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    
    return model

input_shape = (128, 128, 3)  # Changed to 128x128

data_gen = ImageDataGenerator(rotation_range=15, width_shift_range=0.1, height_shift_range=0.1, zoom_range=0.1, horizontal_flip=True, vertical_flip=True)  # Added horizontal_flip and vertical_flip

train_data_gen = data_gen.flow(train_images, train_labels)
val_data_gen = data_gen.flow(val_images, val_labels)

classification_model = create_classification_model(input_shape)
classification_model.compile(optimizer=Adam(lr=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])


callbacks = [EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)]  # Increased patience to 15

history = classification_model.fit(train_data_gen, epochs=10, validation_data=val_data_gen, callbacks=callbacks)  # Increased epochs to 30

# 5. Evaluation and fine-tuning
# Evaluate the classification model on the validation set
val_loss, val_acc = classification_model.evaluate(val_data_gen)
print(f"Validation loss: {val_loss}, Validation accuracy: {val_acc}")

# 6. Testing and deployment
# Evaluate the classification model on the test set
test_loss, test_acc = classification_model.evaluate(test_images, test_labels)
print(f"Test loss: {test_loss}, Test accuracy: {test_acc}")

classification_model.save('./models/resnet50_model.h5')
print(train_labels, test_labels)
