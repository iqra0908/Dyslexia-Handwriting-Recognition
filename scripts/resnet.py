import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50, EfficientNetB0
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import zipfile
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict

# This function takes a byte string representing an image, preprocesses it (resize and convert to RGB),
# and returns a normalized numpy array.
def preprocess_image(image):
    img = cv2.imdecode(np.frombuffer(image, np.uint8), cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (128, 128))
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    img = img / 255.0
    return img

# This function takes a path to a zip file containing images, subfolder names in the zip file, 
# and a number of samples to load from each subfolder. It loads the images, preprocesses them, 
# and returns numpy arrays of images and their labels.
def load_images_labels(zip_file, subfolders, num_samples=10):
    images, labels = [], []
    count = defaultdict(lambda: defaultdict(int))
    with zipfile.ZipFile(zip_file) as archive:
        for subfolder in subfolders:
            for filename in archive.namelist():
                if 'png' in filename and not filename.endswith('/') and ('-' in filename or '_' in filename):
                    if '-' in filename:
                        label_name = filename.split('/')[-1].split('-')[0]
                    elif '_' in filename:
                        label_name = filename.split('/')[-1].split('_')[0]
                    else:
                        continue
                    if count[label_name][subfolder] < num_samples:
                        img_data = archive.read(filename)
                        img = preprocess_image(img_data)
                        images.append(img)
                        labels.append(label_name)
                        count[label_name][subfolder] += 1
                    if all(count[label_name][sf] >= num_samples for sf in subfolders for label_name in count):
                        break

    print(np.unique(labels))
    return np.array(images), np.array(labels)

# This class provides a static method to create a classification model based on a given base model type.
# It also provides a static method to create a detection model.
class ClassifierModel:

    @staticmethod
    def create_classification_model(input_shape, num_classes, base_model_type='ResNet50'):
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

# This is the main function that loads the data, preprocesses the images, creates a classification model,
# trains and evaluates the model, and saves it to a file.
def main():
    zip_file = 'Gambo.zip'
    train_subfolders = ['Train/Normal', 'Train/Reversal']
    test_subfolders = ['Test/Normal', 'Test/Reversal']
    train_images, train_labels = load_images_labels(zip_file, train_subfolders, num_samples=500)
    test_images, test_labels = load_images_labels(zip_file, test_subfolders, num_samples=100)

    train_images, val_images, train_labels, val_labels = train_test_split(train_images, train_labels, test_size=0.2, random_state=42)

    le = LabelEncoder()
    train_labels = le.fit_transform(train_labels)
    val_labels = le.transform(val_labels)
    test_labels = le.transform(test_labels)

    unique_labels = np.unique(train_labels)
    num_classes = len(unique_labels)

    input_shape = (128, 128, 3)

    data_gen = ImageDataGenerator(rotation_range=15, width_shift_range=0.1, height_shift_range=0.1, zoom_range=0.1, horizontal_flip=True, vertical_flip=True)

    train_data_gen = data_gen.flow(train_images, train_labels)
    val_data_gen = data_gen.flow(val_images, val_labels)

    classification_model = ClassifierModel.create_classification_model(input_shape, num_classes)
    classification_model.compile(optimizer=Adam(lr=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    callbacks = [EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)]

    history = classification_model.fit(train_data_gen, epochs=100, validation_data=val_data_gen, callbacks=callbacks)

    val_loss, val_acc = classification_model.evaluate(val_data_gen)
    print(f"Validation loss: {val_loss}, Validation accuracy: {val_acc}")

    test_loss, test_acc = classification_model.evaluate(test_images, test_labels)
    print(f"Test loss: {test_loss}, Test accuracy: {test_acc}")

    classification_model.save('./models/resnet50_model.h5')


if __name__ == '__main__':
    main()
