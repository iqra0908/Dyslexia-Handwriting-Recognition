import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import zipfile
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
import pickle

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

# This is the main function that loads the data, preprocesses the images, trains an SVM model,
# and evaluates the model.
def main():
    zip_file = './data/Gambo.zip'
    train_subfolders = ['Train/Normal', 'Train/Reversal']
    test_subfolders = ['Test/Normal', 'Test/Reversal']
    train_images, train_labels = load_images_labels(zip_file, train_subfolders, num_samples=10)
    test_images, test_labels = load_images_labels(zip_file, test_subfolders, num_samples=2)

    train_images, val_images, train_labels, val_labels = train_test_split(train_images, train_labels, test_size=0.2, random_state=42)

    le = LabelEncoder()
    train_labels = le.fit_transform(train_labels)
    val_labels = le.transform(val_labels)
    test_labels = le.transform(test_labels)

    unique_labels = np.unique(train_labels)

    # Flatten the images to 1D arrays
    train_images = train_images.reshape(train_images.shape[0], -1)
    val_images = val_images.reshape(val_images.shape[0], -1)
    test_images = test_images.reshape(test_images.shape[0], -1)

    # Train the SVM model
    svm_model = SVC(kernel='linear', C=1, gamma='auto')
    svm_model.fit(train_images, train_labels)
    
    # Evaluate the model on the validation set
    val_preds = svm_model.predict(val_images)
    val_acc = accuracy_score(val_labels, val_preds)
    print(f"Validation accuracy: {val_acc}")
    
    y_pred = svm_model.predict(test_images)
    acc = accuracy_score(test_labels, y_pred)
    print(f"Test accuracy: {acc}")
    
    # Save SVM model using pickle
    with open('./models/svm_model.pkl', 'wb') as f:
        pickle.dump(svm_model, f)
if __name__ == '__main__':
    main()
