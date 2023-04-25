import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import zipfile
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
import pickle
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score

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
    subfolders = ['Train/Normal', 'Train/Reversal']
    images, labels = load_images_labels(zip_file, subfolders, num_samples=50)

    le = LabelEncoder()
    labels = le.fit_transform(labels)

    unique_labels = np.unique(labels)

    # Flatten the images to 1D arrays
    images = images.reshape(images.shape[0], -1)

    # Split data into training and testing sets
    train_images, test_images, train_labels, test_labels = train_test_split(images, labels, test_size=0.2, random_state=42)

    # Train the SVM model
    svm_model = SVC(kernel='linear', C=1, gamma='auto', probability=True)  # Add probability=True
    svm_model.fit(train_images, train_labels)

    # Evaluate the model on the test set
    test_preds = svm_model.predict(test_images)
    test_acc = accuracy_score(test_labels, test_preds)

    # Print the test accuracy
    print(f"Test accuracy: {test_acc}")

    # Save the best SVM model using pickle
    with open('./app/models/svm_model.pkl', 'wb') as f:
        pickle.dump(svm_model, f)

if __name__ == '__main__':
    main()

