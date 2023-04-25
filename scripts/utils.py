import cv2
import numpy as np
import tensorflow as tf
from scripts.bounding_box import BoundingBox
from sklearn.svm import SVC
import pickle
import os

encoder_path = os.environ.get("LABEL_ENCODER_PATH")
#encoder_path = '/Users/iqraimtiaz/Documents/duke/Courses/540-DL/Dyslexia-Handwriting-Recognition/scripts/label_encoder.pkl'

with open(encoder_path, 'rb') as f:
    le = pickle.load(f)
        
def load_image(image_path):
    """
    Loads an image from a given path and returns it in RGB format.

    Args:
        image_path (str): Path of the image file to load.

    Returns:
        numpy.ndarray: The loaded image in RGB format.
    """
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.bitwise_not(image)
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    return image

def preprocess_image(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.bitwise_not(image)
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    return image

def preprocess_template_letter(template_letter):
    """
    Preprocesses the given template letter image by converting it to grayscale and performing thresholding.

    Args:
        template_letter (numpy.ndarray): The template letter image to preprocess.

    Returns:
        numpy.ndarray: The preprocessed template letter image.
    """
    template_letter = cv2.cvtColor(template_letter, cv2.COLOR_BGR2GRAY)
    _, template_letter = cv2.threshold(template_letter, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return template_letter

def sliding_window(image, window_size, step_size, model):
    """
    Applies a sliding window to the given image with the given window size and step size, and uses the given model to
    classify each window as a character or not.

    Args:
        image (numpy.ndarray): The image to apply the sliding window on.
        window_size (int): The size of the sliding window.
        step_size (int): The step size of the sliding window.
        model (tensorflow.keras.models.Model): The model to use for classification.

    Returns:
        list: A list of detected windows that are classified as characters, in the format (x, y, w, h, predicted_class, score).
    """
    detections = []
    for y in range(0, image.shape[0] - window_size, step_size):
        for x in range(0, image.shape[1] - window_size, step_size):
            patch = image[y:y+window_size, x:x+window_size]
            resized_patch = cv2.resize(patch, (128, 128), interpolation=cv2.INTER_AREA).astype('float32')
            resized_patch = resized_patch / 255.0
            if isinstance(model, SVC):  # If the model is an SVM model
                flattened_patch = resized_patch.reshape(1, -1)
                prediction = model.predict(flattened_patch)[0]
            else:
                prediction = model.predict(resized_patch[np.newaxis, ...])[0]
                
            predicted_class = np.argmax(prediction)

            if isinstance(model, SVC):  # If the model is an SVM model
                if predicted_class != 0:
                    detections.append((x, y, window_size, window_size, predicted_class, prediction[predicted_class]))
            else:
                if predicted_class != 0 and prediction[predicted_class] > 0.5:
                    detections.append((x, y, window_size, window_size, predicted_class, prediction[predicted_class]))
        
    return detections

'''This function performs non-maximum suppression on a set of bounding boxes and 
corresponding scores. It removes any boxes that overlap with a higher-scoring box 
above a certain threshold and returns the remaining boxes and scores.'''
def non_max_suppression(boxes, classes, scores, threshold):
    indices = tf.image.non_max_suppression(boxes, scores, max_output_size=boxes.shape[0], iou_threshold=threshold)
    return [boxes[i] for i in indices], [classes[i] for i in indices], [scores[i] for i in indices]

'''This function draws bounding boxes and corresponding characters on an input image and 
returns the resulting image. The boxes are colored green and labeled with the corresponding 
character, which is selected based on the maximum score for each box.'''
def display_detections(image, nms_boxes, nms_classes, nms_scores, characters):
    output_image = image.copy()
    
    for (x, y, w, h), predicted_class, score in zip(nms_boxes, nms_classes, nms_scores):
        letter = str(le.inverse_transform([predicted_class])[0])
        cv2.rectangle(output_image, (x, y), (x+w, y+h), (0, 0, 255), 2)
        cv2.putText(output_image, letter, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2, cv2.LINE_AA)
    return output_image

# This function generate a list characters containing all uppercase and lowercase letters 
# as well as digits.
def get_characters():
    
    '''# Generate a list of uppercase and lowercase letters
    uppercase_letters = list(string.ascii_uppercase)
    lowercase_letters = list(string.ascii_lowercase)

    # Generate a list of digits
    digits = list(string.digits)

    # Combine the lists to create a list of all characters
    characters = uppercase_letters + lowercase_letters + digits'''
    characters = ['1', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K','L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'b', 'c', 'd','e', 'f', 'g', 'j', 'k', 'l', 'n', 'p', 'q', 'r', 's', 'z']
    return characters

def get_letters(nms_boxes, nms_classes, nms_scores, characters):
    letters = []
    classes = []
    for (x, y, w, h), predicted_class, score in zip(nms_boxes, nms_classes, nms_scores):
        letter = str(le.inverse_transform([predicted_class])[0])
        letters.append(letter)
        classes.append(predicted_class)
    print(letters,classes)
    return letters