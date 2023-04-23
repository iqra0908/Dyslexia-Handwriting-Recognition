import cv2
import numpy as np
import tensorflow as tf
from bounding_box import BoundingBox

def load_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.bitwise_not(image)
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    return image

def preprocess_template_letter(template_letter):
    template_letter = cv2.cvtColor(template_letter, cv2.COLOR_BGR2GRAY)
    _, template_letter = cv2.threshold(template_letter, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return template_letter

def sliding_window(image, window_size, step_size, model):
    detections = []
    for y in range(0, image.shape[0] - window_size, step_size):
        for x in range(0, image.shape[1] - window_size, step_size):
            patch = image[y:y+window_size, x:x+window_size]
            resized_patch = cv2.resize(patch, (128, 128), interpolation=cv2.INTER_AREA).astype('float32')
            resized_patch = resized_patch / 255.0
            prediction = model.predict(resized_patch[np.newaxis, ...])[0]
            predicted_class = np.argmax(prediction)

            if predicted_class != 0 and prediction[predicted_class] > 0.1:
                detections.append((x, y, window_size, window_size, predicted_class, prediction[predicted_class]))
    return detections

def non_max_suppression(boxes, scores, threshold):
    indices = tf.image.non_max_suppression(boxes, scores, max_output_size=boxes.shape[0], iou_threshold=threshold)
    return [boxes[i] for i in indices], [scores[i] for i in indices]

def display_detections(image, nms_boxes, nms_scores, characters):
    output_image = image.copy()
    for (x, y, w, h), score in zip(nms_boxes, nms_scores):
        letter = characters[int(score)-1]
        cv2.rectangle(output_image, (x, y), (x+w, y+h), (0, 0, 255), 2)
        cv2.putText(output_image, letter, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    return output_image
