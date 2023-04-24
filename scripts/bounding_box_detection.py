import cv2
import numpy as np
from tensorflow.keras.models import load_model
from scripts.bounding_box import BoundingBox
import scripts.utils as utils

'''
This function performs manual detection of handwritten letters in an image using a trained model. 
It loads the trained model, reads the input image, and sets up a bounding box to select a 
template letter from the input image. It then creates four trackbars to allow the user to 
adjust the position and size of the bounding box. When the user selects a bounding box, 
the function crops the template letter from the input image, preprocesses it, and performs 
sliding window detection on the input image using the trained model. 
It then performs non-maximum suppression to remove overlapping detections and displays the 
final results in a separate window.'''

def object_detection(image, template_letters):
    model = load_model('./models/resnet50_model.h5')
    all_detected_characters = []

    for template_letter in template_letters:
        # Preprocess the template letter and perform sliding window detection
        template_letter = utils.preprocess_template_letter(template_letter)
        window_size = template_letter.shape[0]
        step_size = int(window_size / 2)
        detections = utils.sliding_window(image, window_size, step_size, model)

        if len(detections) > 0:
            boxes, classes, scores = zip(*[(det[:4], det[4], det[-1]) for det in detections])
            nms_boxes, nms_classes, nms_scores = utils.non_max_suppression(np.array(boxes), np.array(classes), np.array(scores), threshold=0.5)
        else:
            nms_boxes, nms_classes, nms_scores = [], [], []

        output_image = utils.display_detections(image, nms_boxes, nms_classes, nms_scores, utils.get_characters())
        # Get the detected characters
        detected_characters = utils.get_letters(nms_boxes, nms_classes, nms_scores, utils.get_characters())
        
        # Append the detected characters to the list of all detected characters
        all_detected_characters.extend(detected_characters)

    # Return the output image and all detected characters
    return output_image, all_detected_characters

if __name__ == "__main__":
    image_path = './data/11_year_old_boy_first_week_at_oak_hill.png'
    image = utils.load_image(image_path)
    object_detection(image)
