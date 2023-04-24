import cv2
import numpy as np
from tensorflow.keras.models import load_model
from bounding_box import BoundingBox
import string
from utils import load_image, preprocess_template_letter, sliding_window, non_max_suppression, display_detections, get_characters

'''
This function performs manual detection of handwritten letters in an image using a trained model. 
It loads the trained model, reads the input image, and sets up a bounding box to select a 
template letter from the input image. It then creates four trackbars to allow the user to 
adjust the position and size of the bounding box. When the user selects a bounding box, 
the function crops the template letter from the input image, preprocesses it, and performs 
sliding window detection on the input image using the trained model. 
It then performs non-maximum suppression to remove overlapping detections and displays the 
final results in a separate window.'''

def manual_detection():
    model = load_model('./models/resnet50_model.h5')
    image_path = './data/11_year_old_boy_first_week_at_oak_hill.png'
    image = load_image(image_path)
    template_box = BoundingBox()

    # This function is called when a trackbar is moved by the user. It updates the 
    # corresponding parameter of the bounding box based on the new value of the trackbar 
    # and calls update_selected_image() to redraw the selected image.
    # Replace the on_x_trackbar, on_y_trackbar, on_w_trackbar, and on_h_trackbar functions
    # with a single function that takes a parameter to identify which trackbar it should update
    def on_trackbar(val, param):
        if param == "x":
            template_box.update(x=val)
        elif param == "y":
            template_box.update(y=val)
        elif param == "w":
            template_box.update(w=val)
        elif param == "h":
            template_box.update(h=val)
        update_selected_image()

    # This function updates the selected image window to show the current position and 
    # size of the bounding box.
    def update_selected_image():
        selected_image = image.copy()
        x, y, w, h = template_box.to_list()
        cv2.rectangle(selected_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.imshow('Input Image', selected_image)

    # Modify the trackbar creation to include the param argument for the on_trackbar function
    cv2.namedWindow('Input Image')
    cv2.createTrackbar('X', 'Input Image', 0, image.shape[1] - 1, lambda val: on_trackbar(val, "x"))
    cv2.createTrackbar('Y', 'Input Image', 0, image.shape[0] - 1, lambda val: on_trackbar(val, "y"))
    cv2.createTrackbar('W', 'Input Image', 1, image.shape[1], lambda val: on_trackbar(val, "w"))
    cv2.createTrackbar('H', 'Input Image', 1, image.shape[0], lambda val: on_trackbar(val, "h"))

    update_selected_image()
    cv2.imshow('Input Image', image)
    cv2.waitKey(0)

    x, y, w, h = template_box.to_list()
    template_letter = image[y:y+h, x:x+w]
    cv2.destroyAllWindows()
    template_letter = preprocess_template_letter(template_letter)
    window_size = template_letter.shape[0]
    step_size = int(window_size / 2)
    detections = sliding_window(image, window_size, step_size, model)

    if len(detections) > 0:
        boxes, scores = zip(*[(det[:4], det[-1]) for det in detections])
        nms_boxes, nms_scores = non_max_suppression(np.array(boxes), np.array(scores), threshold=0.5)
    else:
        nms_boxes, nms_scores = [], []

    
    output_image = display_detections(image, nms_boxes, nms_scores, get_characters())
    cv2.imshow("Detections", output_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    manual_detection()
