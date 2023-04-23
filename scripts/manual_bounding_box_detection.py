import cv2
import numpy as np
from tensorflow.keras.models import load_model
from bounding_box import BoundingBox
from utils import load_image, preprocess_template_letter, sliding_window, non_max_suppression, display_detections

def manual_detection():
    model = load_model('./models/resnet50_model.h5')
    image_path = './data/11_year_old_boy_first_week_at_oak_hill.png'
    image = load_image(image_path)
    template_box = BoundingBox()

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

    characters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    output_image = display_detections(image, nms_boxes, nms_scores, characters)
    cv2.imshow("Detections", output_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    manual_detection()
