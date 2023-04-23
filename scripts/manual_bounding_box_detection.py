import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
global template_box

class BoundingBox:
    def __init__(self, x=0, y=0, w=1, h=1):
        self.x = x
        self.y = y
        self.w = w
        self.h = h

    def update(self, x=None, y=None, w=None, h=None):
        if x is not None:
            self.x = x
        if y is not None:
            self.y = y
        if w is not None:
            self.w = w
        if h is not None:
            self.h = h

    def to_list(self):
        return [self.x, self.y, self.w, self.h]


# Load the pre-trained letter recognition model
model = load_model('./models/resnet50_model.h5')

# Load the worksheet image
#image_path = './data/11_year_old_boy_first_week_at_oak_hill.png'
image_path = './data/bb-writing.png'
image = cv2.imread(image_path,cv2.IMREAD_GRAYSCALE)
image = cv2.bitwise_not(image)
image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

cv2.imshow('Input Image', image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Create a copy of the input image for displaying the selected box
selected_image = image.copy()

# Initialize the template box and letter
template_letter = None

template_box = BoundingBox()

def on_x_trackbar(val):
    template_box.update(x=val)
    update_selected_image()

def on_y_trackbar(val):
    template_box.update(y=val)
    update_selected_image()

def on_w_trackbar(val):
    template_box.update(w=val)
    update_selected_image()

def on_h_trackbar(val):
    template_box.update(h=val)
    update_selected_image()

def update_selected_image():
    global selected_image
    global template_box

    selected_image = image.copy()
    x, y, w, h = template_box.to_list()
    cv2.rectangle(selected_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.imshow('Input Image', selected_image)

# Create trackbars for selecting the template box
cv2.namedWindow('Input Image')
cv2.createTrackbar('X', 'Input Image', 0, image.shape[1] - 1, on_x_trackbar)
cv2.createTrackbar('Y', 'Input Image', 0, image.shape[0] - 1, on_y_trackbar)
cv2.createTrackbar('W', 'Input Image', 1, image.shape[1], on_w_trackbar)
cv2.createTrackbar('H', 'Input Image', 1, image.shape[0], on_h_trackbar)

update_selected_image()

# Display the input image
cv2.imshow('Input Image', image)

# Wait for the user to select the template box using the mouse
cv2.waitKey(0)

x, y, w, h = template_box.to_list()
template_letter = image[y:y+h, x:x+w]

# Close all windows
cv2.destroyAllWindows()

print(template_letter)
# Preprocess the template letter
template_letter = cv2.cvtColor(template_letter, cv2.COLOR_BGR2GRAY)
_, template_letter = cv2.threshold(template_letter, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# Use the template letter dimensions as the sliding window parameters
window_size = template_letter.shape[0]
step_size = int(window_size / 2)

# Sliding window approach
detections = []
for y in range(0, image.shape[0] - window_size, step_size):
    for x in range(0, image.shape[1] - window_size, step_size):
        # Extract a patch from the input image
        patch = image[y:y+window_size, x:x+window_size]

        # Resize the patch to the expected input size of the model
        resized_patch = cv2.resize(patch, (128, 128), interpolation=cv2.INTER_AREA).astype('float32')

        # Normalize the resized_patch
        resized_patch = resized_patch / 255.0

        # Predict the class of the patch using the classification model
        prediction = model.predict(resized_patch[np.newaxis, ...])[0]
        predicted_class = np.argmax(prediction)

        # If the predicted class is not the background class, store the detection
        if predicted_class != 0 and prediction[predicted_class] > 0.1:
            detections.append((x, y, window_size, window_size, predicted_class, prediction[predicted_class]))

# Apply non-maximum suppression (NMS)
def non_max_suppression(boxes, scores, threshold):
    indices = tf.image.non_max_suppression(boxes, scores, max_output_size=boxes.shape[0], iou_threshold=threshold)
    return [boxes[i] for i in indices], [scores[i] for i in indices]

if len(detections) > 0:
    boxes, scores = zip(*[(det[:4], det[-1]) for det in detections])
    nms_boxes, nms_scores = non_max_suppression(np.array(boxes), np.array(scores), threshold=0.5)
else:
    nms_boxes, nms_scores = [], []

characters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z','0','1','2','3','4','5','6','7','8','9']

# Display the detections on the input image
output_image = image.copy()
for (x, y, w, h), score in zip(nms_boxes, nms_scores):
    letter = characters[int(score)-1]

    # Draw bounding box around the detected letter with red color
    cv2.rectangle(output_image, (x, y), (x+w, y+h), (0, 0, 255), 2)
    
    # Add letter label to the bounding box
    cv2.putText(output_image, letter, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

print(detections)
cv2.imshow("Detections", output_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
