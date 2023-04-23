import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

# Load the pre-trained letter recognition model
model = load_model('./models/resnet50_model.h5')

# Preprocess the input image
image_path = './data/11_year_old_boy_first_week_at_oak_hill.png'
#image_path = './data/download.jpeg'
image = cv2.imread(image_path)
image = image / 255.0

# Resize the image to new dimensions (width, height)
image = cv2.resize(image, (image.shape[0]*10, image.shape[1]*10))

# Parameters for the sliding window
window_sizes = [64, 128, 256]
step_sizes = [8, 16, 32]

# Sliding window approach
detections = []
for window_size in window_sizes:
    for step_size in step_sizes:
        for y in range(0, image.shape[0] - window_size, step_size):
            for x in range(0, image.shape[1] - window_size, step_size):
                # Extract a patch from the input image
                patch = image[y:y+window_size, x:x+window_size]

                # Resize the patch to the expected input size of the model
                resized_patch = cv2.resize(patch, (128, 128), interpolation=cv2.INTER_AREA).astype('float32')

                # Predict the class of the patch using the classification model
                prediction = model.predict(patch[np.newaxis, ...])[0]
                predicted_class = np.argmax(prediction)
                #print(f"Predicted class: {predicted_class}, Confidence: {prediction[predicted_class]}")

                # If the predicted class is not the background class, store the detection
                if predicted_class != 0 and prediction[predicted_class] > 0.1:
                    detections.append((x, y, window_size, window_size, predicted_class, prediction[predicted_class]))

print(f"Number of detections: {len(detections)}")

# Apply non-maximum suppression (NMS)
def non_max_suppression(boxes, scores, threshold):
    indices = tf.image.non_max_suppression(boxes, scores, max_output_size=boxes.shape[0], iou_threshold=threshold)
    return [boxes[i] for i in indices], [scores[i] for i in indices]

if len(detections) > 0:
    boxes, scores = zip(*[(det[:4], det[-1]) for det in detections])
    nms_boxes, nms_scores = non_max_suppression(np.array(boxes), np.array(scores), threshold=0.5)
else:
    nms_boxes, nms_scores = [], []

characters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
# Display the detections on the input image
output_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
for (x, y, w, h), score in zip(nms_boxes, nms_scores):
    letter = characters[score-1]

    cv2.rectangle(output_image, (x, y), (x+w, y+h), (0, 255, 0), 2)
    cv2.putText(output_image, letter, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

cv2.imshow("Detections", output_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

