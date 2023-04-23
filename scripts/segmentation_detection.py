import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

# Load the pre-trained letter recognition model
model = load_model('./models/resnet50_model.h5')

# Load the worksheet image
image_path = './data/11_year_old_boy_first_week_at_oak_hill.png'
#image_path = './data/download.jpeg'
image = cv2.imread(image_path,cv2.IMREAD_GRAYSCALE)
image = cv2.bitwise_not(image)
image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

# Perform thresholding to binarize the image
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
_, thresh = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# Perform connected component analysis to extract individual characters
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

print(image.shape)
# Parameters for the sliding window
window_size = image.shape[0]/10
step_size = 10

# Sliding window approach
detections = []
for contour in contours:
    x, y, w, h = cv2.boundingRect(contour)

    # Extract a patch from the input image
    patch = image[y:y+h, x:x+w]

    # Resize the patch to the expected input size of the model
    resized_patch = cv2.resize(patch, (128, 128), interpolation=cv2.INTER_AREA).astype('float32')
    
    # Normalize the resized_patch
    resized_patch = resized_patch / 255.0

    # Predict the class of the patch using the classification model
    prediction = model.predict(resized_patch[np.newaxis, ...])[0]
    predicted_class = np.argmax(prediction)

    # If the predicted class is not the background class, store the detection
    if predicted_class != 0:# and prediction[predicted_class] > 0.1:
        detections.append((x, y, w, h, predicted_class, prediction[predicted_class]))

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
output_image = image.copy()
for (x, y, w, h), score in zip(nms_boxes, nms_scores):
    letter = characters[int(score)-1]

    cv2.rectangle(output_image, (x, y), (x+w, y+h), (0, 255, 0), 2)
    cv2.putText(output_image, letter, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

print(detections)
cv2.imshow("Detections", output_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
