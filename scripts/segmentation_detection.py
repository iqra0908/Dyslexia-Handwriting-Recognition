import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

# Load the pre-trained letter recognition model
model = load_model('./models/resnet50_model.h5')

# Load the worksheet image
image_path = './data/11_year_old_boy_first_week_at_oak_hill.png'
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
image = cv2.bitwise_not(image)
image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

# Threshold the image and find contours
_, thresh = cv2.threshold(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Preprocess image patches
def preprocess_patch(patch):
    resized_patch = cv2.resize(patch, (128, 128), interpolation=cv2.INTER_AREA).astype('float32')
    resized_patch = resized_patch / 255.0
    return resized_patch

# Predict character for each contour
detections = []
for cnt in contours:
    x, y, w, h = cv2.boundingRect(cnt)

    # Extract a patch from the input image
    patch = image[y:y+h, x:x+w]

    # Preprocess the patch
    resized_patch = preprocess_patch(patch)

    # Predict the class of the patch using the classification model
    prediction = model.predict(resized_patch[np.newaxis, ...])[0]
    predicted_class = np.argmax(prediction)

    # If the predicted class is not the background class, store the detection
    if predicted_class != 0 and prediction[predicted_class] > 0.1:
        detections.append((x, y, w, h, predicted_class, prediction[predicted_class]))

characters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z','0','1','2','3','4','5','6','7','8','9']

# Display the detections on the input image
output_image = image.copy()
for (x, y, w, h, predicted_class, score) in detections:
    letter = characters[predicted_class - 1]

    # Draw bounding box around the detected letter with red color
    cv2.rectangle(output_image, (x, y), (x+w, y+h), (0, 0, 255), 2)

    # Add letter label to the bounding box
    cv2.putText(output_image, letter, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

cv2.imshow("Detections", output_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

