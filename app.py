import streamlit as st
import cv2
import numpy as np
from PIL import Image
from streamlit_drawable_canvas import st_canvas
from scripts.bounding_box_detection import object_detection

# Function to perform handwriting recognition on an image
def recognize_handwriting(image):
    #recognized_image = manual_detection(image)
    return image

# Set the title and the sidebar header of the Streamlit app
st.set_page_config(page_title="Dyslexia Handwriting Recognition", page_icon=":pencil2:")
st.sidebar.title("Dyslexia Handwriting Recognition")

# Add a file uploader to the sidebar
image_file = st.sidebar.file_uploader(label="Upload an image of handwriting", type=['jpg', 'jpeg', 'png'])

# Create a list to store the bounding boxes
bounding_boxes = []

# If an image has been uploaded, perform handwriting recognition and display the results
if image_file is not None:
    # Convert the uploaded file to a PIL Image
    image = Image.open(image_file)

    # Convert the PIL Image to an OpenCV image
    image_cv2 = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    # Create a canvas for drawing the bounding box
    canvas_result = st_canvas(
        fill_color="rgba(0, 255, 0, 0.3)",  # Color of the rectangle
        stroke_width=0,
        stroke_color="rgba(0, 255, 0, 0.3)",
        background_image=image,
        width=image.width,
        height=image.height,
        drawing_mode="rect",
        key="canvas",
    )

# Check if bounding boxes have been drawn
    if canvas_result.json_data["objects"]:
        # Clear the bounding_boxes list
        bounding_boxes = []

        # Loop through each drawn bounding box
        for obj in canvas_result.json_data["objects"]:
            x = int(obj["left"])
            y = int(obj["top"])
            w = int(obj["width"])
            h = int(obj["height"])

            # Add the bounding box to the list
            bounding_boxes.append((x, y, w, h))
        
    start_detection = st.button("Start Detection")


    # If there are bounding boxes, perform handwriting recognition and display the results
    if start_detection:
        # Crop the regions within the bounding boxes
        template_letters = [image_cv2[y:y+h, x:x+w] for x, y, w, h in bounding_boxes]

        # Perform handwriting recognition on the cropped regions
        recognized_image, detected_characters = object_detection(image_cv2, template_letters)

        # Display the uploaded image and the recognized image
        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption='Uploaded image', use_column_width=True)
        with col2:
            st.image(recognized_image, caption='Recognized image', use_column_width=True)

        # Display the detected text
        st.write("Detected text: ", "".join(detected_characters))