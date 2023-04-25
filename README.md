# Dyslexia Handwriting Recognition

This project aims to create a handwriting recognition system for people with dyslexia, using a sliding window approach and deep learning techniques. The goal is to detect and recognize letters in handwriting samples from individuals with dyslexia.

## Prerequisites
To run the project, you need to have the following installed:

To run the project, you need to have the following installed:

- Python 3.7.15
- keras==2.11.0
- matplotlib==3.5.3
- numpy==1.21.6
- opencv-python==4.7.0.72
- pandas==1.3.5
- Pillow==9.5.0
- requests==2.28.2
- scikit-learn==1.0.2
- scipy==1.7.3
- streamlit==1.21.0
- TensorFlow==2.11.0
- torch==1.13.1
- torchvision==0.14.1
- zipp==3.15.0

## How to run
1. Load the pre-trained letter recognition model.
2. Load the worksheet image you want to process.
3. Set the template letter by selecting a bounding box around a letter in the worksheet image.
4. The script will then use a sliding window approach to detect and recognize letters in the image.
5. The detected letters will be displayed on the input image with bounding boxes.

### Example
To run the code with the provided example image, simply execute the following command:

```bash
python manual_bounding_box_detection.py
```

## Description of the code
The script contains the following steps:

1. Loading the pre-trained letter recognition model (ResNet50 or EfficientNetB0).
2. Loading the worksheet image and converting it to grayscale.
3. Initializing the template box and letter.
4. Creating trackbars for selecting the template box.
5. Preprocessing the template letter.
6. Using the template letter dimensions as the sliding window parameters.
7. Implementing a sliding window approach to detect and recognize letters.
8. Applying non-maximum suppression (NMS) to remove duplicate detections.
9. Displaying the detections on the input image.
10. Training a classification model on a dataset of dyslexic handwriting.
11. Evaluating the model on a test set and saving the trained model.

## Dataset

The dataset used in this project is a collection of handwriting samples from people with dyslexia, obtained from the Kaggle website at https://www.kaggle.com/datasets/drizasazanitaisa/dyslexia-handwriting-dataset.

It includes both normal and reversal handwriting samples. The dataset is contained in the Gambo.zip file.
The dataset used in this project was collected from three sources:

- Uppercase letters were obtained from NIST Special Database 19 [1].
- Lowercase letters were obtained from a Kaggle dataset [2].
- Additional testing datasets were collected from dyslexic students at Seberang Jaya Primary School, Penang, Malaysia.

The dataset contains a total of **78,275** samples for the normal class, **52,196** for the reversal class, and **8,029** for the corrected class. Access to the dataset requires the password.

### Data Sampling
To reduce the computational requirements for training the model, we used data sampling to select a subset of images from the dataset. Specifically, we selected **500** images for each class from the training set and **100** images for each class from the test set.

## Results

| Model | Number of Epochs | Loss (Validation) | Accuracy (Validation) | Loss (Test) | Accuracy (Test) |
|-------|-----------------|-------------------|-----------------------|-------------|-----------------|
| ResNet50 | 100 | 2.5033 | 0.2851 | 2.4646 | 0.2912 |
| ResNet101 | 30 | 2.1344 | 0.3318 | 2.15 | 0.3056 |
| ResNet152 | 30 | 2.27 | 0.306 | 2.265 | 0.296 |
| LeNet-5 | 30 | 1.400 | 0.4711 | 1.349 | 0.493 |
| AlexNet | 30 | 3.875 | 0.01 | 3.87 | 0.0208 |
| LeNet-5-modified | 30 | 1.306 | 0.5135 | 1.0726 | 0.5812 |

SVM Model:
Test accuracy: 0.93125

## Streamlit app
You can also run the handwriting recognition system using our Streamlit app. To run the app, make sure you have installed all the required packages including Streamlit. Then, run the following command:

```bash
streamlit run app.py
```

The app will launch in your browser. You can upload an image of handwriting using the file uploader in the sidebar. Once an image is uploaded, you can draw bounding boxes around letters in the image using the canvas tool. Then, you can choose a detection method (ResNet50, ResNet101, or SVM) and perform handwriting recognition on the selected letters. The recognized image and detected text will be displayed in the app.

## References
The following papers are related to this dataset:
1. P. J. Grother, “NIST Special Database 19,” NIST, 2016. [Online]. Available: https://www.nist.gov/srd/nist-special-database-19. [Accessed: 22-May-2019].
2. S. Patel, “A-Z Handwritten Alphabets in .csv format,” Kaggle, 2017. [Online]. Available: https://www.kaggle.com/sachinpatel21/az-handwritten-alphabets-in-csv-format. [Accessed: 22-May-2019].
3. Rosli, M. S. A. B., Isa, I. S., Ramlan, S. A., Sulaiman, S. N., & Maruzuki, M. I. F. (2021). Development of CNN Transfer Learning for Dyslexia Handwriting Recognition. 2021 11th IEEE International Conference on Control System, Computing and Engineering (ICCSCE), 194-199. doi: 10.1109/ICCSCE52189.2021.9530971.
4. Seman, N. S. L., Isa, I. S., Ramlan, S. A., Li-Chih, W., & Maruzuki, M. I. F. (2021). Notice of Removal: Classification of Handwriting Impairment Using CNN for Potential Dyslexia Symptom. 2021 11th IEEE International Conference on Control System, Computing and Engineering (ICCSCE), 188-193. doi: 10.1109/ICCSCE52189.2021.9530989.
5. Isa, I. S., Sazanita, I., Rahimi, W. N. S., Ramlan, S. A., Sulaiman, S. N., & Mohamad, F. A. (2021). CNN Comparisons Models On Dyslexia Handwriting Classification. Universiti Teknologi MARA Cawangan Pulau Pinang.
6. Isa, I. S., Rahimi, W. N. S., Ramlan, S. A., & Sulaiman, S. N. (2019). Automated detection of dyslexia symptom based on handwriting image for primary school children. Procedia Computer Science, 163, 440-449. doi: 10.1016/j.procs.2019.12.202.