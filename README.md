# Satellite-Image-Enhancement-Feature-Extraction-and-Land-Classification

# Satellite Image Land Classification System

## 1. Overview

This project is a deep learning-based system that classifies satellite images into different land categories such as Forest, River, SeaLake, Residential, and Industrial.

It integrates image preprocessing, enhancement, and classification into a single pipeline. A Streamlit-based interface allows users to upload images and get real-time predictions with confidence scores.

---

## 2. Project Architecture

```id="8gcb0t"
satellite_image_land_classification/
│
├── app.py
├── model/
│   ├── land_classifier_model.keras
│   └── class_labels.json
│
├── utils/
│   ├── classification.py
│   ├── image_enhancement.py
│   └── feature_extraction.py
│
├── data/
│   ├── sample_images/
│   └── outputs/
│
├── requirements.txt
└── README.md
```

---

## 3. Dataset

### EuroSAT Dataset

This project uses the EuroSAT dataset for training the model.

Dataset Link:
[https://github.com/phelber/eurosat](https://github.com/phelber/eurosat)

### Dataset Details

* 27,000+ labeled satellite images
* 10 classes
* RGB images
* Size: 64×64 (can be resized to 224×224 for training)

---

## 4. Model Details

### Model Type

* Convolutional Neural Network (CNN)
* Based on MobileNetV2 architecture

### Input Specifications

* Input Size: 224 × 224 × 3
* Color Format: RGB
* Normalization: Pixel values scaled to [0, 1]

### Output

* Softmax classification layer
* Output shape: (1, 10)

### Classes

* AnnualCrop
* Forest
* HerbaceousVegetation
* Highway
* Industrial
* Pasture
* PermanentCrop
* Residential
* River
* SeaLake

### Training Details

* Loss Function: Categorical Crossentropy
* Optimizer: Adam
* Metric: Accuracy

### Important Notes (Critical for Accuracy)

* Preprocessing during inference must match training
* Always use RGB format
* Normalize image (divide by 255)
* Do not use enhanced image for prediction
* Ensure label order is correct

---

## 5. Tech Stack

### Language

* Python

### Libraries

* TensorFlow / Keras
* OpenCV
* NumPy
* Streamlit
* PIL

---

## 6. Installation

```id="2m7k6j"
pip install tensorflow opencv-python numpy streamlit pillow matplotlib scikit-learn
```

---

## 7. Complete Steps (Training + Running)

### Step 1: Download Dataset

* Download EuroSAT dataset from:
  [https://github.com/phelber/eurosat](https://github.com/phelber/eurosat)

* Extract into:

```
data/dataset/
```

---

### Step 2: Train the Model

You need to train your model before running prediction.

Basic training flow:

1. Load dataset
2. Resize images to 224×224
3. Normalize images (divide by 255)
4. Encode labels
5. Train MobileNetV2 model
6. Save model

Example (basic idea):

```id="nd4oaf"
model.save("model/land_classifier_model.keras")
```

---

### Step 3: Verify Model & Labels

* Ensure:

  * Model file exists
  * Label order matches training
  * Classes = 10

---

### Step 4: Run the Application

```id="u9p1p8"
streamlit run app.py
```

---

### Step 5: Upload Image

* Upload satellite image
* System will:

  * Preprocess image
  * Predict class
  * Show confidence

---

## 8. Alternative Run Methods

### Using .bat File (Windows)

Create `run_app.bat`

```id="2bzn4x"
@echo off
cd /d %~dp0
streamlit run app.py
pause
```

---

### Using Python Module

```id="y7l2ts"
python -m streamlit run app.py
```

---

### Using VS Code

* Open folder
* Open terminal
* Run:

```id="e49c3t"
streamlit run app.py
```

---

## 9. System Workflow

1. User uploads image
2. Image is validated
3. Resized to model size
4. Converted to RGB
5. Normalized
6. Passed into CNN
7. Predictions generated
8. Results displayed

---

## 10. Image Enhancement Module

Used for improving image quality (not for model input)

### Techniques:

* CLAHE
* Denoising
* Sharpening
* Gamma correction
* Histogram processing

---

## 11. Syllabus Coverage

### Digital Image Formation and Low-Level Processing

* Image representation
* Image transformation
* Convolution and filtering
* Image enhancement
* Histogram processing

---

### Depth Estimation and Multi-Camera Views

* Perspective transformation
* Geometric understanding
* Spatial transformations

---

### Feature Extraction and Image Segmentation

* Edge detection concepts
* Texture analysis
* CNN-based feature extraction
* Scale-space concepts

---

### Pattern Analysis and Motion Analysis

* Supervised learning
* CNN classification
* Probability prediction
* Accuracy evaluation
* Confusion matrix

---

### Shape from X

* Light and surface interaction
* Texture and color understanding
* Reflectance concepts

---

## 12. Common Issues and Fixes

### Issue: Same prediction (SeaLake)

**Reasons:**

* Wrong preprocessing
* Label mismatch
* Model overfitting
* Using enhanced image

**Fix:**

* Use RGB format
* Normalize properly
* Use original image
* Check label order

---

### Issue: Model not loading

**Fix:**

* Use `.keras` format
* Match TensorFlow version
* Handle custom layers properly

---

## 13. Applications

* Environmental monitoring
* Urban planning
* Agriculture analysis
* Water detection
* Remote sensing

---

## 14. Future Improvements

* Improve model accuracy
* Add more classes
* Add Grad-CAM visualization
* Deploy on cloud
* Real-time satellite API integration

---

## 15. Conclusion

This project demonstrates how computer vision and deep learning can be applied to solve real-world land classification problems using satellite imagery. It integrates theory and practical implementation effectively.

