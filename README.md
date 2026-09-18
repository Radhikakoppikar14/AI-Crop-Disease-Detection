# 🍅 AI Crop Disease Detection

An AI-powered image classification project that detects common tomato leaf diseases from images using **transfer learning with MobileNetV3-Small** and provides predictions through a **Streamlit web application**.

## 📌 Project Overview

This project classifies tomato leaf images into four categories:

* 🦠 Bacterial Spot
* 🍂 Early Blight
* 🍃 Late Blight
* 🌿 Healthy

The model was trained using images from the **PlantVillage dataset** and deployed as an interactive web application where users can upload a tomato leaf image and receive a predicted disease along with the model's confidence.

---

## ✨ Features

* Tomato leaf disease classification
* 4-class image classification
* Transfer learning using MobileNetV3-Small
* Image preprocessing and normalization
* Model validation and evaluation
* Classification report
* Confusion matrix
* Prediction confidence
* Interactive Streamlit web interface

---

## 🛠️ Technologies Used

* Python
* PyTorch
* Torchvision
* MobileNetV3-Small
* Scikit-learn
* Streamlit
* Pillow
* PlantVillage Dataset
* Git & GitHub

---

## 🧠 Machine Learning Approach

### 1. Dataset

The project uses tomato leaf images from the PlantVillage dataset.

The four classes used are:

```text
Tomato___Bacterial_spot
Tomato___Early_blight
Tomato___Late_blight
Tomato___healthy
```

Total images used:

**6,627**

### 2. Train / Validation Split

The dataset was divided into:

* Training: **5,301 images**
* Validation: **1,326 images**

A fixed random seed was used to make the split reproducible.

### 3. Transfer Learning

The project uses **MobileNetV3-Small** with pretrained ImageNet weights.

The pretrained feature layers were frozen, while the final classifier was replaced with a new layer for the four tomato classes.

This approach reduces training time and computational requirements compared with training a deep neural network from scratch.

### 4. Training

Training configuration:

* Epochs: **3**
* Batch size: **32**
* Optimizer: **Adam**
* Learning rate: **0.001**
* Loss function: **CrossEntropyLoss**
* Device: **CPU**

---

## 📊 Model Performance

The model achieved:

### Validation Accuracy

**95.32%**

Classification results:

| Class                | Precision | Recall | F1-Score |
| -------------------- | --------: | -----: | -------: |
| Bacterial Spot       |      1.00 |   0.95 |     0.97 |
| Early Blight         |      0.77 |   0.97 |     0.86 |
| Late Blight          |      0.96 |   0.90 |     0.93 |
| Healthy              |      1.00 |   0.99 |     1.00 |
| **Overall Accuracy** |           |        | **0.95** |

The main classification confusion occurred between **Early Blight and Late Blight**, which is visible in the confusion matrix.

---

## 🌐 Streamlit Application

The trained model is integrated into a Streamlit web application.

Users can:

1. Upload a tomato leaf image.
2. Click **Predict Disease**.
3. View the predicted class.
4. View the prediction confidence.
5. View probabilities for all four classes.

Example prediction:

```text
Disease Detected: Early Blight
Confidence: 98.17%
```

---

## 📂 Project Structure

```text
AI-Crop-Disease-Detection/
│
├── PlantVillage-Dataset/
│
├── data/
│
├── models/
│   └── tomato_disease_mobilenetv3.pth
│
├── src/
│   ├── train.py
│   └── evaluate.py
│
├── app/
│   └── app.py
│
├── notebooks/
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd AI-Crop-Disease-Detection
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

### 3. Activate the environment

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Train the Model

To train the model:

```bash
python src/train.py
```

The trained model will be saved to:

```text
models/tomato_disease_mobilenetv3.pth
```

---

## 📊 Evaluate the Model

Run:

```bash
python src/evaluate.py
```

This generates:

* Classification report
* Precision
* Recall
* F1-score
* Confusion matrix

---

## 🚀 Run the Web Application

Start Streamlit:

```bash
streamlit run app/app.py
```

The application will open locally in your browser.

Usually:

```text
http://localhost:8501
```

---

## ⚠️ Limitations

This project is a machine learning prototype and should not be considered a professional agricultural diagnostic system.

The model was trained using PlantVillage images, which can have relatively controlled backgrounds and conditions. Performance may differ on real-world photographs taken using mobile phones under different lighting, backgrounds, camera angles, or image quality.

The prediction should therefore be treated as an AI-assisted classification result rather than a definitive agricultural diagnosis.

---

## 🔮 Future Improvements

Possible future improvements include:

* Data augmentation
* Larger and more diverse datasets
* Fine-tuning more MobileNetV3 layers
* Additional crop diseases
* Support for multiple crops
* Real-world field image testing
* Model explainability using Grad-CAM
* Cloud deployment
* Mobile application
* Disease treatment recommendations
* Integration with agricultural advisory systems

---

## 🎯 Learning Outcomes

Through this project, I practiced:

* Python for machine learning
* Image classification
* PyTorch
* Transfer learning
* CNN-based models
* Dataset preparation
* Model training
* Model evaluation
* Precision, recall and F1-score
* Confusion matrix analysis
* Model saving and loading
* Streamlit deployment
* Git/GitHub project organization

---

## 👩‍💻 Author

**Radhika Koppikar**

Computer Science Engineering Graduate

Interested in:

* Artificial Intelligence
* Machine Learning
* Full-Stack Development
* AI-powered applications

---

## 📜 Disclaimer

This project is created for educational and portfolio purposes. It is not intended to replace professional agricultural diagnosis or expert advice.
