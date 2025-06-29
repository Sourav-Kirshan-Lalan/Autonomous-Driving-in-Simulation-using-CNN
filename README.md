
# 🚗 Autonomous Driving in Simulation using CNN

This project implements an end-to-end deep learning model that learns to drive a simulated car using only front-facing camera images. The model is trained to predict steering angles and autonomously navigate a car in the **Udacity Self-Driving Car Simulator**.

---

## 📌 Overview

The goal is to mimic human driving behavior by using a convolutional neural network (CNN) that takes in road images and outputs steering angles. The model is trained on a dataset of driving images and corresponding steering measurements collected from the simulator.

---

## 🧠 Model Architecture

The model is inspired by NVIDIA's end-to-end self-driving car architecture. It includes:

- Image normalization
- 5 Convolutional layers
- Flatten + Fully Connected layers
- Output: a single value (steering angle)

---

## 🧰 Tech Stack

- **Language**: Python
- **Frameworks**: TensorFlow / Keras
- **Computer Vision**: OpenCV, matplotlib
- **Simulator**: Udacity Self-Driving Car Simulator
- **Libraries**: NumPy, Pandas, scikit-learn

---

## 🛠️ How It Works

### 1. **Data Collection**
Drive manually in the Udacity simulator to record:
- Center, left, right camera images
- Steering angle, throttle, brake, speed

### 2. **Data Preprocessing**
- Crop unneeded image regions (sky, hood)
- Resize to (200x66)
- Normalize to [-1, 1]

### 3. **Data Augmentation**
During training:
- Random brightness
- Horizontal flipping (steering angle inversion)
- Translation (simulate car being off-center)

### 4. **Model Training**
Use Mean Squared Error (MSE) loss and the Adam optimizer to train a regression model on image → steering angle.

### 5. **Model Evaluation**
Visualize loss curves and validate the model on unseen images.

---

## 📈 Training Results

| Metric       | Value     |
|--------------|-----------|
| Train Loss   | 0.0442    |
| Val Loss     | 0.0303    |
| Epochs       | 10        |
