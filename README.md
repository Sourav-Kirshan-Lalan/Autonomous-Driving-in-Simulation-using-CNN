
# 🚗 Autonomous Driving in Simulation using CNN

This project implements an end-to-end deep learning model that learns to drive a simulated car using only front-facing camera images. The model is trained to predict steering angles and autonomously navigate a car in the **Udacity Self-Driving Car Simulator**.

---

## 📌 Overview

The goal is to mimic human driving behavior by using a convolutional neural network (CNN) that takes in road images and outputs steering angles. The model is trained on a dataset of driving images and corresponding steering measurements collected from the simulator.

---

## 🎥 Demo

🚗 **[Watch Demo Video](#)** *(Add your YouTube link or GIF here)*

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

## 🗂️ Project Structure

```
autonomous-driving-cnn/
├── data/
│   ├── IMG/                       # Driving images
│   └── driving_log.csv            # Image paths + steering data
├── model.py                       # CNN model architecture
├── train.py                       # Training script
├── drive.py                       # Real-time inference for simulator
├── utils.py                       # Preprocessing & augmentation
├── model.h5                       # Trained model
└── README.md                      # Project documentation
```

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

### 6. **Inference**
Use `drive.py` to deploy the trained model. It runs in real-time with the simulator and controls the car automatically.

---

## 🧪 Example: Preprocessing Function

```python
def img_preprocess(img):
    img = img[60:-25, :, :]  # Crop sky and hood
    img = cv2.resize(img, (200, 66))
    img = img / 127.5 - 1.0  # Normalize to [-1, 1]
    return img
```

---

## 📈 Training Results

| Metric       | Value     |
|--------------|-----------|
| Train Loss   | 0.0004    |
| Val Loss     | 0.0006    |
| Epochs       | 10        |

*(Add your actual values after training)*

---

## 🚀 How to Run

### 🖥️ 1. Train the Model

```bash
python train.py
```

### 🕹️ 2. Run Simulator

- Download [Udacity Self-Driving Car Simulator](https://github.com/udacity/self-driving-car-sim)
- Open the simulator → Autonomous Mode

### 🤖 3. Drive Autonomously

```bash
python drive.py model.h5
```

---

## 💡 Future Improvements

- Add lane detection as a safety constraint
- Use LSTM for temporal prediction
- Deploy in Carla Simulator for real-world conditions
- Add object detection (traffic signs, pedestrians)

---

## 📚 References

- NVIDIA: [End to End Learning for Self-Driving Cars](https://developer.nvidia.com/blog/deep-learning-self-driving-cars/)
- [Udacity Self-Driving Car Simulator](https://github.com/udacity/self-driving-car-sim)

---

## 👨‍💻 Author

**Sourav Kirshan Lalan**  
📧 souravkirshan401@gmail.com  
🔗 [LinkedIn](https://linkedin.com/in/souravkirshan) | [GitHub](https://github.com/yourusername)
