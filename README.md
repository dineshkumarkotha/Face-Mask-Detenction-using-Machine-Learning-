# Face Mask Detection using Machine Learning

## Overview
This project implements a real-time face mask detection system using machine learning techniques. Developed as a response to the COVID-19 pandemic, the system identifies whether a person is wearing a face mask and provides audio feedback if they are not. It aims to enhance public safety by promoting mask usage in public areas.

## Features
- Real-time face mask detection using webcam input.
- High accuracy achieved with Convolutional Neural Networks (CNN).
- Utilizes Python libraries like OpenCV, TensorFlow, Keras, and NumPy.
- Audio notification for individuals without a mask.
- Scalable design for integration with automated doors and monitoring systems.

## System Design
### Architecture
The project employs a CNN-based architecture for feature extraction and classification:
- **Data Preprocessing**: Images are resized and normalized.
- **Model Training**: Training with labeled datasets of masked and unmasked faces.
- **Prediction**: Real-time detection and classification of mask usage.

### Technologies Used
- **Python**: Core programming language.
- **TensorFlow and Keras**: For model building and training.
- **OpenCV**: For real-time video capture and image processing.
- **NumPy**: For efficient array computations.

### Workflow
1. Input from a webcam or image dataset.
2. Preprocessing using OpenCV for resizing and normalization.
3. Feature extraction and classification using a trained CNN model.
4. Output: Visual and audio feedback based on detection.

## Installation
### Prerequisites
- Python 3.7 or later
- Libraries: TensorFlow, Keras, OpenCV, NumPy, tqdm
- A system with a webcam for real-time detection

### Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/face-mask-detection.git
   cd face-mask-detection

2. Install dependencies:
   ```bash
  Copy code
  pip install -r requirements.txt

3. Run the detection script:
  ```bash
  Copy code
  python detect_mask.py
