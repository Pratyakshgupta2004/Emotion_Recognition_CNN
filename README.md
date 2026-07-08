# Emotion Recognition using Mini-Xception

## Overview

This project is developed to recognize human facial expressions in real time using a webcam. The model is trained on the FER2013 dataset and classifies facial expressions into seven different emotions.

## Features

- Real-time emotion detection
- Mini-Xception CNN model
- Face detection using OpenCV
- Live emotion prediction with confidence score
- Training and validation graphs
- Lightweight model for real-time performance

## Emotions Detected

- Angry
- Disgust
- Fear
- Happy
- Sad
- Surprise
- Neutral

## Project Structure

```
Emotion_Recognition_MiniXception
│
├── dataset
│   └── fer2013.csv
│
├── models
│   └── emotion_model.keras
│
├── plots
│   ├── accuracy.png
│   └── loss.png
│
├── src
│   ├── preprocess.py
│   ├── model.py
│   ├── train.py
│   ├── utils.py
│   └── real_time_detection.py
│
├── requirements.txt
└── README.md
```

## Technologies Used

- Python
- TensorFlow
- OpenCV
- NumPy
- Pandas
- Matplotlib
- Scikit-learn

## Dataset

The project uses the FER2013 dataset. Each image is a 48×48 grayscale facial image labelled with one of seven emotion categories.

## Model

Mini-Xception is a lightweight convolutional neural network designed for facial expression recognition. It uses depthwise separable convolutions and residual connections, making it suitable for real-time emotion detection.

## Output

The application detects faces from the webcam and displays the predicted emotion along with its confidence score in real time.

## Future Scope

- Improve model accuracy
- Support video file input
- Add emotion statistics
- Develop a graphical user interface
- Deploy as a web or mobile application

## Author

**Pratyaksh Gupta**

B.Tech Computer Science (AI & ML)

GLA University, Mathura
