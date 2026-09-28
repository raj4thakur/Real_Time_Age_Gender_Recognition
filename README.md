# VisionAge AI — Real-Time Age & Gender Recognition

VisionAge AI is a real-time computer vision web application that uses deep learning to estimate a person's age and gender from a webcam image.

The application combines a TensorFlow/Keras deep learning model with a FastAPI backend and a modern browser-based frontend.

The project is designed as an end-to-end machine learning deployment project covering:

- Computer Vision
- Deep Learning
- TensorFlow/Keras
- FastAPI
- REST API
- HTML/CSS/JavaScript
- Webcam integration
- Docker
- CI/CD
- Cloud deployment

---

## Features

- Real-time webcam access
- Age estimation
- Gender classification
- Gender confidence score
- Live confidence progress bar
- FastAPI REST API
- Automatic frontend serving through FastAPI
- No separate frontend server required
- Responsive modern UI
- TensorFlow/Keras inference
- Docker-ready architecture
- CI/CD-ready project structure
- Render deployment support

---

## Demo

The application provides a dashboard where the user can:

1. Start the webcam.
2. Allow browser camera permission.
3. Capture frames from the webcam.
4. Send frames to the FastAPI backend.
5. Run the trained TensorFlow models.
6. Display the predicted age.
7. Display the predicted gender.
8. Display the gender confidence score.



Link to try: https://real-time-age-gender-recognition.onrender.com/
Example prediction:

```text
Estimated Age
31

Detected Gender
Male

Gender Confidence
61.3%