# Vision Assistance System
This project aims to assist visually impaired individuals by providing real-time information about their surroundings through wearable smart glasses. Equipped with a Raspberry Pi, camera, and speakers, the glasses offer audible feedback about detected objects, text, faces, and more.

## Features
- **Predefined Colors Recognition**: Identifies specific colors to aid in identifying objects.
- **Optical Character Recognition (OCR)**: Reads printed text using Tesseract OCR.
- **Currency Recognition**: Identifies different banknotes to help distinguish currency.
- **Face Recognition**: Recognizes familiar faces using a trained model.
- **Text-to-Speech Feedback**: Communicates the recognized objects, text, or faces to the user via audio.

## Technologies Used
- **OpenCV**: For image processing and object detection.
- **face_recognition**: For identifying familiar faces.
- **pytesseract**: For text extraction.
- **pyttsx3**: For text-to-speech functionality.
- **Picamera Module**: To interface with the Raspberry Pi camera.
- **Raspberry Pi GPIO**: To manage input and output on the Raspberry Pi.

## Libraries
```python
import cv2
import argparse
import time
import numpy as np
import imutils
import pickle
from imutils.video import VideoStream, FPS
import face_recognition
import pyttsx3
from picamera.array import PiRGBArray
from picamera import PiCamera
from imutils.object_detection import non_max_suppression
import pytesseract
import RPi.GPIO as GPIO
```

## How It Works
1. The user wears the glasses, which are equipped with a camera.
2. The camera captures the surroundings and sends the images to the Raspberry Pi for processing.
3. Based on the detected object, text, or face, a corresponding audible message is generated using text-to-speech and played through the speakers.
