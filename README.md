# Face Detection Project

This project is a simple tool that captures a single image from the webcam, detects human faces in the image, face detector, and displays the result with bounding boxes and confidence scores.

## Features

- Capture each single image from the default camera.
- Detect all visible faces in the image.
- Draw bounding boxes around each detected face with confidence score.
- Display the result in a window.

## Requirements

- Python 3.7+
- OpenCV
- MTCNN
- NumPy

## Installation

Create a virtual environment:

```
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

Install library
```
pip install -r requirement.txt
```

Run project
```
python3 DEMO.py
```

===> If you want to dectect with high precition. Reduce infromation noise in background.

Author: Quang Minh


--------


Link to get dataset: http://shuoyang1213.me/WIDERFACE/

Train images: https://drive.google.com/file/d/15hGDLhsx8bLgLcIRD5DhYt5iBxnjNF1M/view

Validation images: https://drive.google.com/file/d/1GUCogbp16PMGa39thoMMeWxp7Rp5oM8Q/view

===> After install image -> entract rar to this project