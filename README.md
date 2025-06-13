traffic_sign_detection
Traffic Sign Recognition System
This project is a hybrid traffic sign recognition system that combines:

YOLOv8 for real-time detection of traffic signs.
A custom PyTorch classifier to classify detected signs into 43 categories.
It works on static images, webcam feeds, or video files.

Model Architecture
1. Detection Model
Framework: YOLOv8 (Ultralytics)
Purpose: Detect bounding boxes around traffic signs.
2. Classification Model
Framework: PyTorch
Model: Custom CNN trained on the GTSRB dataset
Input size: 64x64
Output: Class ID (0–42)
Requirements
pip install torch torchvision opencv-python ultralytics
