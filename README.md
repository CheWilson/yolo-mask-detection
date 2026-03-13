# YOLOv8 Face Mask Detection

This project implements a face mask detection system using YOLOv8 to identify whether people are wearing masks correctly.

## Features
- Detect people wearing masks
- Detect people without masks
- Detect incorrectly worn masks

## Tech Stack
- Python
- YOLOv8
- PyTorch
- Computer Vision

## Dataset
The dataset is annotated in Pascal VOC XML format and converted to YOLO format using a custom script.

## Training
To train the model:

```bash
python train_yolo.py
