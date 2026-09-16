# Electronic Waste Detection and Classification

## Project Overview

This project develops a computer vision system for detecting and classifying electronic waste (e-waste) objects using an object detection model.

The system is intended to identify electronic waste objects from images and, ultimately, from a real-time camera feed. The project covers the complete machine learning workflow from dataset preparation and preprocessing through model training, evaluation, and real-time inference.

## Project Objectives

- Prepare and analyze an electronic waste object-detection dataset.
- Train a YOLO-based object detection model using transfer learning.
- Detect electronic waste objects and classify them into their corresponding device classes.
- Evaluate the trained model using appropriate computer vision metrics.
- Analyze model errors, including false positives and false negatives.
- Develop a real-time detection demonstration using a camera.
- Document the complete development process.

## Dataset

The project uses the Balanced E-Waste Dataset, Version 3, obtained from Roboflow.

Dataset:
Balanced E-Waste Dataset

Source:
https://universe.roboflow.com/electronic-waste-detection/balanced-e-waste-dataset/dataset/3

The dataset contains annotated images of electronic devices and is based on the UNU-KEY classification structure.

## Model

The project uses YOLO26s for object detection.

The model will be trained using transfer learning from pretrained weights and fine-tuned on the electronic waste dataset.

## Project Structure

```text
.
├── data/
├── demo/
├── docs/
├── models/
├── notebooks/
├── results/
├── src/
├── .gitignore
└── README.md
