# Human Pose Detection using MMVR Datasets

## Overview

This project presents a deep learning–based human pose detection system that predicts 17 body joint locations directly from millimeter-wave radar data. Using a convolutional neural network (**RadarPoseNet**), the model learns to map radar signals to 2D pose keypoints, providing a privacy-preserving alternative to camera-based tracking while maintaining accurate human pose estimation.

---

## Model Architecture

### RadarPoseNet

The model uses a CNN-based architecture consisting of:

- Single-channel radar tensor input
- Convolutional feature extraction layers
- Batch normalization and ReLU activations
- Pooling layers for dimensionality reduction
- Fully connected regression layers
- Dropout regularization

**Input:** Single-channel radar tensor

**Output:** 34 values representing the coordinates of 17 body joints

```
(x₁, y₁, x₂, y₂, ..., x₁₇, y₁₇)
```

---

## Dataset

The project utilizes the MMVR dataset, which contains synchronized radar measurements and human pose annotations.

Dataset characteristics:

- Radar and pose annotations stored in `.npz` files
- 17 keypoints per person
- Confidence values removed prior to training
- Coordinates normalized before model training

Dataset source:

https://zenodo.org/records/12611978

---

## Training and Evaluation

The model was trained using:

- **Loss Function:** Mean Squared Error (MSE)
- **Optimizer:** Adam
- **Learning Rate:** 1e-3
- **Epochs:** 10

Model performance was evaluated using:

### Percentage of Correct Keypoints (PCK)

PCK measures the percentage of predicted joints that fall within a specified distance threshold from the ground truth location.

---

## Results

Training demonstrated consistent convergence throughout the learning process.

### Observations

- Training loss decreased steadily over multiple epochs.
- PCK scores improved throughout training.
- Higher prediction errors were observed for wrists and ankles, likely due to limitations in radar spatial resolution.

---

## Visualizations

The project generates the following visualizations:

- Training loss curve
- PCK accuracy graph
- Keypoint localization scatter plot comparing predicted and ground truth joint locations

---

## Authors

- Claire Aiken
- Talan Malicote

**CSE 486/586 – Introduction to AI**
