# Human-Pose-Detection-using-MMVR-Datasets Overview
A deep learning–based human pose detection system that predicts 17 body joint locations directly from radar data. Using a CNN architecture (RadarPoseNet), the model maps radar signals to 2D pose keypoints, offering a privacy-friendly alternative to camera-based tracking while maintaining accurate human pose estimation


# Model Architecture
- CNN- Based architecture (RaderPoseNet)
- Input: Single-channel radar tensor
- Convolutional feature extraction
- Fully connected regression layers
- Output: 34 values representing 17 (x,y) joint coordinates

# Dataset
- Radar and pose annotation stored in a .npz file
- 17 keypoints per person
- Confidence values removed
- Coordinates normalized before training

# Results
- Trained using Mean Squared Error (MSE)
- Evaluated using Percentage of Correct Keypoints (PCK)
- Trained loss decreased consistently
- PCK improved throughout training
- Higher errors observed on writs and ankles due to radar spatial resolution limitations

# Visualizations
- Loss Curve and PCK accuracy graph
- Keypoint localization scatter plot




