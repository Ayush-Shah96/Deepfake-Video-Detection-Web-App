# Deepfake-Video-Detection-Web-App
This repository provides a Deep Fake Video Ddection built with Deep Learning and OpenCV. The project leverages powerful convolutional neural network (CNN) architectures for accurate deepfake detection and recognition, paired with the flexible image processing capabilities of OpenCV.

## Key Features:

<b>Video Upload & Analysis : </b> Upload videos in multiple formats (MP4, AVI, MOV, MKV)
Real-time video analysis with progress tracking
Face detection using Haar cascades

<b>Deep Learning Detection : </b> Custom CNN model based on ResNet50
Processes face crops from video frames
Generates confidence scores for fake detection

<b>Interactive Dashboard : </b> Real-time confidence visualization over time
Distribution charts of predictions
Detailed frame-by-frame analysis
Adjustable detection threshold

<b>Batch Processing :</b> Analyze multiple videos at once
Summary statistics and comparisons
Batch results visualization

<b>Professional UI : </b> Clean, modern interface with custom CSS
Color-coded results (green for real, red for fake)
Responsive layout with multiple tabs

## Important Notes:

1. Model Training: The current implementation uses a simulated detection model. For production use, you'd need to train the model on a large dataset of real and fake videos <br>
2. Face Detection: Uses OpenCV's Haar cascades for face detection <br>
3. Performance: Optimized for real-time analysis with frame sampling <br>
4. Privacy: No data is stored permanently - all processing is done locally <br>
