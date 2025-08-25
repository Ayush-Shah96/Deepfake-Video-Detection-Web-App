import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import os
import glob
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')



def upload_and_analyze_custom_video():
    """
    Function to analyze a custom uploaded video
    """
    from google.colab import files
    
    print("📁 Upload your video file:")
    uploaded = files.upload()
    
    if not uploaded:
        print("No file uploaded!")
        return
    
    video_path = list(uploaded.keys())[0]
    print(f"Processing uploaded video: {video_path}")
    
    # Initialize components
    model = create_dummy_model()
    face_extractor = FaceExtractor()
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    processor = VideoProcessor(model, face_extractor, transform)
    
    # Process the uploaded video
    predictions = processor.process_video(video_path, max_frames=50, frame_skip=2)
    
    if predictions:
        visualize_results(predictions, video_path)
    else:
        print("❌ No faces detected in the uploaded video!")
