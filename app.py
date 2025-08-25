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

# Check if CUDA is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

class DeepfakeDetector(nn.Module):
    """
    A neural network model for deepfake detection using EfficientNet backbone
    """
    def __init__(self, num_classes=2):
        super(DeepfakeDetector, self).__init__()
        # Use EfficientNet-B0 as backbone
        self.backbone = models.efficientnet_b0(pretrained=True)
        
        # Replace the classifier
        num_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        return self.backbone(x)

class FaceExtractor:
    """
    Extract faces from video frames using OpenCV
    """
    def __init__(self):
        # Load face cascade classifier
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
    def extract_faces(self, frame, min_size=(64, 64)):
        """Extract faces from a single frame"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4, minSize=min_size)
        
        face_crops = []
        for (x, y, w, h) in faces:
            # Add some padding around the face
            padding = int(0.1 * min(w, h))
            x_start = max(0, x - padding)
            y_start = max(0, y - padding)
            x_end = min(frame.shape[1], x + w + padding)
            y_end = min(frame.shape[0], y + h + padding)
            
            face_crop = frame[y_start:y_end, x_start:x_end]
            if face_crop.size > 0:
                face_crops.append(face_crop)
        
        return face_crops, faces

class VideoProcessor:
    """
    Process videos for deepfake detection
    """
    def __init__(self, model, face_extractor, transform):
        self.model = model
        self.face_extractor = face_extractor
        self.transform = transform
        
    def process_video(self, video_path, max_frames=30, frame_skip=5):
        """Process video and return deepfake predictions"""
        cap = cv2.VideoCapture(video_path)
        predictions = []
        frame_count = 0
        processed_frames = 0
        
        while cap.isOpened() and processed_frames < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
                
            if frame_count % frame_skip == 0:  # Skip frames for efficiency
                faces, face_boxes = self.face_extractor.extract_faces(frame)
                
                for face in faces:
                    # Preprocess face
                    face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                    face_pil = Image.fromarray(face_rgb)
                    face_tensor = self.transform(face_pil).unsqueeze(0).to(device)
                    
                    # Make prediction
                    with torch.no_grad():
                        output = self.model(face_tensor)
                        prob = torch.softmax(output, dim=1)
                        fake_prob = prob[0][1].item()  # Probability of being fake
                        predictions.append(fake_prob)
                
                processed_frames += 1
            
            frame_count += 1
        
        cap.release()
        return predictions

def create_dummy_model():
    """
    Create and return a dummy pre-trained model for demonstration
    Since we don't have access to actual deepfake detection weights,
    this creates a model with random weights for demonstration purposes
    """
    model = DeepfakeDetector(num_classes=2)
    model = model.to(device)
    model.eval()
    
    print("Note: Using randomly initialized model for demonstration.")
    print("In practice, you would load pre-trained weights from a deepfake detection dataset.")
    
    return model

def download_sample_video():
    """
    Download a sample video for testing
    """
    # For demonstration, we'll create a simple video using OpenCV
    # In practice, you would upload your own video or download from a dataset
    
    print("Creating a sample video for demonstration...")
    
    # Create a simple test video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('sample_video.mp4', fourcc, 20.0, (640, 480))
    
    for i in range(100):
        # Create a frame with a simple pattern
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        # Add a simple face-like rectangle
        cv2.rectangle(frame, (200, 150), (400, 350), (255, 255, 255), 2)
        cv2.circle(frame, (250, 200), 10, (0, 0, 0), -1)  # Left eye
        cv2.circle(frame, (350, 200), 10, (0, 0, 0), -1)  # Right eye
        cv2.rectangle(frame, (275, 250), (325, 280), (0, 0, 0), -1)  # Mouth
        
        out.write(frame)
    
    out.release()
    print("Sample video created: sample_video.mp4")
    return 'sample_video.mp4'

def visualize_results(predictions, video_path):
    """
    Visualize the deepfake detection results
    """
    plt.figure(figsize=(15, 5))
    
    # Plot 1: Prediction timeline
    plt.subplot(1, 3, 1)
    plt.plot(predictions, marker='o', linewidth=2, markersize=4)
    plt.axhline(y=0.5, color='r', linestyle='--', alpha=0.7, label='Threshold')
    plt.title('Deepfake Probability Over Time')
    plt.xlabel('Frame/Face Index')
    plt.ylabel('Fake Probability')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Distribution of predictions
    plt.subplot(1, 3, 2)
    plt.hist(predictions, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    plt.axvline(x=0.5, color='r', linestyle='--', alpha=0.7, label='Threshold')
    plt.title('Distribution of Predictions')
    plt.xlabel('Fake Probability')
    plt.ylabel('Frequency')
    plt.legend()
    
    # Plot 3: Summary statistics
    plt.subplot(1, 3, 3)
    avg_prob = np.mean(predictions)
    max_prob = np.max(predictions)
    min_prob = np.min(predictions)
    std_prob = np.std(predictions)
    
    stats = ['Average', 'Maximum', 'Minimum', 'Std Dev']
    values = [avg_prob, max_prob, min_prob, std_prob]
    colors = ['blue', 'red', 'green', 'orange']
    
    bars = plt.bar(stats, values, color=colors, alpha=0.7)
    plt.title('Prediction Statistics')
    plt.ylabel('Probability')
    plt.xticks(rotation=45)
    
    # Add value labels on bars
    for bar, value in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{value:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()
    
    # Print detailed results
    print(f"\n{'='*50}")
    print(f"DEEPFAKE DETECTION RESULTS")
    print(f"{'='*50}")
    print(f"Video: {video_path}")
    print(f"Total faces analyzed: {len(predictions)}")
    print(f"Average fake probability: {avg_prob:.3f}")
    print(f"Maximum fake probability: {max_prob:.3f}")
    print(f"Minimum fake probability: {min_prob:.3f}")
    print(f"Standard deviation: {std_prob:.3f}")
    
    threshold = 0.5
    fake_count = sum(1 for p in predictions if p > threshold)
    real_count = len(predictions) - fake_count
    
    print(f"\nUsing threshold of {threshold}:")
    print(f"Faces classified as FAKE: {fake_count} ({fake_count/len(predictions)*100:.1f}%)")
    print(f"Faces classified as REAL: {real_count} ({real_count/len(predictions)*100:.1f}%)")
    
    if avg_prob > threshold:
        print(f"\n🚨 VERDICT: This video is likely a DEEPFAKE (confidence: {avg_prob:.1%})")
    else:
        print(f"\n✅ VERDICT: This video appears to be REAL (confidence: {1-avg_prob:.1%})")

def main():
    """
    Main function to run the deepfake detection system
    """
    print("🔍 Deepfake Video Detection System")
    print("==================================")
    
    # Initialize components
    print("\n1. Initializing model...")
    model = create_dummy_model()
    
    print("\n2. Setting up face extractor...")
    face_extractor = FaceExtractor()
    
    print("\n3. Preparing image transforms...")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    print("\n4. Creating video processor...")
    processor = VideoProcessor(model, face_extractor, transform)
    
    # Get or create sample video
    print("\n5. Preparing sample video...")
    video_path = download_sample_video()
    
    # Process video
    print(f"\n6. Processing video: {video_path}")
    print("Extracting faces and making predictions...")
    predictions = processor.process_video(video_path, max_frames=20, frame_skip=3)
    
    if not predictions:
        print("❌ No faces detected in the video!")
        return
    
    # Visualize results
    print(f"\n7. Analyzing {len(predictions)} face detections...")
    visualize_results(predictions, video_path)
    
    print("\n✨ Analysis complete!")

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

# Run the main demonstration
if __name__ == "__main__":
    main()
    
    upload_and_analyze_custom_video()
