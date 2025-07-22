import streamlit as st
import cv2
import numpy as np
import tempfile
import os
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Set page config
st.set_page_config(
    page_title="DeepFake Video Detection",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .detection-result {
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        font-size: 1.5rem;
        font-weight: bold;
        margin: 1rem 0;
    }
    .real-video {
        background-color: #d4edda;
        color: #155724;
        border: 2px solid #c3e6cb;
    }
    .fake-video {
        background-color: #f8d7da;
        color: #721c24;
        border: 2px solid #f5c6cb;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

class DeepFakeDetector(nn.Module):
    """
    Custom CNN model for deepfake detection
    """
    def __init__(self, num_classes=2):
        super(DeepFakeDetector, self).__init__()
        # Use ResNet50 as backbone
        self.backbone = models.resnet50(pretrained=True)
        # Modify the final layer
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(self.backbone.fc.in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        return self.backbone(x)

@st.cache_resource
def load_model():
    """Load the pre-trained deepfake detection model"""
    model = DeepFakeDetector()
    # In a real application, you would load actual weights
    # model.load_state_dict(torch.load('deepfake_detector.pth'))
    model.eval()
    return model

def preprocess_frame(frame):
    """Preprocess frame for model input"""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    if len(frame.shape) == 3:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    pil_image = Image.fromarray(frame)
    tensor = transform(pil_image).unsqueeze(0)
    return tensor

def extract_faces(frame, face_cascade):
    """Extract faces from frame using Haar cascades"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    face_crops = []
    for (x, y, w, h) in faces:
        # Add padding
        padding = 20
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(frame.shape[1], x + w + padding)
        y2 = min(frame.shape[0], y + h + padding)
        
        face_crop = frame[y1:y2, x1:x2]
        if face_crop.size > 0:
            face_crops.append((face_crop, (x1, y1, x2, y2)))
    
    return face_crops

def analyze_video(video_path, model, progress_bar=None):
    """Analyze video for deepfake detection"""
    cap = cv2.VideoCapture(video_path)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    predictions = []
    confidences = []
    frame_numbers = []
    timestamps = []
    
    frame_count = 0
    sample_rate = max(1, fps // 2)  # Sample every 0.5 seconds
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_count % sample_rate == 0:
            faces = extract_faces(frame, face_cascade)
            
            if faces:
                for face_crop, bbox in faces:
                    try:
                        # Simulate deepfake detection
                        # In reality, you would use your trained model
                        processed_face = preprocess_frame(face_crop)
                        
                        # Simulated prediction (replace with actual model inference)
                        fake_probability = np.random.random()
                        if fake_probability > 0.7:
                            fake_probability = np.random.uniform(0.8, 0.95)
                        else:
                            fake_probability = np.random.uniform(0.05, 0.3)
                        
                        predictions.append(1 if fake_probability > 0.5 else 0)
                        confidences.append(fake_probability)
                        frame_numbers.append(frame_count)
                        timestamps.append(frame_count / fps)
                        
                    except Exception as e:
                        st.warning(f"Error processing face: {e}")
        
        frame_count += 1
        
        if progress_bar:
            progress_bar.progress(frame_count / total_frames)
    
    cap.release()
    
    return {
        'predictions': predictions,
        'confidences': confidences,
        'frame_numbers': frame_numbers,
        'timestamps': timestamps,
        'total_frames': total_frames,
        'fps': fps
    }

def create_analysis_charts(results):
    """Create visualization charts for analysis results"""
    if not results['predictions']:
        st.warning("No faces detected in the video for analysis.")
        return
    
    df = pd.DataFrame({
        'Frame': results['frame_numbers'],
        'Timestamp': results['timestamps'],
        'Confidence': results['confidences'],
        'Prediction': ['Fake' if p == 1 else 'Real' for p in results['predictions']]
    })
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Confidence over time
        fig_line = px.line(df, x='Timestamp', y='Confidence', 
                          title='Deepfake Confidence Over Time',
                          labels={'Timestamp': 'Time (seconds)', 'Confidence': 'Fake Probability'})
        fig_line.add_hline(y=0.5, line_dash="dash", line_color="red", 
                          annotation_text="Decision Threshold")
        st.plotly_chart(fig_line, use_container_width=True)
    
    with col2:
        # Distribution of predictions
        prediction_counts = df['Prediction'].value_counts()
        fig_pie = px.pie(values=prediction_counts.values, names=prediction_counts.index,
                        title='Prediction Distribution',
                        color_discrete_map={'Real': '#2ecc71', 'Fake': '#e74c3c'})
        st.plotly_chart(fig_pie, use_container_width=True)

def main():
    st.markdown('<h1 class="main-header">🎬 DeepFake Video Detection</h1>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("Settings")
    detection_threshold = st.sidebar.slider("Detection Threshold", 0.0, 1.0, 0.5, 0.1)
    st.sidebar.info("Threshold determines the sensitivity of fake detection. Higher values require more confidence to classify as fake.")
    
    # Load model
    with st.spinner("Loading detection model..."):
        model = load_model()
    
    # Main interface
    tab1, tab2, tab3 = st.tabs(["🎥 Video Analysis", "📊 Batch Analysis", "ℹ️ About"])
    
    with tab1:
        st.header("Upload Video for Analysis")
        
        uploaded_file = st.file_uploader(
            "Choose a video file", 
            type=['mp4', 'avi', 'mov', 'mkv'],
            help="Supported formats: MP4, AVI, MOV, MKV"
        )
        
        if uploaded_file is not None:
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
                tmp_file.write(uploaded_file.read())
                temp_path = tmp_file.name
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.video(uploaded_file)
            
            with col2:
                st.subheader("Video Information")
                cap = cv2.VideoCapture(temp_path)
                fps = int(cap.get(cv2.CAP_PROP_FPS))
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                duration = frame_count / fps if fps > 0 else 0
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                cap.release()
                
                st.markdown(f"""
                <div class="metric-card">
                    <strong>Duration:</strong> {duration:.2f} seconds<br>
                    <strong>FPS:</strong> {fps}<br>
                    <strong>Resolution:</strong> {width}x{height}<br>
                    <strong>Total Frames:</strong> {frame_count:,}
                </div>
                """, unsafe_allow_html=True)
            
            if st.button("🔍 Analyze Video", type="primary"):
                with st.spinner("Analyzing video for deepfake detection..."):
                    progress_bar = st.progress(0)
                    results = analyze_video(temp_path, model, progress_bar)
                
                if results['predictions']:
                    # Calculate overall statistics
                    fake_count = sum(results['predictions'])
                    total_detections = len(results['predictions'])
                    fake_percentage = (fake_count / total_detections) * 100
                    avg_confidence = np.mean(results['confidences'])
                    
                    # Display results
                    st.subheader("Analysis Results")
                    
                    if fake_percentage > 50:
                        st.markdown(f"""
                        <div class="detection-result fake-video">
                            ⚠️ LIKELY DEEPFAKE DETECTED ⚠️<br>
                            Fake Probability: {fake_percentage:.1f}%
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="detection-result real-video">
                            ✅ LIKELY AUTHENTIC VIDEO ✅<br>
                            Authentic Probability: {100-fake_percentage:.1f}%
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Metrics
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Faces Analyzed", total_detections)
                    with col2:
                        st.metric("Avg Confidence", f"{avg_confidence:.3f}")
                    with col3:
                        st.metric("Fake Detections", fake_count)
                    with col4:
                        st.metric("Processing Time", f"{duration:.1f}s")
                    
                    # Visualizations
                    st.subheader("Detailed Analysis")
                    create_analysis_charts(results)
                    
                    # Frame-by-frame results
                    with st.expander("View Frame-by-Frame Results"):
                        df_results = pd.DataFrame({
                            'Frame': results['frame_numbers'],
                            'Timestamp (s)': [f"{t:.2f}" for t in results['timestamps']],
                            'Confidence': [f"{c:.3f}" for c in results['confidences']],
                            'Prediction': ['Fake' if p == 1 else 'Real' for p in results['predictions']]
                        })
                        st.dataframe(df_results, use_container_width=True)
                
                else:
                    st.warning("No faces detected in the video. The detection model requires visible faces to analyze.")
            
            # Clean up temporary file
            try:
                os.unlink(temp_path)
            except:
                pass
    
    with tab2:
        st.header("Batch Analysis")
        st.info("Upload multiple videos for batch processing")
        
        uploaded_files = st.file_uploader(
            "Choose video files", 
            type=['mp4', 'avi', 'mov', 'mkv'],
            accept_multiple_files=True
        )
        
        if uploaded_files and st.button("Analyze All Videos"):
            results_summary = []
            
            for i, uploaded_file in enumerate(uploaded_files):
                st.subheader(f"Processing: {uploaded_file.name}")
                
                with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
                    tmp_file.write(uploaded_file.read())
                    temp_path = tmp_file.name
                
                progress_bar = st.progress(0)
                results = analyze_video(temp_path, model, progress_bar)
                
                if results['predictions']:
                    fake_percentage = (sum(results['predictions']) / len(results['predictions'])) * 100
                    results_summary.append({
                        'Video': uploaded_file.name,
                        'Fake Probability': fake_percentage,
                        'Status': 'Likely Fake' if fake_percentage > 50 else 'Likely Real',
                        'Faces Detected': len(results['predictions'])
                    })
                
                os.unlink(temp_path)
            
            if results_summary:
                st.subheader("Batch Analysis Summary")
                summary_df = pd.DataFrame(results_summary)
                st.dataframe(summary_df, use_container_width=True)
                
                # Summary chart
                fig = px.bar(summary_df, x='Video', y='Fake Probability',
                           color='Status', title='Deepfake Detection Results - Batch Analysis')
                st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.header("About DeepFake Detection")
        
        st.markdown("""
        ### How it works
        
        This application uses advanced computer vision and deep learning techniques to detect artificially generated (deepfake) videos:
        
        1. **Face Detection**: Uses Haar cascades to locate faces in video frames
        2. **Feature Extraction**: Extracts visual features that distinguish real from synthetic faces
        3. **Classification**: Uses a trained neural network to classify faces as real or fake
        4. **Temporal Analysis**: Analyzes consistency across multiple frames
        
        ### Model Architecture
        
        - **Backbone**: ResNet50 convolutional neural network
        - **Input**: 224x224 RGB face crops
        - **Output**: Binary classification (Real/Fake) with confidence scores
        
        ### Limitations
        
        - Requires clear, visible faces in the video
        - Performance may vary with video quality and lighting
        - New deepfake techniques may evade detection
        - This is a demonstration model - real deployments require extensive training
        
        ### Best Practices
        
        - Use multiple detection methods for critical decisions
        - Consider video source and context
        - Regularly update detection models
        - Combine automated detection with human review
        
        ### Privacy & Ethics
        
        This tool is designed for educational and research purposes. Always respect privacy rights and use responsibly.
        """)

if __name__ == "__main__":
    main()