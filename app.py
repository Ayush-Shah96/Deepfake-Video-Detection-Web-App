import streamlit as st
import cv2
import numpy as np
import tempfile
import os
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from main import DeepfakeDetector
import time
from PIL import Image
import io

# Page configuration
st.set_page_config(
    page_title="Deepfake Video Detection",
    page_icon="🎥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
        background: linear-gradient(90deg, #ff6b6b, #4ecdc4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .detection-card {
        background-color: #f0f0f0;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #4CAF50;
        margin: 1rem 0;
    }
    
    .deepfake-alert {
        background-color: #ffebee;
        border-left: 5px solid #f44336;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    
    .real-alert {
        background-color: #e8f5e8;
        border-left: 5px solid #4CAF50;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    
    .stProgress .st-bo {
        background-color: #4CAF50;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'detector' not in st.session_state:
    st.session_state.detector = None
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = None
if 'processing' not in st.session_state:
    st.session_state.processing = False

def initialize_detector():
    """Initialize the deepfake detector"""
    if st.session_state.detector is None:
        with st.spinner("Loading deepfake detection model..."):
            st.session_state.detector = DeepfakeDetector()
            # Train with minimal synthetic data for demo
            X_train, y_train = np.random.random((50, 224, 224, 3)), np.random.randint(0, 2, 50)
            st.session_state.detector.train_model(X_train, y_train, epochs=1)

def create_confidence_gauge(confidence, prediction):
    """Create a confidence gauge chart"""
    color = "#f44336" if prediction == "DEEPFAKE" else "#4CAF50"
    
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = confidence,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': f"Confidence: {prediction}"},
        gauge = {
            'axis': {'range': [None, 100]},
            'bar': {'color': color},
            'steps': [
                {'range': [0, 50], 'color': "lightgray"},
                {'range': [50, 80], 'color': "yellow"},
                {'range': [80, 100], 'color': color}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    
    fig.update_layout(height=300)
    return fig

def create_frame_analysis_chart(results):
    """Create a chart showing frame-by-frame analysis"""
    if 'frame_predictions' not in results or 'frame_confidences' not in results:
        return None
    
    df = pd.DataFrame({
        'Frame': range(len(results['frame_predictions'])),
        'Prediction': results['frame_predictions'],
        'Confidence': results['frame_confidences']
    })
    
    # Create color mapping
    df['Color'] = df['Prediction'].map({'DEEPFAKE': 'red', 'REAL': 'green'})
    
    fig = px.scatter(df, x='Frame', y='Confidence', color='Prediction',
                     color_discrete_map={'DEEPFAKE': '#f44336', 'REAL': '#4CAF50'},
                     title="Frame-by-Frame Analysis",
                     labels={'Confidence': 'Confidence (%)'})
    
    fig.update_layout(height=400)
    return fig

def create_summary_pie_chart(results):
    """Create a pie chart showing the distribution of predictions"""
    if 'deepfake_frames' not in results or 'real_frames' not in results:
        return None
    
    labels = ['Real Frames', 'Deepfake Frames']
    values = [results['real_frames'], results['deepfake_frames']]
    colors = ['#4CAF50', '#f44336']
    
    fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.3)])
    fig.update_traces(hoverinfo='label+percent', textinfo='value+percent', 
                      marker=dict(colors=colors, line=dict(color='#FFFFFF', width=2)))
    fig.update_layout(title_text="Frame Analysis Summary", height=400)
    
    return fig

def process_video(uploaded_file):
    """Process uploaded video file"""
    if uploaded_file is None:
        return None
    
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_file_path = tmp_file.name
    
    try:
        # Initialize detector if needed
        initialize_detector()
        
        # Analyze video
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        status_text.text("Analyzing video frames...")
        results = st.session_state.detector.analyze_video(tmp_file_path, sample_rate=30)
        progress_bar.progress(100)
        status_text.text("Analysis complete!")
        
        return results
    
    except Exception as e:
        st.error(f"Error processing video: {str(e)}")
        return None
    
    finally:
        # Clean up temporary file
        if os.path.exists(tmp_file_path):
            os.unlink(tmp_file_path)

def main():
    # Header
    st.markdown('<h1 class="main-header">🎥 Deepfake Video Detection</h1>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("📋 Instructions")
        st.markdown("""
        1. Upload a video file (MP4, AVI, MOV)
        2. Click 'Analyze Video' to start detection
        3. View detailed results and confidence scores
        4. Download analysis report (optional)
        """)
        
        st.header("⚙️ Settings")
        sample_rate = st.slider("Frame Sampling Rate", 1, 60, 30, 
                               help="Analyze every Nth frame (higher = faster but less accurate)")
        
        st.header("ℹ️ About")
        st.markdown("""
        This tool uses deep learning to detect deepfake videos by analyzing facial features and temporal inconsistencies.
        
        **Note**: This is a demo implementation. Production systems would use more sophisticated models trained on large datasets.
        """)
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("Upload Video")
        uploaded_file = st.file_uploader(
            "Choose a video file",
            type=['mp4', 'avi', 'mov', 'mkv'],
            help="Upload a video file to analyze for deepfake content"
        )
        
        if uploaded_file is not None:
            # Display video info
            st.success(f"✅ Video uploaded: {uploaded_file.name}")
            st.info(f"📊 File size: {uploaded_file.size / (1024*1024):.2f} MB")
            
            # Analyze button
            if st.button("🔍 Analyze Video", type="primary", disabled=st.session_state.processing):
                st.session_state.processing = True
                
                with st.spinner("Analyzing video for deepfake content..."):
                    results = process_video(uploaded_file)
                    if results:
                        st.session_state.analysis_results = results
                
                st.session_state.processing = False
    
    with col2:
        if uploaded_file is not None:
            st.header("Video Preview")
            st.video(uploaded_file)
    
    # Results section
    if st.session_state.analysis_results:
        results = st.session_state.analysis_results
        
        if 'error' in results:
            st.error(f"❌ Error: {results['error']}")
            return
        
        st.header("🎯 Detection Results")
        
        # Main result display
        prediction = results['overall_prediction']
        confidence = results['overall_confidence']
        
        if prediction == "DEEPFAKE":
            st.markdown(f"""
            <div class="deepfake-alert">
                <h3>⚠️ DEEPFAKE DETECTED</h3>
                <p><strong>Confidence:</strong> {confidence:.1f}%</p>
                <p>This video appears to contain deepfake content.</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="real-alert">
                <h3>✅ AUTHENTIC VIDEO</h3>
                <p><strong>Confidence:</strong> {confidence:.1f}%</p>
                <p>This video appears to be authentic.</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Detailed metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Overall Confidence", f"{confidence:.1f}%")
        
        with col2:
            st.metric("Deepfake Frames", results['deepfake_frames'])
        
        with col3:
            st.metric("Real Frames", results['real_frames'])
        
        with col4:
            st.metric("Total Analyzed", results['total_analyzed_frames'])
        
        # Charts section
        st.header("📊 Detailed Analysis")
        
        tab1, tab2, tab3 = st.tabs(["Confidence Gauge", "Frame Analysis", "Summary"])
        
        with tab1:
            fig_gauge = create_confidence_gauge(confidence, prediction)
            st.plotly_chart(fig_gauge, use_container_width=True)
        
        with tab2:
            fig_frames = create_frame_analysis_chart(results)
            if fig_frames:
                st.plotly_chart(fig_frames, use_container_width=True)
            else:
                st.info("Frame analysis data not available")
        
        with tab3:
            col1, col2 = st.columns(2)
            
            with col1:
                fig_pie = create_summary_pie_chart(results)
                if fig_pie:
                    st.plotly_chart(fig_pie, use_container_width=True)
            
            with col2:
                st.subheader("Analysis Summary")
                st.write(f"**Duration:** {results['duration']:.2f} seconds")
                st.write(f"**Frame Rate:** {results['frame_rate']:.1f} fps")
                st.write(f"**Deepfake Percentage:** {results['deepfake_percentage']:.1f}%")
                st.write(f"**Average Confidence:** {results['average_confidence']:.1f}%")
        
        # Export results
        st.header("📥 Export Results")
        
        # Create downloadable report
        report_data = {
            'video_analysis_report': {
                'overall_prediction': prediction,
                'confidence': confidence,
                'summary': results
            }
        }
        
        report_json = pd.Series(report_data).to_json(indent=2)
        
        col1, col2 = st.columns(2)
        with col1:
            st.download_button(
                label="📋 Download JSON Report",
                data=report_json,
                file_name=f"deepfake_analysis_{int(time.time())}.json",
                mime="application/json"
            )
        
        with col2:
            # Create CSV summary
            summary_df = pd.DataFrame([{
                'Prediction': prediction,
                'Confidence': f"{confidence:.1f}%",
                'Deepfake_Frames': results['deepfake_frames'],
                'Real_Frames': results['real_frames'],
                'Total_Frames_Analyzed': results['total_analyzed_frames'],
                'Duration_Seconds': results['duration'],
                'Deepfake_Percentage': f"{results['deepfake_percentage']:.1f}%"
            }])
            
            csv = summary_df.to_csv(index=False)
            st.download_button(
                label="📊 Download CSV Summary",
                data=csv,
                file_name=f"deepfake_summary_{int(time.time())}.csv",
                mime="text/csv"
            )

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>⚠️ <strong>Disclaimer:</strong> This is a demonstration tool. Results should not be used as definitive proof of deepfake content. 
        Professional verification is recommended for critical applications.</p>
        <p>Built with Streamlit and TensorFlow | © 2025 Deepfake Detection System</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()