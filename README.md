# 🔍 Deepfake Video Detection System

A comprehensive deepfake detection system with both Google Colab notebook and Streamlit web application interfaces. This project uses deep learning models to analyze video content and detect potentially manipulated or artificially generated faces.

## 🌟 Features

- **🎯 Real-time deepfake detection** using EfficientNet backbone
- **👥 Face extraction** with OpenCV Haar cascades
- **📊 Comprehensive analysis** with detailed statistics and visualizations
- **🌐 Web interface** built with Streamlit
- **📱 Responsive design** that works on desktop and mobile
- **🎮 Demo mode** for testing without uploading files
- **⚙️ Configurable parameters** for processing optimization
- **📈 Interactive charts** using Plotly

## 🚀 Quick Start

### Option 1: Streamlit Web App (Recommended)

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/deepfake-detection.git
   cd deepfake-detection
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the web app**
   ```bash
   streamlit run app.py
   ```

4. **Open your browser** and go to `http://localhost:8501`

### Option 2: Google Colab

1. Upload `deepfake_detection_colab.ipynb` to Google Colab
2. Run all cells to install dependencies and initialize the system
3. Use the provided functions to analyze videos

## 🛠️ Installation

### System Requirements

- **Python**: 3.8 or higher
- **Operating System**: Windows, macOS, or Linux
- **Memory**: Minimum 4GB RAM (8GB recommended)
- **Storage**: At least 2GB free space
- **GPU**: CUDA-compatible GPU (optional, for faster processing)

### Dependencies Installation

1. **Create virtual environment** (recommended)
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. **Install PyTorch** (choose your platform)
   ```bash
   # CPU only
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
   
   # CUDA 11.8
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   
   # CUDA 12.1
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
   ```

3. **Install other dependencies**
   ```bash
   pip install -r requirements.txt
   ```

### Docker Installation (Alternative)

```bash
# Build the image
docker build -t deepfake-detection .

# Run the container
docker run -p 8501:8501 deepfake-detection
```

## 📖 Usage

### Web Application

1. **Start the Streamlit app**
   ```bash
   streamlit run app.py
   ```

2. **Upload a video** using the file uploader
3. **Adjust settings** in the sidebar (optional)
4. **Click "Analyze Video"** to start detection
5. **View results** with confidence scores and visualizations

### Google Colab

Open `deepfake_detection_colab.ipynb` in Google Colab and follow the notebook instructions.

## 🎯 How It Works

### 1. **Face Detection**
- Uses OpenCV Haar Cascade classifiers to detect faces in video frames
- Extracts face regions with padding for better analysis
- Supports multiple faces per frame

### 2. **Preprocessing**
- Resizes detected faces to 224x224 pixels
- Applies normalization using ImageNet statistics
- Converts to PyTorch tensors for model input

### 3. **Deep Learning Model**
- **Backbone**: EfficientNet-B0 for feature extraction
- **Architecture**: Custom classifier with dropout layers
- **Output**: Binary classification (Real vs Fake)
- **Activation**: Softmax for probability scores

### 4. **Post-processing**
- Aggregates predictions across all detected faces
- Calculates confidence scores and statistics
- Applies threshold-based classification

## 📊 Performance Metrics

The system provides several metrics to evaluate detection quality:

- **Average Probability**: Mean fake probability across all faces
- **Confidence Score**: System confidence in the prediction
- **Face Count**: Number of faces analyzed
- **Classification Accuracy**: Percentage above/below threshold
- **Temporal Consistency**: Variation in predictions over time


### Analysis Results
- Interactive charts showing prediction timeline
- Statistical summaries with confidence scores
- Classification breakdown (Real vs Fake)

## ⚠️ Important Notes

### Model Limitations

- **Demo Model**: Current implementation uses randomly initialized weights for demonstration
- **Training Data**: For production use, train on datasets like DFDC or FaceForensics++
- **Accuracy**: Detection accuracy depends on video quality and face visibility
- **Performance**: Processing time varies with video length and hardware

### Ethical Considerations

- **Privacy**: Ensure you have permission to analyze uploaded videos
- **Bias**: Be aware of potential biases in training data
- **Misuse**: This tool should be used responsibly and ethically
- **Legal**: Comply with local laws regarding deepfake detection

### Security

- **Data Handling**: Uploaded videos are processed locally and not stored permanently
- **Model Security**: Use trusted model weights and validate inputs
- **Dependencies**: Keep dependencies updated for security patches

## 🔮 Future Enhancements

### Planned Features
- [ ] **Advanced Face Detection** using MTCNN or RetinaFace
- [ ] **Temporal Analysis** for video-level consistency
- [ ] **Ensemble Models** combining multiple detection approaches
- [ ] **Real-time Processing** for live video streams
- [ ] **Mobile App** for on-device detection
- [ ] **API Endpoints** for programmatic access
- [ ] **Batch Processing** for multiple videos
- [ ] **Cloud Storage** integration

### Model Improvements
- [ ] **Pre-trained Weights** from established datasets
- [ ] **Multi-scale Analysis** for different face sizes
- [ ] **Attention Mechanisms** for focus on key features
- [ ] **Transfer Learning** from domain-specific models
- [ ] **Adversarial Training** for robustness

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](docs/CONTRIBUTING.md) for guidelines.

### Development Setup

1. **Fork the repository**
2. **Create feature branch**
   ```bash
   git checkout -b feature/amazing-feature
   ```
3. **Make changes and test**
4. **Submit pull request**


## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

### Datasets
- **DFDC**: Deepfake Detection Challenge Dataset
- **FaceForensics++**: Large-scale face manipulation dataset
- **Celeb-DF**: High-quality deepfake dataset
- **DeeperForensics-1.0**: Large-scale challenging dataset


**⭐ Star this repository if you find it helpful!**

**🔗 Share with others who might be interested in deepfake detection!**
