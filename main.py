import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import img_to_array
from sklearn.metrics import classification_report
import os
import logging
from typing import Tuple, List, Optional
import tempfile

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DeepfakeDetector:
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize the deepfake detector
        
        Args:
            model_path: Path to pre-trained model. If None, creates a new model.
        """
        self.model = None
        self.img_size = (224, 224)
        self.model_path = model_path
        
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
        else:
            self.model = self._create_model()
    
    def _create_model(self) -> tf.keras.Model:
        """
        Create a CNN model for deepfake detection
        
        Returns:
            Compiled Keras model
        """
        model = Sequential([
            # First convolutional block
            Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
            BatchNormalization(),
            MaxPooling2D(2, 2),
            
            # Second convolutional block
            Conv2D(64, (3, 3), activation='relu'),
            BatchNormalization(),
            MaxPooling2D(2, 2),
            
            # Third convolutional block
            Conv2D(128, (3, 3), activation='relu'),
            BatchNormalization(),
            MaxPooling2D(2, 2),
            
            # Fourth convolutional block
            Conv2D(256, (3, 3), activation='relu'),
            BatchNormalization(),
            MaxPooling2D(2, 2),
            
            # Flatten and dense layers
            Flatten(),
            Dense(512, activation='relu'),
            Dropout(0.5),
            Dense(256, activation='relu'),
            Dropout(0.3),
            Dense(1, activation='sigmoid')  # Binary classification
        ])
        
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        logger.info("Model created successfully")
        return model
    
    def preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Preprocess a single frame for prediction
        
        Args:
            frame: Input frame as numpy array
            
        Returns:
            Preprocessed frame
        """
        # Resize frame
        frame_resized = cv2.resize(frame, self.img_size)
        
        # Convert BGR to RGB (OpenCV uses BGR)
        frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        
        # Normalize pixel values
        frame_normalized = frame_rgb.astype('float32') / 255.0
        
        # Add batch dimension
        frame_batch = np.expand_dims(frame_normalized, axis=0)
        
        return frame_batch
    
    def extract_faces(self, frame: np.ndarray) -> List[np.ndarray]:
        """
        Extract faces from frame using Haar cascade
        
        Args:
            frame: Input frame
            
        Returns:
            List of face regions
        """
        # Load Haar cascade for face detection
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Convert to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        
        face_regions = []
        for (x, y, w, h) in faces:
            # Extract face region with some padding
            padding = 20
            x1 = max(0, x - padding)
            y1 = max(0, y - padding)
            x2 = min(frame.shape[1], x + w + padding)
            y2 = min(frame.shape[0], y + h + padding)
            
            face_region = frame[y1:y2, x1:x2]
            if face_region.size > 0:
                face_regions.append(face_region)
        
        return face_regions
    
    def predict_frame(self, frame: np.ndarray) -> Tuple[float, str]:
        """
        Predict if a frame contains a deepfake
        
        Args:
            frame: Input frame
            
        Returns:
            Tuple of (confidence_score, prediction_label)
        """
        if self.model is None:
            raise ValueError("Model not loaded or created")
        
        # Extract faces from frame
        faces = self.extract_faces(frame)
        
        if not faces:
            # If no faces detected, analyze the whole frame
            preprocessed = self.preprocess_frame(frame)
        else:
            # Use the largest face detected
            largest_face = max(faces, key=lambda x: x.shape[0] * x.shape[1])
            preprocessed = self.preprocess_frame(largest_face)
        
        # Make prediction
        prediction = self.model.predict(preprocessed, verbose=0)[0][0]
        
        # Convert to confidence and label
        if prediction > 0.5:
            confidence = prediction * 100
            label = "DEEPFAKE"
        else:
            confidence = (1 - prediction) * 100
            label = "REAL"
        
        return confidence, label
    
    def analyze_video(self, video_path: str, sample_rate: int = 30) -> dict:
        """
        Analyze a video file for deepfake content
        
        Args:
            video_path: Path to video file
            sample_rate: Analyze every nth frame
            
        Returns:
            Dictionary with analysis results
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
        
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        duration = frame_count / fps if fps > 0 else 0
        
        predictions = []
        confidences = []
        frame_idx = 0
        
        logger.info(f"Analyzing video: {frame_count} frames, {duration:.2f}s duration")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Sample frames at specified rate
            if frame_idx % sample_rate == 0:
                try:
                    confidence, label = self.predict_frame(frame)
                    predictions.append(label)
                    confidences.append(confidence)
                except Exception as e:
                    logger.warning(f"Error processing frame {frame_idx}: {e}")
            
            frame_idx += 1
        
        cap.release()
        
        if not predictions:
            return {
                'error': 'No frames could be analyzed',
                'frame_count': frame_count,
                'duration': duration
            }
        
        # Calculate statistics
        deepfake_count = predictions.count('DEEPFAKE')
        real_count = predictions.count('REAL')
        avg_confidence = np.mean(confidences)
        
        # Determine overall prediction
        if deepfake_count > real_count:
            overall_prediction = 'DEEPFAKE'
            overall_confidence = np.mean([c for i, c in enumerate(confidences) if predictions[i] == 'DEEPFAKE'])
        else:
            overall_prediction = 'REAL'
            overall_confidence = np.mean([c for i, c in enumerate(confidences) if predictions[i] == 'REAL'])
        
        results = {
            'overall_prediction': overall_prediction,
            'overall_confidence': overall_confidence,
            'deepfake_frames': deepfake_count,
            'real_frames': real_count,
            'total_analyzed_frames': len(predictions),
            'frame_rate': fps,
            'duration': duration,
            'deepfake_percentage': (deepfake_count / len(predictions)) * 100,
            'average_confidence': avg_confidence,
            'frame_predictions': predictions,
            'frame_confidences': confidences
        }
        
        return results
    
    def save_model(self, path: str):
        """Save the trained model"""
        if self.model:
            self.model.save(path)
            logger.info(f"Model saved to {path}")
    
    def load_model(self, path: str):
        """Load a pre-trained model"""
        try:
            self.model = tf.keras.models.load_model(path)
            logger.info(f"Model loaded from {path}")
        except Exception as e:
            logger.error(f"Error loading model from {path}: {e}")
            self.model = self._create_model()
    
    def train_model(self, train_data, train_labels, validation_data=None, epochs=10):
        """
        Train the model with provided data
        
        Args:
            train_data: Training data
            train_labels: Training labels
            validation_data: Optional validation data
            epochs: Number of training epochs
        """
        if self.model is None:
            self.model = self._create_model()
        
        history = self.model.fit(
            train_data,
            train_labels,
            epochs=epochs,
            validation_data=validation_data,
            batch_size=32,
            verbose=1
        )
        
        return history

# Utility function for demo purposes
def create_synthetic_training_data(num_samples: int = 1000):
    """
    Create synthetic training data for demonstration
    This would normally be replaced with real deepfake/real video data
    """
    X = np.random.random((num_samples, 224, 224, 3))
    y = np.random.randint(0, 2, num_samples)  # 0 for real, 1 for deepfake
    return X, y

if __name__ == "__main__":
    # Example usage
    detector = DeepfakeDetector()
    
    # Create and train with synthetic data (for demo)
    X_train, y_train = create_synthetic_training_data(100)
    X_val, y_val = create_synthetic_training_data(20)
    
    print("Training model with synthetic data (for demo purposes)...")
    detector.train_model(X_train, y_train, (X_val, y_val), epochs=2)
    
    print("Model training completed!")
    print("Note: In production, use real deepfake detection datasets like FaceForensics++, DFDC, etc.")