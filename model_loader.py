# model_loader_tf.py
import tensorflow as tf
from tensorflow import keras
from PIL import Image
import numpy as np
import os

class CatDogClassifier:
    """Cat/Dog classifier wrapper for TensorFlow/Keras model"""
    
    def __init__(self, model_path: str):
        self.model = self._load_model(model_path)
        self.image_size = (128, 128)
        self.class_names = ['cat', 'dog']
        
    def _load_model(self, model_path: str):
        """Load the trained Keras model"""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at {model_path}")
        
        try:
            model = keras.models.load_model(model_path)
            return model
        except Exception as e:
            raise RuntimeError(f"Failed to load Keras model: {e}")
    
    def preprocess_image(self, image: Image.Image):
        """Preprocess image for model input"""
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Resize and convert to array
        image = image.resize(self.image_size)
        image_array = np.array(image) / 255.0  # Normalize to [0,1]
        
        # Add batch dimension
        return np.expand_dims(image_array, axis=0)
    
    def predict(self, image: Image.Image):
        """Make prediction on a single image"""
        input_tensor = self.preprocess_image(image)
        
        # Make prediction
        predictions = self.model.predict(input_tensor, verbose=0)
        pred_class = np.argmax(predictions[0])
        confidence = float(np.max(predictions[0]))
        
        return {
            'prediction': self.class_names[pred_class],
            'confidence': confidence,
            'probabilities': {
                'cat': float(predictions[0][0]),
                'dog': float(predictions[0][1])
            }
        }