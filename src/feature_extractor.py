from src.utils.logger import get_logger
import numpy as np
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import img_to_array, load_img

class FeatureExtractor:
    def __init__(self):
        self.logger = get_logger("FeatureExtractor")  # Initialize logger
        weights_path = "/Users/bahadirgolcuk/bahadir/project/facial_emotion_analyze_pyspark/models/mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_128_no_top.h5"
        self.model = MobileNetV2(weights=weights_path, include_top=False, input_shape=(128, 128, 3))

    def extract_features(self, image_path: str) -> np.ndarray:
        try:
            image = load_img(image_path, target_size=(128, 128))
            image = img_to_array(image)
            image = np.expand_dims(image, axis=0)
            features = self.model.predict(image)
            self.logger.info(f"Extracted features from {image_path}")
            return features.flatten()
        except Exception as e:
            self.logger.error(f"Error extracting features from {image_path}: {e}")
            return np.array([])
    
    def extract_features_from_array(self, image_array: np.ndarray) -> np.ndarray:
        """
        Extract features from a preprocessed image (in-memory NumPy array).
        """
        try:
            image = np.expand_dims(image_array, axis=0)  # Expand dimensions for batch processing
            features = self.model.predict(image)
            return features.flatten()
        except Exception as e:
            return f"Error extracting features: {e}"