from src.feature_extractor import FeatureExtractor
from src.utils.logger import get_logger
import os
import numpy as np

def main():
    # Initialize logger
    logger = get_logger("FeatureExtractorTrigger")

    # Initialize FeatureExtractor
    extractor = FeatureExtractor()

    # Directory containing preprocessed images
    preprocessed_dir = "data/processed"
    features_dir = "data/features"
    os.makedirs(features_dir, exist_ok=True)

    # Extract features for each image
    logger.info("Starting feature extraction...")
    for root, _, files in os.walk(preprocessed_dir):
        for file in files:
            image_path = os.path.join(root, file)
            features = extractor.extract_features(image_path)
            if features.size > 0:
                feature_path = os.path.join(features_dir, f"{os.path.splitext(file)[0]}.npy")
                np.save(feature_path, features)
                logger.info(f"Features saved for {file} at {feature_path}")
            else:
                logger.warning(f"Failed to extract features for {file}")

    logger.info("Feature extraction completed!")

if __name__ == "__main__":
    main()