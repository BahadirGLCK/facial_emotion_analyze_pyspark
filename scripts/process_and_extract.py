from multiprocessing import Pool, cpu_count
import numpy as np
import os
from tqdm import tqdm
from src.preprocessor import ImagePreprocessor
from src.feature_extractor import FeatureExtractor

# Global variable to hold the model
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
EXTRACTOR = FeatureExtractor()

def preprocess_and_extract_multiprocess(args):
    """
    Preprocess an image and extract its features without saving the preprocessed image.
    """
    image_path, bbox, features_dir, name, image_id = args

    # Initialize Preprocessor and Feature Extractor
    preprocessor = ImagePreprocessor(size=(128, 128))
    global EXTRACTOR

    # Preprocess the image
    preprocessed_image, status = preprocessor.preprocess_image(image_path, bbox)
    if preprocessed_image is None:
        return status  # Return error message if preprocessing fails

    # Extract features
    try:
        features = EXTRACTOR.extract_features_from_array(preprocessed_image)
    except Exception as e:
        return f"Feature extraction failed for {image_path}: {e}"

    # Save features
    feature_save_path = os.path.join(features_dir, name)
    os.makedirs(feature_save_path, exist_ok=True)
    feature_file = os.path.join(feature_save_path, f"{image_id}.npy")

    try:
        np.save(feature_file, features)
        return f"Features extracted and saved for {image_path}"
    except Exception as e:
        return f"Failed to save features for {image_path}: {e}"

def main():
    # Directories for raw images and feature storage
    raw_dir = "data/raw/images"
    features_dir = "data/features"

    # Metadata files
    metadata_files = [
        "data/facescrub_actors.txt",
        "data/facescrub_actresses.txt"
    ]

    # Collect all tasks
    tasks = []
    for metadata_path in metadata_files:
        if not os.path.exists(metadata_path):
            print(f"Metadata file not found: {metadata_path}")
            continue

        # Read metadata
        with open(metadata_path, "r") as f:
            next(f)  # Skip header
            for line in f:
                fields = line.strip().split("\t")
                name, image_id, _, _, bbox, _ = fields
                name = name.replace(" ", "_")  # Replace spaces with underscores
                image_path = os.path.join(raw_dir, name, f"{image_id}.jpg")

                # Skip if raw image does not exist
                if not os.path.exists(image_path):
                    print(f"Raw image not found: {image_path}")
                    continue

                # Add task
                tasks.append((image_path, bbox, features_dir, name, image_id))

    # Use multiprocessing for faster processing
    print(f"Starting multiprocessing with {cpu_count()} cores...")
    with Pool(cpu_count()) as pool:
        # Use tqdm to display progress
        results = list(tqdm(pool.imap(preprocess_and_extract_multiprocess, tasks), total=len(tasks)))

    # Log results
    for result in results:
        print(result)

    print("Preprocessing and feature extraction completed!")

if __name__ == "__main__":
    main()