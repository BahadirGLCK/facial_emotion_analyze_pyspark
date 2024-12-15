from pyspark.sql import SparkSession
import os
from src.preprocessor import ImagePreprocessor
from src.utils.logger import get_logger

def preprocess_image(row, raw_dir, processed_dir, size):
    """
    Preprocess a single image by cropping and resizing it based on bbox.
    """
    from src.preprocessor import ImagePreprocessor  # Import required for PySpark workers
    import os

    name = row["name"].replace(" ", "_")  # Replace spaces with underscores
    image_id = row["image_id"]
    bbox = row["bbox"]

    # Path to the raw image
    image_path = os.path.join(raw_dir, name, f"{image_id}.jpg")

    # Skip if raw image does not exist
    if not os.path.exists(image_path):
        return f"Raw image not found: {image_path}"

    # Save name for the processed image
    save_name = f"{image_id}.jpg"

    # Initialize Preprocessor
    preprocessor = ImagePreprocessor(save_dir=processed_dir, size=size)

    # Preprocess the image
    return preprocessor.preprocess_image(image_path=image_path, bbox=bbox, save_name=save_name, person_name=name)


def main():
    # Initialize logger
    logger = get_logger("PreprocessorTrigger")

    # Initialize Spark Session
    spark = SparkSession.builder \
        .appName("FaceScrub Preprocessor") \
        .config("spark.executor.memory", "4g") \
        .config("spark.driver.memory", "4g") \
        .config("spark.executor.cores", "2") \
        .getOrCreate()

    # Define metadata files
    metadata_files = [
        "data/facescrub_actors.txt",
        "data/facescrub_actresses.txt"
    ]

    # Directories for raw and processed images
    raw_dir = "data/raw/images"
    processed_dir = "data/processed/images"

    # Preprocessing size
    size = (128, 128)

    # Process each metadata file
    for metadata_path in metadata_files:
        # Check if the metadata file exists
        if not os.path.exists(metadata_path):
            logger.warning(f"Metadata file not found: {metadata_path}")
            continue

        # Load metadata as a DataFrame
        logger.info(f"Loading metadata from {metadata_path}...")
        df = spark.read.option("header", True).option("delimiter", "\t").csv(metadata_path)
        df.show(5)

        # Convert DataFrame to RDD for distributed processing
        rdd = df.rdd.map(lambda row: preprocess_image(row.asDict(), raw_dir, processed_dir, size))

        # Trigger preprocessing and collect results
        logger.info(f"Starting preprocessing for {metadata_path}...")
        results = rdd.collect()

        # Log the results
        for result in results:
            logger.info(result)

    logger.info("Image preprocessing completed for all files!")

if __name__ == "__main__":
    main()