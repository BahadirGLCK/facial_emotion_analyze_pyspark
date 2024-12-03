from pyspark.sql import SparkSession
from src.preprocessor import ImagePreprocessor
from src.utils.logger import get_logger

def main():
    # Initialize logger
    logger = get_logger("PreprocessorTrigger")

    # Initialize Spark Session
    spark = SparkSession.builder \
        .appName("FaceScrub Preprocessor") \
        .getOrCreate()

    # Path to metadata file
    metadata_path = "data/facescrub_actors.txt"

    # Load metadata as a DataFrame
    logger.info("Loading metadata...")
    df = spark.read.option("header", True).option("delimiter", "\t").csv(metadata_path)
    df.show(5)

    # Initialize Preprocessor
    preprocessor = ImagePreprocessor(save_dir="data/processed", size=(128, 128))

    # Process each image
    logger.info("Starting image preprocessing...")
    for row in df.collect():
        image_path = f"data/images/{row['image_id']}.jpg"
        bbox = row["bbox"]
        save_name = f"{row['image_id']}.jpg"
        status = preprocessor.preprocess_image(image_path, bbox, save_name)
        logger.info(f"Processed {save_name}: {status}")

    logger.info("Image preprocessing completed!")

if __name__ == "__main__":
    main()