from pyspark.sql import SparkSession
from pyspark.sql.functions import col
import os
import requests
from src.utils.logger import get_logger

def download_image(row, save_dir):
    """
    Download a single image using the URL and save it in the appropriate directory.
    """
    name = row["name"].replace(" ", "_")  # Replace spaces with underscores
    image_id = row["image_id"]
    url = row["url"]

    # Create a directory for the person
    person_dir = os.path.join(save_dir, name)
    os.makedirs(person_dir, exist_ok=True)

    # Define the image save path
    save_path = os.path.join(person_dir, f"{image_id}.jpg")

    try:
        response = requests.get(url, stream=True)
        if response.status_code == 200:
            with open(save_path, 'wb') as f:
                for chunk in response.iter_content(1024):
                    f.write(chunk)
            return f"Downloaded {image_id} for {name}"
        else:
            return f"Failed {image_id} for {name}: HTTP {response.status_code}"
    except Exception as e:
        return f"Failed {image_id} for {name}: {e}"

def main():
    # Initialize logger
    logger = get_logger("DownloaderTrigger")

    # Initialize Spark Session
    spark = SparkSession.builder \
        .appName("FaceScrub Downloader with PySpark") \
        .getOrCreate()

    # Define metadata files
    metadata_files = [
        "data/facescrub_actors.txt",
        "data/facescrub_actresses.txt"
    ]

    # Output directory
    save_dir = "data/raw/images"

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

        # Convert DataFrame to RDD for parallel processing
        rdd = df.select("name", "image_id", "url").rdd

        # Broadcast the save directory to all workers
        save_dir_broadcast = spark.sparkContext.broadcast(save_dir)

        # Perform parallel downloads
        logger.info(f"Starting parallel image downloads for {metadata_path}...")
        results = rdd.map(lambda row: download_image(row.asDict(), save_dir_broadcast.value)).collect()

        # Log results
        for result in results:
            logger.info(result)

    logger.info("Image downloads completed for all files!")

if __name__ == "__main__":
    main()