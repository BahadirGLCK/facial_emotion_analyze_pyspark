from pyspark.sql import SparkSession
import aiohttp
import asyncio
import os
from src.utils.logger import get_logger

async def async_download_image(url, save_path):
    async with aiohttp.ClientSession() as session:
        try:
            async with session.get(url) as response:
                if response.status == 200:
                    with open(save_path, 'wb') as f:
                        f.write(await response.read())
                    return "success"
                else:
                    return f"failed: HTTP {response.status}"
        except Exception as e:
            return f"failed: {e}"

def download_image(row, save_dir):
    name = row["name"].replace(" ", "_")  # Replace spaces with underscores
    image_id = row["image_id"]
    url = row["url"]

    # Create a directory for the person
    person_dir = os.path.join(save_dir, name)
    os.makedirs(person_dir, exist_ok=True)

    # Define the image save path
    save_path = os.path.join(person_dir, f"{image_id}.jpg")

    # Check if the file already exists
    if os.path.exists(save_path):
        return f"File already exists: {save_path}"

    # Run the async download in an event loop
    loop = asyncio.get_event_loop()
    return loop.run_until_complete(async_download_image(url, save_path))

def main():
    # Initialize logger
    logger = get_logger("DownloaderTrigger")

    # Initialize Spark Session
    spark = SparkSession.builder \
        .appName("FaceScrub Downloader with PySpark") \
        .config("spark.executor.memory", "4g") \
        .config("spark.driver.memory", "4g") \
        .config("spark.executor.cores", "2") \
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
        if not os.path.exists(metadata_path):
            logger.warning(f"Metadata file not found: {metadata_path}")
            continue

        logger.info(f"Loading metadata from {metadata_path}...")
        df = spark.read.option("header", True).option("delimiter", "\t").csv(metadata_path)
        df.show(5)

        rdd = df.select("name", "image_id", "url").rdd.coalesce(4)
        logger.info(f"Starting parallel image downloads for {metadata_path}...")
        results = rdd.map(lambda row: download_image(row.asDict(), save_dir)).collect()

        for result in results:
            logger.info(result)

    logger.info("Image downloads completed for all files!")

if __name__ == "__main__":
    main()