from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, FloatType, IntegerType
from src.trainer import EmotionClassifierTrainer
from src.utils.logger import get_logger
import numpy as np
import os

def main():
    # Initialize logger
    logger = get_logger("TrainerTrigger")

    # Initialize Spark Session
    spark = SparkSession.builder \
        .appName("FaceScrub Trainer") \
        .getOrCreate()

    # Directory containing features
    features_dir = "data/features"

    # Prepare labeled data
    logger.info("Loading labeled data...")
    rows = []
    for root, _, files in os.walk(features_dir):
        for file in files:
            feature_path = os.path.join(root, file)
            features = np.load(feature_path)
            label = 0 if "neutral" in file else 1  # Example labeling
            rows.append((label, features.tolist()))

    # Convert to PySpark DataFrame
    schema = StructType([
        StructField("label", IntegerType(), True),
        StructField("features", FloatType(), True)
    ])
    df = spark.createDataFrame(rows, schema=schema)

    # Initialize Trainer
    trainer = EmotionClassifierTrainer()

    # Train the classifier
    logger.info("Training the classifier...")
    trainer.train(df, feature_col="features", label_col="label")

    # Evaluate the model
    accuracy = trainer.evaluate(df, feature_col="features", label_col="label")
    logger.info(f"Model evaluation completed with accuracy: {accuracy}")

    # Save the model
    trainer.model.save("output/models/emotion_classifier")
    logger.info("Model saved successfully!")

if __name__ == "__main__":
    main()