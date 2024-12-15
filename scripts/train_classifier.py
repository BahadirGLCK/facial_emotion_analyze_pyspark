from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, FloatType, IntegerType, ArrayType
from sklearn.model_selection import train_test_split
import numpy as np
import os
import json
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
from src.trainer import EmotionClassifierTrainer, TensorFlowEmotionTrainer
from src.utils.logger import get_logger

label_mapping = {
    "angry": 0,
    "fear": 1,
    "neutral": 2,
    "sad": 3,
    "disgust": 4,
    "happy": 5,
    "surprise": 6
}

def process_batch(batch_data, features_dir, label_mapping):
    rows = []
    for item in batch_data:
        image_path = item.get("image_path")
        emotion = item.get("emotion")

        if "error" in item or emotion not in label_mapping:
            continue

        person_name = image_path.split("/")[-2]
        image_id = os.path.splitext(os.path.basename(image_path))[0]
        feature_path = os.path.join(features_dir, person_name, f"{image_id}.npy")

        if os.path.exists(feature_path):
            features = np.load(feature_path).tolist()
            label = label_mapping[emotion]
            rows.append((label, features))
    return rows

def main_spark_model():
    # Initialize logger
    logger = get_logger("TrainerTrigger")

    # Initialize Spark Session with Memory Configuration
    spark = SparkSession.builder \
        .appName("FaceScrub Trainer") \
        .config("spark.driver.memory", "6g") \
        .config("spark.executor.memory", "6g") \
        .config("spark.sql.shuffle.partitions", "8") \
        .getOrCreate()

    # Load labels
    labeled_data_path = "data/labeled_emotions.json"
    logger.info("Loading labeled data...")
    with open(labeled_data_path, "r") as f:
        labeled_data = json.load(f)

    # Collect feature paths and labels
    features_dir = "data/features"
    schema = StructType([
        StructField("label", IntegerType(), True),
        StructField("features", ArrayType(FloatType()), True)
    ])
    batch_size = 500  # Reduce batch size for memory optimization
    parquet_dir = "data/intermediate_batches"

    if not os.path.exists(parquet_dir):
        os.makedirs(parquet_dir, exist_ok=True)
        for i in range(0, len(labeled_data), batch_size):
            batch_data = labeled_data[i:i + batch_size]
            logger.info(f"Processing batch {i // batch_size + 1}")
            rows = process_batch(batch_data, features_dir, label_mapping)

            # Write each batch to Parquet
            if rows:
                batch_df = spark.createDataFrame(rows, schema=schema)
                batch_file = os.path.join(parquet_dir, f"batch_{i // batch_size + 1}.parquet")
                batch_df.write.mode("overwrite").parquet(batch_file)
                logger.info(f"Batch {i // batch_size + 1} saved as {batch_file}")

        logger.info("All batches processed and saved successfully!")

    # Explicitly specify schema when reading Parquet files
    schema = StructType([
        StructField("label", IntegerType(), True),
        StructField("features", ArrayType(FloatType()), True)
    ])

    logger.info("Loading all batches from Parquet files...")
    df = spark.read.schema(schema).parquet(parquet_dir + "/*.parquet")
    logger.info(f"All batches loaded successfully! Total rows: {df.count()}")

    # Initialize Trainer
    trainer = EmotionClassifierTrainer()

    # Split data
    #train_df, eval_df, test_df = trainer.split_data(df)

    # Train the classifier
    logger.info("Training the classifier...")
    trainer.train(train_df, eval_df, feature_col="features", label_col="label")

    # Test the classifier
    logger.info("Testing the classifier...")
    trainer.test(test_df, feature_col="features", label_col="label")

    # Save metrics graph
    output_dir = "output/metrics"
    trainer.save_metrics_plot(output_dir)
    logger.info("Training metrics saved successfully!")

    # Save the model
    os.makedirs("output/models/", exist_ok=True)
    trainer.model.save("output/models/emotion_rf_classifier")
    logger.info("Model saved successfully!")

def main():
    logger = get_logger("TrainerTriggerwithTensorflow")

    spark = SparkSession.builder \
        .appName("FaceScrub Trainer") \
        .config("spark.driver.memory", "6g") \
        .config("spark.executor.memory", "6g") \
        .config("spark.sql.shuffle.partitions", "8") \
        .getOrCreate()

    # Load data from saved Parquet files
    logger.info("Loading all batches...")
    parquet_dir = "data/intermediate_batches"

    schema = StructType([
        StructField("label", IntegerType(), True),
        StructField("features", ArrayType(FloatType()), True)
    ])
    df = spark.read.schema(schema).parquet(parquet_dir + "/*.parquet").toPandas()
    
    x_data = np.array(df["features"].tolist())
    y_data = np.array(df["label"].tolist())
    
    # Split data into train, eval, and test sets
    x_train, x_temp, y_train, y_temp = train_test_split(x_data, y_data, test_size=0.3, random_state=42)
    x_eval, x_test, y_eval, y_test = train_test_split(x_temp, y_temp, test_size=0.5, random_state=42)

    # Initialize TensorFlow Trainer
    input_shape = x_train.shape[1]
    num_classes = len(np.unique(y_data))
    trainer = TensorFlowEmotionTrainer(input_shape, num_classes)

    # Train and Evaluate
    trainer.train((x_train, y_train), (x_eval, y_eval), batch_size=4, epochs=20)

    # Test the Model
    trainer.test((x_test, y_test))

    # Save Metrics Plot
    trainer.save_metrics_plot("output/metrics")

    # Save the Model
    trainer.save_model("output/models/emotion_mlp_classifier.h5")

if __name__ == "__main__":
    main()