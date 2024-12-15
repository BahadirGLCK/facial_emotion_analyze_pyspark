import os
import io
import json
from typing import Tuple, List
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import udf, col, regexp_extract, rand, split as spark_split
from pyspark.sql.types import IntegerType, ArrayType, StringType, StructType, StructField
from PIL import Image
from pyspark.sql.functions import udf
from pyspark.sql.types import ArrayType, IntegerType
import base64
import numpy as np

# Note: For image reading/cropping in Spark, one approach is to broadcast images and process them in a UDF. 
# However, PySpark UDFs with PIL are not always straightforward. Another approach is to collect after joining and process locally.
# For demonstration, we show a UDF-based approach. In a production scenario, consider reading images in a mapPartitions function.

def load_and_crop_image_func(image_path: str, bbox: list) -> list:
    if not os.path.exists(image_path):
        return None
    try:
        img = Image.open(image_path)
        x, y, w, h = bbox
        cropped = img.crop((x, y, x + w, y + h))
        cropped = cropped.resize((224, 224))
        arr = np.array(cropped)
        return arr.tolist()  # return a nested list of pixel values
    except:
        return None


class DataPreprocessor:
    def __init__(self, spark: SparkSession, label_mapping: dict):
        self.spark = spark
        self.label_mapping = label_mapping

    def load_emotion_labels(self, json_path: str) -> DataFrame:
        # Load the JSON file with emotions in multiline mode if it's an array of JSON objects
        df = self.spark.read.option("multiline", "true").json(json_path)
        
        # Now df should have columns like 'emotion' or 'error' if they exist in the JSON.
        # Filter out entries that have errors and keep only entries that have an emotion
        df = df.filter((col("error").isNull()) & (col("emotion").isNotNull()))
        
        # Extract image_id from image_path using regex
        df = df.withColumn("image_id", regexp_extract("image_path", r"/(\d+)\.jpg$", 1))
        
        # Convert image_id to int
        df = df.withColumn("image_id", col("image_id").cast(IntegerType()))
        return df

    def load_facescrub_data(self, actors_path: str, actresses_path: str) -> DataFrame:
        schema = StructType([
            StructField("name", StringType(), True),
            StructField("image_id", StringType(), True),
            StructField("face_id", StringType(), True),
            StructField("url", StringType(), True),
            StructField("bbox", StringType(), True),
            StructField("sha256", StringType(), True)
        ])
        
        actors_df = self.spark.read \
            .option("delimiter", "\t") \
            .option("header", "true") \
            .schema(schema) \
            .csv(actors_path)
        
        actresses_df = self.spark.read \
            .option("delimiter", "\t") \
            .option("header", "true") \
            .schema(schema) \
            .csv(actresses_path)

        combined_df = actors_df.unionByName(actresses_df)
        # Convert image_id to int
        combined_df = combined_df.withColumn("image_id", col("image_id").cast(IntegerType()))
        return combined_df

    def join_data(self, emotion_df: DataFrame, facescrub_df: DataFrame) -> DataFrame:
        joined = emotion_df.join(facescrub_df, on="image_id", how="inner")

        # UDF to parse bbox
        parse_bbox_udf = udf(lambda bbox_str: [int(x) for x in bbox_str.split(',')],
                            ArrayType(IntegerType()))
        joined = joined.withColumn("bbox_coords", parse_bbox_udf(col("bbox")))

        # Capture label_mapping locally
        label_mapping = self.label_mapping

        # UDF for emotion to label
        emotion_to_label_udf = udf(lambda e: label_mapping.get(e, -1), IntegerType())
        joined = joined.withColumn("label", emotion_to_label_udf(col("emotion")))

        joined = joined.filter(col("label") >= 0)
        return joined

    def add_image_arrays(self, df):
        # Define UDF without referencing self or spark
        crop_udf = udf(lambda path, bbox: load_and_crop_image_func(path, bbox),
                    ArrayType(ArrayType(ArrayType(IntegerType()))))

        df = df.withColumn("image_array", crop_udf(col("image_path"), col("bbox_coords")))
        df = df.filter(col("image_array").isNotNull())
        return df

    def split_data(self, df: DataFrame) -> Tuple[DataFrame, DataFrame, DataFrame]:
        # Split by random fraction: 75% train, 15% val, 10% test
        # Add a random column
        df = df.withColumn("rand_val", rand())
        train_df = df.filter(col("rand_val") < 0.75)
        val_df = df.filter((col("rand_val") >= 0.75) & (col("rand_val") < 0.90))
        test_df = df.filter(col("rand_val") >= 0.90)
        return train_df, val_df, test_df


if __name__ == "__main__":
    spark = SparkSession.builder \
        .appName("FaceEmotionDataPrep") \
        .getOrCreate()

    label_mapping = {
        "angry": 0,
        "fear": 1,
        "neutral": 2,
        "sad": 3,
        "disgust": 4,
        "happy": 5,
        "surprise": 6
    }

    preprocessor = DataPreprocessor(spark, label_mapping)

    emotion_df = preprocessor.load_emotion_labels("data/labeled_emotion.json")
    facescrub_df = preprocessor.load_facescrub_data("data/facescrub_actors.txt", "data/facescrub_actresses.txt")
    joined_df = preprocessor.join_data(emotion_df, facescrub_df)
    final_df = preprocessor.add_image_arrays(joined_df)

    train_df, val_df, test_df = preprocessor.split_data(final_df)

    # Collect the final data locally for model training with TensorFlow
    train_data = train_df.select("image_array", "label").collect()
    val_data = val_df.select("image_array", "label").collect()
    test_data = test_df.select("image_array", "label").collect()

    spark.stop()

    # Convert collected data into numpy arrays
    import numpy as np

    def convert_collected_to_arrays(collected):
        X = []
        y = []
        for row in collected:
            X.append(row["image_array"])
            y.append(row["label"])
        X = np.array(X)
        y = np.array(y)
        return X, y

    X_train, y_train = convert_collected_to_arrays(train_data)
    X_val, y_val = convert_collected_to_arrays(val_data)
    X_test, y_test = convert_collected_to_arrays(test_data)