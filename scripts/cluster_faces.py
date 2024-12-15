from pyspark.ml.linalg import Vectors, VectorUDT
from pyspark.sql.functions import udf
from pyspark.sql.types import StructType, StructField, StringType, FloatType, ArrayType
from pyspark.sql import SparkSession
import numpy as np
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
from src.cluster import FaceClustering

def load_features(root, file):
    """
    Function to load features from a file and return a tuple of metadata and features.
    """
    # Extract person's name from the subdirectory
    person_name = os.path.basename(root)
    
    # Load features
    feature_path = os.path.join(root, file)
    features = np.load(feature_path).tolist()
    
    return person_name, file, features

def main():
    # Initialize Spark session
    spark = SparkSession.builder \
        .appName("FaceScrub Clustering") \
        .config("spark.driver.memory", "8g") \
        .config("spark.executor.memory", "8g") \
        .getOrCreate()

    # Load features directory
    features_dir = "/Users/bahadirgolcuk/bahadir/project/facial_emotion_analyze_pyspark/data/features"

    # Create RDD from directory structure
    rdd = spark.sparkContext.parallelize([
        (root, file)
        for root, _, files in os.walk(features_dir)
        for file in files if file.endswith(".npy") and not file.startswith(".")
    ])

    # Process RDD to load features
    rdd_processed = rdd.map(lambda x: load_features(x[0], x[1]))

    # Define schema
    schema = StructType([
        StructField("person_name", StringType(), True),
        StructField("filename", StringType(), True),
        StructField("features", ArrayType(FloatType()), True)
    ])

    # Convert to DataFrame
    df = spark.createDataFrame(rdd_processed, schema=schema)

    # Convert features array to Vector
    if "features_vector" not in df.columns:
        to_vector_udf = udf(lambda features: Vectors.dense(features), VectorUDT())
        df = df.withColumn("features_vector", to_vector_udf(df["features"]))

    # Initialize and run clustering
    k = 5  # Number of clusters
    clustering = FaceClustering(k=k, features_col="features_vector")
    clustered_data = clustering.run(spark, df)

    # Save results
    output_path = "data/clusters.csv"
    clustering.save_clusters(clustered_data, output_path)
    print(f"Clustering completed. Results saved to {output_path}")

if __name__ == "__main__":
    main()