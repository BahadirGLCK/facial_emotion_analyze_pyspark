from pyspark.ml.feature import VectorAssembler
from pyspark.ml.clustering import KMeans
from pyspark.sql import DataFrame
from pyspark.sql import SparkSession
from src.utils.logger import get_logger

class FaceClustering:
    def __init__(self, k: int, features_col: str = "features_vector", output_col: str = "cluster", seed: int = 42):
        """
        Initialize clustering parameters.
        :param k: Number of clusters.
        :param features_col: Column name for feature vectors.
        :param output_col: Column name for cluster assignments.
        :param seed: Random seed for clustering.
        """
        self.k = k
        self.features_col = features_col
        self.output_col = output_col
        self.seed = seed
        self.logger = get_logger("FaceClustering")

    def run(self, spark: SparkSession, data: DataFrame) -> DataFrame:
        """
        Run clustering on the input DataFrame.
        :param spark: Spark session.
        :param data: Input DataFrame containing features.
        :return: DataFrame with cluster assignments.
        """
        self.logger.info("Initializing VectorAssembler...")
        
        # Ensure we only add the features_vector column if it doesn't exist
        if self.features_col not in data.columns:
            assembler = VectorAssembler(inputCols=["features"], outputCol=self.features_col)
            data = assembler.transform(data)

        self.logger.info(f"Starting KMeans clustering with k={self.k}...")
        kmeans = KMeans(k=self.k, seed=self.seed, featuresCol=self.features_col, predictionCol=self.output_col)
        model = kmeans.fit(data)

        self.logger.info("Assigning cluster labels to data...")
        clustered_data = model.transform(data)
        return clustered_data

    def save_clusters(self, data: DataFrame, output_path: str):
        """
        Save the clustered data to the specified path.
        :param data: Clustered DataFrame.
        :param output_path: Path to save the data.
        """
        self.logger.info(f"Saving clustered data to {output_path}...")
        data.select("filename", self.output_col).write.csv(output_path, header=True)