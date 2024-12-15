import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf
from pyspark.sql.types import StringType
os.environ["TF_USE_LEGACY_KERAS"] = "1"
from deepface import DeepFace
from src.utils.logger import get_logger

class DeepFaceLabeler:
    def __init__(self, images_dir: str, output_path: str, spark: SparkSession):
        """
        Initialize the labeler with paths and Spark session.
        """
        self.images_dir = images_dir
        self.output_path = output_path
        self.spark = spark
        self.logger = get_logger("DeepFaceLabeler")

    @staticmethod
    def analyze_emotion(image_path: str) -> str:
        """
        Analyze the emotion of a face in an image using DeepFace.
        """
        try:
            result = DeepFace.analyze(img_path=image_path, actions=['emotion'], enforce_detection=False)
            return result['dominant_emotion']
        except Exception as e:
            return "unknown"

    def load_images(self):
        """
        Collect all image paths into a Spark DataFrame.
        """
        image_paths = []
        for root, _, files in os.walk(self.images_dir):
            for file in files:
                if file.endswith(".jpg") or file.endswith(".png"):
                    image_paths.append(os.path.join(root, file))

        if not image_paths:
            self.logger.error("No images found in the directory.")
            raise FileNotFoundError("No images found in the directory.")

        self.logger.info(f"Loaded {len(image_paths)} images.")
        return self.spark.createDataFrame([(path,) for path in image_paths], ["image_path"])

    def label_images(self):
        """
        Label images using DeepFace and save results to a CSV file.
        """
        # Load images
        df = self.load_images()

        # Define UDF for emotion analysis
        analyze_emotion_udf = udf(lambda path: self.analyze_emotion(path), StringType())

        # Apply UDF to label images
        self.logger.info("Starting DeepFace emotion analysis...")
        df_with_labels = df.withColumn("emotion", analyze_emotion_udf(df["image_path"]))

        # Save results
        self.logger.info(f"Saving labeled data to {self.output_path}...")
        df_with_labels.write.csv(self.output_path, header=True)
        self.logger.info("Labeling completed successfully!")