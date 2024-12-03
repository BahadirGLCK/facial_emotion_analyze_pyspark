from src.utils.logger import get_logger
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.sql import DataFrame


class EmotionClassifierTrainer:
    def __init__(self):
        self.logger = get_logger("EmotionClassifierTrainer")  # Initialize logger
        self.model = None

    def train(self, df: DataFrame, feature_col: str, label_col: str):
        assembler = VectorAssembler(inputCols=[feature_col], outputCol="features")
        transformed_df = assembler.transform(df)
        lr = LogisticRegression(featuresCol="features", labelCol=label_col)
        self.model = lr.fit(transformed_df)
        self.logger.info(f"Model trained on {df.count()} records")

    def evaluate(self, df: DataFrame, feature_col: str, label_col: str) -> float:
        predictions = self.model.transform(df)
        evaluator = MulticlassClassificationEvaluator(labelCol=label_col, predictionCol="prediction", metricName="accuracy")
        accuracy = evaluator.evaluate(predictions)
        self.logger.info(f"Model evaluation complete: Accuracy = {accuracy}")
        return accuracy