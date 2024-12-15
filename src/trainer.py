from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.sql.types import StructType, StructField, FloatType, ArrayType
from pyspark.sql import DataFrame
from pyspark.ml.linalg import Vectors, VectorUDT
from pyspark.sql.functions import col, log, when, expr, udf
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.optimizers import Adam
import numpy as np
import matplotlib.pyplot as plt
import os
from src.utils.logger import get_logger


class EmotionClassifierTrainer:

    def __init__(self, max_depth=5, num_trees=20):
        self.logger = get_logger("EmotionClassifierTrainer")
        self.model = None
        self.metrics = {
            "train_accuracy": [],
            "eval_accuracy": [],
            "test_accuracy": [],
            "train_loss": [],
            "eval_loss": [],
            "test_loss": []
        }
        self.max_depth = max_depth
        self.num_trees = num_trees

    @staticmethod
    def array_to_vector(df, array_col="features", vector_col="vectorized_features"):
        """
        Converts an array column to a DenseVector column.
        """
        vector_udf = udf(lambda array: Vectors.dense(array), VectorUDT())
        return df.withColumn(vector_col, vector_udf(array_col))

    def split_data(self, df: DataFrame, train_ratio=0.7, eval_ratio=0.2):
        """
        Split data into train, eval, and test sets.
        """
        # Randomly split the data
        train_ratio_adjusted = train_ratio / (train_ratio + eval_ratio)
        train_eval, test = df.randomSplit([train_ratio + eval_ratio, 1 - (train_ratio + eval_ratio)])
        train, eval = train_eval.randomSplit([train_ratio_adjusted, 1 - train_ratio_adjusted])
        self.logger.info(f"Data split: Train = {train.count()}, Eval = {eval.count()}, Test = {test.count()}")

        # Ensure features are vectorized
        train = self.array_to_vector(train)
        eval = self.array_to_vector(eval)
        test = self.array_to_vector(test)

        return train, eval, test

    def train(self, train_df: DataFrame, eval_df: DataFrame, feature_col: str, label_col: str):
        """
        Train Random Forest model and evaluate after each iteration.
        """
        # Assemble feature vectors
        assembler = VectorAssembler(inputCols=[feature_col], outputCol="features")
        train_df = assembler.transform(train_df)
        eval_df = assembler.transform(eval_df)

        # Train model
        self.logger.info("Initializing RandomForestClassifier...")
        rf = RandomForestClassifier(
            featuresCol="features",
            labelCol=label_col,
            maxDepth=self.max_depth,
            numTrees=self.num_trees
        )
        self.model = rf.fit(train_df)

        # Collect training and evaluation metrics
        train_accuracy, train_loss = self._evaluate(train_df, label_col)
        eval_accuracy, eval_loss = self._evaluate(eval_df, label_col)

        self.metrics["train_accuracy"].append(train_accuracy)
        self.metrics["train_loss"].append(train_loss)
        self.metrics["eval_accuracy"].append(eval_accuracy)
        self.metrics["eval_loss"].append(eval_loss)

        self.logger.info(f"Training Accuracy: {train_accuracy:.4f}, Loss: {train_loss:.4f}")
        self.logger.info(f"Evaluation Accuracy: {eval_accuracy:.4f}, Loss: {eval_loss:.4f}")

    def test(self, test_df: DataFrame, feature_col: str, label_col: str) -> float:
        """
        Evaluate model on test data.
        """
        assembler = VectorAssembler(inputCols=[feature_col], outputCol="features")
        test_df = assembler.transform(test_df)

        test_accuracy, test_loss = self._evaluate(test_df, label_col)
        self.metrics["test_accuracy"].append(test_accuracy)
        self.metrics["test_loss"].append(test_loss)
        self.logger.info(f"Test Accuracy: {test_accuracy:.4f}, Loss: {test_loss:.4f}")
        return test_accuracy

    def _evaluate(self, df: DataFrame, label_col: str):
        """
        Evaluate accuracy and log loss of the model.
        """
        predictions = self.model.transform(df)

        # Calculate accuracy
        evaluator = MulticlassClassificationEvaluator(
            labelCol=label_col, predictionCol="prediction", metricName="f1"
        )
        accuracy = evaluator.evaluate(predictions)

        # Calculate log loss (proxy for loss)
        log_loss = self._calculate_log_loss(predictions, label_col)

        return accuracy, log_loss

    def _calculate_log_loss(self, df: DataFrame, label_col: str) -> float:
        """
        Calculate log loss for predictions.
        Log loss is computed as:
        -log(P(correct class)) for each instance.
        """
        # Extract probability for the actual label
        df = df.withColumn(
            "log_loss", (-1 * log(expr(f"element_at(probability, {label_col} + 1)")))
        )
        
        # Handle edge cases where probabilities might be zero or NaN
        df = df.withColumn(
            "log_loss",
            when(col("log_loss").isNull() | (col("log_loss") == float("inf")), 0.0)
            .otherwise(col("log_loss"))
        )
        
        # Compute mean log loss
        log_loss = df.agg({"log_loss": "mean"}).collect()[0]["avg(log_loss)"]
        return log_loss

    def save_metrics_plot(self, output_dir: str):
        """
        Save accuracy and loss graph for train, evaluation, and test sets.
        """
        os.makedirs(output_dir, exist_ok=True)
        plt.figure(figsize=(12, 6))

        # Plot accuracy
        plt.subplot(1, 2, 1)
        plt.plot(self.metrics["train_accuracy"], label="Train Accuracy", marker="o")
        plt.plot(self.metrics["eval_accuracy"], label="Evaluation Accuracy", marker="o")
        plt.plot(self.metrics["test_accuracy"], label="Test Accuracy", marker="o")
        plt.title("Accuracy")
        plt.xlabel("Iterations")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.grid()

        # Plot loss
        plt.subplot(1, 2, 2)
        plt.plot(self.metrics["train_loss"], label="Train Loss", marker="o")
        plt.plot(self.metrics["eval_loss"], label="Evaluation Loss", marker="o")
        plt.plot(self.metrics["test_loss"], label="Test Loss", marker="o")
        plt.title("Log Loss")
        plt.xlabel("Iterations")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid()

        output_path = os.path.join(output_dir, "training_metrics.png")
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        self.logger.info(f"Metrics plot saved to {output_path}")

class TensorFlowEmotionTrainer:

    def __init__(self, input_shape, num_classes, hidden_units=[1024, 128, 64], learning_rate=0.001):
        """
        TensorFlow-based Emotion Classifier Trainer using MLP.
        """
        self.logger = get_logger("TensorFlowEmotionTrainer")
        self.model = self._build_model(input_shape, num_classes, hidden_units, learning_rate)
        self.metrics = {
            "train_accuracy": [],
            "eval_accuracy": [],
            "test_accuracy": [],
            "train_loss": [],
            "eval_loss": [],
            "test_loss": []
        }

    def _build_model(self, input_shape, num_classes, hidden_units, learning_rate):
        """
        Build the Multilayer Perceptron model with TensorFlow.
        """
        model = tf.keras.Sequential()
        model.add(tf.keras.Input(shape=(input_shape,)))
        for units in hidden_units:
            model.add(tf.keras.layers.Dense(units, activation='relu'))
        model.add(tf.keras.layers.Dense(num_classes, activation='softmax'))

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy",]
        )

        self.logger.info("TensorFlow MLP model built successfully.")
        return model

    def train(self, train_data, eval_data, batch_size=32, epochs=10):
        """
        Train the TensorFlow model.
        """
        x_train, y_train = train_data
        x_eval, y_eval = eval_data

        history = self.model.fit(
            x_train, y_train,
            validation_data=(x_eval, y_eval),
            batch_size=batch_size,
            epochs=epochs,
            verbose=1
        )

        # Log metrics
        self.metrics["train_loss"].extend(history.history["loss"])
        self.metrics["train_accuracy"].extend(history.history["accuracy"])
        self.metrics["eval_loss"].extend(history.history["val_loss"])
        self.metrics["eval_accuracy"].extend(history.history["val_accuracy"])

        self.logger.info("Training completed successfully.")

    def test(self, test_data):
        """
        Evaluate the model on test data.
        """
        x_test, y_test = test_data
        loss, accuracy = self.model.evaluate(x_test, y_test, verbose=1)

        self.metrics["test_loss"].append(loss)
        self.metrics["test_accuracy"].append(accuracy)

        self.logger.info(f"Test Accuracy: {accuracy:.4f}, Test Loss: {loss:.4f}")
        return accuracy, loss

    def save_metrics_plot(self, output_dir: str):
        """
        Save accuracy and loss graphs for train, evaluation, and test sets.
        """
        os.makedirs(output_dir, exist_ok=True)
        plt.figure(figsize=(12, 6))

        # Plot accuracy
        plt.subplot(1, 2, 1)
        plt.plot(self.metrics["train_accuracy"], label="Train Accuracy", marker="o")
        plt.plot(self.metrics["eval_accuracy"], label="Eval Accuracy", marker="o")
        plt.title("Accuracy")
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.grid()

        # Plot loss
        plt.subplot(1, 2, 2)
        plt.plot(self.metrics["train_loss"], label="Train Loss", marker="o")
        plt.plot(self.metrics["eval_loss"], label="Eval Loss", marker="o")
        plt.title("Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid()

        output_path = os.path.join(output_dir, "training_metrics.png")
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        self.logger.info(f"Metrics plot saved to {output_path}")

    def save_model(self, output_path: str):
        """
        Save the trained TensorFlow model.
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        self.model.save(output_path)
        self.logger.info(f"Model saved to {output_path}")


class TensorflowMobileEmotionTrainer:
    def __init__(self, num_classes, input_shape = (128,128,3), learning_rate=0.001):
        self.logger = get_logger("MobileEmotionModelTrainer")
        self.model = self._build_model(num_classes, input_shape, learning_rate)
    
    def _build_model(self, num_classes, input_shape, learning_rate):
        self.logger.info("Building the model ..")
        base_model = MobileNetV2(weights="imagenet", include_top=False, input_shape=input_shape)
        base_model.trainable = True

        x = Flatten()(base_model.output)
        x = Dense(128, activation="relu")(x)
        x = Dropout(0.5)(x)
        output = Dense(num_classes, activation="softmax")(x)

        model = Model(inputs=base_model.input, outputs=output)
        model.compile(optimizer=Adam(learning_rate=learning_rate),
                      loss="sparse_categorical_crossentropy",
                      metrics=["accuracy"])
        self.logger.info("Model built successfully.")
        return model

    def train(self, train_data, eval_data, batch_size=32, epochs=10):
        self.logger.info("Starting model training...")
        X_train, y_train = train_data
        X_eval, y_eval = eval_data

        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_eval, y_eval),
            batch_size=batch_size,
            epochs=epochs,
            verbose=1
        )
        self.logger.info("Training completed.")
    
    def test(self, test_data):
        self.logger.info("Evaluating the model on the test set...")
        X_test, y_test = test_data
        loss, accuracy = self.model.evaluate(X_test, y_test, verbose=1)
        self.logger.info(f"Test Accuracy: {accuracy:.4f}, Test Loss: {loss:.4f}")
    
    def save_metrics_plot(self, output_dir):
        self.logger.info("Saving metrics plot...")
        os.makedirs(output_dir, exist_ok=True)

        plt.figure(figsize=(12, 6))

        # Plot accuracy
        plt.subplot(1, 2, 1)
        plt.plot(self.history.history["accuracy"], label="Train Accuracy")
        plt.plot(self.history.history["val_accuracy"], label="Eval Accuracy")
        plt.title("Accuracy")
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        plt.legend()

        # Plot loss
        plt.subplot(1, 2, 2)
        plt.plot(self.history.history["loss"], label="Train Loss")
        plt.plot(self.history.history["val_loss"], label="Eval Loss")
        plt.title("Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()

        plt.tight_layout()
        plot_path = os.path.join(output_dir, "metrics.png")
        plt.savefig(plot_path)
        self.logger.info(f"Metrics plot saved to {plot_path}")

    def save_model(self, model_dir):
        self.logger.info("Saving the trained model...")
        os.makedirs(model_dir, exist_ok=True)
        self.model.save(model_dir)
        self.logger.info(f"Model saved to {model_dir}")