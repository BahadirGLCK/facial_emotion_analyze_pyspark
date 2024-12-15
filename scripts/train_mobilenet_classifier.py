from pyspark.sql import SparkSession
import sys
import os
import numpy as np

import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras import Model
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, classification_report

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
from src.data_preprocessing import DataPreprocessor
from src.utils.logger import get_logger

def process_partition(iterator):
    from PIL import Image
    import numpy as np
    import os
    results = []
    for row in iterator:
        image_path = row["image_path"]
        bbox = row["bbox_coords"]
        if os.path.exists(image_path):
            try:
                img = Image.open(image_path)
                x, y, w, h = bbox
                cropped = img.crop((x, y, x + w, y + h))
                cropped = cropped.resize((224, 224))
                
                # Convert to grayscale
                cropped = cropped.convert("L")
                # Convert to numpy array (shape: (224, 224))
                arr = np.array(cropped)
                # Expand dimensions to (224, 224, 1)
                arr = np.expand_dims(arr, axis=-1)
                # Repeat the channel 3 times to get (224, 224, 3)
                arr = np.repeat(arr, 3, axis=-1)
                
                arr = arr.tolist()
                results.append((row["image_id"], arr, row["label"]))
            except:
                pass
    return iter(results)

def convert_collected_to_arrays(collected):
    X = []
    y = []
    for row in collected:
        X.append(row["image_array"])
        y.append(row["label"])
    X = np.array(X)
    y = np.array(y)
    return X, y

def create_model(num_classes: int):
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    # Make the base model trainable
    base_model.trainable = True

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.2)(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    return model

label_mapping = {
    "angry": 0,
    "fear": 1,
    "neutral": 2,
    "sad": 3,
    "disgust": 4,
    "happy": 5,
    "surprise": 6
}

logger = get_logger("MobilenetTraining")

spark = SparkSession.builder.appName("FaceEmotionMobilenet").getOrCreate()

preprocessor = DataPreprocessor(spark, label_mapping)
emotion_df = preprocessor.load_emotion_labels("/Users/bahadirgolcuk/bahadir/project/facial_emotion_analyze_pyspark/data/labeled_emotions.json")
facescrub_df = preprocessor.load_facescrub_data("data/facescrub_actors.txt", "data/facescrub_actresses.txt")
joined_df = preprocessor.join_data(emotion_df, facescrub_df)

# Process images using mapPartitions
rdd = joined_df.rdd.mapPartitions(process_partition)
processed_df = rdd.toDF(["image_id", "image_array", "label"])

train_df, val_df, test_df = preprocessor.split_data(processed_df)

# Collect the final data locally for model training with TensorFlow
train_data = train_df.select("image_array", "label").collect()
val_data = val_df.select("image_array", "label").collect()
test_data = test_df.select("image_array", "label").collect()

spark.stop()

X_train, y_train = convert_collected_to_arrays(train_data)
X_val, y_val = convert_collected_to_arrays(val_data)
X_test, y_test = convert_collected_to_arrays(test_data)

num_classes = len(label_mapping)

model = create_model(num_classes)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Training
history = model.fit(X_train, y_train,
                    validation_data=(X_val, y_val),
                    epochs=15,
                    batch_size=4)
logger.info("Training was completed!!!")

# Plot Accuracy and Loss
output_dir = "output/metrics"
os.makedirs(output_dir, exist_ok=True)
plt.figure(figsize=(12, 6))

# Plot accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history["accuracy"], label="Train Accuracy")
plt.plot(history.history["val_accuracy"], label="Eval Accuracy")
plt.title("Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()

# Plot loss
plt.subplot(1, 2, 2)
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Eval Loss")
plt.title("Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()

plt.tight_layout()
plot_path = os.path.join(output_dir, "metrics_mobilenet.png")
plt.savefig(plot_path)
logger.info(f"Metrics plot saved to {plot_path}")

# Evaluate on test set
test_loss, test_acc = model.evaluate(X_test, y_test)
logger.info(f"Test Accuracy: {test_acc}")

# F1-score
y_pred = model.predict(X_test)
y_pred_labels = y_pred.argmax(axis=1)
f1 = f1_score(y_test, y_pred_labels, average='weighted')
logger.info(f"F1-score: {f1}")

# Detailed classification report
logger.info(classification_report(y_test, y_pred_labels))

# Save the model
model.save("face_emotion_model.h5")
logger.info("Model saved successfully!")