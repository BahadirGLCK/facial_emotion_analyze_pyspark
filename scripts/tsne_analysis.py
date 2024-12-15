import numpy as np
import os
import sys 
from sklearn.manifold import TSNE

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
from src.utils.logger import get_logger

def process_tsne_batch(batch, perplexity=30):
    """
    Perform t-SNE on a batch of feature vectors.
    """
    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
    return tsne.fit_transform(batch)

def main():
    # Initialize logger
    logger = get_logger("TSNEAnalysis")

    # Load features directory
    features_dir = "/Users/bahadirgolcuk/bahadir/project/facial_emotion_analyze_pyspark/data/features"

    # Collect all features and filenames
    features = []
    filenames = []
    for root, _, files in os.walk(features_dir):
        for file in files:
            if file.endswith(".npy") and not file.startswith("."):
                filenames.append(os.path.join(root, file))
                features.append(np.load(os.path.join(root, file)))

    # Ensure consistent number of features and filenames
    if len(features) != len(filenames):
        logger.error(f"Mismatch: {len(features)} features and {len(filenames)} filenames.")
        return

    # Convert features to a single NumPy array
    features = np.array(features)
    total_features = len(features)
    logger.info(f"Total features loaded: {total_features}")

    # Perform t-SNE in batches
    batch_size = 36458
    tsne_results = []
    for start in range(0, total_features, batch_size):
        end = min(start + batch_size, total_features)
        logger.info(f"Processing t-SNE for batch {start} to {end}...")
        tsne_results.append(process_tsne_batch(features[start:end]))

    # Combine all t-SNE results
    tsne_results = np.vstack(tsne_results)

    # Save t-SNE results
    output_dir = "/Users/bahadirgolcuk/bahadir/project/facial_emotion_analyze_pyspark/data/tsne_results"
    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, "tsne_results.npy"), tsne_results)

    logger.info(f"t-SNE analysis completed. Results saved to {output_dir}/tsne_results.npy")

if __name__ == "__main__":
    main()