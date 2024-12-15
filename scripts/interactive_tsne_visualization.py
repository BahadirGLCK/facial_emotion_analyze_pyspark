import numpy as np
import plotly.express as px
import pandas as pd
import os

def main():
    # Load t-SNE results
    tsne_results_path = "/Users/bahadirgolcuk/bahadir/project/facial_emotion_analyze_pyspark/data/tsne_results/tsne_results.npy"
    tsne_results = np.load(tsne_results_path)

    # Load filenames (for mapping dots to files)
    features_dir = "/Users/bahadirgolcuk/bahadir/project/facial_emotion_analyze_pyspark/data/features"
    filenames = []
    for root, _, files in os.walk(features_dir):
        for file in files:
            if file.endswith(".npy") and not file.startswith("."):
                filenames.append(os.path.join(root, file))

    # Ensure filenames match t-SNE results
    if len(tsne_results) != len(filenames):
        print(f"Mismatch: {len(tsne_results)} t-SNE results and {len(filenames)} filenames.")
        return

    # Create a DataFrame for Plotly
    tsne_df = pd.DataFrame({
        "x": tsne_results[:, 0],
        "y": tsne_results[:, 1],
        "filename": [os.path.basename(f) for f in filenames]
    })

    # Create an interactive scatter plot
    fig = px.scatter(
        tsne_df,
        x="x",
        y="y",
        hover_name="filename",
        title="Interactive t-SNE Visualization of Face Features",
        labels={"x": "t-SNE Dimension 1", "y": "t-SNE Dimension 2"}
    )

    # Show the plot
    output_path = "/Users/bahadirgolcuk/bahadir/project/facial_emotion_analyze_pyspark/data/tsne_results/tsne_visualization.html"
    fig.write_html(output_path)
    print(f"t-SNE visualization saved to {output_path}. Open it in a browser to view the details.")

if __name__ == "__main__":
    main()