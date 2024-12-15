import os
from multiprocessing import Pool, cpu_count
from deepface import DeepFace
from tqdm import tqdm
import json

def label_image(image_path):
    """
    Labels an image using DeepFace and returns the result.
    """
    try:
        analysis = DeepFace.analyze(img_path=image_path, actions=["emotion"], enforce_detection=False)
        emotion = analysis[0]["dominant_emotion"]
        return {"image_path": image_path, "emotion": emotion}
    except Exception as e:
        return {"image_path": image_path, "error": str(e)}

def process_images(image_paths, output_file):
    """
    Processes images in parallel using multiprocessing.
    """
    # Create a pool of workers
    with Pool(cpu_count()) as pool:
        results = list(tqdm(pool.imap(label_image, image_paths), total=len(image_paths)))

    # Save results to a JSON file
    with open(output_file, "w") as f:
        json.dump(results, f, indent=4)

    print(f"Results saved to {output_file}")

def main():
    # Directory containing images
    images_dir = "data/raw/images"
    output_file = "data/labeled_emotions.json"

    # Collect all image paths
    image_paths = [
        os.path.join(root, file)
        for root, _, files in os.walk(images_dir)
        for file in files if file.endswith(".jpg")
    ]

    print(f"Found {len(image_paths)} images to process.")

    # Process images in parallel
    process_images(image_paths, output_file)

if __name__ == "__main__":
    main()