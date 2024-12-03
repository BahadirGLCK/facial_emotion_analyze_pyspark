import argparse
from scripts.download_images import main as download_images
from scripts.preprocess_images import main as preprocess_images
from scripts.extract_features import main as extract_features
from scripts.train_classifier import main as train_classifier

def main():
    parser = argparse.ArgumentParser(description="Facial Emotion Analysis Pipeline")
    parser.add_argument("--step", type=str, required=True, help="Step to execute: downloader | preprocessor | feature_extractor | trainer")
    args = parser.parse_args()

    if args.step == "downloader":
        print("Starting downloader process...")
        download_images()
    elif args.step == "preprocessor":
        print("Starting preprocessor process...")
        preprocess_images()
    elif args.step == "feature_extractor":
        print("Starting feature extraction process...")
        extract_features()
    elif args.step == "trainer":
        print("Starting trainer process...")
        train_classifier()
    else:
        print("Invalid step. Choose from: downloader | preprocessor | feature_extractor | trainer")

if __name__ == "__main__":
    main()