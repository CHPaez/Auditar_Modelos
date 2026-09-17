"""Download a Hugging Face image-classification model and run a one-image sanity check.

Usage:
    python scripts/download_model.py --model microsoft/resnet-50 --image path/to/photo.jpg
"""
import argparse

from transformers import pipeline


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="microsoft/resnet-50", help="Hugging Face model id")
    parser.add_argument("--image", help="Local image path to classify as a sanity check")
    args = parser.parse_args()

    print(f"Downloading '{args.model}' (cached locally after this run)...")
    classifier = pipeline(task="image-classification", model=args.model)

    if args.image:
        print(classifier(args.image))
    else:
        print("Model downloaded and ready. Pass --image <path> to classify a photo.")


if __name__ == "__main__":
    main()
