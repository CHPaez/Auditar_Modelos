"""Robustness test: re-run the baseline audit after applying blur, rotation and
brightness/contrast noise to each image, and report the accuracy drop.

Usage:
    python scripts/robustness_test.py --model nateraw/vit-base-beans --dataset AI-Lab-Makerere/beans --split test
"""
import argparse
import json

import albumentations as A
import numpy as np
from PIL import Image
from transformers import pipeline

from audit_lib import compute_metrics, load_test_set, predict_all

_TRANSFORM = A.Compose([
    A.GaussianBlur(blur_limit=(3, 7), p=1.0),
    A.Rotate(limit=25, p=1.0),
    A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=1.0),
])


def perturb(image):
    array = np.array(image.convert("RGB"))
    augmented = _TRANSFORM(image=array)["image"]
    return Image.fromarray(augmented)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="nateraw/vit-base-beans")
    parser.add_argument("--dataset", default="AI-Lab-Makerere/beans")
    parser.add_argument("--split", default="test")
    parser.add_argument("--limit", type=int, default=40)
    parser.add_argument("--label-column", default=None, help="Dataset column with the class labels (auto-detected if omitted)")
    parser.add_argument("--out", default="robustness_report.json")
    args = parser.parse_args()

    print(f"Loading model '{args.model}'...")
    classifier = pipeline(task="image-classification", model=args.model)

    print(f"Loading dataset '{args.dataset}' (split={args.split})...")
    dataset, label_names, label_column = load_test_set(args.dataset, args.split, args.limit, args.label_column)
    references = dataset[label_column]

    print("Running baseline predictions...")
    baseline_predictions = predict_all(classifier, dataset["image"], label_names)
    baseline_metrics = compute_metrics(baseline_predictions, references)

    print("Applying blur + rotation + brightness/contrast noise and re-running predictions...")
    perturbed_images = [perturb(image) for image in dataset["image"]]
    perturbed_predictions = predict_all(classifier, perturbed_images, label_names)
    perturbed_metrics = compute_metrics(perturbed_predictions, references)

    report = {
        "model": args.model,
        "dataset": args.dataset,
        "images_evaluated": len(dataset),
        "baseline": baseline_metrics,
        "perturbed": perturbed_metrics,
        "accuracy_drop": baseline_metrics["accuracy"] - perturbed_metrics["accuracy"],
    }

    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)

    print(json.dumps(report, indent=2))
    print(f"\nSaved to {args.out}")


if __name__ == "__main__":
    main()
