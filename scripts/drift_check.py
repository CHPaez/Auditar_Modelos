"""Data-drift check: compares the model's predictions on the original test
images ("reference") against the same images with blur/rotation/brightness
noise applied ("current") -- a stand-in for "the photos reaching the model
today look different from when it was deployed". Produces an HTML drift
report via Evidently.

In a real deployment, "current" would be recent production photos instead of
a synthetic perturbation -- this script demos the mechanics with what the
rest of this kit already has on hand.

Usage:
    python scripts/drift_check.py --model nateraw/vit-base-beans --dataset beans --split test
"""
import argparse

import pandas as pd


def _predict_with_confidence(classifier, images):
    labels, confidences = [], []
    for image in images:
        result = classifier(image)[0]
        labels.append(result["label"])
        confidences.append(result["score"])
    return pd.DataFrame({"predicted_label": labels, "confidence": confidences})


def main():
    # Imported here, not at module level, so unit-testing _predict_with_confidence
    # doesn't require evidently/transformers/torch/albumentations installed.
    from evidently.metric_preset import DataDriftPreset
    from evidently.report import Report
    from transformers import pipeline

    from audit_lib import load_test_set
    from robustness_test import perturb

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="nateraw/vit-base-beans")
    parser.add_argument("--dataset", default="beans")
    parser.add_argument("--split", default="test")
    parser.add_argument("--limit", type=int, default=40)
    parser.add_argument("--out", default="drift_report.html")
    args = parser.parse_args()

    print(f"Loading model '{args.model}'...")
    classifier = pipeline(task="image-classification", model=args.model)

    print(f"Loading dataset '{args.dataset}' (split={args.split})...")
    dataset, _ = load_test_set(args.dataset, args.split, args.limit)

    print("Scoring the reference (original) images...")
    reference_data = _predict_with_confidence(classifier, dataset["image"])

    print("Scoring the 'current' images (blur + rotation + brightness noise applied)...")
    perturbed_images = [perturb(image) for image in dataset["image"]]
    current_data = _predict_with_confidence(classifier, perturbed_images)

    print("Building the drift report...")
    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=reference_data, current_data=current_data)
    report.save_html(args.out)

    print(f"Saved to {args.out} -- open it in any browser.")


if __name__ == "__main__":
    main()
