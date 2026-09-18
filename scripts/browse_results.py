"""Browse predictions image-by-image in FiftyOne's own local app -- the deeper,
interactive alternative to report.html's confusion matrix, from the "stack
minimo" of the audit research doc. Filter by correct/incorrect, click into
individual misclassified images, etc.

This installs and launches FiftyOne's own local web app (heavier than the
rest of this kit on purpose) instead of writing a static file -- see
requirements-visual.txt.

Usage:
    python scripts/browse_results.py --model nateraw/vit-base-beans --dataset AI-Lab-Makerere/beans --split test
"""
import argparse
from pathlib import Path

import fiftyone as fo
from transformers import pipeline

from audit_lib import load_test_set

_CACHE_DIR = Path("_fiftyone_cache")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="nateraw/vit-base-beans")
    parser.add_argument("--dataset", default="AI-Lab-Makerere/beans")
    parser.add_argument("--split", default="test")
    parser.add_argument("--limit", type=int, default=40)
    parser.add_argument("--label-column", default=None, help="Dataset column with the class labels (auto-detected if omitted)")
    args = parser.parse_args()

    print(f"Loading model '{args.model}'...")
    classifier = pipeline(task="image-classification", model=args.model)

    print(f"Loading dataset '{args.dataset}' (split={args.split})...")
    dataset_hf, label_names, label_column = load_test_set(args.dataset, args.split, args.limit, args.label_column)

    _CACHE_DIR.mkdir(exist_ok=True)
    samples = []
    for i, example in enumerate(dataset_hf):
        image = example["image"]
        true_label = label_names[example[label_column]]
        predicted_label = classifier(image)[0]["label"]

        image_path = _CACHE_DIR / f"{i}.png"
        image.save(image_path)

        sample = fo.Sample(filepath=str(image_path))
        sample["ground_truth"] = fo.Classification(label=true_label)
        sample["prediction"] = fo.Classification(label=predicted_label)
        sample["correct"] = true_label == predicted_label
        samples.append(sample)

    fo_dataset = fo.Dataset(name=f"audit-{args.model.replace('/', '-')}", overwrite=True)
    fo_dataset.add_samples(samples)

    print(f"Launching FiftyOne app for {len(samples)} images (opens in your browser)...")
    session = fo.launch_app(fo_dataset)
    session.wait()


if __name__ == "__main__":
    main()
