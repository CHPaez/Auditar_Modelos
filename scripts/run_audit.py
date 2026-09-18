"""Baseline audit: accuracy, precision, recall, F1 and confusion matrix for an
image-classification model against a labeled test set.

Usage:
    python scripts/run_audit.py --model nateraw/vit-base-beans --dataset AI-Lab-Makerere/beans --split test
"""
import argparse
import json

from transformers import pipeline

from audit_lib import compute_metrics, load_test_set, predict_all


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="nateraw/vit-base-beans", help="Hugging Face model id")
    parser.add_argument("--dataset", default="AI-Lab-Makerere/beans", help="Hugging Face dataset id")
    parser.add_argument("--split", default="test")
    parser.add_argument("--limit", type=int, default=40, help="Max images to evaluate")
    parser.add_argument("--label-column", default=None, help="Dataset column with the class labels (auto-detected if omitted)")
    parser.add_argument("--out", default="audit_report.json")
    args = parser.parse_args()

    print(f"Loading model '{args.model}'...")
    classifier = pipeline(task="image-classification", model=args.model)

    print(f"Loading dataset '{args.dataset}' (split={args.split})...")
    dataset, label_names, label_column = load_test_set(args.dataset, args.split, args.limit, args.label_column)

    print(f"Running predictions on {len(dataset)} images...")
    predictions = predict_all(classifier, dataset["image"], label_names)
    references = dataset[label_column]

    metrics = compute_metrics(predictions, references)
    matrix_label_names = [label_names[i] if 0 <= i < len(label_names) else "unknown" for i in metrics["matrix_labels"]]
    report = {
        "model": args.model,
        "dataset": args.dataset,
        "images_evaluated": len(dataset),
        "label_names": label_names,
        "confusion_matrix_labels": matrix_label_names,
        **metrics,
    }

    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)

    print(json.dumps(report, indent=2))
    print(f"\nSaved to {args.out}")


if __name__ == "__main__":
    main()
