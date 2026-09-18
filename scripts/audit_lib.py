"""Shared helpers for running an image-classification model against a labeled dataset."""
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support


def load_test_set(dataset_id, split, limit):
    from datasets import load_dataset  # imported lazily: only needed here, not for compute_metrics

    dataset = load_dataset(dataset_id, split=split)
    if limit:
        dataset = dataset.select(range(min(limit, len(dataset))))
    label_names = dataset.features["label"].names
    return dataset, label_names


def predict_all(classifier, images, label_names):
    predictions = []
    for image in images:
        result = classifier(image)
        predicted_label = result[0]["label"]
        predictions.append(label_names.index(predicted_label) if predicted_label in label_names else -1)
    return predictions


def compute_metrics(predictions, references):
    accuracy = accuracy_score(references, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        references, predictions, average="weighted", zero_division=0
    )
    matrix = confusion_matrix(references, predictions).tolist()
    return {
        "accuracy": accuracy,
        "precision_weighted": precision,
        "recall_weighted": recall,
        "f1_weighted": f1,
        "confusion_matrix": matrix,
    }
