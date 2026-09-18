"""Shared helpers for running an image-classification model against a labeled dataset."""
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support


def load_test_set(dataset_id, split, limit, label_column=None):
    from datasets import load_dataset  # imported lazily: only needed here, not for compute_metrics

    dataset = load_dataset(dataset_id, split=split)
    if limit:
        dataset = dataset.select(range(min(limit, len(dataset))))

    if label_column is None:
        # Different datasets name their class column differently (e.g. "label"
        # vs "labels") -- try the two common ones before giving up.
        for candidate in ("label", "labels"):
            if candidate in dataset.features:
                label_column = candidate
                break
        else:
            raise ValueError(
                f"Could not find a 'label' or 'labels' column in '{dataset_id}' "
                f"(found: {list(dataset.features)}). Pass --label-column explicitly."
            )
    elif label_column not in dataset.features:
        raise ValueError(f"Column '{label_column}' not found in '{dataset_id}' (found: {list(dataset.features)}).")

    label_names = dataset.features[label_column].names
    return dataset, label_names, label_column


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
    # Fix the matrix's labels explicitly to whatever actually appears in this
    # run (not the full class list): with a small --limit, not every class may
    # show up, and the matrix must stay aligned with the labels reported
    # alongside it rather than silently assuming it always covers every class.
    matrix_labels = sorted(set(references) | set(predictions))
    matrix = confusion_matrix(references, predictions, labels=matrix_labels).tolist()
    return {
        "accuracy": accuracy,
        "precision_weighted": precision,
        "recall_weighted": recall,
        "f1_weighted": f1,
        "confusion_matrix": matrix,
        "matrix_labels": matrix_labels,
    }
