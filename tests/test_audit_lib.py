from audit_lib import compute_metrics


def test_compute_metrics_perfect_predictions():
    predictions = [0, 1, 2, 0, 1]
    references = [0, 1, 2, 0, 1]

    metrics = compute_metrics(predictions, references)

    assert metrics["accuracy"] == 1.0
    assert metrics["precision_weighted"] == 1.0
    assert metrics["recall_weighted"] == 1.0
    assert metrics["f1_weighted"] == 1.0
    assert metrics["confusion_matrix"] == [[2, 0, 0], [0, 2, 0], [0, 0, 1]]


def test_compute_metrics_with_errors():
    predictions = [0, 1, 1, 0]
    references = [0, 1, 0, 0]

    metrics = compute_metrics(predictions, references)

    assert metrics["accuracy"] == 0.75
    assert metrics["confusion_matrix"] == [[2, 1], [0, 1]]


def test_compute_metrics_unmatched_label_counts_as_wrong():
    # -1 is what audit_lib.predict_all returns when the model's predicted
    # label text isn't one of the dataset's class names.
    predictions = [-1, 1]
    references = [0, 1]

    metrics = compute_metrics(predictions, references)

    assert metrics["accuracy"] == 0.5
