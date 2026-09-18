from drift_check import _predict_with_confidence


class _FakeClassifier:
    def __init__(self, results):
        self._results = iter(results)

    def __call__(self, image):
        return next(self._results)


def test_predict_with_confidence_builds_dataframe():
    classifier = _FakeClassifier([
        [{"label": "healthy", "score": 0.9}],
        [{"label": "bean_rust", "score": 0.7}],
    ])

    df = _predict_with_confidence(classifier, images=["img1", "img2"])

    assert list(df["predicted_label"]) == ["healthy", "bean_rust"]
    assert list(df["confidence"]) == [0.9, 0.7]
