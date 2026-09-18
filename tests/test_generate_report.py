from generate_report import _code_quality_section, _effectiveness_section, _robustness_section


def test_effectiveness_section_renders_metrics():
    report = {
        "model": "nateraw/vit-base-beans",
        "dataset": "beans",
        "images_evaluated": 40,
        "accuracy": 0.9,
        "precision_weighted": 0.91,
        "recall_weighted": 0.9,
        "f1_weighted": 0.9,
        "confusion_matrix": [[10, 1], [0, 9]],
        "confusion_matrix_labels": ["healthy", "bean_rust"],
    }

    html = _effectiveness_section(report)

    assert "90.0%" in html
    assert "nateraw/vit-base-beans" in html
    assert "healthy" in html


def test_effectiveness_section_pending_when_missing():
    html = _effectiveness_section(None)

    assert "Todavía no se corrió" in html
    assert "run_audit.py" in html


def test_robustness_section_flags_a_drop():
    report = {
        "model": "m",
        "dataset": "d",
        "baseline": {"accuracy": 0.9},
        "perturbed": {"accuracy": 0.6},
        "accuracy_drop": 0.3,
    }

    html = _robustness_section(report)

    assert "30.0%" in html
    assert "frágil" in html


def test_code_quality_section_reports_clean_lint():
    report = {
        "path": "/some/repo",
        "lint": {"exit_code": 0, "output": ""},
        "dependency_audit": {"skipped": "no requirements.txt found at the given path"},
    }

    html = _code_quality_section(report)

    assert "Limpio" in html
    assert "Omitido" in html
