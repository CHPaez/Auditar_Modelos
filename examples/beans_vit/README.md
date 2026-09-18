# Example: bean leaf disease classifier

- **Model**: [`nateraw/vit-base-beans`](https://huggingface.co/nateraw/vit-base-beans) — a ViT fine-tuned on the `beans` dataset.
- **Dataset**: [`AI-Lab-Makerere/beans`](https://huggingface.co/datasets/AI-Lab-Makerere/beans) (test split) — 3 classes: `angular_leaf_spot`, `bean_rust`, `healthy`.

This pair is chosen deliberately: the model was fine-tuned on exactly this dataset's classes, so the reported metrics are meaningful out of the box (unlike pairing a general ImageNet classifier with this dataset, which would produce nonsense accuracy).

## Run it

From the repo root, with dependencies installed (see the main `README.md`):

```bash
python scripts/run_audit.py
python scripts/robustness_test.py
```

Both commands use the defaults above, so no flags are required. They download the model and dataset automatically on first run and write a JSON report (`audit_report.json` / `robustness_report.json`) in the directory you ran them from.

## What to expect

- `run_audit.py` reports accuracy, precision, recall, F1 and a 3x3 confusion matrix over 40 test images (adjust with `--limit`).
- `robustness_test.py` reports the same metrics twice — once on the original images, once after blur/rotation/brightness noise — plus the accuracy drop between the two. A meaningful drop here is the signal worth flagging in an audit, not the raw baseline number alone.
