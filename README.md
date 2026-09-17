# Image Model Audit Kit

A small, free, open-source toolkit to audit an already-trained image-classification model: baseline accuracy/precision/recall/F1, a confusion matrix, and a robustness test against blur/rotation/brightness noise — without touching the model's training pipeline.

## Why

Auditing a model is independent of training it. This kit treats any image-classification model as a black box: give it a model id and a labeled test set, and it reports how well the model actually performs — including how much it degrades under imperfect real-world photos.

## What it audits

1. **Effectiveness** — accuracy, precision, recall, F1, confusion matrix (`scripts/run_audit.py`)
2. **Robustness** — accuracy drop under blur/rotation/brightness noise (`scripts/robustness_test.py`)

Equity/subgroup analysis, drift monitoring, and implementation/code review are separate, larger dimensions of a full model audit and are out of scope for this kit.

## Requirements

- Python 3.10+
- A couple GB of free disk space for model weights + dataset cache (varies by model)
- No paid accounts, no API keys — everything here is free and open source

## Install

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Quickstart (bundled example)

Runs a real audit end-to-end on a public model + dataset — nothing to configure:

```bash
python scripts/run_audit.py
python scripts/robustness_test.py
```

This downloads `nateraw/vit-base-beans` (a model fine-tuned to classify bean leaf disease) and the `beans` test dataset automatically on first run, then prints and saves a JSON report. See `examples/beans_vit/README.md` for what to expect.

To just sanity-check that a general-purpose model downloads and classifies a photo (no metrics, no dataset needed):

```bash
python scripts/download_model.py --image path/to/photo.jpg
```

## Using it with your own model

```bash
python scripts/run_audit.py --model <your-model-id> --dataset <your-dataset-id> --split test
python scripts/robustness_test.py --model <your-model-id> --dataset <your-dataset-id> --split test
```

- `--model`: any Hugging Face model id that supports the `image-classification` pipeline.
- `--dataset`: any Hugging Face dataset id with an `image` field and a `label` `ClassLabel` field.
- `--limit`: cap how many images to evaluate (default 40, for a quick run).

**Important — matching predictions to ground truth**: this kit compares the model's predicted label text to the dataset's class names. This works out of the box when the model was fine-tuned on that exact dataset's classes — which is the normal case for a company's own production model, since it was trained on its own labeled categories. It will **not** give meaningful metrics if you pair a general-purpose model (e.g. an ImageNet classifier like `microsoft/resnet-50`) with a dataset whose classes it never learned. `scripts/download_model.py` is for that kind of quick single-photo sanity check instead — it doesn't need matching classes.

To audit against your own local photos instead of a Hugging Face dataset, organize them as `data/<class_name>/*.jpg` and load them with:

```python
from datasets import load_dataset
dataset = load_dataset("imagefolder", data_dir="data")
```

See the [`imagefolder` docs](https://huggingface.co/docs/datasets/en/image_load#imagefolder) for details.

## Cost

Every tool this kit depends on (`transformers`, `datasets`, `evaluate`, `albumentations`, `scikit-learn`) is free and open source, with no licensing restrictions for internal or commercial use. If you later extend this kit to object-detection models via Ultralytics/YOLO, note that Ultralytics requires either open-sourcing your project under AGPL-3.0 or a paid Enterprise license for any non-open-source use — that caveat does not apply to anything shipped in this kit.

## License

MIT — see `LICENSE`.
