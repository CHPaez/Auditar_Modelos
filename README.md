# Image Model Audit Kit

A small, free, open-source toolkit to audit an already-trained image-classification model: baseline accuracy/precision/recall/F1, a confusion matrix, and a robustness test against blur/rotation/brightness noise — without touching the model's training pipeline.

> Guía en español, paso a paso y sin tecnicismos: [`GUIA_RAPIDA.md`](GUIA_RAPIDA.md).

## Why

Auditing a model is independent of training it. This kit treats any image-classification model as a black box: give it a model id and a labeled test set, and it reports how well the model actually performs — including how much it degrades under imperfect real-world photos.

## What it audits

1. **Effectiveness** — accuracy, precision, recall, F1, confusion matrix (`scripts/run_audit.py`)
2. **Robustness** — accuracy drop under blur/rotation/brightness noise (`scripts/robustness_test.py`)
3. **Implementation/code** — lint + dependency vulnerability scan of the codebase behind the model (`scripts/code_quality_check.py`)
4. **Drift** — compares predictions on the original images against the same images after noise, as a stand-in for "today's photos look different from deployment" (`scripts/drift_check.py`)

Equity/subgroup analysis is a separate, larger dimension of a full model audit and is out of scope for this kit.

## What it uses

Runtime (in `requirements.txt`):

| Package | What it's for |
|---|---|
| `transformers` | Downloads and runs the Hugging Face model (`image-classification` pipeline) |
| `torch` | The ML framework `transformers` runs on |
| `datasets` | Downloads the labeled test dataset from Hugging Face |
| `scikit-learn` | Computes accuracy, precision, recall, F1 and the confusion matrix |
| `albumentations` | Generates the blur/rotation/brightness noise for the robustness test |
| `Pillow`, `numpy` | Image handling and array operations |

Dev/audit tools (in `requirements-dev.txt`):

| Package | What it's for |
|---|---|
| `pytest` | Runs the unit tests |
| `ruff` | Lints the codebase behind a model (implementation/code check) |
| `pip-audit` | Scans that codebase's dependencies for known vulnerabilities |

`generate_report.py` uses only the Python standard library — no extra dependency to view results.

Optional, heavier (in `requirements-visual.txt`): `fiftyone`, for browsing predictions image-by-image instead of just the confusion matrix — see "Browsing results image-by-image" below.

Optional (in `requirements-drift.txt`): `evidently`, `pandas` — for the drift check, see "Checking for drift" below.

## Requirements

- Python 3.10+
- A couple GB of free disk space for model weights + dataset cache (varies by model)
- No paid accounts, no API keys — everything here is free and open source

## Install

```bash
git clone https://github.com/CHPaez/Auditar_Modelos.git
cd Auditar_Modelos

python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Running on a small or shared server

If you're installing this on a small box (2 vCPU / 4 GB RAM class) that's also running other live services (nginx, PHP-FPM, MySQL, etc.), two things help it stay light:

- Install the CPU-only PyTorch build instead of the default one (skips ~2GB of unused CUDA libraries):
  ```bash
  pip install torch --index-url https://download.pytorch.org/whl/cpu
  pip install -r requirements.txt
  ```
- Keep `--limit` low (10–20 images) for a first run, and run one script at a time rather than both scripts in parallel, so the audit doesn't compete for RAM/CPU with whatever else the box is already serving.

## Quickstart (bundled example)

Runs a real audit end-to-end on a public model + dataset — nothing to configure:

```bash
python scripts/run_audit.py
python scripts/robustness_test.py
```

This downloads `nateraw/vit-base-beans` (a model fine-tuned to classify bean leaf disease) and the `AI-Lab-Makerere/beans` test dataset automatically on first run, then prints and saves a JSON report. See `examples/beans_vit/README.md` for what to expect.

Then turn those JSON reports into one simple visual page — no setup, no server, just open the file:

```bash
python scripts/generate_report.py
```

Open the resulting `report.html` in any browser — double-click it, or from the terminal: `start report.html` (Windows) / `open report.html` (macOS) / `xdg-open report.html` (Linux). Each audited dimension gets its own section (metrics as big numbers, a color-shaded confusion matrix, a before/after bar for robustness); a dimension you haven't run yet shows as "not run yet" with the exact command to fill it in, instead of just being missing. Re-run it any time after generating new reports to refresh the page.

### Browsing results image-by-image (optional, heavier)

`report.html` gives you the confusion matrix; if you want to click through the actual misclassified images instead, this launches [FiftyOne](https://voxel51.com/fiftyone/)'s own local app in your browser:

```bash
pip install -r requirements-visual.txt
python scripts/browse_results.py
```

This is a heavier, separate install on purpose — it pulls in FiftyOne's full app stack instead of writing a static file, so it's kept out of the default `requirements.txt`.

### Checking for drift (optional)

Compares the model's predictions on the original images against the same images after blur/rotation/brightness noise, as a stand-in for "the photos reaching the model today look different from when it was deployed" — real drift monitoring would compare against actual recent production photos instead:

```bash
pip install -r requirements-drift.txt
python scripts/drift_check.py
```

Opens as `drift_report.html`, an Evidently data-drift report.

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
- `--dataset`: any Hugging Face dataset id with an `image` field and a `ClassLabel` field for the class.
- `--limit`: cap how many images to evaluate (default 40, for a quick run).
- `--label-column`: name of that class column, if it isn't `label` or `labels` (auto-detected otherwise — some datasets use other names).

**Important — matching predictions to ground truth**: this kit compares the model's predicted label text to the dataset's class names. This works out of the box when the model was fine-tuned on that exact dataset's classes — which is the normal case for a company's own production model, since it was trained on its own labeled categories. It will **not** give meaningful metrics if you pair a general-purpose model (e.g. an ImageNet classifier like `microsoft/resnet-50`) with a dataset whose classes it never learned. `scripts/download_model.py` is for that kind of quick single-photo sanity check instead — it doesn't need matching classes.

To audit against your own local photos instead of a Hugging Face dataset, organize them as `data/<class_name>/*.jpg` and load them with:

```python
from datasets import load_dataset
dataset = load_dataset("imagefolder", data_dir="data")
```

See the [`imagefolder` docs](https://huggingface.co/docs/datasets/en/image_load#imagefolder) for details.

## Checking the code behind the model

Separate from accuracy/robustness, `scripts/code_quality_check.py` audits the codebase behind your model (its training pipeline or serving code) — lint issues via `ruff`, plus a dependency vulnerability scan via `pip-audit` if the target has a `requirements.txt`:

```bash
pip install -r requirements-dev.txt
python scripts/code_quality_check.py --path /path/to/model/codebase
```

This does **not** apply to a public pretrained checkpoint downloaded from Hugging Face — there's no local source code to lint there, just weights and the `transformers` library's own (already maintained) implementation. Point it at your own model's repository instead.

## Tests

Fast, offline unit tests cover the metrics math (`audit_lib.compute_metrics`), the robustness perturbation (`robustness_test.perturb`), the subprocess wrapper used by `code_quality_check.py`, and the prediction table used by `drift_check.py` — no model download or network access required:

```bash
pip install -r requirements-dev.txt
pytest
```

`run_audit.py` and `robustness_test.py` themselves are exercised end-to-end by actually running them (see Quickstart above) rather than by a mocked test, since their entire point is to really download a model and dataset and report on them.

## Cost

Every tool this kit depends on (`transformers`, `datasets`, `albumentations`, `scikit-learn`) is free and open source, with no licensing restrictions for internal or commercial use. If you later extend this kit to object-detection models via Ultralytics/YOLO, note that Ultralytics requires either open-sourcing your project under AGPL-3.0 or a paid Enterprise license for any non-open-source use — that caveat does not apply to anything shipped in this kit.

## License

MIT — see `LICENSE`.
