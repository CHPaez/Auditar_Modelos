"""Implementation/code-quality check: lint + dependency vulnerability scan for the
codebase behind a model (its training pipeline or serving code) — the
"implementation" dimension of a model audit, separate from the accuracy/
robustness checks in run_audit.py and robustness_test.py.

This does NOT apply to a public pretrained checkpoint downloaded from Hugging
Face (there's no local source code to lint there) — point it at your own
model's training/serving repo instead.

Requires `ruff` and `pip-audit` (see requirements-dev.txt).

Usage:
    python scripts/code_quality_check.py --path /path/to/model/codebase
"""
import argparse
import json
import subprocess
from pathlib import Path


def run(cmd):
    result = subprocess.run(cmd, capture_output=True, text=True)
    return {
        "command": " ".join(cmd),
        "exit_code": result.returncode,
        "output": (result.stdout + result.stderr).strip(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path", required=True, help="Path to the codebase to check")
    parser.add_argument("--out", default="code_quality_report.json")
    args = parser.parse_args()

    target = Path(args.path)
    if not target.exists():
        raise SystemExit(f"Path not found: {target}")

    report = {"path": str(target)}

    print(f"Running ruff against {target}...")
    report["lint"] = run(["ruff", "check", str(target)])

    requirements_file = target / "requirements.txt"
    if requirements_file.exists():
        print(f"Running pip-audit against {requirements_file}...")
        report["dependency_audit"] = run(["pip-audit", "-r", str(requirements_file)])
    else:
        print("No requirements.txt found at that path, skipping dependency audit.")
        report["dependency_audit"] = {"skipped": "no requirements.txt found at the given path"}

    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)

    print(json.dumps(report, indent=2))
    print(f"\nSaved to {args.out}")


if __name__ == "__main__":
    main()
