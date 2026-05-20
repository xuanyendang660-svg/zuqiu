"""CLI entry point for the Football-Data residual training pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .training import run_training_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Train football_complete_v3 from Football-Data.co.uk CSV files.")
    parser.add_argument("--root", type=Path, default=Path.cwd(), help="Project root. Defaults to current directory.")
    parser.add_argument("--min-samples", type=int, default=3000, help="Minimum usable historical matches required.")
    parser.add_argument("--data-dir", type=Path, default=None, help="Output data directory. Defaults to <root>/data.")
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=None,
        help="Output model directory. Defaults to <root>/models/football_complete_v3.",
    )
    parser.add_argument("--json", action="store_true", help="Print machine-readable summary.")
    args = parser.parse_args()

    summary = run_training_pipeline(
        args.root.resolve(),
        min_samples=args.min_samples,
        data_dir=args.data_dir,
        model_dir=args.model_dir,
    )
    payload = {
        "sample_count": summary.sample_count,
        "train_count": summary.train_count,
        "holdout_count": summary.holdout_count,
        "seasons": summary.seasons,
        "leagues": summary.leagues,
        "dataset_path": summary.dataset_path,
        "metrics_path": summary.metrics_path,
        "model_dir": summary.model_dir,
        "package_path": summary.package_path,
    }
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return

    print("football_complete_v3 training finished")
    print(f"samples: {summary.sample_count} | train: {summary.train_count} | holdout: {summary.holdout_count}")
    print(f"dataset: {summary.dataset_path}")
    print(f"metrics: {summary.metrics_path}")
    print(f"package: {summary.package_path}")


if __name__ == "__main__":
    main()

