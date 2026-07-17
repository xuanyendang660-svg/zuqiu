from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .benchmark import baseline_distributions
from .dataset import HistoricalDataset, build_historical_dataset, load_premier_league_frame
from .persistence import ModelBundle, load_bundle, save_bundle
from .residual import MarketResidualScoreModel, ResidualConfig
from .residual_benchmark import walk_forward_residual_benchmark


def _season_list(value: str | None) -> list[str] | None:
    if value is None or not value.strip():
        return None
    return [item.strip() for item in value.split(",") if item.strip()]


def _json_dump(payload: object, path: Path | None = None) -> None:
    text = json.dumps(payload, ensure_ascii=False, indent=2, default=str)
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text + "\n", encoding="utf-8")
    print(text)


def _backtest_real(args: argparse.Namespace) -> None:
    frame = load_premier_league_frame(_season_list(args.seasons))
    dataset = build_historical_dataset(frame, max_goals=args.max_goals)
    if args.start_date:
        selected = dataset.frame[dataset.frame["date"] >= args.start_date].reset_index(drop=True)
        dataset = HistoricalDataset(selected, dataset.feature_columns)
    config = ResidualConfig(
        max_goals=args.max_goals,
        n_estimators=args.trees,
        min_samples_leaf=args.min_leaf,
        random_state=args.seed,
    )
    report, model = walk_forward_residual_benchmark(
        dataset,
        folds=args.folds,
        initial_train_fraction=args.initial_train_fraction,
        residual_config=config,
        top_k=args.top_k,
    )
    payload = report.to_dict()
    payload["dataset"] = {
        "rows": len(dataset.frame),
        "first_date": str(dataset.frame["date"].min().date()),
        "last_date": str(dataset.frame["date"].max().date()),
        "feature_count": len(dataset.feature_columns),
        "source": "premier-league-data / football-data.co.uk",
    }
    payload["model_config"] = asdict(config)
    payload["walk_forward_residual_selections"] = getattr(
        model, "walk_forward_selections_", []
    )
    payload["final_residual_selection"] = {
        "alpha": model.selection_.alpha,
        "beta": model.selection_.beta,
        "tail_boost": model.selection_.tail_boost,
    }
    output = Path(args.output)
    bundle_path = Path(args.model_output)
    save_bundle(
        ModelBundle(
            model=model,
            feature_columns=dataset.feature_columns,
            metadata={
                "version": "0.2.0",
                "model_type": "market_residual",
                "training_rows": len(dataset.frame),
                "first_date": payload["dataset"]["first_date"],
                "last_date": payload["dataset"]["last_date"],
                "benchmark": payload,
            },
        ),
        bundle_path,
    )
    payload["model_artifact"] = str(bundle_path)
    _json_dump(payload, output)


def _records_to_matrix(
    records: list[dict[str, object]], feature_columns: tuple[str, ...]
) -> np.ndarray:
    matrix = np.full((len(records), len(feature_columns)), np.nan, dtype=float)
    for row_index, record in enumerate(records):
        for column_index, name in enumerate(feature_columns):
            value = record.get(name)
            if value is not None:
                matrix[row_index, column_index] = float(value)
    return matrix


def _predict_json(args: argparse.Namespace) -> None:
    bundle = load_bundle(args.model)
    if not isinstance(bundle.model, MarketResidualScoreModel):
        raise TypeError("saved model is not the market-residual v2 model")
    raw = json.loads(Path(args.features).read_text(encoding="utf-8"))
    records = raw if isinstance(raw, list) else [raw]
    if not all(isinstance(record, dict) for record in records):
        raise ValueError("each feature record must be a JSON object")
    typed_records = [dict(record) for record in records]
    matrix = _records_to_matrix(typed_records, bundle.feature_columns)
    feature_frame = pd.DataFrame(matrix, columns=bundle.feature_columns)
    base = baseline_distributions(
        HistoricalDataset(feature_frame, bundle.feature_columns), kind="market"
    )
    distributions = bundle.model.predict_distribution(matrix, base)
    output: list[dict[str, object]] = []
    for distribution in distributions:
        order = np.argsort(distribution.ravel())[::-1][: args.top_k]
        ranked = [
            {
                "score": list(np.unravel_index(int(index), distribution.shape)),
                "probability": float(distribution.ravel()[index]),
            }
            for index in order
        ]
        output.append(
            {
                "final_score": ranked[0]["score"],
                "ranked_scores": ranked,
                "residual_selection": {
                    "alpha": bundle.model.selection_.alpha,
                    "beta": bundle.model.selection_.beta,
                    "tail_boost": bundle.model.selection_.tail_boost,
                },
            }
        )
    _json_dump(
        output if isinstance(raw, list) else output[0],
        Path(args.output) if args.output else None,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Data-driven football score model")
    subparsers = parser.add_subparsers(dest="command", required=True)

    backtest = subparsers.add_parser(
        "backtest-real",
        help="train and walk-forward test on bundled real Premier League results and odds",
    )
    backtest.add_argument("--seasons", help="comma-separated season labels; default: all")
    backtest.add_argument("--start-date", default="2000-01-01")
    backtest.add_argument("--folds", type=int, default=3)
    backtest.add_argument("--initial-train-fraction", type=float, default=0.65)
    backtest.add_argument("--trees", type=int, default=180)
    backtest.add_argument("--min-leaf", type=int, default=3)
    backtest.add_argument("--max-goals", type=int, default=7)
    backtest.add_argument("--top-k", type=int, default=5)
    backtest.add_argument("--seed", type=int, default=42)
    backtest.add_argument("--output", default="artifacts/epl_backtest.json")
    backtest.add_argument("--model-output", default="artifacts/epl_model.joblib")
    backtest.set_defaults(handler=_backtest_real)

    predict = subparsers.add_parser("predict-json", help="predict from a saved model bundle")
    predict.add_argument("--model", required=True)
    predict.add_argument("--features", required=True)
    predict.add_argument("--top-k", type=int, default=12)
    predict.add_argument("--output")
    predict.set_defaults(handler=_predict_json)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
