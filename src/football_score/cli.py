"""football-score: train/backtest, freeze a prediction, and review a frozen week."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .review import review_files
from .workflow import (load_history, predict, read_quotes, train_and_backtest,
                       write_new)


def main(argv=None):
    parser = argparse.ArgumentParser(prog="football-score")
    commands = parser.add_subparsers(dest="command", required=True)
    train = commands.add_parser("train", help="Fit on past scores; report a held-out future period")
    train.add_argument("history")
    train.add_argument("--holdout-start", required=True)
    train.add_argument("--quotes", help="Optional verified as-of snapshots keyed by match_id")
    train.add_argument("--model-out", required=True)
    train.add_argument("--report-out", required=True)
    freeze = commands.add_parser("freeze", help="Create a new, immutable pre-kickoff prediction")
    freeze.add_argument("match_json")
    freeze.add_argument("--model", required=True)
    freeze.add_argument("--out", required=True)
    audit = commands.add_parser("review", help="Compare frozen scores to verified results")
    audit.add_argument("freezes_json")
    audit.add_argument("results_json")
    audit.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "train":
            if Path(args.model_out).exists() or Path(args.report_out).exists():
                raise FileExistsError("model/report output already exists; preserve earlier runs")
            rows, sha = load_history(args.history)
            model, report = train_and_backtest(rows, sha, args.holdout_start,
                                                read_quotes(args.quotes))
            write_new(args.model_out, model)
            write_new(args.report_out, report)
            result = {"model": args.model_out, "report": args.report_out,
                      "joint_exact_hit_rate": report["joint"]["exact_hit_rate"]}
        elif args.command == "freeze":
            model = json.loads(Path(args.model).read_text(encoding="utf-8"))
            row = json.loads(Path(args.match_json).read_text(encoding="utf-8"))
            item = predict(row, model)
            write_new(args.out, {"predictions": [item]})
            result = {"out": args.out, "score": item["score"],
                      "market_status": item["market_status"]}
        else:
            result = review_files(args.freezes_json, args.results_json)
            write_new(args.out, result)
        print(json.dumps(result, ensure_ascii=False))
        return 0
    except (ValueError, KeyError, FileNotFoundError, FileExistsError, RuntimeError) as exc:
        print(str(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
