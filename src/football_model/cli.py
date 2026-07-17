from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from .calibration import MarketTargets
from .pipeline import MatchModelInput, run_model
from .score_matrix import ScenarioWeights
from .selection import GoalLicenses


def _pairs(values: list[list[int]] | None) -> tuple[tuple[int, int], ...]:
    return tuple((int(item[0]), int(item[1])) for item in (values or []))


def load_input(path: Path) -> MatchModelInput:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return MatchModelInput(
        home_lambda=float(data["home_lambda"]),
        away_lambda=float(data["away_lambda"]),
        scenario_weights=ScenarioWeights(**data.get("scenario_weights", {})),
        market_targets=MarketTargets(**data["market_targets"])
        if data.get("market_targets")
        else None,
        licenses=GoalLicenses(**data.get("licenses", {})),
        market_cluster=_pairs(data.get("market_cluster")),
        max_goals=int(data.get("max_goals", 7)),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the football exact-score model")
    parser.add_argument("config", type=Path)
    args = parser.parse_args()
    result = run_model(load_input(args.config))
    payload = {
        "exact_score": list(result.selection.score),
        "raw_probability": result.selection.probability,
        "adjusted_score": result.selection.adjusted_score,
        "summary": result.summary,
        "comfort_cluster_warning": result.selection.comfort_cluster_warning,
        "ranked_scores": [
            {"score": list(score), "adjusted_probability": probability}
            for score, probability in result.selection.ranked_scores
        ],
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
