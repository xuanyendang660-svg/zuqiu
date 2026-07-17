from __future__ import annotations

import argparse
import json
from pathlib import Path

from football_v2.live_cold_blowout_ranker import (
    leave_one_league_out_cold_blowout_ranker_test,
)
from football_v2.live_snapshots import build_live_snapshot_dataset
from football_v2.market_event_data import load_market_1718
from football_v2.strict_market_join import join_market_events_without_score
from football_v2.wyscout_events import (
    build_wyscout_event_dataset,
    load_wyscout_index,
    load_wyscout_league_matches,
)
from run_wyscout_cold_blowout_strict_audit import (
    _baseline_report,
    _json_default,
    _normalise,
    _strict_dataset,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Outcome-independent unresolved 30-minute cold-blowout audit"
    )
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--market-cache", default=".cache/football-data")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--output", default="artifacts/wyscout_cold_blowout_clean_audit.json"
    )
    args = parser.parse_args()

    records = _normalise(load_wyscout_index(args.data_root))
    matches = load_wyscout_league_matches(args.data_root, workers=args.workers)
    event_dataset = build_wyscout_event_dataset(matches)
    market_event_dataset = join_market_events_without_score(
        event_dataset,
        records,
        load_market_1718(args.market_cache),
    )
    live_dataset = build_live_snapshot_dataset(
        args.data_root,
        market_event_dataset,
        records,
        cutoffs=(30,),
    )
    strict_dataset, eligibility = _strict_dataset(live_dataset, cutoff=30)
    budgets = (0.02, 0.04, 0.06)
    model_report = leave_one_league_out_cold_blowout_ranker_test(
        strict_dataset,
        cutoff=30,
        budgets=budgets,
    )
    payload = {
        "clean_unresolved_model": model_report.to_dict(),
        "simple_live_baselines": _baseline_report(strict_dataset.frame, budgets),
        "eligibility": eligibility,
        "dataset": {
            "market_join_matches": len(market_event_dataset.frame),
            "market_join_uses_final_score": False,
        },
        "audit_rules": {
            "cutoff": 30,
            "excluded_if_already_true": "underdog goals >= 3 and underdog lead >= 2",
            "selection_budget_applied_within_each_held_out_league": True,
            "market_match_keys": ["division", "date +/- 1 day", "team names"],
            "true_cluster_only_metrics_are_diagnostic_not_deployable": True,
        },
    }
    text = json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
