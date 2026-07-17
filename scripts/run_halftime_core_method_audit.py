from __future__ import annotations

import argparse
import json
from pathlib import Path

from football_v2.halftime_core_audit import run_core_method_audit
from football_v2.halftime_two_nil import load_halftime_market_data, season_codes


_TOP_DIVISIONS = ("E0", "D1", "F1", "I1", "SP1")
_TRANSFER_DIVISIONS = ("E1", "D2", "F2", "I2", "SP2")
_REPLICATION_DIVISIONS = ("N1", "P1", "B1", "SC0", "T1", "G1")
_SECOND_SHADOW_DIVISIONS = ("E2", "E3", "EC", "SC1", "SC2", "SC3")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit high-sample core exact-score rules on a second untouched shadow"
    )
    parser.add_argument("--cache-dir", default=".cache/football-data-halftime")
    parser.add_argument(
        "--output", default="artifacts/halftime_core_method_audit.json"
    )
    args = parser.parse_args()

    discovery, discovery_loading = load_halftime_market_data(
        args.cache_dir,
        seasons=season_codes(2000, 2025),
        divisions=_TOP_DIVISIONS,
    )
    transfer, transfer_loading = load_halftime_market_data(
        args.cache_dir,
        seasons=season_codes(2018, 2021),
        divisions=_TRANSFER_DIVISIONS,
    )
    replication, replication_loading = load_halftime_market_data(
        args.cache_dir,
        seasons=season_codes(2022, 2025),
        divisions=_REPLICATION_DIVISIONS,
    )
    second_shadow, second_shadow_loading = load_halftime_market_data(
        args.cache_dir,
        seasons=season_codes(2018, 2025),
        divisions=_SECOND_SHADOW_DIVISIONS,
    )
    audit = run_core_method_audit(
        discovery,
        transfer,
        replication,
        second_shadow,
    )
    payload = {
        "core_method_audit": audit.to_dict(),
        "dataset": {
            "source": "football-data.co.uk",
            "discovery_matches": len(discovery),
            "transfer_matches": len(transfer),
            "replication_matches": len(replication),
            "second_shadow_matches": len(second_shadow),
            "core_rules_selected_only_from_transfer_metrics": True,
            "second_shadow_never_used_for_discovery_filtering_or_thresholds": True,
            "top_divisions": list(_TOP_DIVISIONS),
            "transfer_divisions": list(_TRANSFER_DIVISIONS),
            "replication_divisions": list(_REPLICATION_DIVISIONS),
            "second_shadow_divisions": list(_SECOND_SHADOW_DIVISIONS),
            "discovery_loading": discovery_loading,
            "transfer_loading": transfer_loading,
            "replication_loading": replication_loading,
            "second_shadow_loading": second_shadow_loading,
        },
    }
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text + "\n", encoding="utf-8")
    print(text)
    if not audit.gate.passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
