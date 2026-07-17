from __future__ import annotations

import argparse
import json
from pathlib import Path

from football_v2.halftime_transfer_validation import (
    validate_halftime_rule_transfer,
)
from football_v2.halftime_two_nil import (
    load_halftime_market_data,
    season_codes,
)


_TOP_DIVISIONS = ("E0", "D1", "F1", "I1", "SP1")
_TRANSFER_DIVISIONS = ("E1", "D2", "F2", "I2", "SP2")
_SHADOW_DIVISIONS = ("N1", "P1", "B1", "SC0", "T1", "G1")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate exact-score halftime rules across tiers and leagues"
    )
    parser.add_argument("--cache-dir", default=".cache/football-data-halftime")
    parser.add_argument(
        "--output", default="artifacts/halftime_transfer_validation.json"
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
    shadow, shadow_loading = load_halftime_market_data(
        args.cache_dir,
        seasons=season_codes(2022, 2025),
        divisions=_SHADOW_DIVISIONS,
    )
    report = validate_halftime_rule_transfer(
        discovery,
        transfer,
        shadow,
        discovery_train_end_year=2011,
        discovery_calibration_end_year=2017,
        transfer_start_year=2018,
        transfer_end_year=2021,
        shadow_start_year=2022,
    )
    payload = {
        "transfer_validation": report.to_dict(),
        "dataset": {
            "source": "football-data.co.uk",
            "discovery_matches": len(discovery),
            "transfer_matches": len(transfer),
            "shadow_matches": len(shadow),
            "top_divisions": list(_TOP_DIVISIONS),
            "transfer_divisions": list(_TRANSFER_DIVISIONS),
            "shadow_divisions": list(_SHADOW_DIVISIONS),
            "final_shadow_never_used_for_rule_discovery_or_filtering": True,
            "discovery_loading": discovery_loading,
            "transfer_loading": transfer_loading,
            "shadow_loading": shadow_loading,
        },
    }
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
