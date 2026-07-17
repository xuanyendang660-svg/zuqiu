from __future__ import annotations

import argparse
import json
from pathlib import Path

from football_v2.halftime_final_audit import run_final_method_audit
from football_v2.halftime_two_nil import load_halftime_market_data, season_codes


_TOP_DIVISIONS = ("E0", "D1", "F1", "I1", "SP1")
_TRANSFER_DIVISIONS = ("E1", "D2", "F2", "I2", "SP2")
_SHADOW_DIVISIONS = ("N1", "P1", "B1", "SC0", "T1", "G1")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Final paired audit of frozen selective halftime exact-score rules"
    )
    parser.add_argument("--cache-dir", default=".cache/football-data-halftime")
    parser.add_argument(
        "--output", default="artifacts/halftime_final_method_audit.json"
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
    audit = run_final_method_audit(discovery, transfer, shadow)
    payload = {
        "final_method_audit": audit.to_dict(),
        "dataset": {
            "source": "football-data.co.uk",
            "discovery_matches": len(discovery),
            "transfer_matches": len(transfer),
            "shadow_matches": len(shadow),
            "rules_frozen_before_final_shadow_evaluation": True,
            "final_shadow_used_only_for_accept_or_reject": True,
            "top_divisions": list(_TOP_DIVISIONS),
            "transfer_divisions": list(_TRANSFER_DIVISIONS),
            "shadow_divisions": list(_SHADOW_DIVISIONS),
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
    if not audit.gate.passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
