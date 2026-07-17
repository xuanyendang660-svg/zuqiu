from __future__ import annotations

import argparse
import json
from pathlib import Path

from football_v2.halftime_topflight_audit import run_topflight_domain_audit
from football_v2.halftime_two_nil import load_halftime_market_data, season_codes


_TOP_DIVISIONS = ("E0", "D1", "F1", "I1", "SP1")
_TRANSFER_DIVISIONS = ("E1", "D2", "F2", "I2", "SP2")
_TOPFLIGHT_DOMAIN_DIVISIONS = ("N1", "P1", "B1", "SC0", "T1", "G1")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Final blind audit of the top-flight halftime exact-score domain"
    )
    parser.add_argument("--cache-dir", default=".cache/football-data-halftime")
    parser.add_argument(
        "--output", default="artifacts/halftime_topflight_domain_audit.json"
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
    modern, modern_loading = load_halftime_market_data(
        args.cache_dir,
        seasons=season_codes(2022, 2025),
        divisions=_TOPFLIGHT_DOMAIN_DIVISIONS,
    )
    historical, historical_loading = load_halftime_market_data(
        args.cache_dir,
        seasons=season_codes(2002, 2017),
        divisions=_TOPFLIGHT_DOMAIN_DIVISIONS,
    )
    audit = run_topflight_domain_audit(
        discovery,
        transfer,
        modern,
        historical,
    )
    payload = {
        "topflight_domain_audit": audit.to_dict(),
        "dataset": {
            "source": "football-data.co.uk",
            "discovery_matches": len(discovery),
            "transfer_matches": len(transfer),
            "modern_replication_matches": len(modern),
            "historical_blind_matches": len(historical),
            "core_rules_selected_only_from_transfer_metrics": True,
            "historical_blind_set_never_used_for_discovery_or_filtering": True,
            "top_divisions": list(_TOP_DIVISIONS),
            "transfer_divisions": list(_TRANSFER_DIVISIONS),
            "topflight_domain_divisions": list(_TOPFLIGHT_DOMAIN_DIVISIONS),
            "discovery_loading": discovery_loading,
            "transfer_loading": transfer_loading,
            "modern_loading": modern_loading,
            "historical_loading": historical_loading,
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
