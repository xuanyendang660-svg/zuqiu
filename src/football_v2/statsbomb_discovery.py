from __future__ import annotations

import json
from pathlib import Path

from .statsbomb_events import CompetitionSeason, _RAW_ROOT, _fetch_json


def discover_male_competition_seasons(
    cache_dir: str | Path,
) -> list[CompetitionSeason]:
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    path = cache / "competitions.json"
    if path.exists():
        competitions = json.loads(path.read_text(encoding="utf-8"))
    else:
        competitions = _fetch_json(f"{_RAW_ROOT}/competitions.json")
        path.write_text(json.dumps(competitions), encoding="utf-8")

    pairs = {
        CompetitionSeason(
            int(row["competition_id"]),
            int(row["season_id"]),
        )
        for row in competitions
        if row.get("competition_gender") == "male"
        and not bool(row.get("competition_youth"))
        and row.get("match_available")
    }
    return sorted(pairs, key=lambda item: (item.competition_id, item.season_id))
