from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from .statsbomb_events import EventMatch
from .strict_side_resolution import resolve_wyscout_sides_strict
from .wyscout_events import load_wyscout_index, parse_wyscout_file


def load_wyscout_league_matches_strict(
    repository_root: str | Path,
    *,
    workers: int = 8,
    max_matches: int | None = None,
) -> list[EventMatch]:
    records = load_wyscout_index(repository_root)
    sides = resolve_wyscout_sides_strict(records)
    if max_matches is not None:
        sides = sides[:max_matches]

    matches: list[EventMatch] = []
    failures: list[str] = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(parse_wyscout_file, record): record for record in sides}
        for future in as_completed(futures):
            record = futures[future]
            try:
                matches.append(future.result())
            except Exception as exc:  # pragma: no cover - integration diagnostics
                failures.append(f"{record.index.path.name}: {exc}")

    if len(matches) != len(sides):
        sample = "; ".join(failures[:10])
        raise RuntimeError(
            f"strict Wyscout loader parsed {len(matches)} of {len(sides)} matches: {sample}"
        )
    matches.sort(key=lambda match: (match.date, match.match_id))
    return matches
