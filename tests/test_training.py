import tempfile
import unittest
from pathlib import Path

from score_model.training import (
    FEATURE_COLUMNS,
    DownloadedCsv,
    build_historical_dataset,
    run_training_pipeline,
)


class FootballDataTrainingTests(unittest.TestCase):
    def test_feature_contract_excludes_same_match_event_fields(self):
        forbidden = {"HS", "AS", "HST", "AST", "HC", "AC", "HY", "AY", "HR", "AR", "HTHG", "HTAG"}

        self.assertTrue(forbidden.isdisjoint(set(FEATURE_COLUMNS)))

    def test_synthetic_pipeline_writes_required_artifacts(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            raw_file = root / "synthetic.csv"
            _write_synthetic_csv(raw_file, match_count=180)

            def fake_download(raw_dir, seasons, primary_leagues, fallback_leagues, min_samples):
                return [
                    DownloadedCsv(
                        season="2425",
                        league="E0",
                        path=str(raw_file),
                        url="synthetic://football-data",
                        rows=180,
                    )
                ]

            import score_model.training as training

            original_download = training.download_football_data
            training.download_football_data = fake_download
            try:
                summary = run_training_pipeline(root, min_samples=100)
            finally:
                training.download_football_data = original_download

            self.assertGreaterEqual(summary.sample_count, 100)
            self.assertTrue((root / "data" / "historical_football_dataset.csv").exists())
            self.assertTrue((root / "data" / "historical_football_dataset.profile.json").exists())
            self.assertTrue((root / "models" / "football_complete_v3" / "football_residual_model.joblib").exists())
            self.assertTrue((root / "models" / "football_complete_v3" / "metrics.json").exists())
            self.assertTrue((root / "models" / "football_complete_v3" / "holdout_predictions.csv").exists())
            self.assertTrue((root / "model_ready_pack.zip").exists())

    def test_build_dataset_marks_lineup_and_tracking_missing(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            raw_file = Path(temp_dir) / "synthetic.csv"
            _write_synthetic_csv(raw_file, match_count=40)
            rows, profile = build_historical_dataset(
                [
                    DownloadedCsv(
                        season="2425",
                        league="E0",
                        path=str(raw_file),
                        url="synthetic://football-data",
                        rows=40,
                    )
                ],
                min_samples=20,
            )

            self.assertEqual(rows[0]["lineup_missing"], 1)
            self.assertEqual(rows[0]["tracking_missing"], 1)
            self.assertFalse(profile["leakage_policy"]["lineups_not_fabricated"] is False)


def _write_synthetic_csv(path: Path, match_count: int) -> None:
    teams = [f"Team {index}" for index in range(12)]
    headers = [
        "Div",
        "Date",
        "Time",
        "HomeTeam",
        "AwayTeam",
        "FTHG",
        "FTAG",
        "B365CH",
        "B365CD",
        "B365CA",
        "B365C>2.5",
        "B365C<2.5",
        "AHCh",
        "B365CAHH",
        "B365CAHA",
        "HS",
        "AS",
        "HC",
        "AC",
        "HY",
        "AY",
        "HR",
        "AR",
    ]
    lines = [",".join(headers)]
    for index in range(match_count):
        home = teams[index % len(teams)]
        away = teams[(index * 5 + 3) % len(teams)]
        if home == away:
            away = teams[(index + 1) % len(teams)]
        day = 1 + (index % 27)
        month = 8 + ((index // 27) % 5)
        date = f"{day:02d}/{month:02d}/2024"
        home_goals = (index + len(home)) % 4
        away_goals = (index * 2 + len(away)) % 3
        home_odds = 1.80 + (index % 5) * 0.15
        draw_odds = 3.10 + (index % 4) * 0.10
        away_odds = 2.70 + (index % 6) * 0.20
        over = 1.85 + (index % 3) * 0.08
        under = 1.95 + (index % 4) * 0.08
        handicap = -0.25 if index % 2 == 0 else 0.25
        values = [
            "E0",
            date,
            "15:00",
            home,
            away,
            str(home_goals),
            str(away_goals),
            f"{home_odds:.2f}",
            f"{draw_odds:.2f}",
            f"{away_odds:.2f}",
            f"{over:.2f}",
            f"{under:.2f}",
            f"{handicap:.2f}",
            "1.90",
            "1.95",
            "99",
            "99",
            "99",
            "99",
            "9",
            "9",
            "1",
            "1",
        ]
        lines.append(",".join(values))
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    unittest.main()

