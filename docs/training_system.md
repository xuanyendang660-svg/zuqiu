# football_complete_v3 Training System

This pipeline builds a leakage-safe residual exact-score training pack from
Football-Data.co.uk public CSV files.

## One-command Run

```powershell
$env:PYTHONPATH="src"
python -m score_model.training_cli --min-samples 3000
```

Windows double-click entry:

```text
train_football_complete_v3.bat
```

## Outputs

- `data/historical_football_dataset.csv`
- `data/historical_football_dataset.profile.json`
- `models/football_complete_v3/football_residual_model.joblib`
- `models/football_complete_v3/metrics.json`
- `models/football_complete_v3/holdout_predictions.csv`
- `models/football_complete_v3/PR_DESCRIPTION.md`
- `model_ready_pack.zip`

## Data Source

The downloader uses:

```text
https://www.football-data.co.uk/mmz4281/{season}/{league}.csv
```

Default seasons:

```text
1920, 2021, 2122, 2223, 2324, 2425
```

Default primary leagues:

```text
E0, D1, I1, SP1, F1, N1, P1
```

Fallback leagues are attempted automatically if the usable sample count is
below the requested threshold.

## Leakage Rules

The dataset builder only uses pre-match or prior-match information.

Excluded same-match post-event fields:

```text
HS, AS, HST, AST, HF, AF, HC, AC, HY, AY, HR, AR, HTHG, HTAG, HTR
```

Rolling features are built before each kickoff group is written back into team
state, so a match cannot use its own final score, shots, corners, cards, or
halftime score.

## Feature Blocks

- Recent 8-match goals for and against.
- Home-only and away-only rolling attack/defense.
- League pre-match averages.
- Pre-match Elo.
- Rest days and missing rest flags.
- 1X2 market implied probabilities.
- Over/under 2.5 implied probabilities.
- Asian handicap depth.
- Base lambda from market totals and 1X2, with rolling league fallback.

## Model

The model trains two RidgeCV-style residual regressions:

```text
residual_home = actual_home_goals - xg_base_home
residual_away = actual_away_goals - xg_base_away
```

The market-implied lambda remains the base estimate. Residual shrinkage is
capped so the residual layer corrects the base instead of overpowering it.

## Score Tree and Audits

Each holdout prediction includes a normalized 0-7 score tree and audited top
score selections.

Audits included in `metrics.json`:

- `top5_template_cluster`
- `high_variance_audit`
- `cold_big_ball_audit`
- `lineup_missing_no_A_grade`
- `tracking_missing`

Football-Data CSV files do not contain verified lineups, injury reports, real
pre-game xG, or tracking pressure. The pipeline does not fabricate them. The
model is therefore grade-limited to `B` until those live feeds are added.

