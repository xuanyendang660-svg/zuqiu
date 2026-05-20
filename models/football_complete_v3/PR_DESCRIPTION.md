# Football Complete V3 Training System

## Training Data
- Total samples: 14305
- Time range: 2019-08-02 to 2025-05-25
- Seasons: 1920, 2021, 2122, 2223, 2324, 2425
- Leagues: England Premier League, France Ligue 1, Germany Bundesliga, Italy Serie A, Netherlands Eredivisie, Portugal Primeira Liga, Spain La Liga

## Holdout Metrics
- adjusted_goal_MAE below base_goal_MAE: True (0.878539 vs 0.883862)
- adjusted_over25_brier below base_over25_brier: True (0.236620 vs 0.237345)
- adjusted_btts_brier below base_btts_brier: True (0.245174 vs 0.245417)
- top1_comfort_score_share: 0.719543

## Model Grade Limit
- Max grade: B
- Reason: Football-Data CSV has no verified lineups, injuries, or tracking pressure data; A grade disabled.

## Leakage Controls
- Same-match shots, shots on target, corners, fouls, cards, and halftime score fields are excluded.
- Rolling features are built before each kickoff group is written back into team state.
- Lineups, injuries, real pregame xG, and tracking pressure are not fabricated; missing flags are retained.