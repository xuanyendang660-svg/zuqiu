"""Dual-engine exact-score model."""

from __future__ import annotations

import math
from collections import Counter, defaultdict

from .schema import FinalPick, MatchInput, ScoreCandidate


BETTING_BOARD_SCORES = (
    (1, 0),
    (2, 0),
    (2, 1),
    (3, 0),
    (3, 1),
    (3, 2),
    (4, 0),
    (4, 1),
    (4, 2),
    (5, 0),
    (5, 1),
    (5, 2),
    (0, 0),
    (1, 1),
    (2, 2),
    (3, 3),
    (0, 1),
    (0, 2),
    (1, 2),
    (0, 3),
    (1, 3),
    (2, 3),
    (0, 4),
    (1, 4),
    (2, 4),
    (0, 5),
    (1, 5),
    (2, 5),
)

SCOREBOOK_SCORES = (
    (0, 0),
    (1, 0),
    (0, 1),
    (1, 1),
    (2, 0),
    (0, 2),
    (2, 1),
    (1, 2),
    (2, 2),
    (3, 0),
    (0, 3),
    (3, 1),
    (1, 3),
    (3, 2),
    (2, 3),
    (3, 3),
    (4, 0),
    (0, 4),
    (4, 1),
    (1, 4),
    (4, 2),
    (2, 4),
    (4, 3),
    (3, 4),
    (4, 4),
    (5, 0),
    (0, 5),
    (5, 1),
    (1, 5),
    (5, 2),
    (2, 5),
    (5, 3),
    (3, 5),
    (6, 0),
    (0, 6),
    (6, 1),
    (1, 6),
    (6, 2),
    (2, 6),
    (7, 0),
    (0, 7),
)


class ExactScoreModel:
    def __init__(
        self,
        max_total_goals: int = 10,
        max_team_goals: int = 9,
        mode: str = "adaptive",
        score_profile: str = "scorebook",
    ) -> None:
        self.max_total_goals = max_total_goals
        self.max_team_goals = max_team_goals
        self.score_profile = score_profile
        if mode not in {"adaptive", "balanced", "aggressive", "contrarian"}:
            raise ValueError("mode must be 'adaptive', 'balanced', 'aggressive', or 'contrarian'.")
        if score_profile not in {"scorebook", "full"}:
            raise ValueError("score_profile must be 'scorebook' or 'full'.")
        self.mode = mode
        self.score_grid = self._build_score_grid()

    def _build_score_grid(self) -> list[tuple[int, int]]:
        if self.score_profile == "scorebook":
            return [
                score
                for score in SCOREBOOK_SCORES
                if score[0] <= self.max_team_goals
                and score[1] <= self.max_team_goals
                and score[0] + score[1] <= self.max_total_goals
            ]

        return [
            (home, away)
            for home in range(self.max_team_goals + 1)
            for away in range(self.max_team_goals + 1)
            if home + away <= self.max_total_goals
        ]

    def predict(self, match: MatchInput) -> FinalPick:
        home_mu, away_mu, candidates, layer_winners = self._score_match(match)
        selected = self._select_final(match, home_mu, away_mu, candidates, layer_winners)
        return self._build_pick(match, home_mu, away_mu, selected, layer_winners)

    def predict_many(self, matches: list[MatchInput], diversify: bool = True) -> list[FinalPick]:
        contexts = []
        for match in matches:
            home_mu, away_mu, candidates, layer_winners = self._score_match(match)
            selected = self._select_final(match, home_mu, away_mu, candidates, layer_winners)
            contexts.append((match, home_mu, away_mu, candidates, layer_winners, selected))

        if not diversify or len(contexts) <= 1:
            return [
                self._build_pick(match, home_mu, away_mu, selected, layer_winners)
                for match, home_mu, away_mu, _candidates, layer_winners, selected in contexts
            ]

        slate_selections = self._select_slate_scores(contexts)
        return [
            self._build_pick(match, home_mu, away_mu, slate_selections[index], layer_winners)
            for index, (match, home_mu, away_mu, _candidates, layer_winners, _selected) in enumerate(contexts)
        ]

    def _score_match(
        self,
        match: MatchInput,
    ) -> tuple[float, float, list[ScoreCandidate], list[ScoreCandidate]]:
        home_mu, away_mu = self._expected_goals(match)
        real_distribution = self._real_distribution(match, home_mu, away_mu)
        market_distribution = self._market_distribution(match, real_distribution)
        candidates = [
            self._candidate(match, home, away, home_mu, away_mu, real_distribution, market_distribution)
            for home, away in self.score_grid
        ]
        layer_winners = self._select_layer_winners(candidates)
        return home_mu, away_mu, candidates, layer_winners

    def _build_pick(
        self,
        match: MatchInput,
        home_mu: float,
        away_mu: float,
        selected: ScoreCandidate,
        layer_winners: list[ScoreCandidate],
    ) -> FinalPick:
        return FinalPick(
            match_id=match.match_id,
            league=match.league,
            home_team=match.home_team,
            away_team=match.away_team,
            expected_home_goals=home_mu,
            expected_away_goals=away_mu,
            score=selected.score,
            confidence_band=self._confidence_band(selected, layer_winners),
            candidate=selected,
            layer_winners=tuple(layer_winners),
        )

    def _select_slate_scores(
        self,
        contexts: list[
            tuple[MatchInput, float, float, list[ScoreCandidate], list[ScoreCandidate], ScoreCandidate]
        ],
    ) -> dict[int, ScoreCandidate]:
        selected_by_index: dict[int, ScoreCandidate] = {}
        score_counts: Counter[str] = Counter()
        family_counts: Counter[str] = Counter()
        public_lane_count = 0
        slate_size = len(contexts)

        order = sorted(
            range(slate_size),
            key=lambda index: self._selection_strength(contexts[index][5], contexts[index][4]),
            reverse=True,
        )

        for index in order:
            match, home_mu, away_mu, candidates, layer_winners, base_selected = contexts[index]
            pool = self._slate_candidate_pool(match, home_mu, away_mu, candidates, layer_winners, base_selected)
            allowed_pool = [
                candidate
                for candidate in pool
                if score_counts[candidate.score] < _score_repeat_limit(candidate.score, slate_size)
            ]
            if not allowed_pool:
                allowed_pool = pool

            selected = max(
                allowed_pool,
                key=lambda candidate: self._slate_adjusted_value(
                    candidate,
                    match,
                    home_mu,
                    away_mu,
                    base_selected,
                    score_counts,
                    family_counts,
                    public_lane_count,
                    slate_size,
                ),
            )
            selected_by_index[index] = selected
            score_counts[selected.score] += 1
            family_counts[_score_family(selected.home_goals, selected.away_goals)] += 1
            if self._is_public_lane_score(selected, match, home_mu, away_mu):
                public_lane_count += 1

        if self._uses_attack_bias():
            selected_by_index = self._audit_slate_structure(contexts, selected_by_index)
            selected_by_index = self._audit_slate_goal_balance(contexts, selected_by_index)

        return selected_by_index

    def _audit_slate_goal_balance(
        self,
        contexts: list[
            tuple[MatchInput, float, float, list[ScoreCandidate], list[ScoreCandidate], ScoreCandidate]
        ],
        selected_by_index: dict[int, ScoreCandidate],
    ) -> dict[int, ScoreCandidate]:
        slate_size = len(contexts)
        if slate_size < 6:
            return selected_by_index

        audited = dict(selected_by_index)
        high_total_target = self._slate_high_total_target(contexts)
        high_indices = [
            index
            for index, selected in audited.items()
            if selected.total_goals >= 4
        ]
        if len(high_indices) <= high_total_target:
            return audited

        order = sorted(
            high_indices,
            key=lambda index: self._slate_high_total_trim_pressure(contexts[index], audited[index]),
            reverse=True,
        )

        for index in order:
            if sum(1 for selected in audited.values() if selected.total_goals >= 4) <= high_total_target:
                break

            match, home_mu, away_mu, candidates, _layer_winners, _base_selected = contexts[index]
            selected = audited[index]
            if self._is_high_total_protected(match, home_mu, away_mu, selected):
                continue

            challenger = self._slate_low_total_challenger(
                match,
                home_mu,
                away_mu,
                selected,
                candidates,
                audited,
                slate_size,
            )
            if challenger is not None:
                audited[index] = challenger

        return audited

    def _slate_high_total_trim_pressure(
        self,
        context: tuple[MatchInput, float, float, list[ScoreCandidate], list[ScoreCandidate], ScoreCandidate],
        selected: ScoreCandidate,
    ) -> float:
        match, home_mu, away_mu, _candidates, _layer_winners, _base_selected = context
        profile = self._scenario_profile(match, home_mu, away_mu)
        lane = self._score_exit_lane(selected, match, home_mu, away_mu)
        support = self._route_support(match, selected, home_mu, away_mu, lane)
        expected_total = home_mu + away_mu

        pressure = 0.18
        pressure += _clamp(4.25 - selected.total_goals, 0.0, 2.0) * 0.03
        if support < 0.44:
            pressure += 0.20
        elif support < 0.54:
            pressure += 0.10
        if selected.edge_ratio < 1.05:
            pressure += 0.08
        if expected_total < 2.65:
            pressure += 0.16
        if match.market.under_money_heat > match.market.over_money_heat + 0.12:
            pressure += 0.12
        if lane == "draw_ladder" and selected.total_goals == 4 and float(profile["draw"]) < 0.56:
            pressure += 0.10
        if lane == "exchange_war" and float(profile["exchange"]) < 0.54:
            pressure += 0.10
        if self._is_high_total_protected(match, home_mu, away_mu, selected):
            pressure -= 0.46
        return pressure

    def _is_high_total_protected(
        self,
        match: MatchInput,
        home_mu: float,
        away_mu: float,
        selected: ScoreCandidate,
    ) -> bool:
        lane = self._score_exit_lane(selected, match, home_mu, away_mu)
        support = self._route_support(match, selected, home_mu, away_mu, lane)
        profile = self._scenario_profile(match, home_mu, away_mu)
        expected_total = home_mu + away_mu
        handicap = self._handicap_profile(match, home_mu, away_mu)
        true_deep = float(handicap["true_deep"]) if handicap else 0.0

        if selected.total_goals >= 5 and support >= 0.66:
            return True
        if expected_total >= 3.05 and support >= 0.56 and match.market.over_money_heat >= 0.50:
            return True
        if match.is_final_round and _endgame_chaos(match) >= 0.58 and support >= 0.54:
            return True
        if max(_survival_pressure(match, "home"), _survival_pressure(match, "away")) >= 0.62 and support >= 0.54:
            return True
        if lane in {"single_side_blowout", "favorite_cover"} and true_deep >= 0.58 and support >= 0.48:
            return True
        if lane == "exchange_war" and float(profile["exchange"]) >= 0.68 and support >= 0.55:
            return True
        if (
            float(profile["normal"]) >= 0.80
            and float(profile["fake_hot"]) < 0.25
            and _outcome(selected.home_goals, selected.away_goals) == str(profile["favorite"])
            and selected.total_goals <= 4
            and expected_total >= 2.65
        ):
            return True
        return False

    def _slate_low_total_challenger(
        self,
        match: MatchInput,
        home_mu: float,
        away_mu: float,
        selected: ScoreCandidate,
        candidates: list[ScoreCandidate],
        selected_by_index: dict[int, ScoreCandidate],
        slate_size: int,
    ) -> ScoreCandidate | None:
        score_counts = Counter(candidate.score for candidate in selected_by_index.values())
        family_counts = Counter(_score_family(candidate.home_goals, candidate.away_goals) for candidate in selected_by_index.values())
        public_count = sum(
            1
            for candidate in selected_by_index.values()
            if self._is_public_or_comfort_pick(candidate, match, home_mu, away_mu)
        )
        public_target = max(2, math.floor(slate_size * 0.45))
        selected_public = self._is_public_or_comfort_pick(selected, match, home_mu, away_mu)
        selected_value = self._route_adjusted_value(selected, match, home_mu, away_mu)
        challengers: list[tuple[float, str, float, ScoreCandidate]] = []

        for candidate in candidates:
            if candidate.score == selected.score or candidate.total_goals >= selected.total_goals:
                continue
            if candidate.total_goals > 3:
                continue
            if score_counts[candidate.score] >= _score_repeat_limit(candidate.score, slate_size):
                continue

            lane = self._score_exit_lane(candidate, match, home_mu, away_mu)
            support = self._route_support(match, candidate, home_mu, away_mu, lane)
            candidate_public = self._is_public_or_comfort_pick(candidate, match, home_mu, away_mu)
            if not selected_public and public_count >= public_target and candidate_public:
                continue
            if not self._passes_goal_balance_quality(candidate, lane, support, selected_value):
                continue

            value = self._route_adjusted_value(candidate, match, home_mu, away_mu)
            if candidate.total_goals == 3:
                value *= 1.12
            if lane in {"low_lock", "clean_sheet", "favorite_noncover"} and support >= 0.38:
                value *= 1.12
            if lane == "upset_cold" and support >= 0.48:
                value *= 1.08
            if candidate.total_goals <= 2:
                value *= 1.06
            if candidate.total_goals == 3 and selected.total_goals >= 5:
                value *= 1.04
            if score_counts[candidate.score] == 0:
                value *= 1.05
            family_count = family_counts[_score_family(candidate.home_goals, candidate.away_goals)]
            if family_count >= max(2, round(slate_size * 0.28)):
                value *= 0.78

            challengers.append((value, lane, support, candidate))

        if not challengers:
            return None

        challenger_value, lane, support, challenger = max(challengers, key=lambda item: item[0])
        threshold = 0.58
        if selected.total_goals >= 5:
            threshold -= 0.06
        if support >= 0.50:
            threshold -= 0.05
        if challenger.edge_ratio >= 1.05:
            threshold -= 0.03
        if challenger.total_goals <= 1 and support < 0.46:
            threshold += 0.08

        if challenger_value < selected_value * _clamp(threshold, 0.42, 0.72):
            return None

        reason = f"slate goal balance audit: high-total share too high; lane={lane} support={support:.2f}"
        return _with_extra_reasons(challenger, reason)

    def _slate_high_total_target(
        self,
        contexts: list[
            tuple[MatchInput, float, float, list[ScoreCandidate], list[ScoreCandidate], ScoreCandidate]
        ],
    ) -> int:
        slate_size = len(contexts)
        pressures = [
            self._match_high_total_pressure(match, home_mu, away_mu)
            for match, home_mu, away_mu, _candidates, _layers, _base in contexts
        ]
        average_pressure = sum(pressures) / max(len(pressures), 1)
        rate = _clamp(0.30 + average_pressure * 0.32, 0.34, 0.58)
        return max(2, round(slate_size * rate))

    def _match_high_total_pressure(self, match: MatchInput, home_mu: float, away_mu: float) -> float:
        profile = self._scenario_profile(match, home_mu, away_mu)
        expected_total = home_mu + away_mu
        handicap = self._handicap_profile(match, home_mu, away_mu)
        true_deep = float(handicap["true_deep"]) if handicap else 0.0
        trap = float(handicap["trap"]) if handicap else 0.0

        pressure = (
            _clamp((expected_total - 2.35) / 1.10, 0.0, 1.0) * 0.30
            + _clamp(match.league_volatility - 0.42, 0.0, 0.58) * 0.32
            + float(profile["exchange"]) * 0.22
            + _clamp(match.market.over_money_heat, 0.0, 1.0) * 0.16
            + _endgame_chaos(match) * 0.14
            + max(_survival_pressure(match, "home"), _survival_pressure(match, "away")) * 0.14
            + true_deep * 0.12
            - _clamp(match.market.under_money_heat, 0.0, 1.0) * 0.12
            - trap * 0.08
        )
        if match.home_draw_sufficient and match.away_draw_sufficient:
            pressure -= 0.18
        if expected_total <= 2.20:
            pressure -= 0.14
        if match.weather_goal_drag >= 0.30:
            pressure -= 0.12
        return _clamp(pressure, 0.0, 1.0)

    def _passes_goal_balance_quality(
        self,
        candidate: ScoreCandidate,
        lane: str,
        support: float,
        selected_value: float,
    ) -> bool:
        if candidate.real_probability < 0.022 and candidate.final_value < selected_value * 0.18:
            return False
        if candidate.structure_score < 0.74 and candidate.edge_ratio < 1.02 and support < 0.38:
            return False
        if candidate.conflict_penalty > 0.20 and support < 0.54:
            return False
        if lane == "low_lock" and support < 0.30:
            return False
        if lane == "draw_ladder" and support < 0.38:
            return False
        if lane == "clean_sheet" and support < 0.26:
            return False
        if lane in {"single_side_blowout", "favorite_cover"} and support < 0.42:
            return False
        if lane == "upset_cold" and support < 0.42 and candidate.edge_ratio < 1.12:
            return False
        return True

    def _audit_slate_structure(
        self,
        contexts: list[
            tuple[MatchInput, float, float, list[ScoreCandidate], list[ScoreCandidate], ScoreCandidate]
        ],
        selected_by_index: dict[int, ScoreCandidate],
    ) -> dict[int, ScoreCandidate]:
        slate_size = len(contexts)
        if slate_size < 6:
            return selected_by_index

        audited = dict(selected_by_index)
        target_public = max(2, math.floor(slate_size * 0.45))

        public_indices = [
            index
            for index, (match, home_mu, away_mu, _candidates, _layers, _base) in enumerate(contexts)
            if self._is_public_or_comfort_pick(audited[index], match, home_mu, away_mu)
        ]
        if len(public_indices) <= target_public:
            return audited

        order = sorted(
            public_indices,
            key=lambda index: self._slate_reaudit_pressure(contexts[index], audited[index]),
            reverse=True,
        )

        for index in order:
            if len(
                [
                    item_index
                    for item_index, (match, home_mu, away_mu, _candidates, _layers, _base) in enumerate(contexts)
                    if self._is_public_or_comfort_pick(audited[item_index], match, home_mu, away_mu)
                ]
            ) <= target_public:
                break

            match, home_mu, away_mu, candidates, _layer_winners, _base_selected = contexts[index]
            selected = audited[index]
            challenger = self._slate_structure_challenger(
                match,
                home_mu,
                away_mu,
                selected,
                candidates,
                audited,
                slate_size,
            )
            if challenger is not None:
                audited[index] = challenger

        return audited

    def _slate_reaudit_pressure(
        self,
        context: tuple[MatchInput, float, float, list[ScoreCandidate], list[ScoreCandidate], ScoreCandidate],
        selected: ScoreCandidate,
    ) -> float:
        match, home_mu, away_mu, _candidates, _layer_winners, _base_selected = context
        profile = self._scenario_profile(match, home_mu, away_mu)
        plan = self._forced_exit_plan(match, home_mu, away_mu)
        lane = self._score_exit_lane(selected, match, home_mu, away_mu)
        support = self._route_support(match, selected, home_mu, away_mu, lane)
        outcome = _outcome(selected.home_goals, selected.away_goals)
        favorite = str(profile["favorite"])

        pressure = 0.0
        if _is_comfort_score(selected.home_goals, selected.away_goals):
            pressure += 0.26
        elif _is_broad_public_score(selected.home_goals, selected.away_goals):
            pressure += 0.18
        if outcome == favorite and selected.total_goals <= 3:
            pressure += 0.14
        pressure += float(profile["fake_hot"]) * 0.26
        pressure += float(profile["upset"]) * 0.18
        pressure += float(profile["exchange"]) * 0.14
        pressure += float(profile["draw"]) * 0.10
        if bool(plan["active"]):
            pressure += float(plan["strength"]) * 0.30
        if support < 0.44:
            pressure += 0.12
        if selected.real_probability < 0.080:
            pressure += 0.08

        if (
            float(profile["normal"]) >= 0.80
            and float(profile["fake_hot"]) < 0.25
            and outcome == favorite
            and support >= 0.46
        ):
            pressure -= 0.44
        return pressure

    def _slate_structure_challenger(
        self,
        match: MatchInput,
        home_mu: float,
        away_mu: float,
        selected: ScoreCandidate,
        candidates: list[ScoreCandidate],
        selected_by_index: dict[int, ScoreCandidate],
        slate_size: int,
    ) -> ScoreCandidate | None:
        profile = self._scenario_profile(match, home_mu, away_mu)
        plan = self._forced_exit_plan(match, home_mu, away_mu)
        preferred_lanes = self._slate_pressure_lanes(match, home_mu, away_mu, profile, plan)
        if not preferred_lanes:
            return None

        score_counts = Counter(candidate.score for candidate in selected_by_index.values())
        family_counts = Counter(_score_family(candidate.home_goals, candidate.away_goals) for candidate in selected_by_index.values())
        selected_value = self._route_adjusted_value(selected, match, home_mu, away_mu)
        challengers: list[tuple[float, str, float, ScoreCandidate]] = []

        for candidate in candidates:
            if candidate.score == selected.score:
                continue
            if score_counts[candidate.score] >= _score_repeat_limit(candidate.score, slate_size):
                continue

            lane = self._score_exit_lane(candidate, match, home_mu, away_mu)
            if lane not in preferred_lanes:
                continue
            support = self._route_support(match, candidate, home_mu, away_mu, lane)
            if not self._passes_slate_audit_quality(candidate, lane, support, selected_value, match, home_mu, away_mu):
                continue
            if not self._is_slate_audit_shape(candidate, lane, support, match, home_mu, away_mu):
                continue

            value = self._forced_exit_value(candidate, lane, support, match, home_mu, away_mu, preferred_lanes)
            if score_counts[candidate.score] == 0:
                value *= 1.06
            family_count = family_counts[_score_family(candidate.home_goals, candidate.away_goals)]
            if family_count == 0:
                value *= 1.08
            elif family_count >= max(2, round(slate_size * 0.24)):
                value *= 0.76
            if lane in {"upset_cold", "single_side_blowout", "clean_sheet"} and support >= 0.52:
                value *= 1.10
            if lane == "exchange_war" and candidate.total_goals >= 5:
                value *= 1.08

            challengers.append((value, lane, support, candidate))

        if not challengers:
            return None

        challenger_value, lane, support, challenger = max(challengers, key=lambda item: item[0])
        threshold = 0.70
        if bool(plan["active"]):
            threshold -= 0.07
        if support >= 0.56:
            threshold -= 0.06
        if challenger.edge_ratio >= 1.08:
            threshold -= 0.03
        if (
            float(profile["normal"]) >= 0.80
            and float(profile["fake_hot"]) < 0.25
            and _outcome(selected.home_goals, selected.away_goals) == str(profile["favorite"])
        ):
            threshold += 0.18
        if challenger.total_goals >= 6 and support < 0.68:
            threshold += 0.08

        if challenger_value < selected_value * _clamp(threshold, 0.46, 0.86):
            return None

        reason = f"slate structure audit: comfort share too high; lane={lane} support={support:.2f}"
        return _with_extra_reasons(challenger, reason)

    def _slate_pressure_lanes(
        self,
        match: MatchInput,
        home_mu: float,
        away_mu: float,
        profile: dict[str, float | str],
        plan: dict[str, object],
    ) -> set[str]:
        lanes = set(plan["lanes"]) if bool(plan["active"]) else set()
        expected_total = home_mu + away_mu

        if float(profile["fake_hot"]) >= 0.32 or float(profile["upset"]) >= 0.34:
            lanes.update({"upset_cold", "draw_ladder"})
        if float(profile["exchange"]) >= 0.46 and expected_total >= 2.52:
            lanes.add("exchange_war")
        if float(profile["draw"]) >= 0.50:
            lanes.add("draw_ladder")
            if expected_total <= 2.35 or match.market.under_money_heat >= 0.50:
                lanes.add("low_lock")
        if expected_total <= 2.30 or match.market.under_money_heat >= 0.58:
            lanes.update({"low_lock", "clean_sheet"})
        if expected_total >= 2.72 or match.league_volatility >= 0.58:
            lanes.update({"exchange_war", "single_side_blowout", "clean_sheet"})
        if match.market.over_money_heat >= 0.55 and expected_total >= 2.55:
            lanes.add("exchange_war")
        if max(_survival_pressure(match, "home"), _survival_pressure(match, "away")) >= 0.45:
            lanes.update({"single_side_blowout", "clean_sheet", "exchange_war", "upset_cold"})

        handicap = self._handicap_profile(match, home_mu, away_mu)
        if handicap:
            if float(handicap["trap"]) >= 0.34:
                lanes.update({"favorite_noncover", "upset_cold", "draw_ladder", "low_lock"})
            if float(handicap["true_deep"]) >= 0.46:
                lanes.update({"favorite_cover", "single_side_blowout", "clean_sheet"})

        return lanes

    def _passes_slate_audit_quality(
        self,
        candidate: ScoreCandidate,
        lane: str,
        support: float,
        selected_value: float,
        match: MatchInput,
        home_mu: float,
        away_mu: float,
    ) -> bool:
        if self._is_public_or_comfort_pick(candidate, match, home_mu, away_mu) and lane not in {"clean_sheet", "low_lock"}:
            return False
        probability_floor = 0.014 if support >= 0.54 else 0.022
        if candidate.real_probability < probability_floor and candidate.final_value < selected_value * 0.14:
            return False
        if candidate.structure_score < 0.72 and candidate.edge_ratio < 1.04 and support < 0.50:
            return False
        if candidate.conflict_penalty > 0.22 and support < 0.60:
            return False
        if lane == "exchange_war" and support < 0.38:
            return False
        if lane == "draw_ladder" and support < 0.40:
            return False
        if lane == "clean_sheet" and support < 0.34:
            return False
        if lane in {"single_side_blowout", "upset_cold"} and support < 0.42:
            return False
        if lane == "single_side_blowout" and candidate.total_goals >= 6 and support < 0.66:
            return False
        if lane == "upset_cold" and candidate.conflict_penalty > 0.16 and candidate.edge_ratio < 1.16 and support < 0.62:
            return False
        return True

    def _is_slate_audit_shape(
        self,
        candidate: ScoreCandidate,
        lane: str,
        support: float,
        match: MatchInput,
        home_mu: float,
        away_mu: float,
    ) -> bool:
        home = candidate.home_goals
        away = candidate.away_goals
        total = candidate.total_goals
        if lane == "exchange_war":
            return total >= 4 and min(home, away) >= 1
        if lane == "single_side_blowout":
            return abs(home - away) >= 3 or (abs(home - away) >= 2 and support >= 0.62)
        if lane == "clean_sheet":
            return min(home, away) == 0 and total >= 2 and (not _is_broad_public_score(home, away) or support >= 0.56)
        if lane == "low_lock":
            return (home, away) == (0, 0) or (total == 1 and support >= 0.60)
        if lane == "draw_ladder":
            return home == away and total >= 4
        if lane == "upset_cold":
            return _outcome(home, away) not in {_favorite_side(match, home_mu, away_mu), "draw"}
        if lane in {"favorite_cover", "favorite_noncover"}:
            return not _is_comfort_score(home, away)
        return not _is_comfort_score(home, away)

    def _selection_strength(self, selected: ScoreCandidate, layer_winners: list[ScoreCandidate]) -> float:
        ordered = sorted(layer_winners, key=lambda item: item.final_value, reverse=True)
        runner_up = ordered[1].final_value if len(ordered) > 1 else 0.0
        separation = max(selected.final_value - runner_up, 0.0)
        separation_ratio = separation / max(selected.final_value, 0.0001)
        return selected.real_probability + separation_ratio * 0.08 + selected.structure_score * 0.01

    def _slate_candidate_pool(
        self,
        match: MatchInput,
        home_mu: float,
        away_mu: float,
        candidates: list[ScoreCandidate],
        layer_winners: list[ScoreCandidate],
        base_selected: ScoreCandidate,
    ) -> list[ScoreCandidate]:
        by_score = {base_selected.score: base_selected}
        candidate_limit = len(candidates) if self.score_profile == "scorebook" else 32
        for candidate in sorted(candidates, key=lambda item: item.final_value, reverse=True)[:candidate_limit]:
            by_score.setdefault(candidate.score, candidate)
        for candidate in layer_winners:
            by_score.setdefault(candidate.score, candidate)
        for candidate in self._route_specialists(match, home_mu, away_mu, candidates):
            by_score.setdefault(candidate.score, candidate)

        floor_probability = max(0.010, base_selected.real_probability * 0.18)
        floor_value = base_selected.final_value * 0.16
        pool = []
        for candidate in by_score.values():
            if candidate.real_probability < floor_probability and candidate.final_value < floor_value:
                continue
            if candidate.structure_score < 0.70 and candidate.edge_ratio < 1.02:
                continue
            pool.append(candidate)

        return pool or [base_selected]

    def _slate_adjusted_value(
        self,
        candidate: ScoreCandidate,
        match: MatchInput,
        home_mu: float,
        away_mu: float,
        base_selected: ScoreCandidate,
        score_counts: Counter[str],
        family_counts: Counter[str],
        public_lane_count: int,
        slate_size: int,
    ) -> float:
        value = candidate.final_value
        repeat_count = score_counts[candidate.score]
        family = _score_family(candidate.home_goals, candidate.away_goals)
        family_count = family_counts[family]
        is_public_lane = self._is_public_lane_score(candidate, match, home_mu, away_mu)
        public_lane_limit = max(2, round(slate_size * 0.34))
        scenario = self._scenario_profile(match, home_mu, away_mu) if self.mode == "adaptive" else None
        favorite = str(scenario["favorite"]) if scenario else _favorite_side(match, home_mu, away_mu)
        outcome = _outcome(candidate.home_goals, candidate.away_goals)

        if repeat_count:
            value *= 0.38**repeat_count
        if family_count and candidate.score == base_selected.score:
            value *= 0.96 ** min(family_count, 4)
        elif family_count:
            value *= 0.86 ** min(family_count, 5)

        draw_family_limit = max(2, round(slate_size * 0.28))
        if family == "draw_ladder" and family_count >= draw_family_limit:
            value *= 0.48
        if candidate.score == "2-2":
            value *= 0.82
            if repeat_count:
                value *= 0.16
            if family_count:
                value *= 0.72
        if _is_broad_public_score(candidate.home_goals, candidate.away_goals):
            value *= 0.92
            if repeat_count:
                value *= 0.28
            if family_count:
                value *= 0.82

        if candidate.score == base_selected.score:
            value *= 1.10
        else:
            relative_value = candidate.final_value / max(base_selected.final_value, 0.0001)
            if relative_value < 0.34:
                value *= 0.70
            elif relative_value >= 0.74:
                value *= 1.08

        if family_count == 0 and candidate.score != "2-2":
            value *= 1.04
        if self.mode == "adaptive":
            if scenario and scenario["normal"] >= 0.82 and favorite != "draw" and outcome != favorite:
                value *= 0.18
                if outcome == "draw" and candidate.total_goals >= 4:
                    value *= 1.35
            if scenario and scenario["normal"] >= 0.82 and outcome == favorite and candidate.total_goals >= 3:
                value *= 1.18
            if is_public_lane:
                value *= 0.94 if scenario and scenario["normal"] >= 0.82 else 0.90
                if public_lane_count >= public_lane_limit:
                    if candidate.total_goals <= 2:
                        value *= 0.24
                    else:
                        value *= 0.72 if scenario and scenario["normal"] >= 0.82 else 0.34
            elif candidate.real_probability >= 0.026 and candidate.structure_score >= 0.78:
                value *= 1.16
                if public_lane_count >= public_lane_limit:
                    value *= 1.34
                if candidate.total_goals >= 4:
                    value *= 1.08
            if _is_scorebook_edge_score(candidate.home_goals, candidate.away_goals):
                value *= 1.06
            lane = self._score_exit_lane(candidate, match, home_mu, away_mu)
            support = self._route_support(match, candidate, home_mu, away_mu, lane)
            if lane in {"low_lock", "clean_sheet", "single_side_blowout"} and support >= 0.48:
                value *= 1.10
            if lane in {"upset_cold", "exchange_war"} and support >= 0.52:
                value *= 1.08
            if _is_broad_public_score(candidate.home_goals, candidate.away_goals) and support < 0.54:
                value *= 0.86

        return value

    def _route_specialists(
        self,
        match: MatchInput,
        home_mu: float,
        away_mu: float,
        candidates: list[ScoreCandidate],
    ) -> list[ScoreCandidate]:
        """Keep one strong candidate from every script lane in the slate pool."""

        specialists: dict[str, tuple[float, ScoreCandidate]] = {}
        for candidate in candidates:
            if candidate.real_probability < 0.010 and candidate.edge_ratio < 1.03:
                continue
            if candidate.structure_score < 0.68 and candidate.edge_ratio < 1.08:
                continue
            lane = self._score_exit_lane(candidate, match, home_mu, away_mu)
            support = self._route_support(match, candidate, home_mu, away_mu, lane)
            if support < 0.24 and lane in {"low_lock", "single_side_blowout", "upset_cold"}:
                continue
            adjusted = self._route_adjusted_value(candidate, match, home_mu, away_mu)
            current = specialists.get(lane)
            if current is None or adjusted > current[0]:
                specialists[lane] = (adjusted, candidate)
        return [candidate for _value, candidate in specialists.values()]

    def _route_adjusted_value(
        self,
        candidate: ScoreCandidate,
        match: MatchInput,
        home_mu: float,
        away_mu: float,
    ) -> float:
        lane = self._score_exit_lane(candidate, match, home_mu, away_mu)
        support = self._route_support(match, candidate, home_mu, away_mu, lane)
        value = candidate.final_value

        if support >= 0.30:
            value *= 1.0 + support * 0.34
        if lane in {"low_lock", "clean_sheet", "single_side_blowout"} and support >= 0.46:
            value *= 1.12
        if lane in {"upset_cold", "exchange_war"} and support >= 0.52:
            value *= 1.10
        if _is_broad_public_score(candidate.home_goals, candidate.away_goals) and support < 0.54:
            value *= 0.84
        if candidate.total_goals >= 6 and support < 0.62:
            value *= 0.78
        if candidate.real_probability < 0.018 and support < 0.55:
            value *= 0.70
        return value

    def _score_exit_lane(
        self,
        candidate: ScoreCandidate,
        match: MatchInput,
        home_mu: float,
        away_mu: float,
    ) -> str:
        return _score_exit_lane_from_values(candidate.home_goals, candidate.away_goals, match, home_mu, away_mu)

    def _route_support(
        self,
        match: MatchInput,
        candidate: ScoreCandidate,
        home_mu: float,
        away_mu: float,
        lane: str,
    ) -> float:
        home = candidate.home_goals
        away = candidate.away_goals
        total = home + away
        expected_total = home_mu + away_mu
        outcome = _outcome(home, away)
        profile = self._scenario_profile(match, home_mu, away_mu)
        favorite = str(profile["favorite"])
        handicap = self._handicap_profile(match, home_mu, away_mu)
        edge_support = _clamp((candidate.edge_ratio - 1.0) * 0.55, 0.0, 0.22)
        under_heat = _clamp(match.market.under_money_heat, 0.0, 1.0)
        over_heat = _clamp(match.market.over_money_heat, 0.0, 1.0)
        chaos = _endgame_chaos(match)
        home_survival = _survival_pressure(match, "home")
        away_survival = _survival_pressure(match, "away")

        if lane == "low_lock":
            draw_script = 0.25 if match.home_draw_sufficient and match.away_draw_sufficient else 0.0
            return _clamp(
                under_heat * 0.34
                + _clamp(match.weather_goal_drag, 0.0, 1.0) * 0.24
                + _clamp(2.45 - expected_total, 0.0, 1.0) * 0.30
                + _clamp(0.55 - match.league_volatility, 0.0, 0.55) * 0.18
                + draw_script
                + (chaos * 0.08 if match.is_final_round and (match.home_settled or match.away_settled) else 0.0)
                + edge_support,
                0.0,
                1.0,
            )

        if lane == "draw_ladder":
            draw_script = 0.22 if match.home_draw_sufficient and match.away_draw_sufficient else 0.0
            return _clamp(
                float(profile["draw"]) * 0.52
                + _clamp(match.market.draw_money_heat, 0.0, 1.0) * 0.14
                + _clamp(match.league_draw_bias, -0.4, 0.4) * 0.16
                + draw_script
                + (chaos * 0.10 if match.is_final_round and total >= 4 else 0.0)
                + edge_support,
                0.0,
                1.0,
            )

        if lane == "exchange_war":
            weak_side_mu = min(home_mu, away_mu)
            high_loser_goals_penalty = 0.0
            if min(home, away) >= 2 and weak_side_mu < 1.20:
                high_loser_goals_penalty = 0.16
            if match.is_final_round and abs(home - away) <= 1:
                high_loser_goals_penalty += max(match.home_collapse_risk, match.away_collapse_risk) * 0.10
            return _clamp(
                float(profile["exchange"]) * 0.48
                + _clamp(match.league_volatility - 0.48, 0.0, 0.52) * 0.40
                + _clamp(expected_total - 2.65, 0.0, 1.6) * 0.16
                + over_heat * 0.14
                + (chaos * 0.18 if match.is_final_round and total >= 5 else 0.0)
                + (max(home_survival, away_survival) * 0.10 if total >= 4 else 0.0)
                + edge_support
                - high_loser_goals_penalty,
                0.0,
                1.0,
            )

        if lane == "upset_cold":
            underdog_pressure = _stake_pressure(match, outcome)
            underdog_survival = home_survival if outcome == "home" else away_survival
            return _clamp(
                float(profile["upset"]) * 0.48
                + float(profile["fake_hot"]) * 0.20
                + _clamp(underdog_pressure, 0.0, 1.0) * 0.18
                + _clamp(underdog_survival, 0.0, 1.0) * 0.22
                + _side_money_heat(match, outcome) * 0.10
                + (
                    _clamp(match.home_collapse_risk if outcome == "away" else match.away_collapse_risk, 0.0, 1.0)
                    * 0.22
                    if match.is_final_round
                    else 0.0
                )
                + edge_support,
                0.0,
                1.0,
            )

        if lane in {"clean_sheet", "single_side_blowout", "favorite_cover"}:
            winner = "home" if home > away else "away"
            loser = "away" if winner == "home" else "home"
            loser_mu = away_mu if winner == "home" else home_mu
            winner_big_need = match.home_big_win_need if winner == "home" else match.away_big_win_need
            loser_settled = match.away_settled if loser == "away" else match.home_settled
            loser_collapse = match.away_collapse_risk if loser == "away" else match.home_collapse_risk
            winner_celebration = match.home_celebration_risk if winner == "home" else match.away_celebration_risk
            winner_survival = home_survival if winner == "home" else away_survival
            true_deep = 0.0
            trap = 0.0
            if handicap and str(handicap["favorite"]) == winner:
                true_deep = float(handicap["true_deep"])
                trap = float(handicap["trap"])
            return _clamp(
                _clamp(1.12 - loser_mu, 0.0, 1.0) * 0.26
                + _clamp(_stake_pressure(match, winner), 0.0, 1.0) * 0.16
                + _clamp(winner_big_need, 0.0, 1.0) * 0.18
                + true_deep * 0.22
                + (0.10 if loser_settled else 0.0)
                + (_clamp(loser_collapse, 0.0, 1.0) * 0.24 if match.is_final_round else 0.0)
                + (_clamp(winner_celebration, 0.0, 1.0) * 0.10 if match.is_final_round and total >= 5 else 0.0)
                + _clamp(winner_survival, 0.0, 1.0) * 0.26
                + edge_support
                - trap * 0.10,
                0.0,
                1.0,
            )

        if lane == "favorite_noncover":
            trap = float(handicap["trap"]) if handicap else _clamp(match.market.bookmaker_trap_risk, 0.0, 1.0)
            underdog = "away" if favorite == "home" else "home"
            favorite_settled = match.home_settled if favorite == "home" else match.away_settled
            return _clamp(
                trap * 0.36
                + _clamp(_stake_pressure(match, underdog), 0.0, 1.0) * 0.20
                + under_heat * 0.14
                + (0.12 if favorite_settled else 0.0)
                + edge_support,
                0.0,
                1.0,
            )

        return _clamp(0.18 + edge_support, 0.0, 1.0)

    def _is_public_lane_score(
        self,
        candidate: ScoreCandidate,
        match: MatchInput,
        home_mu: float,
        away_mu: float,
    ) -> bool:
        favorite = _market_favorite_side(match, home_mu, away_mu)
        score = (candidate.home_goals, candidate.away_goals)
        if candidate.home_goals == candidate.away_goals:
            return score in {(0, 0), (1, 1), (2, 2)}
        if favorite == "home" and candidate.home_goals > candidate.away_goals:
            return score in {(1, 0), (2, 0), (2, 1), (3, 1)}
        if favorite == "away" and candidate.away_goals > candidate.home_goals:
            return score in {(0, 1), (0, 2), (1, 2), (1, 3)}
        return False

    def _expected_goals(self, match: MatchInput) -> tuple[float, float]:
        base_total = _clamp(match.league_avg_goals, 1.55, 3.75)
        tempo = 1.0 + (_clamp(match.league_volatility, 0.0, 1.0) - 0.5) * 0.18
        base_total *= tempo
        base_total *= 1.0 + _endgame_chaos(match) * 0.045
        base_total *= 1.0 - _clamp(match.weather_goal_drag, 0.0, 0.6) * 0.22
        base_total *= 1.0 + _clamp(match.referee_goal_bias, -1.0, 1.0) * 0.08

        home_goal_edge = _clamp(match.league_home_advantage, -0.2, 0.35)
        home_mu = _clamp(base_total / 2.0 + home_goal_edge, 0.15, base_total - 0.15)
        away_mu = max(base_total - home_mu, 0.15)

        home_mu *= math.sqrt(_safe_ratio(match.home_attack, match.away_defense))
        away_mu *= math.sqrt(_safe_ratio(match.away_attack, match.home_defense))

        if match.recent_home_xg is not None and match.recent_away_xga is not None:
            home_mu = home_mu * 0.62 + ((match.recent_home_xg + match.recent_away_xga) / 2.0) * 0.38
        if match.recent_away_xg is not None and match.recent_home_xga is not None:
            away_mu = away_mu * 0.62 + ((match.recent_away_xg + match.recent_home_xga) / 2.0) * 0.38

        home_mu *= 1.0 + _clamp(match.home_form, -1.0, 1.0) * 0.08
        away_mu *= 1.0 + _clamp(match.away_form, -1.0, 1.0) * 0.08
        home_mu *= 1.0 + _clamp(match.home_motivation, -1.0, 1.0) * 0.07
        away_mu *= 1.0 + _clamp(match.away_motivation, -1.0, 1.0) * 0.07

        home_stakes = _stake_pressure(match, "home")
        away_stakes = _stake_pressure(match, "away")
        home_survival = _survival_pressure(match, "home")
        away_survival = _survival_pressure(match, "away")
        home_mu *= 1.0 + home_stakes * 0.075
        away_mu *= 1.0 + away_stakes * 0.075
        home_mu *= 1.0 + home_survival * 0.105
        away_mu *= 1.0 + away_survival * 0.105
        if home_survival > 0:
            away_mu *= 1.0 + home_survival * 0.025
        if away_survival > 0:
            home_mu *= 1.0 + away_survival * 0.025

        home_mu *= 1.0 + _clamp(match.home_big_win_need, 0.0, 1.0) * 0.075
        away_mu *= 1.0 + _clamp(match.away_big_win_need, 0.0, 1.0) * 0.075
        if match.home_big_win_need > 0:
            away_mu *= 1.0 + _clamp(match.home_big_win_need, 0.0, 1.0) * 0.025
        if match.away_big_win_need > 0:
            home_mu *= 1.0 + _clamp(match.away_big_win_need, 0.0, 1.0) * 0.025

        if match.home_draw_sufficient and match.away_draw_sufficient:
            home_mu *= 0.93
            away_mu *= 0.93
        elif match.home_draw_sufficient and away_stakes <= 0.15:
            home_mu *= 0.97
            away_mu *= 0.95
        elif match.away_draw_sufficient and home_stakes <= 0.15:
            home_mu *= 0.95
            away_mu *= 0.97

        if match.home_settled:
            home_mu *= 0.965
            away_mu *= 1.0 + _clamp(away_stakes, 0.0, 1.0) * 0.035
        if match.away_settled:
            away_mu *= 0.965
            home_mu *= 1.0 + _clamp(home_stakes, 0.0, 1.0) * 0.035

        if match.is_final_round:
            home_mu *= 1.0 + _clamp(match.home_celebration_risk, 0.0, 1.0) * 0.035
            away_mu *= 1.0 + _clamp(match.away_celebration_risk, 0.0, 1.0) * 0.035
            home_mu *= 1.0 + _clamp(match.away_collapse_risk, 0.0, 1.0) * 0.085
            away_mu *= 1.0 + _clamp(match.home_collapse_risk, 0.0, 1.0) * 0.085
            if match.home_celebration_risk > 0:
                away_mu *= 1.0 + _clamp(match.home_celebration_risk, 0.0, 1.0) * 0.025
            if match.away_celebration_risk > 0:
                home_mu *= 1.0 + _clamp(match.away_celebration_risk, 0.0, 1.0) * 0.025

        home_mu *= 1.0 - _clamp(match.home_absence_impact, 0.0, 1.0) * 0.12
        away_mu *= 1.0 - _clamp(match.away_absence_impact, 0.0, 1.0) * 0.12
        home_mu *= 1.0 + _clamp(match.away_absence_impact, 0.0, 1.0) * 0.09
        away_mu *= 1.0 + _clamp(match.home_absence_impact, 0.0, 1.0) * 0.09

        home_mu *= 1.0 - _clamp(match.home_rotation_risk, 0.0, 1.0) * 0.09
        away_mu *= 1.0 - _clamp(match.away_rotation_risk, 0.0, 1.0) * 0.09
        home_mu *= 1.0 + _clamp(match.home_rest_edge - match.away_rest_edge, -1.0, 1.0) * 0.05
        away_mu *= 1.0 + _clamp(match.away_rest_edge - match.home_rest_edge, -1.0, 1.0) * 0.05

        if match.market.expected_home_goals is not None and match.market.expected_away_goals is not None:
            market_weight = 0.22 + _clamp(match.market.confidence, 0.0, 1.0) * 0.22
            market_weight *= 1.0 - _market_independence_hint(match, home_mu, away_mu) * 0.55
            home_mu = home_mu * (1.0 - market_weight) + match.market.expected_home_goals * market_weight
            away_mu = away_mu * (1.0 - market_weight) + match.market.expected_away_goals * market_weight

        return _clamp(home_mu, 0.12, 4.8), _clamp(away_mu, 0.12, 4.8)

    def _real_distribution(
        self,
        match: MatchInput,
        home_mu: float,
        away_mu: float,
    ) -> dict[tuple[int, int], float]:
        probabilities: dict[tuple[int, int], float] = {}
        volatility = _clamp(match.league_volatility, 0.0, 1.0)

        for home, away in self.score_grid:
            total = home + away
            probability = _poisson(home, home_mu) * _poisson(away, away_mu)
            if home == away:
                probability *= 1.0 + _clamp(match.league_draw_bias, -1.0, 1.0) * 0.18
            if total >= 4:
                probability *= 1.0 + (volatility - 0.5) * 0.38
            if total <= 1:
                probability *= 1.0 - (volatility - 0.5) * 0.16
            probability *= self._script_distribution_multiplier(match, home, away)
            probabilities[(home, away)] = max(probability, 0.0)

        return _normalize(probabilities)

    def _script_distribution_multiplier(self, match: MatchInput, home: int, away: int) -> float:
        """Adjust the raw score distribution for table-state scripts."""

        value = 1.0
        total = home + away
        outcome = _outcome(home, away)
        home_pressure = _stake_pressure(match, "home")
        away_pressure = _stake_pressure(match, "away")
        home_survival = _survival_pressure(match, "home")
        away_survival = _survival_pressure(match, "away")
        chaos = _endgame_chaos(match)

        if match.home_draw_sufficient and match.away_draw_sufficient:
            if outcome == "draw":
                value *= 1.18
                if total <= 2:
                    value *= 1.10
            elif total >= 4:
                value *= 0.90
        elif match.home_draw_sufficient and outcome != "away" and total <= 2:
            value *= 1.05
        elif match.away_draw_sufficient and outcome != "home" and total <= 2:
            value *= 1.05

        if home_pressure >= 0.55:
            if outcome == "home":
                value *= 1.0 + min(home_pressure, 1.0) * 0.10
            if total >= 4:
                value *= 1.0 + _clamp(match.home_big_win_need, 0.0, 1.0) * 0.08
            if outcome == "away" and match.home_big_win_need >= 0.35:
                value *= 1.06

        if away_pressure >= 0.55:
            if outcome == "away":
                value *= 1.0 + min(away_pressure, 1.0) * 0.10
            if total >= 4:
                value *= 1.0 + _clamp(match.away_big_win_need, 0.0, 1.0) * 0.08
            if outcome == "home" and match.away_big_win_need >= 0.35:
                value *= 1.06

        if home_survival >= 0.45:
            if outcome == "home":
                value *= 1.0 + home_survival * 0.16
                if home - away >= 2:
                    value *= 1.0 + home_survival * 0.13
            if total >= 4 and home > 0:
                value *= 1.0 + home_survival * 0.08
        if away_survival >= 0.45:
            if outcome == "away":
                value *= 1.0 + away_survival * 0.16
                if away - home >= 2:
                    value *= 1.0 + away_survival * 0.13
            if total >= 4 and away > 0:
                value *= 1.0 + away_survival * 0.08

        if match.home_settled and away_pressure >= 0.35:
            if outcome == "away":
                value *= 1.10
            if outcome == "home" and abs(home - away) >= 2:
                value *= 0.88
        if match.away_settled and home_pressure >= 0.35:
            if outcome == "home":
                value *= 1.10
            if outcome == "away" and abs(home - away) >= 2:
                value *= 0.88

        if match.is_final_round:
            if total >= 5:
                value *= 1.0 + chaos * 0.24
            if outcome == "draw" and total >= 4:
                value *= 1.0 + chaos * 0.18
            if total == 0 and (match.home_draw_sufficient or match.away_draw_sufficient or match.home_settled or match.away_settled):
                value *= 1.0 + chaos * 0.14

            if outcome == "home":
                value *= 1.0 + _clamp(match.away_collapse_risk, 0.0, 1.0) * min(home, 5) * 0.045
                if home - away >= 3:
                    value *= 1.0 + _clamp(match.away_collapse_risk, 0.0, 1.0) * 0.18
            if outcome == "away":
                value *= 1.0 + _clamp(match.home_collapse_risk, 0.0, 1.0) * min(away, 5) * 0.045
                if away - home >= 3:
                    value *= 1.0 + _clamp(match.home_collapse_risk, 0.0, 1.0) * 0.18
            if total >= 4 and home > 0 and away > 0:
                value *= 1.0 + max(match.home_celebration_risk, match.away_celebration_risk) * 0.10

        return _clamp(value, 0.72, 1.62)

    def _market_distribution(
        self,
        match: MatchInput,
        real_distribution: dict[tuple[int, int], float],
    ) -> dict[tuple[int, int], float]:
        market = match.market
        exact = _exact_score_market_distribution(market.exact_score_odds, self.score_grid)

        if market.expected_home_goals is not None and market.expected_away_goals is not None:
            market_mu = self._poisson_grid(market.expected_home_goals, market.expected_away_goals)
        else:
            market_mu = real_distribution

        if exact:
            coverage = len(exact) / max(len(self.score_grid), 1)
            exact_weight = 0.35 + coverage * 0.35
            blended = {
                score: exact.get(score, market_mu.get(score, 0.0)) * exact_weight
                + market_mu.get(score, 0.0) * (1.0 - exact_weight)
                for score in self.score_grid
            }
            return _normalize(blended)

        return market_mu

    def _poisson_grid(self, home_mu: float, away_mu: float) -> dict[tuple[int, int], float]:
        probabilities = {
            (home, away): _poisson(home, home_mu) * _poisson(away, away_mu)
            for home, away in self.score_grid
        }
        return _normalize(probabilities)

    def _candidate(
        self,
        match: MatchInput,
        home: int,
        away: int,
        home_mu: float,
        away_mu: float,
        real_distribution: dict[tuple[int, int], float],
        market_distribution: dict[tuple[int, int], float],
    ) -> ScoreCandidate:
        score = f"{home}-{away}"
        real_probability = real_distribution[(home, away)]
        market_probability = max(market_distribution.get((home, away), 0.0), 0.0001)
        edge_ratio = real_probability / market_probability
        edge_gap = real_probability - market_probability
        structure_score, structure_reasons = self._structure_score(match, home, away, home_mu, away_mu)
        cold_bonus, cold_reasons = self._cold_bonus(match, home, away, real_probability, edge_ratio, structure_score)
        conflict_penalty, conflict_reasons = self._conflict_penalty(match, home, away, edge_ratio)
        comfort_penalty, comfort_reasons = self._comfort_penalty(home, away, real_probability, edge_ratio, structure_score)
        tail_penalty, tail_reasons = self._tail_penalty(match, home, away, home_mu, away_mu, edge_ratio)
        public_penalty, public_reasons = self._public_penalty(match, home, away, market_probability, edge_ratio)
        scorebook_bonus, scorebook_reasons = self._scorebook_bonus(match, home, away, home_mu, away_mu, edge_ratio)
        market_independence_multiplier, market_independence_reasons = self._market_independence_multiplier(
            match,
            home,
            away,
            home_mu,
            away_mu,
        )
        scenario_multiplier, scenario_reasons = self._scenario_multiplier(
            match,
            home,
            away,
            home_mu,
            away_mu,
            edge_ratio,
        )

        market_confidence = _clamp(match.market.confidence, 0.0, 1.0)
        market_confidence *= 1.0 - _market_independence_hint(match, home_mu, away_mu) * 0.45
        if self.mode == "contrarian":
            market_confidence *= 0.55
        market_factor = 1.0 + market_confidence * _clamp((edge_ratio - 1.0) * 0.26 + edge_gap * 5.5, -0.42, 0.78)
        final_value = (
            real_probability
            * market_factor
            * structure_score
            * cold_bonus
            * scorebook_bonus
            * market_independence_multiplier
            * scenario_multiplier
        )
        final_value *= (1.0 - conflict_penalty) * (1.0 - comfort_penalty) * (1.0 - tail_penalty) * (1.0 - public_penalty)

        reasons = tuple(
            structure_reasons
            + cold_reasons
            + scorebook_reasons
            + market_independence_reasons
            + scenario_reasons
            + conflict_reasons
            + comfort_reasons
            + tail_reasons
            + public_reasons
        )
        return ScoreCandidate(
            score=score,
            home_goals=home,
            away_goals=away,
            total_goals=home + away,
            real_probability=real_probability,
            market_probability=market_probability,
            edge_ratio=edge_ratio,
            edge_gap=edge_gap,
            structure_score=structure_score,
            cold_bonus=cold_bonus,
            conflict_penalty=conflict_penalty,
            final_value=final_value,
            reasons=reasons,
        )

    def _market_independence_multiplier(
        self,
        match: MatchInput,
        home: int,
        away: int,
        home_mu: float,
        away_mu: float,
    ) -> tuple[float, list[str]]:
        one_x_two = _implied_probabilities(match.market.one_x_two_odds)
        if not one_x_two:
            return 1.0, []

        market_favorite = max(one_x_two, key=one_x_two.get)
        if market_favorite == "draw":
            return 1.0, []

        football_favorite = _football_favorite_side(match, home_mu, away_mu)
        independence = _market_independence_hint(match, home_mu, away_mu)
        if football_favorite == "draw" or football_favorite == market_favorite or independence < 0.34:
            return 1.0, []

        outcome = _outcome(home, away)
        multiplier = 1.0
        reasons: list[str] = []
        if outcome == football_favorite:
            multiplier += independence * 0.18
            reasons.append("market independence: football-side result outranks public line")
        elif outcome == "draw" and abs(home_mu - away_mu) <= 0.34:
            multiplier += independence * 0.07
            reasons.append("market independence: draw stays live against public line")
        elif outcome == market_favorite:
            multiplier -= independence * 0.12
            reasons.append("market independence: public-line result discounted")

        if reasons:
            reasons.append(f"market independence score={independence:.2f}")
        return _clamp(multiplier, 0.86, 1.22), reasons

    def _structure_score(
        self,
        match: MatchInput,
        home: int,
        away: int,
        home_mu: float,
        away_mu: float,
    ) -> tuple[float, list[str]]:
        reasons: list[str] = []
        total = home + away
        expected_total = home_mu + away_mu
        expected_margin = home_mu - away_mu
        actual_margin = home - away
        outcome = _outcome(home, away)
        home_pressure = _stake_pressure(match, "home")
        away_pressure = _stake_pressure(match, "away")
        home_survival = _survival_pressure(match, "home")
        away_survival = _survival_pressure(match, "away")
        chaos = _endgame_chaos(match)

        total_distance = abs(total - expected_total)
        score = 1.12 - min(total_distance, 3.2) * 0.13
        if total >= 4 and match.league_volatility >= 0.58:
            score += 0.08
            reasons.append("high-volatility league supports 4+ goal layer")
        if match.is_final_round and total >= 5 and chaos >= 0.45:
            score += 0.07
            reasons.append("final-round chaos keeps 5+ goal route live")
        if total <= 1 and match.weather_goal_drag >= 0.25:
            score += 0.06
            reasons.append("weather drag supports low-total layer")
        if total == 0:
            under_lock = (
                _clamp(match.market.under_money_heat, 0.0, 1.0) * 0.28
                + _clamp(match.weather_goal_drag, 0.0, 1.0) * 0.22
                + _clamp(2.30 - expected_total, 0.0, 0.8) * 0.22
                + (0.16 if match.home_draw_sufficient and match.away_draw_sufficient else 0.0)
            )
            if under_lock >= 0.20:
                score += 0.07
                reasons.append("under/draw script keeps 0-0 route live")

        if actual_margin == 0 and abs(expected_margin) <= 0.28:
            score += 0.07
            reasons.append("draw structure matches balanced expected goals")
        if actual_margin == 0 and match.home_draw_sufficient and match.away_draw_sufficient:
            score += 0.10
            reasons.append("table script allows both sides to accept draw")
        if actual_margin == 0 and match.is_final_round and total >= 4 and chaos >= 0.45:
            score += 0.08
            reasons.append("final-round split incentives allow high-score draw")
        elif actual_margin > 0 and expected_margin > 0.18:
            score += 0.06
            reasons.append("home-positive score matches expected goal edge")
        elif actual_margin < 0 and expected_margin < -0.18:
            score += 0.06
            reasons.append("away-positive score matches expected goal edge")
        elif abs(actual_margin) >= 2 and abs(expected_margin) < 0.2:
            score -= 0.11
            reasons.append("margin is aggressive for a balanced match")

        absence_swing = match.home_absence_impact - match.away_absence_impact
        if away > home and absence_swing >= 0.25:
            score += 0.08
            reasons.append("home absences lift away-score structure")
        if home > away and absence_swing <= -0.25:
            score += 0.08
            reasons.append("away absences lift home-score structure")

        if min(home, away) == 0 and total >= 2:
            shutout_side = "home" if home > away else "away"
            opponent_mu = away_mu if shutout_side == "home" else home_mu
            favorite = _favorite_side(match, home_mu, away_mu)
            if opponent_mu <= 0.95:
                score += 0.05
                reasons.append("opponent scoring profile allows clean-sheet route")
            if shutout_side == favorite and abs(actual_margin) >= 2:
                score += 0.04
                reasons.append("favorite clean-sheet margin is a live branch")
            if total >= 3 and _stake_pressure(match, shutout_side) >= 0.45:
                score += 0.05
                reasons.append("stake pressure supports one-sided score branch")

        if match.is_final_round and abs(actual_margin) >= 2:
            collapsing_side = "away" if actual_margin > 0 else "home"
            collapse_risk = match.away_collapse_risk if collapsing_side == "away" else match.home_collapse_risk
            if collapse_risk >= 0.35:
                score += 0.10 if abs(actual_margin) >= 3 else 0.06
                reasons.append(f"final-round {collapsing_side} collapse risk supports blowout branch")
            if max(match.home_celebration_risk, match.away_celebration_risk) >= 0.45 and total >= 5:
                score += 0.05
                reasons.append("celebration script can stretch winning margin past comfort score")

        if outcome == "home" and home_pressure >= 0.45:
            score += 0.05
            reasons.append("home table pressure supports home-positive route")
        if outcome == "away" and away_pressure >= 0.45:
            score += 0.05
            reasons.append("away table pressure supports away-positive route")
        if outcome == "home" and home_survival >= 0.45:
            score += 0.07
            reasons.append("home survival pressure supports proactive attack route")
            if home - away >= 2:
                score += 0.07
                reasons.append("home survival pressure allows multi-goal break")
            if total >= 5:
                score += 0.04
                reasons.append("home survival game state keeps extreme score live")
        if outcome == "away" and away_survival >= 0.45:
            score += 0.07
            reasons.append("away survival pressure supports proactive attack route")
            if away - home >= 2:
                score += 0.07
                reasons.append("away survival pressure allows multi-goal break")
            if total >= 5:
                score += 0.04
                reasons.append("away survival game state keeps extreme score live")
        if outcome == "home" and match.home_big_win_need >= 0.35 and home - away >= 2:
            score += 0.06
            reasons.append("home needs goal difference, wide home route is live")
        if outcome == "away" and match.away_big_win_need >= 0.35 and away - home >= 2:
            score += 0.06
            reasons.append("away needs goal difference, wide away route is live")
        if match.home_settled and outcome == "home" and home - away >= 2:
            score -= 0.06
            reasons.append("settled home side lowers wide-cover structure")
        if match.away_settled and outcome == "away" and away - home >= 2:
            score -= 0.06
            reasons.append("settled away side lowers wide-cover structure")

        handicap = self._handicap_profile(match, home_mu, away_mu)
        if handicap and handicap["depth"] >= 1.0:
            favorite = str(handicap["favorite"])
            favorite_margin = actual_margin if favorite == "home" else -actual_margin
            if outcome == favorite and favorite_margin > float(handicap["depth"]):
                score += 0.06 * float(handicap["true_deep"])
                score -= 0.08 * float(handicap["trap"])
                reasons.append("deep handicap cover route evaluated")
            elif outcome == favorite:
                score += 0.07 * float(handicap["trap"])
                reasons.append("deep handicap win-not-cover route evaluated")
            elif outcome in {"draw", "home", "away"}:
                score += 0.05 * float(handicap["trap"])
                reasons.append("deep handicap upset/draw route evaluated")

        return _clamp(score, 0.66, 1.36), reasons

    def _scorebook_bonus(
        self,
        match: MatchInput,
        home: int,
        away: int,
        home_mu: float,
        away_mu: float,
        edge_ratio: float,
    ) -> tuple[float, list[str]]:
        if self.score_profile != "scorebook":
            return 1.0, []

        family = _score_family(home, away)
        expected_margin = home_mu - away_mu
        total = home + away
        bonus = 1.0
        reasons: list[str] = [f"scorebook family: {family}"]
        chaos = _endgame_chaos(match)
        winner = "home" if home > away else "away" if away > home else "draw"
        winner_survival = _survival_pressure(match, winner) if winner != "draw" else 0.0

        if family == "draw_ladder":
            if home >= 2 and match.league_volatility >= 0.50:
                bonus += 0.08
            if home >= 3 and match.league_volatility >= 0.62:
                bonus += 0.08
        elif family == "narrow_result":
            if abs(expected_margin) <= 0.45:
                bonus += 0.04
        elif family == "one_side_break":
            if abs(home - away) >= 2 and abs(expected_margin) >= 0.42:
                bonus += 0.08
            if total >= 4 and match.league_volatility >= 0.55:
                bonus += 0.07
            if match.is_final_round and abs(home - away) >= 3:
                collapse_risk = match.away_collapse_risk if home > away else match.home_collapse_risk
                bonus += _clamp(collapse_risk, 0.0, 1.0) * 0.10
            if abs(home - away) >= 2 and winner_survival >= 0.45:
                bonus += winner_survival * 0.10
                reasons.append("survival scorebook break")
        elif family == "clean_sheet_break":
            if abs(expected_margin) >= 0.55:
                bonus += 0.08
            if edge_ratio >= 1.05:
                bonus += 0.04
            if match.is_final_round and abs(home - away) >= 3:
                collapse_risk = match.away_collapse_risk if home > away else match.home_collapse_risk
                bonus += _clamp(collapse_risk, 0.0, 1.0) * 0.08
            if abs(home - away) >= 2 and winner_survival >= 0.45:
                bonus += winner_survival * 0.08
                reasons.append("survival clean-sheet break")
        elif family == "exchange_war":
            if match.league_volatility >= 0.54:
                bonus += 0.10
            if total >= 6:
                bonus += 0.05
            if max(_survival_pressure(match, "home"), _survival_pressure(match, "away")) >= 0.45:
                bonus += max(_survival_pressure(match, "home"), _survival_pressure(match, "away")) * 0.06

        if match.is_final_round and total >= 5:
            bonus += chaos * 0.08
            reasons.append("final-round scorebook stretch")
        if match.is_final_round and home == away and total >= 4:
            bonus += chaos * 0.06
            reasons.append("final-round high-draw branch")

        if self.mode == "contrarian" and family in {"draw_ladder", "exchange_war"} and total >= 4:
            bonus += 0.06
        if self.mode == "contrarian" and family in {"one_side_break", "clean_sheet_break"} and edge_ratio >= 1.0:
            bonus += 0.05

        return _clamp(bonus, 0.88, 1.32), reasons

    def _scenario_multiplier(
        self,
        match: MatchInput,
        home: int,
        away: int,
        home_mu: float,
        away_mu: float,
        edge_ratio: float,
    ) -> tuple[float, list[str]]:
        if self.mode != "adaptive":
            return 1.0, []

        profile = self._scenario_profile(match, home_mu, away_mu)
        favorite = str(profile["favorite"])
        outcome = _outcome(home, away)
        total = home + away
        value = 1.0

        if outcome == favorite:
            route = "normal"
            value += 0.16 * profile["normal"] - 0.12 * profile["fake_hot"]
        elif outcome == "draw":
            route = "draw"
            value += 0.18 * profile["draw"] - 0.10 * profile["normal"]
        else:
            route = "upset"
            value += 0.20 * profile["upset"] - 0.12 * profile["normal"]

        if total >= 4 and home > 0 and away > 0:
            value += 0.15 * profile["exchange"]
            route = f"{route}+exchange"

        if outcome == "home" and _survival_pressure(match, "home") >= 0.45:
            value += 0.11 * _survival_pressure(match, "home")
            route = f"{route}+home-survival"
        if outcome == "away" and _survival_pressure(match, "away") >= 0.45:
            value += 0.11 * _survival_pressure(match, "away")
            route = f"{route}+away-survival"
        if total >= 5 and max(_survival_pressure(match, "home"), _survival_pressure(match, "away")) >= 0.45:
            value += 0.07 * max(_survival_pressure(match, "home"), _survival_pressure(match, "away"))
            route = f"{route}+survival-chaos"

        if match.is_final_round:
            chaos = _endgame_chaos(match)
            if total >= 5:
                value += 0.11 * chaos
                route = f"{route}+endgame-stretch"
            if outcome == "draw" and total >= 4:
                value += 0.10 * chaos
                route = f"{route}+endgame-high-draw"
            if outcome == "home" and home - away >= 3:
                value += 0.12 * _clamp(match.away_collapse_risk, 0.0, 1.0)
                route = f"{route}+away-collapse"
            if outcome == "away" and away - home >= 3:
                value += 0.12 * _clamp(match.home_collapse_risk, 0.0, 1.0)
                route = f"{route}+home-collapse"

        if profile["normal"] >= 0.72 and outcome != favorite and edge_ratio < 1.22:
            value -= 0.12
        if profile["fake_hot"] >= 0.58 and outcome == favorite and _is_comfort_score(home, away):
            value -= 0.10
        if profile["draw"] >= 0.62 and outcome == "draw" and total <= 2:
            value += 0.05
        if profile["upset"] >= 0.58 and outcome not in {favorite, "draw"} and edge_ratio >= 1.05:
            value += 0.06

        handicap = self._handicap_profile(match, home_mu, away_mu)
        if handicap and handicap["depth"] >= 1.0:
            handicap_favorite = str(handicap["favorite"])
            favorite_margin = (home - away) if handicap_favorite == "home" else (away - home)
            covers_deep = outcome == handicap_favorite and favorite_margin > float(handicap["depth"])
            wins_not_cover = outcome == handicap_favorite and not covers_deep
            if covers_deep:
                value += 0.11 * float(handicap["true_deep"]) - 0.12 * float(handicap["trap"])
                route = f"{route}+deep-cover"
            elif wins_not_cover:
                value += 0.12 * float(handicap["trap"]) - 0.05 * float(handicap["true_deep"])
                route = f"{route}+deep-noncover"
            elif outcome == "draw":
                value += 0.10 * float(handicap["trap"])
                route = f"{route}+deep-draw"
            else:
                value += 0.08 * float(handicap["trap"])
                route = f"{route}+deep-upset"

        if match.market.total_goals_line is not None:
            total_line = match.market.total_goals_line
            over_heat = _clamp(match.market.over_money_heat, 0.0, 1.0)
            under_heat = _clamp(match.market.under_money_heat, 0.0, 1.0)
            if total >= math.ceil(total_line + 0.25):
                value += 0.04 * over_heat
                value -= 0.05 * under_heat
            if total <= math.floor(total_line - 0.25):
                value += 0.04 * under_heat
                value -= 0.05 * over_heat

        reason = (
            f"scenario gate: {route} "
            f"normal={profile['normal']:.2f} draw={profile['draw']:.2f} "
            f"upset={profile['upset']:.2f} fake_hot={profile['fake_hot']:.2f}"
        )
        reasons = [reason]
        if float(profile.get("market_independence", 0.0)) >= 0.42:
            reasons.append(
                "market independence layer treats line as public-flow signal "
                f"market={profile['market_favorite']} football={profile['football_favorite']}"
            )
        return _clamp(value, 0.70, 1.46), reasons

    def _scenario_profile(self, match: MatchInput, home_mu: float, away_mu: float) -> dict[str, float | str]:
        one_x_two = _implied_probabilities(match.market.one_x_two_odds)
        mu_margin = home_mu - away_mu
        football_favorite = _football_favorite_side(match, home_mu, away_mu)
        market_favorite = "draw"
        market_independence = _market_independence_hint(match, home_mu, away_mu)
        if one_x_two:
            market_favorite = max(one_x_two, key=one_x_two.get)
            favorite = market_favorite
            if (
                football_favorite != "draw"
                and market_favorite != "draw"
                and football_favorite != market_favorite
                and market_independence >= 0.42
            ):
                favorite = football_favorite
            favorite_prob = one_x_two[market_favorite]
            draw_prob = one_x_two.get("draw", 0.30)
        elif mu_margin >= 0.22:
            favorite = "home"
            favorite_prob = 0.43 + min(mu_margin, 1.4) * 0.16
            draw_prob = 0.30
        elif mu_margin <= -0.22:
            favorite = "away"
            favorite_prob = 0.43 + min(abs(mu_margin), 1.4) * 0.16
            draw_prob = 0.30
        else:
            favorite = "draw"
            favorite_prob = 0.34
            draw_prob = 0.34

        support = self._favorite_support(match, favorite, mu_margin)
        underdog_push = self._underdog_push(match, favorite, mu_margin)
        heat = _clamp((favorite_prob - 0.42) / 0.30, 0.0, 1.0)
        money_heat = _side_money_heat(match, market_favorite)
        heat = _clamp(heat * 0.72 + money_heat * 0.20 + _clamp(match.market.favorite_money_heat, 0.0, 1.0) * 0.08, 0.0, 1.0)
        market_side_is_result_side = market_favorite == favorite
        fake_hot = _clamp(
            heat * max(0.0, 0.62 - support) * 1.55
            + underdog_push * 0.34
            + _clamp(match.market.bookmaker_trap_risk, 0.0, 1.0) * 0.24,
            0.0,
            1.0,
        )
        normal_heat = heat * 0.20 if market_side_is_result_side else 0.0
        fake_hot = _clamp(fake_hot + (market_independence * 0.18 if not market_side_is_result_side else 0.0), 0.0, 1.0)
        normal = _clamp(0.18 + support * 0.62 + normal_heat - fake_hot * 0.36, 0.0, 1.0)

        balance = _clamp((0.54 - abs(mu_margin)) * 1.18, 0.0, 1.0)
        draw_script = 0.0
        if match.home_draw_sufficient and match.away_draw_sufficient:
            draw_script = 0.30
        elif match.home_draw_sufficient or match.away_draw_sufficient:
            draw_script = 0.12
        draw = _clamp(
            0.14
            + balance * 0.48
            + draw_prob * 0.36
            + draw_script
            + _clamp(match.market.draw_money_heat, 0.0, 1.0) * 0.08
            + _clamp(match.league_draw_bias, -0.45, 0.45) * 0.24
            + fake_hot * 0.18
            - _clamp(match.league_volatility - 0.55, 0.0, 0.45) * 0.16
            - normal * 0.14,
            0.0,
            1.0,
        )
        upset = _clamp(
            0.10
            + fake_hot * 0.46
            + underdog_push * 0.36
            + heat * 0.08
            + _underdog_money_heat(match, favorite) * 0.10
            - normal * 0.18,
            0.0,
            1.0,
        )
        exchange = _clamp(
            0.12
            + _clamp(match.league_volatility - 0.48, 0.0, 0.52) * 0.86
            + _clamp(home_mu + away_mu - 2.55, 0.0, 1.8) * 0.20
            + _clamp(match.referee_goal_bias, 0.0, 1.0) * 0.08,
            0.0,
            1.0,
        )
        if match.is_final_round:
            chaos = _endgame_chaos(match)
            exchange = _clamp(exchange + chaos * 0.16, 0.0, 1.0)
            upset = _clamp(
                upset
                + chaos * 0.06
                + max(match.home_collapse_risk, match.away_collapse_risk) * 0.10,
                0.0,
                1.0,
            )
            survival_push = max(_survival_pressure(match, "home"), _survival_pressure(match, "away"))
            exchange = _clamp(exchange + survival_push * 0.08, 0.0, 1.0)
            upset = _clamp(upset + survival_push * 0.10, 0.0, 1.0)
            if match.home_draw_sufficient or match.away_draw_sufficient or match.home_settled or match.away_settled:
                draw = _clamp(draw + chaos * 0.08, 0.0, 1.0)

        return {
            "favorite": favorite,
            "market_favorite": market_favorite,
            "football_favorite": football_favorite,
            "normal": normal,
            "draw": draw,
            "upset": upset,
            "exchange": exchange,
            "fake_hot": fake_hot,
            "market_independence": market_independence,
        }

    def _handicap_profile(
        self,
        match: MatchInput,
        home_mu: float,
        away_mu: float,
    ) -> dict[str, float | str] | None:
        if match.market.asian_handicap is None:
            return None

        line = match.market.asian_handicap
        if abs(line) < 0.75:
            return None

        favorite = "home" if line < 0 else "away"
        margin = home_mu - away_mu if favorite == "home" else away_mu - home_mu
        support = self._favorite_support(match, favorite, home_mu - away_mu)
        favorite_pressure = _stake_pressure(match, favorite)
        underdog = "away" if favorite == "home" else "home"
        underdog_pressure = _stake_pressure(match, underdog)
        favorite_settled = match.home_settled if favorite == "home" else match.away_settled
        depth = abs(line)
        heat = _clamp(
            _side_money_heat(match, favorite) * 0.45
            + _clamp(match.market.favorite_money_heat, 0.0, 1.0) * 0.35
            + _clamp(match.market.bookmaker_trap_risk, 0.0, 1.0) * 0.20,
            0.0,
            1.0,
        )

        true_deep = _clamp(
            (support - 0.48) * 1.30
            + _clamp(margin - depth * 0.42, -0.6, 1.2) * 0.28
            + favorite_pressure * 0.18
            - underdog_pressure * 0.10
            - (0.16 if favorite_settled else 0.0),
            0.0,
            1.0,
        )
        trap = _clamp(
            heat * max(0.0, 0.66 - support) * 1.20
            + underdog_pressure * 0.20
            + (0.18 if favorite_settled else 0.0)
            + _clamp(match.market.bookmaker_trap_risk, 0.0, 1.0) * 0.28,
            0.0,
            1.0,
        )
        return {
            "favorite": favorite,
            "depth": depth,
            "true_deep": true_deep,
            "trap": trap,
        }

    def _favorite_support(self, match: MatchInput, favorite: str, mu_margin: float) -> float:
        if favorite == "home":
            raw = (
                mu_margin * 0.54
                + (match.home_form - match.away_form) * 0.16
                + (match.home_motivation - match.away_motivation) * 0.10
                + (_stake_pressure(match, "home") - _stake_pressure(match, "away")) * 0.13
                + (_survival_pressure(match, "home") - _survival_pressure(match, "away")) * 0.08
                + _clamp(match.home_big_win_need, 0.0, 1.0) * 0.05
                - match.home_absence_impact * 0.14
                + match.away_absence_impact * 0.12
                - match.home_rotation_risk * 0.08
                + match.away_rotation_risk * 0.06
                + (match.home_rest_edge - match.away_rest_edge) * 0.06
                - (0.08 if match.home_settled else 0.0)
            )
        elif favorite == "away":
            raw = (
                -mu_margin * 0.54
                + (match.away_form - match.home_form) * 0.16
                + (match.away_motivation - match.home_motivation) * 0.10
                + (_stake_pressure(match, "away") - _stake_pressure(match, "home")) * 0.13
                + (_survival_pressure(match, "away") - _survival_pressure(match, "home")) * 0.08
                + _clamp(match.away_big_win_need, 0.0, 1.0) * 0.05
                - match.away_absence_impact * 0.14
                + match.home_absence_impact * 0.12
                - match.away_rotation_risk * 0.08
                + match.home_rotation_risk * 0.06
                + (match.away_rest_edge - match.home_rest_edge) * 0.06
                - (0.08 if match.away_settled else 0.0)
            )
        else:
            raw = -abs(mu_margin) * 0.30 + _clamp(match.league_draw_bias, -0.4, 0.4) * 0.12
        return _clamp(0.50 + raw, 0.0, 1.0)

    def _underdog_push(self, match: MatchInput, favorite: str, mu_margin: float) -> float:
        if favorite == "home":
            raw = (
                (match.away_form - match.home_form) * 0.18
                + (match.away_motivation - match.home_motivation) * 0.10
                + (_stake_pressure(match, "away") - _stake_pressure(match, "home")) * 0.16
                + (_survival_pressure(match, "away") - _survival_pressure(match, "home")) * 0.12
                + _clamp(match.away_big_win_need, 0.0, 1.0) * 0.06
                + match.home_absence_impact * 0.18
                - match.away_absence_impact * 0.12
                + match.home_rotation_risk * 0.08
                - match.away_rotation_risk * 0.05
                + _clamp(0.30 - mu_margin, 0.0, 0.9) * 0.22
                + (0.08 if match.home_settled else 0.0)
            )
        elif favorite == "away":
            raw = (
                (match.home_form - match.away_form) * 0.18
                + (match.home_motivation - match.away_motivation) * 0.10
                + (_stake_pressure(match, "home") - _stake_pressure(match, "away")) * 0.16
                + (_survival_pressure(match, "home") - _survival_pressure(match, "away")) * 0.12
                + _clamp(match.home_big_win_need, 0.0, 1.0) * 0.06
                + match.away_absence_impact * 0.18
                - match.home_absence_impact * 0.12
                + match.away_rotation_risk * 0.08
                - match.home_rotation_risk * 0.05
                + _clamp(0.30 + mu_margin, 0.0, 0.9) * 0.22
                + (0.08 if match.away_settled else 0.0)
            )
        else:
            raw = abs(mu_margin) * 0.18 + _clamp(match.league_volatility - 0.55, 0.0, 0.45) * 0.18
        return _clamp(0.16 + raw, 0.0, 1.0)

    def _cold_bonus(
        self,
        match: MatchInput,
        home: int,
        away: int,
        real_probability: float,
        edge_ratio: float,
        structure_score: float,
    ) -> tuple[float, list[str]]:
        comfortable_scores = {(0, 0), (1, 0), (0, 1), (1, 1), (2, 0), (2, 1), (1, 2)}
        total = home + away
        reasons: list[str] = []
        bonus = 1.0

        if (home, away) not in comfortable_scores and real_probability >= 0.018:
            bonus += 0.05
            reasons.append("non-comfort score remains above probability floor")
        if total >= 4 and edge_ratio >= 1.08:
            bonus += 0.06
            reasons.append("4+ goal score has market edge")
        if abs(home - away) >= 3 and edge_ratio >= 1.12:
            bonus += 0.04
            reasons.append("wide-margin score has enough edge support")
        if self._uses_attack_bias() and (home, away) not in comfortable_scores:
            if real_probability >= 0.035 and structure_score >= 0.94:
                bonus += 0.08
                reasons.append(f"{self.mode} mode lifts structured non-comfort score")
            if total in {4, 5} and match.league_volatility >= 0.54:
                bonus += 0.05
                reasons.append(f"{self.mode} mode keeps 4-5 goal layer alive")
        if self.mode == "contrarian" and (home, away) not in comfortable_scores:
            if real_probability >= 0.028 and structure_score >= 0.90:
                bonus += 0.10
                reasons.append("contrarian mode lifts anti-public score")
            if total >= 4 and match.league_volatility >= 0.50:
                bonus += 0.06
                reasons.append("contrarian mode keeps high-total alternative live")

        return _clamp(bonus, 0.94, 1.42), reasons

    def _comfort_penalty(
        self,
        home: int,
        away: int,
        real_probability: float,
        edge_ratio: float,
        structure_score: float,
    ) -> tuple[float, list[str]]:
        if not self._uses_attack_bias():
            return 0.0, []

        heavy_comfort = {(1, 1), (1, 0), (0, 1), (2, 1)}
        soft_comfort = {(0, 0), (2, 0), (1, 2), (2, 2)}
        score = (home, away)
        if score not in heavy_comfort and score not in soft_comfort:
            return 0.0, []

        if real_probability >= 0.145 and edge_ratio >= 1.15 and structure_score >= 1.04:
            return 0.0, []

        penalty = 0.0
        if score in heavy_comfort:
            penalty = 0.16
        elif score in soft_comfort:
            penalty = 0.09
        if self.mode == "contrarian":
            penalty += 0.08

        if edge_ratio < 0.95:
            penalty += 0.05
        if structure_score < 1.0:
            penalty += 0.03

        return _clamp(penalty, 0.0, 0.34), [f"{self.mode} mode penalizes crowded comfort score"]

    def _tail_penalty(
        self,
        match: MatchInput,
        home: int,
        away: int,
        home_mu: float,
        away_mu: float,
        edge_ratio: float,
    ) -> tuple[float, list[str]]:
        if not self._uses_attack_bias():
            return 0.0, []

        expected_total = home_mu + away_mu
        total = home + away
        margin_gap = abs((home - away) - (home_mu - away_mu))
        total_gap = abs(total - expected_total)
        penalty = 0.0

        if total_gap >= 1.75 and edge_ratio < 1.35:
            penalty += 0.10
        if total_gap >= 2.35 and edge_ratio < 1.65:
            penalty += 0.10
        if abs(home - away) >= 3 and margin_gap >= 1.75 and edge_ratio < 1.75:
            penalty += 0.08
        if abs(home - away) >= 3 and edge_ratio < 1.35:
            penalty += 0.08

        if penalty == 0.0:
            return 0.0, []
        support = self._route_support(
            match,
            _light_candidate(home, away, edge_ratio),
            home_mu,
            away_mu,
            _score_exit_lane_from_values(home, away, match, home_mu, away_mu),
        )
        reasons = [f"{self.mode} mode trims unsupported tail score"]
        if support >= 0.48:
            penalty *= 0.58
            reasons.append("route support keeps non-standard tail score live")
        elif support >= 0.34:
            penalty *= 0.78
            reasons.append("partial route support softens tail trim")
        return _clamp(penalty, 0.0, 0.24), reasons

    def _public_penalty(
        self,
        match: MatchInput,
        home: int,
        away: int,
        market_probability: float,
        edge_ratio: float,
    ) -> tuple[float, list[str]]:
        if self.mode != "contrarian":
            return 0.0, []

        one_x_two = _implied_probabilities(match.market.one_x_two_odds)
        outcome = _outcome(home, away)
        penalty = 0.0

        if one_x_two:
            market_favorite = max(one_x_two, key=one_x_two.get)
            favorite_prob = one_x_two[market_favorite]
            if outcome == market_favorite and favorite_prob >= 0.46:
                penalty += 0.10
            if outcome == market_favorite and favorite_prob >= 0.58 and home + away <= 3:
                penalty += 0.08

        if market_probability >= 0.115 and edge_ratio <= 1.08:
            penalty += 0.08
        if _is_comfort_score(home, away) and edge_ratio <= 1.18:
            penalty += 0.07

        if penalty == 0.0:
            return 0.0, []
        return _clamp(penalty, 0.0, 0.28), ["contrarian mode fades public-market score"]

    def _conflict_penalty(self, match: MatchInput, home: int, away: int, edge_ratio: float) -> tuple[float, list[str]]:
        one_x_two = _implied_probabilities(match.market.one_x_two_odds)
        if not one_x_two:
            return 0.0, []

        outcome = "draw"
        if home > away:
            outcome = "home"
        elif away > home:
            outcome = "away"

        market_favorite = max(one_x_two, key=one_x_two.get)
        favorite_prob = one_x_two[market_favorite]
        outcome_prob = one_x_two.get(outcome, 0.0)
        market_gap = favorite_prob - outcome_prob
        if outcome != market_favorite and market_gap >= 0.26:
            if self.mode == "adaptive":
                if favorite_prob >= 0.62:
                    if outcome == "draw":
                        if edge_ratio >= 1.25:
                            return 0.08, ["adaptive mode allows draw against strong favorite with edge"]
                        return 0.14, ["adaptive mode trims thin draw against strong favorite"]
                    if edge_ratio >= 1.55:
                        return 0.12, ["adaptive mode allows upset only with large edge"]
                    if edge_ratio >= 1.20:
                        return 0.17, ["adaptive mode trims thin upset direction"]
                    return 0.23, ["adaptive mode blocks unsupported upset direction"]
                if edge_ratio >= 1.05:
                    return 0.06, ["adaptive mode allows market-direction conflict with edge"]
                return 0.12, ["adaptive mode trims market-direction conflict"]
            if self.mode == "contrarian":
                if favorite_prob >= 0.62:
                    if outcome == "draw":
                        if edge_ratio >= 1.45:
                            return 0.08, ["contrarian mode allows strong-favorite draw only with edge"]
                        return 0.16, ["contrarian mode anchors against unsupported strong-favorite draw"]
                    if edge_ratio >= 1.75:
                        return 0.12, ["contrarian mode allows major upset only with large exact-score edge"]
                    if edge_ratio >= 1.25:
                        return 0.20, ["contrarian mode trims thin major-upset direction"]
                    return 0.28, ["contrarian mode blocks unsupported major-upset direction"]
                if edge_ratio >= 0.92:
                    return 0.03, ["contrarian mode allows market-direction conflict"]
                return 0.08, ["contrarian mode discounts but allows market-direction conflict"]
            if edge_ratio < 1.16:
                return 0.18, ["outcome conflicts with strong 1X2 market without enough exact-score edge"]
            if edge_ratio < 1.55:
                return 0.11, ["outcome still conflicts with strong 1X2 market"]
        return 0.0, []

    def _select_layer_winners(self, candidates: list[ScoreCandidate]) -> list[ScoreCandidate]:
        layers: dict[int, list[ScoreCandidate]] = defaultdict(list)
        for candidate in candidates:
            layers[candidate.total_goals].append(candidate)
        return [
            max(layers[total], key=lambda item: item.final_value)
            for total in range(self.max_total_goals + 1)
            if layers.get(total)
        ]

    def _select_final(
        self,
        match: MatchInput,
        home_mu: float,
        away_mu: float,
        candidates: list[ScoreCandidate],
        layer_winners: list[ScoreCandidate],
    ) -> ScoreCandidate:
        selected = max(layer_winners, key=lambda item: item.final_value)
        if self.mode == "aggressive":
            selected = self._market_direction_challenger(match, selected, candidates)
        if self.mode == "contrarian":
            selected = self._public_fade_challenger(match, selected, candidates)
        if self._uses_attack_bias():
            selected = self._script_lane_challenger(match, home_mu, away_mu, selected, candidates)
            selected = self._forced_exit_challenger(match, home_mu, away_mu, selected, candidates)
            selected = self._result_script_challenger(match, home_mu, away_mu, selected, candidates)
            selected = self._shape_arbitration_challenger(match, home_mu, away_mu, selected, candidates)
            selected = self._strong_match_under_challenger(match, home_mu, away_mu, selected, candidates)
        if not self._uses_attack_bias() or not _is_comfort_score(selected.home_goals, selected.away_goals):
            return selected

        challengers = [
            candidate
            for candidate in candidates
            if not _is_comfort_score(candidate.home_goals, candidate.away_goals)
            and not (
                self.mode == "contrarian"
                and _is_broad_public_score(candidate.home_goals, candidate.away_goals)
                and _outcome(candidate.home_goals, candidate.away_goals) == _favorite_side(match, home_mu, away_mu)
            )
            and 2 <= candidate.total_goals <= 5
            and candidate.real_probability >= 0.03
            and candidate.structure_score >= 0.98
            and candidate.conflict_penalty <= 0.05
        ]
        if not challengers:
            return selected

        challenger = max(challengers, key=lambda item: item.final_value)
        threshold = 0.78
        if challenger.edge_ratio >= 1.0 or challenger.total_goals >= 4:
            threshold = 0.74
        if self.mode == "contrarian":
            threshold -= 0.08
        if challenger.final_value >= selected.final_value * threshold:
            selected = challenger
            if self._uses_attack_bias():
                selected = self._strong_match_under_challenger(match, home_mu, away_mu, selected, candidates)
            return selected
        if self._uses_attack_bias():
            selected = self._strong_match_under_challenger(match, home_mu, away_mu, selected, candidates)
        return selected

    def _script_lane_challenger(
        self,
        match: MatchInput,
        home_mu: float,
        away_mu: float,
        selected: ScoreCandidate,
        candidates: list[ScoreCandidate],
    ) -> ScoreCandidate:
        selected_value = self._route_adjusted_value(selected, match, home_mu, away_mu)
        selected_lane = self._score_exit_lane(selected, match, home_mu, away_mu)
        selected_support = self._route_support(match, selected, home_mu, away_mu, selected_lane)
        challengers: list[ScoreCandidate] = []

        for candidate in candidates:
            if candidate.score == selected.score:
                continue
            lane = self._score_exit_lane(candidate, match, home_mu, away_mu)
            support = self._route_support(match, candidate, home_mu, away_mu, lane)
            if (
                self.mode == "contrarian"
                and _is_broad_public_score(candidate.home_goals, candidate.away_goals)
                and _outcome(candidate.home_goals, candidate.away_goals) == _favorite_side(match, home_mu, away_mu)
                and support < 0.70
            ):
                continue
            if (
                selected_lane == "single_side_blowout"
                and selected_support >= 0.48
                and _is_broad_public_score(candidate.home_goals, candidate.away_goals)
                and lane != "single_side_blowout"
            ):
                continue
            if (
                selected_lane == "single_side_blowout"
                and selected_support >= 0.48
                and lane == "exchange_war"
                and abs(candidate.home_goals - candidate.away_goals) < 3
                and support < 0.62
            ):
                continue
            if (
                selected_lane == "upset_cold"
                and selected_support >= 0.55
                and lane == selected_lane
                and selected.total_goals >= 4
                and candidate.total_goals < selected.total_goals
                and candidate.final_value < selected.final_value
            ):
                continue
            if (
                self.mode == "contrarian"
                and lane == "upset_cold"
                and abs(candidate.home_goals - candidate.away_goals) >= 2
                and support < 0.62
                and candidate.edge_ratio < 1.45
            ):
                continue
            min_probability = 0.010 if support >= 0.55 else 0.020
            if candidate.real_probability < min_probability and candidate.final_value < selected.final_value * 0.18:
                continue
            if candidate.structure_score < 0.72 and candidate.edge_ratio < 1.06 and support < 0.42:
                continue
            if candidate.conflict_penalty > 0.20 and support < 0.58:
                continue
            if lane == selected_lane and not _is_broad_public_score(selected.home_goals, selected.away_goals):
                continue
            challengers.append(candidate)

        if not challengers:
            return selected

        challenger = max(challengers, key=lambda item: self._route_adjusted_value(item, match, home_mu, away_mu))
        challenger_value = self._route_adjusted_value(challenger, match, home_mu, away_mu)
        challenger_lane = self._score_exit_lane(challenger, match, home_mu, away_mu)
        support = self._route_support(match, challenger, home_mu, away_mu, challenger_lane)

        threshold = 0.82
        if _is_broad_public_score(selected.home_goals, selected.away_goals):
            threshold = 0.68
        if selected.score == "2-2":
            threshold = 0.62
        if challenger_lane in {"low_lock", "clean_sheet", "single_side_blowout"} and support >= 0.48:
            threshold -= 0.07
        if challenger_lane in {"upset_cold", "exchange_war"} and support >= 0.56:
            threshold -= 0.06
        if selected_lane == "single_side_blowout" and selected_support >= 0.48 and challenger_lane == "exchange_war":
            threshold += 0.28
        if challenger.edge_ratio >= 1.08:
            threshold -= 0.03
        if challenger.total_goals >= 6 and support < 0.68:
            threshold += 0.08
        if challenger_lane == "upset_cold":
            one_x_two = _implied_probabilities(match.market.one_x_two_odds)
            if one_x_two and max(one_x_two.values()) >= 0.62 and challenger.edge_ratio < 1.24 and support < 0.62:
                threshold += 0.08

        if challenger_value >= selected_value * _clamp(threshold, 0.52, 0.92):
            return challenger
        return selected

    def _forced_exit_challenger(
        self,
        match: MatchInput,
        home_mu: float,
        away_mu: float,
        selected: ScoreCandidate,
        candidates: list[ScoreCandidate],
    ) -> ScoreCandidate:
        plan = self._forced_exit_plan(match, home_mu, away_mu)
        if not bool(plan["active"]):
            return selected

        preferred_lanes = set(plan["lanes"])
        selected_lane = self._score_exit_lane(selected, match, home_mu, away_mu)
        selected_support = self._route_support(match, selected, home_mu, away_mu, selected_lane)
        selected_public = self._is_public_or_comfort_pick(selected, match, home_mu, away_mu)
        if (
            not selected_public
            and selected_lane in preferred_lanes
            and selected_support >= 0.46
            and not (
                selected.total_goals < 4
                and selected_lane in {"draw_ladder", "exchange_war"}
                and float(plan["strength"]) >= 0.62
            )
        ):
            return selected

        challengers: list[tuple[float, str, float, ScoreCandidate]] = []
        selected_value = self._route_adjusted_value(selected, match, home_mu, away_mu)
        for candidate in candidates:
            if candidate.score == selected.score:
                continue
            lane = self._score_exit_lane(candidate, match, home_mu, away_mu)
            if lane not in preferred_lanes:
                continue

            support = self._route_support(match, candidate, home_mu, away_mu, lane)
            if not self._passes_forced_exit_quality(candidate, lane, support, selected_value):
                continue
            if not self._is_forced_exit_shape(candidate, lane, support, match, home_mu, away_mu, preferred_lanes):
                continue

            adjusted = self._forced_exit_value(candidate, lane, support, match, home_mu, away_mu, preferred_lanes)
            challengers.append((adjusted, lane, support, candidate))

        if not challengers:
            return selected

        challenger_value, challenger_lane, support, challenger = max(challengers, key=lambda item: item[0])
        strength = float(plan["strength"])
        threshold = 0.78 - strength * 0.30
        if selected_public:
            threshold -= 0.10
        if challenger_lane in {"upset_cold", "single_side_blowout"} and support >= 0.58:
            threshold -= 0.05
        if challenger_lane in {"clean_sheet", "low_lock"} and support >= 0.54:
            threshold -= 0.04
        if challenger.edge_ratio >= 1.08:
            threshold -= 0.03
        if challenger.total_goals >= 6 and support < 0.68:
            threshold += 0.07

        if challenger_value < selected_value * _clamp(threshold, 0.38, 0.76):
            return selected

        reason = (
            "forced exit gate: "
            + ", ".join(str(item) for item in plan["reasons"])
            + f"; lane={challenger_lane} support={support:.2f}"
        )
        return _with_extra_reasons(challenger, reason)

    def _result_script_challenger(
        self,
        match: MatchInput,
        home_mu: float,
        away_mu: float,
        selected: ScoreCandidate,
        candidates: list[ScoreCandidate],
    ) -> ScoreCandidate:
        selected_script = self._result_script_key(selected, match, home_mu, away_mu)
        selected_signal = self._result_script_signal(match, selected, home_mu, away_mu, selected_script)
        selected_value = self._route_adjusted_value(selected, match, home_mu, away_mu)
        challengers: list[tuple[float, str, float, ScoreCandidate]] = []

        for candidate in candidates:
            if candidate.score == selected.score:
                continue
            script = self._result_script_key(candidate, match, home_mu, away_mu)
            if script == selected_script and selected_signal >= 0.56:
                continue

            signal = self._result_script_signal(match, candidate, home_mu, away_mu, script)
            if self._result_script_conflicts_with_protected_pick(
                match,
                home_mu,
                away_mu,
                selected,
                selected_script,
                selected_signal,
                script,
                signal,
            ):
                continue
            if not self._passes_result_script_quality(candidate, script, signal, selected_value):
                continue

            value = self._result_script_adjusted_value(candidate, match, home_mu, away_mu, script, signal)
            challengers.append((value, script, signal, candidate))

        if not challengers:
            return selected

        challenger_value, script, signal, challenger = max(challengers, key=lambda item: item[0])
        threshold = 0.76
        profile = self._scenario_profile(match, home_mu, away_mu)
        public_selected = self._is_public_or_comfort_pick(selected, match, home_mu, away_mu)
        if public_selected and selected_signal < 0.62:
            threshold -= 0.08
        if script in {"small_cold", "big_exchange_cold", "big_cover_cold"}:
            if float(profile["fake_hot"]) >= 0.40 or float(profile["upset"]) >= 0.48:
                threshold -= 0.07
            elif float(profile["normal"]) >= 0.76:
                threshold += 0.08
        if script in {"small_draw", "big_draw"} and float(profile["draw"]) >= 0.56:
            threshold -= 0.05
        if script in {"big_exchange_normal", "big_cover_normal", "big_exchange_cold", "big_cover_cold"} and signal >= 0.68:
            threshold -= 0.04
        if challenger.edge_ratio >= 1.08:
            threshold -= 0.03
        if challenger.conflict_penalty >= 0.18 and signal < 0.70:
            threshold += 0.06
        if selected_script in {"big_cover_normal", "big_cover_cold"} and selected_signal >= 0.68:
            threshold += 0.05

        if challenger_value < selected_value * _clamp(threshold, 0.48, 0.84):
            return selected

        carried_reasons = tuple(
            reason
            for reason in selected.reasons
            if reason.startswith("forced exit gate") or reason.startswith("slate ")
        )
        reason = f"result script arbitration: {script} signal={signal:.2f}"
        return _with_extra_reasons(challenger, *carried_reasons, reason)

    def _result_script_key(
        self,
        candidate: ScoreCandidate,
        match: MatchInput,
        home_mu: float,
        away_mu: float,
    ) -> str:
        home = candidate.home_goals
        away = candidate.away_goals
        total = candidate.total_goals
        margin = abs(home - away)
        outcome = _outcome(home, away)
        favorite = _market_favorite_side(match, home_mu, away_mu)
        big = self._is_big_result_total(match, home_mu, away_mu, total)

        if outcome == "draw":
            return "big_draw" if big else "small_draw"

        normal_direction = favorite == "draw" or outcome == favorite
        if not normal_direction:
            if not big:
                return "small_cold"
            if margin >= 2:
                return "big_cover_cold"
            return "big_exchange_cold"

        if not big:
            return "small_normal"
        if margin >= 2:
            return "big_cover_normal"
        if min(home, away) >= 1 and total >= 4:
            return "big_exchange_normal"
        return "big_normal"

    def _is_big_result_total(
        self,
        match: MatchInput,
        home_mu: float,
        away_mu: float,
        total: int,
    ) -> bool:
        if match.market.total_goals_line is not None:
            line = match.market.total_goals_line
            if total >= math.ceil(line + 0.25):
                return True
            if total <= math.floor(line - 0.25):
                return False
        expected_total = home_mu + away_mu
        if expected_total <= 2.25:
            return total >= 3
        if expected_total >= 3.05:
            return total >= 4
        return total >= 3

    def _result_script_signal(
        self,
        match: MatchInput,
        candidate: ScoreCandidate,
        home_mu: float,
        away_mu: float,
        script: str,
    ) -> float:
        home = candidate.home_goals
        away = candidate.away_goals
        outcome = _outcome(home, away)
        expected_total = home_mu + away_mu
        profile = self._scenario_profile(match, home_mu, away_mu)
        lane = self._score_exit_lane(candidate, match, home_mu, away_mu)
        support = self._route_support(match, candidate, home_mu, away_mu, lane)
        handicap = self._handicap_profile(match, home_mu, away_mu)
        true_deep = 0.0
        trap = _clamp(match.market.bookmaker_trap_risk, 0.0, 1.0)
        if handicap:
            trap = max(trap, float(handicap["trap"]))
            if outcome == str(handicap["favorite"]):
                true_deep = float(handicap["true_deep"])
        over_heat = _clamp(match.market.over_money_heat, 0.0, 1.0)
        under_heat = _clamp(match.market.under_money_heat, 0.0, 1.0)
        volatility = _clamp(match.league_volatility, 0.0, 1.0)
        chaos = _endgame_chaos(match)
        edge_support = _clamp((candidate.edge_ratio - 1.0) * 0.36, 0.0, 0.18)
        total_line = match.market.total_goals_line if match.market.total_goals_line is not None else 2.5
        line_over_fit = _clamp(candidate.total_goals - total_line, 0.0, 2.0) * 0.08
        line_under_fit = _clamp(total_line - candidate.total_goals, 0.0, 2.0) * 0.08
        balance = _clamp(0.52 - abs(home_mu - away_mu), 0.0, 0.52) / 0.52
        both_score_fit = _clamp(min(home_mu, away_mu) - 0.98, 0.0, 0.85)

        if script == "small_normal":
            return _clamp(
                float(profile["normal"]) * 0.34
                + support * 0.22
                + _clamp(2.48 - expected_total, 0.0, 1.0) * 0.18
                + under_heat * 0.18
                + line_under_fit
                + _clamp(0.58 - volatility, 0.0, 0.58) * 0.16
                + edge_support
                - float(profile["fake_hot"]) * 0.14
                - over_heat * 0.10,
                0.0,
                1.0,
            )

        if script == "big_normal":
            return _clamp(
                float(profile["normal"]) * 0.28
                + support * 0.22
                + _clamp(expected_total - 2.35, 0.0, 1.3) * 0.16
                + over_heat * 0.16
                + line_over_fit
                + volatility * 0.12
                + edge_support
                - trap * 0.10
                - under_heat * 0.08,
                0.0,
                1.0,
            )

        if script == "big_exchange_normal":
            return _clamp(
                float(profile["normal"]) * 0.18
                + float(profile["exchange"]) * 0.28
                + support * 0.24
                + _clamp(expected_total - 2.60, 0.0, 1.5) * 0.18
                + both_score_fit * 0.20
                + volatility * 0.16
                + over_heat * 0.14
                + (chaos * 0.12 if match.is_final_round else 0.0)
                + edge_support
                - under_heat * 0.10,
                0.0,
                1.0,
            )

        if script == "big_cover_normal":
            winner = "home" if home > away else "away"
            loser_mu = away_mu if winner == "home" else home_mu
            winner_big_need = match.home_big_win_need if winner == "home" else match.away_big_win_need
            winner_survival = _survival_pressure(match, winner)
            loser = "away" if winner == "home" else "home"
            loser_settled = match.away_settled if loser == "away" else match.home_settled
            loser_collapse = match.away_collapse_risk if loser == "away" else match.home_collapse_risk
            return _clamp(
                float(profile["normal"]) * 0.20
                + support * 0.26
                + true_deep * 0.24
                + _clamp(1.16 - loser_mu, 0.0, 1.0) * 0.14
                + _clamp(winner_big_need, 0.0, 1.0) * 0.18
                + _clamp(winner_survival, 0.0, 1.0) * 0.14
                + (0.08 if loser_settled else 0.0)
                + (_clamp(loser_collapse, 0.0, 1.0) * 0.20 if match.is_final_round else 0.0)
                + over_heat * 0.08
                + edge_support
                - trap * 0.14,
                0.0,
                1.0,
            )

        if script == "small_draw":
            draw_lock = 0.18 if match.home_draw_sufficient and match.away_draw_sufficient else 0.0
            return _clamp(
                float(profile["draw"]) * 0.34
                + support * 0.22
                + balance * 0.16
                + _clamp(2.42 - expected_total, 0.0, 1.0) * 0.18
                + under_heat * 0.18
                + _clamp(match.market.draw_money_heat, 0.0, 1.0) * 0.10
                + _clamp(match.weather_goal_drag, 0.0, 1.0) * 0.14
                + draw_lock
                + edge_support
                - over_heat * 0.12
                - float(profile["exchange"]) * 0.08,
                0.0,
                1.0,
            )

        if script == "big_draw":
            return _clamp(
                float(profile["draw"]) * 0.24
                + float(profile["exchange"]) * 0.22
                + support * 0.24
                + balance * 0.12
                + _clamp(expected_total - 2.75, 0.0, 1.4) * 0.18
                + volatility * 0.18
                + over_heat * 0.14
                + (chaos * 0.18 if match.is_final_round else 0.0)
                + edge_support
                - under_heat * 0.12
                - float(profile["normal"]) * 0.08,
                0.0,
                1.0,
            )

        if script == "small_cold":
            return _clamp(
                float(profile["upset"]) * 0.30
                + float(profile["fake_hot"]) * 0.22
                + support * 0.24
                + trap * 0.16
                + _clamp(2.42 - expected_total, 0.0, 0.9) * 0.16
                + under_heat * 0.18
                + _clamp(_stake_pressure(match, outcome), 0.0, 1.0) * 0.14
                + _clamp(_survival_pressure(match, outcome), 0.0, 1.0) * 0.12
                + edge_support
                - float(profile["normal"]) * 0.12
                - over_heat * 0.10,
                0.0,
                1.0,
            )

        if script == "big_exchange_cold":
            return _clamp(
                float(profile["upset"]) * 0.24
                + float(profile["fake_hot"]) * 0.18
                + float(profile["exchange"]) * 0.24
                + support * 0.24
                + both_score_fit * 0.18
                + _clamp(expected_total - 2.58, 0.0, 1.5) * 0.16
                + volatility * 0.16
                + over_heat * 0.14
                + _clamp(_stake_pressure(match, outcome), 0.0, 1.0) * 0.12
                + (chaos * 0.12 if match.is_final_round else 0.0)
                + edge_support
                - under_heat * 0.10
                - float(profile["normal"]) * 0.10,
                0.0,
                1.0,
            )

        if script == "big_cover_cold":
            favorite = str(profile["favorite"])
            favorite_collapse = match.home_collapse_risk if favorite == "home" else match.away_collapse_risk
            favorite_settled = match.home_settled if favorite == "home" else match.away_settled
            return _clamp(
                float(profile["upset"]) * 0.28
                + float(profile["fake_hot"]) * 0.20
                + support * 0.26
                + trap * 0.18
                + _clamp(_stake_pressure(match, outcome), 0.0, 1.0) * 0.16
                + _clamp(_survival_pressure(match, outcome), 0.0, 1.0) * 0.16
                + (0.08 if favorite_settled else 0.0)
                + (_clamp(favorite_collapse, 0.0, 1.0) * 0.22 if match.is_final_round else 0.0)
                + _clamp(expected_total - 2.50, 0.0, 1.5) * 0.10
                + over_heat * 0.10
                + edge_support
                - float(profile["normal"]) * 0.12
                - under_heat * 0.08,
                0.0,
                1.0,
            )

        return 0.0

    def _result_script_conflicts_with_protected_pick(
        self,
        match: MatchInput,
        home_mu: float,
        away_mu: float,
        selected: ScoreCandidate,
        selected_script: str,
        selected_signal: float,
        challenger_script: str,
        challenger_signal: float,
    ) -> bool:
        if selected_signal < 0.62:
            return False
        if selected_script in {"big_cover_normal", "big_cover_cold"}:
            if challenger_script in {"small_normal", "small_draw", "small_cold"}:
                return challenger_signal < selected_signal + 0.12
            if challenger_script in {"big_exchange_normal", "big_exchange_cold"}:
                return challenger_signal < selected_signal + 0.12
        if selected_script in {"small_draw", "small_normal"}:
            if challenger_script in {"big_exchange_normal", "big_exchange_cold", "big_cover_normal", "big_cover_cold"}:
                return challenger_signal < selected_signal + 0.16
        if selected_script in {"small_cold", "big_cover_cold", "big_exchange_cold"}:
            if challenger_script.endswith("_normal"):
                return challenger_signal < selected_signal + 0.12
        profile = self._scenario_profile(match, home_mu, away_mu)
        if (
            float(profile["normal"]) >= 0.78
            and selected_script.endswith("_normal")
            and challenger_script in {"small_cold", "big_exchange_cold", "big_cover_cold"}
            and challenger_signal < 0.74
        ):
            return True
        return False

    def _passes_result_script_quality(
        self,
        candidate: ScoreCandidate,
        script: str,
        signal: float,
        selected_value: float,
    ) -> bool:
        threshold = {
            "small_normal": 0.50,
            "big_normal": 0.50,
            "big_exchange_normal": 0.68,
            "big_cover_normal": 0.57,
            "small_draw": 0.52,
            "big_draw": 0.58,
            "small_cold": 0.55,
            "big_exchange_cold": 0.64,
            "big_cover_cold": 0.62,
        }[script]
        if signal < threshold:
            return False

        if script in {"big_exchange_normal", "big_exchange_cold"}:
            if candidate.total_goals < 4 or min(candidate.home_goals, candidate.away_goals) < 1:
                return False
            if signal < 0.74 and candidate.structure_score < 0.92 and candidate.edge_ratio < 1.10:
                return False

        probability_floor = 0.018
        if signal >= 0.74:
            probability_floor = 0.008
        elif signal >= 0.66:
            probability_floor = 0.012
        if script in {"small_cold", "big_exchange_cold", "big_cover_cold"}:
            probability_floor += 0.004
        if candidate.real_probability < probability_floor and candidate.final_value < selected_value * 0.16:
            return False
        if candidate.structure_score < 0.72 and candidate.edge_ratio < 1.02 and signal < 0.66:
            return False
        if candidate.conflict_penalty > 0.22 and signal < 0.72:
            return False
        if script in {"big_exchange_cold", "big_cover_cold"} and candidate.edge_ratio < 0.92 and signal < 0.70:
            return False
        return True

    def _result_script_adjusted_value(
        self,
        candidate: ScoreCandidate,
        match: MatchInput,
        home_mu: float,
        away_mu: float,
        script: str,
        signal: float,
    ) -> float:
        value = self._route_adjusted_value(candidate, match, home_mu, away_mu)
        value *= 1.0 + signal * 0.40
        if script in {"small_cold", "big_exchange_cold", "big_cover_cold"}:
            value *= 1.08
        if script in {"big_exchange_normal", "big_exchange_cold", "big_draw"}:
            value *= 1.06
        if script in {"big_cover_normal", "big_cover_cold"} and abs(candidate.home_goals - candidate.away_goals) >= 3:
            value *= 1.06
        if script in {"small_normal", "small_draw", "small_cold"} and candidate.total_goals <= 1:
            value *= 1.05
        if _is_broad_public_score(candidate.home_goals, candidate.away_goals) and signal < 0.62:
            value *= 0.90
        if candidate.edge_ratio >= 1.10:
            value *= 1.04
        if candidate.real_probability < 0.012 and signal < 0.72:
            value *= 0.80
        return value

    def _shape_arbitration_challenger(
        self,
        match: MatchInput,
        home_mu: float,
        away_mu: float,
        selected: ScoreCandidate,
        candidates: list[ScoreCandidate],
    ) -> ScoreCandidate:
        selected_value = self._route_adjusted_value(selected, match, home_mu, away_mu)
        selected_shape = self._score_shape_key(selected, match, home_mu, away_mu)
        selected_signal = (
            self._shape_signal(match, selected, home_mu, away_mu, selected_shape)
            if selected_shape is not None
            else 0.0
        )
        challengers: list[tuple[float, str, float, ScoreCandidate]] = []
        selected_lane = self._score_exit_lane(selected, match, home_mu, away_mu)
        selected_support = self._route_support(match, selected, home_mu, away_mu, selected_lane)

        for candidate in candidates:
            if candidate.score == selected.score:
                continue
            shape = self._score_shape_key(candidate, match, home_mu, away_mu)
            if shape is None:
                continue

            signal = self._shape_signal(match, candidate, home_mu, away_mu, shape)
            if self._shape_challenger_conflicts_with_protected_script(
                selected,
                selected_lane,
                selected_support,
                shape,
                signal,
            ):
                continue
            if not self._passes_shape_arbitration_quality(candidate, selected, shape, signal, selected_value):
                continue

            value = self._shape_adjusted_value(candidate, match, home_mu, away_mu, shape, signal)
            challengers.append((value, shape, signal, candidate))

        if not challengers:
            return selected

        challenger_value, shape, signal, challenger = max(challengers, key=lambda item: item[0])
        if selected_shape == shape and selected_signal >= signal - 0.04:
            return selected

        threshold = 0.74
        if self._is_public_or_comfort_pick(selected, match, home_mu, away_mu):
            threshold -= 0.08
        if _is_broad_public_score(selected.home_goals, selected.away_goals):
            threshold -= 0.04
        if shape in {"small_cold", "high_draw"}:
            threshold -= 0.04
        if shape in {"extreme_clean", "extreme_single"} and signal >= 0.72:
            threshold -= 0.05
        if shape == "wide_exchange" and signal >= 0.68:
            threshold -= 0.03
        if challenger.edge_ratio >= 1.08:
            threshold -= 0.03
        if challenger.conflict_penalty >= 0.18 and signal < 0.70:
            threshold += 0.06
        if max(challenger.home_goals, challenger.away_goals) >= 5 and signal < 0.75:
            threshold += 0.04

        if challenger_value < selected_value * _clamp(threshold, 0.48, 0.82):
            return selected

        carried_reasons = tuple(
            reason
            for reason in selected.reasons
            if reason.startswith("forced exit gate") or reason.startswith("slate ")
        )
        reason = f"shape arbitration: {shape} signal={signal:.2f}"
        return _with_extra_reasons(challenger, *carried_reasons, reason)

    def _strong_match_under_challenger(
        self,
        match: MatchInput,
        home_mu: float,
        away_mu: float,
        selected: ScoreCandidate,
        candidates: list[ScoreCandidate],
    ) -> ScoreCandidate:
        signal, preferred_outcomes = self._strong_match_under_signal(match, home_mu, away_mu)
        if signal < 0.54 and not (
            signal >= 0.44
            and selected.total_goals >= 5
            and _clamp(match.market.under_money_heat, 0.0, 1.0) >= 0.56
        ):
            return selected

        selected_lane = self._score_exit_lane(selected, match, home_mu, away_mu)
        if selected.total_goals <= 2 and selected_lane in {"low_lock", "draw_ladder", "upset_cold", "favorite_noncover"}:
            return selected
        if selected.total_goals < 4 and selected_lane not in {"single_side_blowout", "favorite_cover"}:
            return selected

        selected_value = self._route_adjusted_value(selected, match, home_mu, away_mu)
        under_heat = _clamp(match.market.under_money_heat, 0.0, 1.0)
        expected_total = home_mu + away_mu
        contenders: list[tuple[float, ScoreCandidate]] = []
        for candidate in candidates:
            if candidate.score == selected.score or candidate.total_goals > 2:
                continue
            outcome = _outcome(candidate.home_goals, candidate.away_goals)
            if outcome not in preferred_outcomes:
                continue
            lane = self._score_exit_lane(candidate, match, home_mu, away_mu)
            support = self._route_support(match, candidate, home_mu, away_mu, lane)
            if candidate.real_probability < 0.015 and support < 0.58:
                continue
            if candidate.structure_score < 0.72 and candidate.edge_ratio < 0.94 and support < 0.58:
                continue
            if candidate.conflict_penalty > 0.24 and signal < 0.70:
                continue

            value = self._route_adjusted_value(candidate, match, home_mu, away_mu)
            value *= 1.0 + signal * 0.62 + support * 0.18
            if outcome == preferred_outcomes[0]:
                value *= 2.25 if outcome != "draw" else 1.16
            elif preferred_outcomes[0] != "draw" and outcome == "draw":
                value *= 0.64
            if outcome == "draw":
                if candidate.score == "1-1":
                    value *= 1.14
                elif candidate.score == "0-0" and expected_total > 2.35:
                    value *= 0.82
            else:
                if candidate.total_goals == 1:
                    value *= 1.13
                if abs(candidate.home_goals - candidate.away_goals) >= 2:
                    value *= 0.58 if preferred_outcomes[0] == outcome else 0.86
            if under_heat >= 0.56 and candidate.total_goals <= 1:
                value *= 1.08
            if candidate.edge_ratio >= 1.08:
                value *= 1.04
            contenders.append((value, candidate))

        if not contenders:
            return selected

        challenger_value, challenger = max(contenders, key=lambda item: item[0])
        threshold = 0.74 - signal * 0.30
        if selected.total_goals >= 5:
            threshold -= 0.07
        if selected_lane in {"single_side_blowout", "favorite_cover", "exchange_war"}:
            threshold -= 0.04
        if signal >= 0.55 and _clamp(match.market.under_money_heat, 0.0, 1.0) >= 0.56:
            threshold -= 0.05
        if challenger.edge_ratio < 0.78 and signal < 0.66:
            threshold += 0.04
        if challenger.real_probability < 0.022 and signal < 0.68:
            threshold += 0.05

        if challenger_value < selected_value * _clamp(threshold, 0.44, 0.76):
            return selected

        reason = (
            "strong-match under trap arbitration: "
            f"signal={signal:.2f} outcomes={','.join(preferred_outcomes)}"
        )
        return _with_extra_reasons(challenger, reason)

    def _strong_match_under_signal(
        self,
        match: MatchInput,
        home_mu: float,
        away_mu: float,
    ) -> tuple[float, tuple[str, ...]]:
        one_x_two = _implied_probabilities(match.market.one_x_two_odds)
        if not one_x_two:
            return 0.0, ()

        market_favorite = max(one_x_two, key=one_x_two.get)
        if market_favorite == "draw":
            return 0.0, ()

        underdog = "away" if market_favorite == "home" else "home"
        mu_margin = home_mu - away_mu
        favorite_context = _side_context_strength(match, market_favorite, mu_margin)
        underdog_context = _side_context_strength(match, underdog, mu_margin)
        resistance = _clamp(underdog_context - favorite_context + 0.30, 0.0, 0.70) / 0.70
        balance = _clamp(0.58 - abs(mu_margin), 0.0, 0.58) / 0.58
        favorite_prob = one_x_two[market_favorite]
        favorite_heat = _clamp((favorite_prob - 0.42) / 0.30, 0.0, 1.0)
        favorite_heat = _clamp(
            favorite_heat * 0.48
            + _side_money_heat(match, market_favorite) * 0.30
            + _clamp(match.market.favorite_money_heat, 0.0, 1.0) * 0.22,
            0.0,
            1.0,
        )
        under_heat = _clamp(match.market.under_money_heat, 0.0, 1.0)
        over_heat = _clamp(match.market.over_money_heat, 0.0, 1.0)
        trap = _clamp(match.market.bookmaker_trap_risk, 0.0, 1.0)
        depth = abs(match.market.asian_handicap) if match.market.asian_handicap is not None else 0.0
        line_is_shallow_or_half_deep = 1.0 if 0.50 <= depth <= 1.25 else 0.0
        total_line = match.market.total_goals_line if match.market.total_goals_line is not None else home_mu + away_mu
        under_gate = under_heat >= 0.52 and under_heat >= over_heat - 0.02
        if not under_gate and total_line > 2.65:
            return 0.0, ()

        handicap = self._handicap_profile(match, home_mu, away_mu)
        true_deep = float(handicap["true_deep"]) if handicap else 0.0
        favorite_big_need = match.home_big_win_need if market_favorite == "home" else match.away_big_win_need
        favorite_survival = _survival_pressure(match, market_favorite)
        underdog_pressure = _stake_pressure(match, underdog)
        underdog_attack = match.away_attack if underdog == "away" else match.home_attack
        underdog_recent_xg = match.recent_away_xg if underdog == "away" else match.recent_home_xg
        underdog_form = match.away_form if underdog == "away" else match.home_form
        underdog_motivation = match.away_motivation if underdog == "away" else match.home_motivation
        underdog_quality = _clamp(
            _clamp(underdog_attack - 1.18, 0.0, 0.70) * 0.60
            + _clamp((underdog_recent_xg or underdog_attack * 1.18) - 1.35, 0.0, 0.90) * 0.30
            + _clamp(underdog_form, 0.0, 1.0) * 0.10
            + _clamp(underdog_motivation, 0.0, 1.0) * 0.10,
            0.0,
            1.0,
        )
        big_line_under_tension = (
            0.08
            if total_line >= 3.0 and under_heat >= over_heat + 0.10 and line_is_shallow_or_half_deep
            else 0.0
        )

        signal = (
            0.12
            + resistance * 0.28
            + balance * 0.20
            + under_heat * 0.22
            + trap * 0.18
            + favorite_heat * 0.12
            + line_is_shallow_or_half_deep * 0.12
            + _clamp(total_line - 2.65, 0.0, 0.90) * 0.08
            + underdog_pressure * 0.10
            + underdog_quality * 0.18
            + big_line_under_tension
            - over_heat * 0.15
            - true_deep * 0.24
            - _clamp(favorite_big_need, 0.0, 1.0) * 0.12
            - _clamp(favorite_survival, 0.0, 1.0) * 0.10
        )

        if favorite_prob >= 0.66 and true_deep >= 0.52:
            signal -= 0.22
        if match.is_final_round and max(match.home_collapse_risk, match.away_collapse_risk) >= 0.55:
            signal -= 0.12
        if match.market.asian_handicap is not None and depth >= 1.50:
            signal -= 0.10

        profile = self._scenario_profile(match, home_mu, away_mu)
        draw_first = abs(mu_margin) <= 0.25 or float(profile["draw"]) >= 0.42
        upset_first = (
            underdog_context >= favorite_context - 0.18
            or self._underdog_push(match, market_favorite, mu_margin) >= 0.31
            or float(profile["fake_hot"]) >= 0.28
        )
        if draw_first and not (upset_first and abs(mu_margin) >= 0.34):
            outcomes = ("draw", underdog)
        else:
            outcomes = (underdog, "draw")
        return _clamp(signal, 0.0, 1.0), outcomes

    def _shape_challenger_conflicts_with_protected_script(
        self,
        selected: ScoreCandidate,
        selected_lane: str,
        selected_support: float,
        challenger_shape: str,
        challenger_signal: float,
    ) -> bool:
        if selected_lane in {"single_side_blowout", "clean_sheet", "favorite_cover"} and selected_support >= 0.52:
            if challenger_shape in {"small_cold", "wide_exchange", "high_draw"} and challenger_signal < selected_support + 0.18:
                return True
        if selected_lane == "single_side_blowout" and selected_support >= 0.52:
            if challenger_shape == "wide_exchange" and max(selected.home_goals, selected.away_goals) >= 4:
                return True
        if selected_lane == "single_side_blowout" and selected_support >= 0.78:
            if challenger_shape in {"wide_exchange", "high_draw"}:
                return True
        if selected_lane == "upset_cold" and selected_support >= 0.60:
            if challenger_shape != "small_cold" and challenger_signal < selected_support + 0.14:
                return True
            if challenger_shape == "small_cold" and selected.total_goals >= 3 and selected_support >= 0.72:
                return True
        if selected_lane == "draw_ladder" and selected.total_goals >= 4 and selected_support >= 0.50:
            if challenger_shape == "wide_exchange":
                return True
        if selected_lane == "draw_ladder" and selected.total_goals >= 4 and selected_support >= 0.30:
            if challenger_shape != "high_draw":
                return True
        if selected.total_goals >= 4 and challenger_shape == "small_cold" and selected_support >= 0.52:
            return True
        if selected_lane == "single_side_blowout" and challenger_shape == "wide_exchange" and challenger_signal < 0.78:
            return True
        return False

    def _score_shape_key(
        self,
        candidate: ScoreCandidate,
        match: MatchInput,
        home_mu: float,
        away_mu: float,
    ) -> str | None:
        score = (candidate.home_goals, candidate.away_goals)
        if score in {(4, 0), (0, 4), (5, 0), (0, 5)}:
            return "extreme_clean"
        if score in {(4, 1), (1, 4), (5, 1), (1, 5)}:
            return "extreme_single"
        if score in {(4, 2), (2, 4)}:
            return "wide_exchange"
        if score == (3, 3):
            return "high_draw"

        outcome = _outcome(candidate.home_goals, candidate.away_goals)
        favorite = _market_favorite_side(match, home_mu, away_mu)
        if favorite != "draw" and outcome not in {favorite, "draw"} and candidate.total_goals <= 2:
            return "small_cold"
        return None

    def _shape_signal(
        self,
        match: MatchInput,
        candidate: ScoreCandidate,
        home_mu: float,
        away_mu: float,
        shape: str,
    ) -> float:
        home = candidate.home_goals
        away = candidate.away_goals
        total = candidate.total_goals
        expected_total = home_mu + away_mu
        outcome = _outcome(home, away)
        profile = self._scenario_profile(match, home_mu, away_mu)
        lane = self._score_exit_lane(candidate, match, home_mu, away_mu)
        support = self._route_support(match, candidate, home_mu, away_mu, lane)
        handicap = self._handicap_profile(match, home_mu, away_mu)
        true_deep = 0.0
        trap = _clamp(match.market.bookmaker_trap_risk, 0.0, 1.0)
        if handicap:
            trap = max(trap, float(handicap["trap"]))
            if outcome == str(handicap["favorite"]):
                true_deep = float(handicap["true_deep"])
        edge_support = _clamp((candidate.edge_ratio - 1.0) * 0.40, 0.0, 0.20)
        chaos = _endgame_chaos(match)
        over_heat = _clamp(match.market.over_money_heat, 0.0, 1.0)
        under_heat = _clamp(match.market.under_money_heat, 0.0, 1.0)
        volatility = _clamp(match.league_volatility, 0.0, 1.0)

        if shape in {"extreme_clean", "extreme_single"}:
            winner = "home" if home > away else "away"
            loser = "away" if winner == "home" else "home"
            loser_mu = away_mu if winner == "home" else home_mu
            winner_big_need = match.home_big_win_need if winner == "home" else match.away_big_win_need
            winner_survival = _survival_pressure(match, winner)
            loser_settled = match.away_settled if loser == "away" else match.home_settled
            loser_collapse = match.away_collapse_risk if loser == "away" else match.home_collapse_risk
            loser_absence = match.away_absence_impact if loser == "away" else match.home_absence_impact
            winner_pressure = _stake_pressure(match, winner)
            loser_zero_fit = _clamp(1.15 - loser_mu, 0.0, 1.0)
            if shape == "extreme_single":
                loser_zero_fit = _clamp(1.35 - loser_mu, 0.0, 1.0) * 0.62
            upset_stretch = float(profile["upset"]) * 0.18 if outcome != str(profile["favorite"]) else 0.0
            signal = (
                support * 0.38
                + loser_zero_fit * 0.20
                + true_deep * 0.22
                + _clamp(winner_big_need, 0.0, 1.0) * 0.18
                + _clamp(winner_pressure, 0.0, 1.0) * 0.12
                + _clamp(winner_survival, 0.0, 1.0) * 0.16
                + (0.09 if loser_settled else 0.0)
                + (_clamp(loser_collapse, 0.0, 1.0) * 0.24 if match.is_final_round else 0.0)
                + _clamp(loser_absence, 0.0, 1.0) * 0.08
                + _clamp(expected_total - 2.55, 0.0, 1.4) * 0.08
                + edge_support
                + upset_stretch
                - trap * 0.16
            )
            if max(home, away) >= 5:
                signal += _clamp(max(home_mu, away_mu) - 2.05, 0.0, 0.8) * 0.08
            return _clamp(signal, 0.0, 1.0)

        if shape == "wide_exchange":
            weak_side_fit = _clamp(min(home_mu, away_mu) - 1.08, 0.0, 0.75)
            high_draw_pull = float(profile["draw"]) * 0.10 if abs(home_mu - away_mu) <= 0.30 else 0.0
            signal = (
                support * 0.36
                + float(profile["exchange"]) * 0.26
                + _clamp(volatility - 0.48, 0.0, 0.52) * 0.34
                + _clamp(expected_total - 2.75, 0.0, 1.4) * 0.18
                + weak_side_fit * 0.18
                + over_heat * 0.14
                + (chaos * 0.16 if match.is_final_round else 0.0)
                + max(_survival_pressure(match, "home"), _survival_pressure(match, "away")) * 0.10
                + edge_support
                - under_heat * 0.14
                - high_draw_pull
            )
            return _clamp(signal, 0.0, 1.0)

        if shape == "high_draw":
            balance = _clamp(0.46 - abs(home_mu - away_mu), 0.0, 0.46) / 0.46
            signal = (
                support * 0.34
                + float(profile["draw"]) * 0.24
                + float(profile["exchange"]) * 0.20
                + _clamp(volatility - 0.52, 0.0, 0.48) * 0.30
                + _clamp(expected_total - 2.85, 0.0, 1.35) * 0.18
                + balance * 0.14
                + _clamp(match.market.draw_money_heat, 0.0, 1.0) * 0.10
                + (chaos * 0.17 if match.is_final_round else 0.0)
                + edge_support
                - float(profile["normal"]) * 0.10
                - under_heat * 0.12
            )
            return _clamp(signal, 0.0, 1.0)

        if shape == "small_cold":
            favorite = str(profile["favorite"])
            favorite_settled = match.home_settled if favorite == "home" else match.away_settled
            favorite_collapse = match.home_collapse_risk if favorite == "home" else match.away_collapse_risk
            underdog_survival = _survival_pressure(match, outcome)
            signal = (
                support * 0.40
                + float(profile["upset"]) * 0.28
                + float(profile["fake_hot"]) * 0.22
                + _clamp(2.42 - expected_total, 0.0, 0.9) * 0.22
                + under_heat * 0.18
                + trap * 0.16
                + _clamp(_stake_pressure(match, outcome), 0.0, 1.0) * 0.16
                + _clamp(underdog_survival, 0.0, 1.0) * 0.14
                + (0.08 if favorite_settled else 0.0)
                + (_clamp(favorite_collapse, 0.0, 1.0) * 0.16 if match.is_final_round else 0.0)
                + edge_support
                - over_heat * 0.12
                - float(profile["exchange"]) * 0.08
            )
            if candidate.total_goals <= 1:
                signal += _clamp(2.20 - expected_total, 0.0, 0.8) * 0.08
            return _clamp(signal, 0.0, 1.0)

        return 0.0

    def _passes_shape_arbitration_quality(
        self,
        candidate: ScoreCandidate,
        selected: ScoreCandidate,
        shape: str,
        signal: float,
        selected_value: float,
    ) -> bool:
        threshold = {
            "extreme_clean": 0.60,
            "extreme_single": 0.60,
            "wide_exchange": 0.58,
            "high_draw": 0.60,
            "small_cold": 0.54,
        }[shape]
        if max(candidate.home_goals, candidate.away_goals) >= 5:
            threshold += 0.06
        if candidate.total_goals >= 6 and shape in {"wide_exchange", "high_draw"}:
            threshold += 0.04
        if signal < threshold:
            return False

        probability_floor = 0.014
        if signal >= 0.72:
            probability_floor = 0.008
        elif signal >= 0.64:
            probability_floor = 0.011
        if shape == "small_cold":
            probability_floor += 0.004
        if candidate.real_probability < probability_floor and candidate.final_value < selected_value * 0.14:
            return False
        if candidate.structure_score < 0.70 and candidate.edge_ratio < 1.02 and signal < 0.66:
            return False
        if candidate.conflict_penalty > 0.24 and signal < 0.72:
            return False
        if shape in {"extreme_clean", "extreme_single"} and max(candidate.home_goals, candidate.away_goals) >= 5 and signal < 0.72:
            return False
        if shape == "small_cold" and selected.total_goals >= 5 and signal < 0.62:
            return False
        return True

    def _shape_adjusted_value(
        self,
        candidate: ScoreCandidate,
        match: MatchInput,
        home_mu: float,
        away_mu: float,
        shape: str,
        signal: float,
    ) -> float:
        value = self._route_adjusted_value(candidate, match, home_mu, away_mu)
        value *= 1.0 + signal * 0.48
        if shape in {"extreme_clean", "extreme_single"}:
            value *= 1.08 if max(candidate.home_goals, candidate.away_goals) == 4 else 1.12
        if shape in {"wide_exchange", "high_draw"}:
            value *= 1.08
        if shape == "high_draw" and match.is_final_round and abs(home_mu - away_mu) <= 0.30:
            value *= 1.30
        if shape == "wide_exchange" and candidate.total_goals >= 6 and abs(home_mu - away_mu) <= 0.25:
            value *= 0.74
        if shape == "small_cold":
            value *= 1.10
        if candidate.edge_ratio >= 1.12:
            value *= 1.04
        if candidate.real_probability < 0.012 and signal < 0.74:
            value *= 0.78
        return value

    def _forced_exit_plan(self, match: MatchInput, home_mu: float, away_mu: float) -> dict[str, object]:
        profile = self._scenario_profile(match, home_mu, away_mu)
        lanes: set[str] = set()
        reasons: list[str] = []
        strength = 0.0
        expected_total = home_mu + away_mu

        normal = float(profile["normal"])
        fake_hot = float(profile["fake_hot"])
        if fake_hot >= 0.40 and normal <= 0.66:
            lanes.update({"upset_cold", "draw_ladder", "favorite_noncover"})
            strength = max(strength, 0.44 + fake_hot * 0.42)
            reasons.append("fake-hot favorite")

        upset = float(profile["upset"])
        if upset >= 0.46:
            lanes.add("upset_cold")
            strength = max(strength, 0.40 + upset * 0.42)
            reasons.append("upset pressure")

        draw = float(profile["draw"])
        if draw >= 0.58:
            lanes.add("draw_ladder")
            if expected_total <= 2.25 or match.market.under_money_heat >= 0.55:
                lanes.add("low_lock")
            strength = max(strength, 0.38 + draw * 0.36)
            reasons.append("draw ladder")

        exchange = float(profile["exchange"])
        if exchange >= 0.56 and expected_total >= 2.62:
            lanes.add("exchange_war")
            strength = max(strength, 0.34 + exchange * 0.42)
            reasons.append("open exchange")

        handicap = self._handicap_profile(match, home_mu, away_mu)
        if handicap:
            trap = float(handicap["trap"])
            true_deep = float(handicap["true_deep"])
            depth = float(handicap["depth"])
            if trap >= 0.43:
                lanes.update({"favorite_noncover", "upset_cold", "draw_ladder", "low_lock"})
                strength = max(strength, 0.43 + trap * 0.44)
                reasons.append("deep-handicap trap")
            if depth >= 1.25 and true_deep >= 0.56:
                lanes.update({"favorite_cover", "single_side_blowout", "clean_sheet"})
                strength = max(strength, 0.42 + true_deep * 0.38)
                reasons.append("true-deep cover")

        chaos = _endgame_chaos(match)
        survival = max(_survival_pressure(match, "home"), _survival_pressure(match, "away"))
        collapse = max(_clamp(match.home_collapse_risk, 0.0, 1.0), _clamp(match.away_collapse_risk, 0.0, 1.0))
        if match.is_final_round and chaos >= 0.42:
            lanes.update({"single_side_blowout", "upset_cold", "exchange_war", "draw_ladder", "low_lock"})
            strength = max(strength, 0.42 + chaos * 0.38)
            reasons.append("final-round chaos")
        if survival >= 0.55:
            lanes.update({"single_side_blowout", "clean_sheet", "exchange_war", "upset_cold"})
            strength = max(strength, 0.42 + survival * 0.40)
            reasons.append("survival attack")
        if match.is_final_round and collapse >= 0.55:
            lanes.update({"single_side_blowout", "clean_sheet", "upset_cold"})
            strength = max(strength, 0.40 + collapse * 0.38)
            reasons.append("collapse risk")

        if match.home_draw_sufficient and match.away_draw_sufficient and expected_total <= 2.25:
            lanes.add("low_lock")
            strength = max(strength, 0.58)
            reasons.append("mutual draw lock")

        if normal >= 0.78 and fake_hot < 0.26 and chaos < 0.38 and survival < 0.45:
            handicap_trap = float(handicap["trap"]) if handicap else 0.0
            if handicap_trap < 0.40:
                return {"active": False, "strength": 0.0, "lanes": (), "reasons": ()}

        return {
            "active": strength >= 0.54 and bool(lanes),
            "strength": _clamp(strength, 0.0, 1.0),
            "lanes": tuple(lanes),
            "reasons": tuple(reasons),
        }

    def _passes_forced_exit_quality(
        self,
        candidate: ScoreCandidate,
        lane: str,
        support: float,
        selected_value: float,
    ) -> bool:
        probability_floor = 0.012 if support >= 0.56 else 0.020
        if candidate.real_probability < probability_floor and candidate.final_value < selected_value * 0.16:
            return False
        if candidate.structure_score < 0.74 and candidate.edge_ratio < 1.05 and support < 0.50:
            return False
        if candidate.conflict_penalty > 0.24 and support < 0.60:
            return False
        if lane == "upset_cold" and candidate.conflict_penalty > 0.18 and candidate.edge_ratio < 1.18 and support < 0.62:
            return False
        if lane == "single_side_blowout" and candidate.total_goals >= 6 and support < 0.64:
            return False
        return True

    def _is_forced_exit_shape(
        self,
        candidate: ScoreCandidate,
        lane: str,
        support: float,
        match: MatchInput,
        home_mu: float,
        away_mu: float,
        preferred_lanes: set[str],
    ) -> bool:
        home = candidate.home_goals
        away = candidate.away_goals
        total = candidate.total_goals
        score = (home, away)

        if lane == "exchange_war":
            return total >= 4 and min(home, away) >= 1
        if lane == "single_side_blowout":
            return abs(home - away) >= 3 or (abs(home - away) >= 2 and support >= 0.62)
        if lane == "clean_sheet":
            if total < 2:
                return False
            if _is_broad_public_score(home, away):
                return support >= 0.54 and abs(home - away) >= 2
            return True
        if lane == "low_lock":
            if (
                match.home_draw_sufficient
                and match.away_draw_sufficient
                and (match.market.under_money_heat >= 0.55 or match.weather_goal_drag >= 0.30)
                and total == 1
            ):
                return False
            return score == (0, 0) or (total == 1 and support >= 0.58)
        if lane == "draw_ladder":
            if score == (1, 1):
                return False
            if score == (2, 2):
                return support >= 0.44
            return total >= 4 or (score == (0, 0) and "low_lock" in preferred_lanes)
        if lane == "favorite_noncover":
            return not _is_comfort_score(home, away) or support >= 0.58
        if lane == "favorite_cover":
            if _is_broad_public_score(home, away) and support < 0.62:
                return False
            return abs(home - away) >= 2
        if lane == "upset_cold":
            if _is_comfort_score(home, away) and support < 0.62:
                return False
            market_favorite = _favorite_side(match, home_mu, away_mu)
            return _outcome(home, away) not in {market_favorite, "draw"}
        return not _is_comfort_score(home, away)

    def _forced_exit_value(
        self,
        candidate: ScoreCandidate,
        lane: str,
        support: float,
        match: MatchInput,
        home_mu: float,
        away_mu: float,
        preferred_lanes: set[str],
    ) -> float:
        value = self._route_adjusted_value(candidate, match, home_mu, away_mu)
        value *= 1.0 + support * 0.36
        if lane in preferred_lanes:
            value *= 1.08
        if lane in {"upset_cold", "single_side_blowout", "clean_sheet", "low_lock"} and support >= 0.54:
            value *= 1.10
        if lane == "exchange_war" and candidate.total_goals >= 5 and support >= 0.50:
            value *= 1.08
        if lane == "draw_ladder" and candidate.total_goals >= 4 and support >= 0.44:
            value *= 1.06
        if _is_scorebook_edge_score(candidate.home_goals, candidate.away_goals):
            value *= 1.07
        if _is_broad_public_score(candidate.home_goals, candidate.away_goals) and lane not in {"low_lock", "clean_sheet"}:
            value *= 0.86
        return value

    def _is_public_or_comfort_pick(
        self,
        candidate: ScoreCandidate,
        match: MatchInput,
        home_mu: float,
        away_mu: float,
    ) -> bool:
        if _is_comfort_score(candidate.home_goals, candidate.away_goals):
            return True
        if _is_broad_public_score(candidate.home_goals, candidate.away_goals):
            return True
        if min(candidate.home_goals, candidate.away_goals) == 0 and abs(candidate.home_goals - candidate.away_goals) >= 3:
            return False
        favorite = _favorite_side(match, home_mu, away_mu)
        return _outcome(candidate.home_goals, candidate.away_goals) == favorite and candidate.total_goals <= 3

    def _market_direction_challenger(
        self,
        match: MatchInput,
        selected: ScoreCandidate,
        candidates: list[ScoreCandidate],
    ) -> ScoreCandidate:
        one_x_two = _implied_probabilities(match.market.one_x_two_odds)
        if not one_x_two:
            return selected

        market_favorite = max(one_x_two, key=one_x_two.get)
        market_gap = one_x_two[market_favorite] - one_x_two.get(_outcome(selected.home_goals, selected.away_goals), 0.0)
        if _outcome(selected.home_goals, selected.away_goals) == market_favorite or market_gap < 0.26:
            return selected

        same_direction = [
            candidate
            for candidate in candidates
            if _outcome(candidate.home_goals, candidate.away_goals) == market_favorite
            and candidate.real_probability >= 0.04
            and candidate.conflict_penalty <= 0.05
        ]
        if not same_direction:
            return selected

        def adjusted_value(candidate: ScoreCandidate) -> float:
            value = candidate.final_value
            if candidate.total_goals >= 2:
                value *= 1.05
            if not _is_comfort_score(candidate.home_goals, candidate.away_goals):
                value *= 1.08
            if (candidate.home_goals, candidate.away_goals) in {(1, 0), (0, 1)}:
                value *= 0.94
            return value

        challenger = max(same_direction, key=adjusted_value)
        if adjusted_value(challenger) >= selected.final_value * 0.72:
            return challenger
        return selected

    def _public_fade_challenger(
        self,
        match: MatchInput,
        selected: ScoreCandidate,
        candidates: list[ScoreCandidate],
    ) -> ScoreCandidate:
        one_x_two = _implied_probabilities(match.market.one_x_two_odds)
        selected_outcome = _outcome(selected.home_goals, selected.away_goals)
        market_favorite = max(one_x_two, key=one_x_two.get) if one_x_two else None

        is_public_pick = _is_comfort_score(selected.home_goals, selected.away_goals)
        if market_favorite and selected_outcome == market_favorite:
            is_public_pick = True
        if selected.market_probability >= 0.12 and selected.edge_ratio <= 1.12:
            is_public_pick = True
        if not is_public_pick:
            return selected

        challengers = []
        for candidate in candidates:
            candidate_outcome = _outcome(candidate.home_goals, candidate.away_goals)
            if candidate.real_probability < 0.025:
                continue
            if candidate.structure_score < 0.88:
                continue
            if candidate.total_goals < 2 or candidate.total_goals > 6:
                continue
            if market_favorite and candidate_outcome == market_favorite and _is_comfort_score(candidate.home_goals, candidate.away_goals):
                continue
            if candidate.conflict_penalty > 0.10:
                continue
            challengers.append(candidate)

        if not challengers:
            return selected

        def adjusted_value(candidate: ScoreCandidate) -> float:
            value = candidate.final_value
            if not _is_comfort_score(candidate.home_goals, candidate.away_goals):
                value *= 1.18
            if market_favorite and _outcome(candidate.home_goals, candidate.away_goals) != market_favorite:
                value *= 1.15
            if candidate.total_goals >= 4:
                value *= 1.08
            if candidate.edge_ratio >= 1.0:
                value *= 1.05
            return value

        challenger = max(challengers, key=adjusted_value)
        if adjusted_value(challenger) >= selected.final_value * 0.58:
            return challenger
        return selected

    def _uses_attack_bias(self) -> bool:
        return self.mode in {"adaptive", "aggressive", "contrarian"}

    def _confidence_band(self, selected: ScoreCandidate, layer_winners: list[ScoreCandidate]) -> str:
        ordered = sorted(layer_winners, key=lambda item: item.final_value, reverse=True)
        runner_up = ordered[1].final_value if len(ordered) > 1 else 0.0
        separation = selected.final_value - runner_up
        if selected.real_probability >= 0.095 and separation >= 0.01:
            return "A"
        if selected.real_probability >= 0.065 and separation >= 0.004:
            return "B"
        if selected.edge_ratio >= 1.18:
            return "C+"
        return "C"


def _exact_score_market_distribution(
    exact_score_odds: dict[str, float],
    score_grid: list[tuple[int, int]],
) -> dict[tuple[int, int], float]:
    raw: dict[tuple[int, int], float] = {}
    valid_scores = set(score_grid)
    for score, odds in exact_score_odds.items():
        parsed = _parse_score(score)
        if parsed in valid_scores and odds > 1.0:
            raw[parsed] = 1.0 / odds
    return _normalize(raw) if raw else {}


def _implied_probabilities(odds: dict[str, float]) -> dict[str, float]:
    mapped_keys = {
        "1": "home",
        "home": "home",
        "x": "draw",
        "draw": "draw",
        "2": "away",
        "away": "away",
    }
    raw: dict[str, float] = {}
    for key, value in odds.items():
        normalized_key = mapped_keys.get(key.lower())
        if normalized_key and value > 1.0:
            raw[normalized_key] = 1.0 / value
    return _normalize(raw)


def _parse_score(score: str) -> tuple[int, int] | None:
    if "-" not in score:
        return None
    left, right = score.split("-", 1)
    if not left.isdigit() or not right.isdigit():
        return None
    return int(left), int(right)


def _is_comfort_score(home: int, away: int) -> bool:
    return (home, away) in {(0, 0), (1, 0), (0, 1), (1, 1), (2, 0), (2, 1), (1, 2)}


def _is_broad_public_score(home: int, away: int) -> bool:
    return (home, away) in {
        (0, 0),
        (1, 0),
        (0, 1),
        (1, 1),
        (2, 0),
        (0, 2),
        (2, 1),
        (1, 2),
        (2, 2),
        (3, 1),
        (1, 3),
    }


def _is_scorebook_edge_score(home: int, away: int) -> bool:
    return (home, away) in {
        (3, 2),
        (2, 3),
        (4, 3),
        (3, 4),
        (4, 2),
        (2, 4),
        (5, 2),
        (2, 5),
        (5, 3),
        (3, 5),
        (4, 1),
        (1, 4),
        (5, 1),
        (1, 5),
        (6, 1),
        (1, 6),
        (3, 0),
        (0, 3),
        (4, 0),
        (0, 4),
        (5, 0),
        (0, 5),
        (6, 0),
        (0, 6),
    }


def _outcome(home: int, away: int) -> str:
    if home > away:
        return "home"
    if away > home:
        return "away"
    return "draw"


def _favorite_side(match: MatchInput, home_mu: float, away_mu: float) -> str:
    one_x_two = _implied_probabilities(match.market.one_x_two_odds)
    football_favorite = _football_favorite_side(match, home_mu, away_mu)
    if one_x_two:
        favorite = max(one_x_two, key=one_x_two.get)
        if (
            favorite != "draw"
            and football_favorite != "draw"
            and favorite != football_favorite
            and _market_independence_hint(match, home_mu, away_mu) >= 0.44
        ):
            return football_favorite
        if favorite != "draw":
            return favorite
    return football_favorite


def _market_favorite_side(match: MatchInput, home_mu: float, away_mu: float) -> str:
    one_x_two = _implied_probabilities(match.market.one_x_two_odds)
    if one_x_two:
        favorite = max(one_x_two, key=one_x_two.get)
        if favorite != "draw":
            return favorite
    return _football_favorite_side(match, home_mu, away_mu)


def _football_favorite_side(match: MatchInput, home_mu: float, away_mu: float) -> str:
    margin = home_mu - away_mu
    if margin >= 0.18:
        return "home"
    if margin <= -0.18:
        return "away"

    home_context = _side_context_strength(match, "home", margin)
    away_context = _side_context_strength(match, "away", margin)
    if home_context - away_context >= 0.16:
        return "home"
    if away_context - home_context >= 0.16:
        return "away"
    return "draw"


def _market_independence_hint(match: MatchInput, home_mu: float, away_mu: float) -> float:
    one_x_two = _implied_probabilities(match.market.one_x_two_odds)
    if not one_x_two:
        return _clamp(match.market.bookmaker_trap_risk, 0.0, 1.0) * 0.32

    market_favorite = max(one_x_two, key=one_x_two.get)
    if market_favorite == "draw":
        return _clamp(match.market.bookmaker_trap_risk, 0.0, 1.0) * 0.28

    football_favorite = _football_favorite_side(match, home_mu, away_mu)
    mu_margin = home_mu - away_mu
    market_support = _side_context_strength(match, market_favorite, mu_margin)
    football_support = (
        _side_context_strength(match, football_favorite, mu_margin)
        if football_favorite != "draw"
        else _clamp(0.54 - abs(mu_margin), 0.0, 0.54)
    )
    market_heat = _clamp((one_x_two[market_favorite] - 0.42) / 0.30, 0.0, 1.0)
    market_heat = _clamp(
        market_heat * 0.62
        + _side_money_heat(match, market_favorite) * 0.22
        + _clamp(match.market.favorite_money_heat, 0.0, 1.0) * 0.16,
        0.0,
        1.0,
    )
    conflict = 0.0
    if football_favorite != "draw" and football_favorite != market_favorite:
        conflict = 0.26
    weak_market = _clamp(0.64 - market_support, 0.0, 0.64)
    support_gap = _clamp(football_support - market_support, 0.0, 0.8)

    value = (
        conflict
        + market_heat * weak_market * 0.76
        + support_gap * 0.34
        + _clamp(match.market.bookmaker_trap_risk, 0.0, 1.0) * 0.30
    )
    if match.market.asian_handicap is not None and abs(match.market.asian_handicap) >= 1.25 and market_support < 0.56:
        value += 0.10
    return _clamp(value, 0.0, 1.0)


def _side_context_strength(match: MatchInput, side: str, mu_margin: float) -> float:
    if side == "home":
        raw = (
            mu_margin * 0.54
            + (match.home_form - match.away_form) * 0.16
            + (match.home_motivation - match.away_motivation) * 0.10
            + (_stake_pressure(match, "home") - _stake_pressure(match, "away")) * 0.13
            + (_survival_pressure(match, "home") - _survival_pressure(match, "away")) * 0.08
            + _clamp(match.home_big_win_need, 0.0, 1.0) * 0.05
            - match.home_absence_impact * 0.14
            + match.away_absence_impact * 0.12
            - match.home_rotation_risk * 0.08
            + match.away_rotation_risk * 0.06
            + (match.home_rest_edge - match.away_rest_edge) * 0.06
            - (0.08 if match.home_settled else 0.0)
        )
    elif side == "away":
        raw = (
            -mu_margin * 0.54
            + (match.away_form - match.home_form) * 0.16
            + (match.away_motivation - match.home_motivation) * 0.10
            + (_stake_pressure(match, "away") - _stake_pressure(match, "home")) * 0.13
            + (_survival_pressure(match, "away") - _survival_pressure(match, "home")) * 0.08
            + _clamp(match.away_big_win_need, 0.0, 1.0) * 0.05
            - match.away_absence_impact * 0.14
            + match.home_absence_impact * 0.12
            - match.away_rotation_risk * 0.08
            + match.home_rotation_risk * 0.06
            + (match.away_rest_edge - match.home_rest_edge) * 0.06
            - (0.08 if match.away_settled else 0.0)
        )
    else:
        raw = -abs(mu_margin) * 0.30 + _clamp(match.league_draw_bias, -0.4, 0.4) * 0.12
    return _clamp(0.50 + raw, 0.0, 1.0)


def _score_exit_lane_from_values(home: int, away: int, match: MatchInput, home_mu: float, away_mu: float) -> str:
    total = home + away
    outcome = _outcome(home, away)
    favorite = _favorite_side(match, home_mu, away_mu)

    if total <= 1:
        return "low_lock"
    if home == away:
        return "draw_ladder"
    if favorite != "draw" and outcome not in {favorite, "draw"}:
        return "upset_cold"
    if abs(home - away) >= 3:
        return "single_side_blowout"
    if total >= 4 and home > 0 and away > 0:
        return "exchange_war"
    if min(home, away) == 0:
        return "clean_sheet"
    if favorite != "draw" and outcome == favorite:
        margin = abs(home - away)
        depth = abs(match.market.asian_handicap) if match.market.asian_handicap is not None else 0.0
        if depth >= 1.0 and margin <= depth:
            return "favorite_noncover"
        if margin >= 2:
            return "favorite_cover"
        return "favorite_narrow"
    return "open_result"


def _score_family(home: int, away: int) -> str:
    if home == away:
        return "draw_ladder"
    if abs(home - away) == 1 and min(home, away) >= 1:
        return "narrow_result"
    if min(home, away) == 0:
        return "clean_sheet_break"
    if {home, away} in ({3, 2}, {4, 2}, {5, 2}, {4, 3}, {5, 3}, {6, 2}):
        return "exchange_war"
    return "one_side_break"


def _endgame_chaos(match: MatchInput) -> float:
    if not match.is_final_round:
        return _clamp(match.endgame_chaos, 0.0, 1.0) * 0.35

    pressure_gap = abs(_stake_pressure(match, "home") - _stake_pressure(match, "away"))
    locked_or_draw = 0.0
    if match.home_settled or match.away_settled:
        locked_or_draw += 0.18
    if match.home_draw_sufficient or match.away_draw_sufficient:
        locked_or_draw += 0.10

    raw = (
        _clamp(match.endgame_chaos, 0.0, 1.0) * 0.42
        + max(_clamp(match.home_collapse_risk, 0.0, 1.0), _clamp(match.away_collapse_risk, 0.0, 1.0)) * 0.26
        + max(_clamp(match.home_celebration_risk, 0.0, 1.0), _clamp(match.away_celebration_risk, 0.0, 1.0)) * 0.16
        + max(_survival_pressure(match, "home"), _survival_pressure(match, "away")) * 0.18
        + _clamp(pressure_gap, 0.0, 1.0) * 0.12
        + locked_or_draw
    )
    return _clamp(raw, 0.0, 1.0)


def _survival_pressure(match: MatchInput, side: str) -> float:
    if side == "home":
        explicit = _clamp(match.home_survival_pressure, 0.0, 1.0)
        table_pressure = _clamp(match.home_table_pressure, 0.0, 1.0)
        motivation = _clamp(match.home_motivation, 0.0, 1.0)
        settled = match.home_settled
        draw_sufficient = match.home_draw_sufficient
    elif side == "away":
        explicit = _clamp(match.away_survival_pressure, 0.0, 1.0)
        table_pressure = _clamp(match.away_table_pressure, 0.0, 1.0)
        motivation = _clamp(match.away_motivation, 0.0, 1.0)
        settled = match.away_settled
        draw_sufficient = match.away_draw_sufficient
    else:
        return 0.0

    inferred = 0.0
    if match.is_final_round and not settled and not draw_sufficient and table_pressure >= 0.70:
        inferred = 0.28 + table_pressure * 0.22 + motivation * 0.12
    raw = max(explicit, inferred)
    if settled:
        raw *= 0.35
    if draw_sufficient:
        raw *= 0.55
    return _clamp(raw, 0.0, 1.0)


def _stake_pressure(match: MatchInput, side: str) -> float:
    if side == "home":
        raw = (
            match.home_motivation * 0.42
            + match.home_table_pressure * 0.46
            + match.home_survival_pressure * 0.24
            + match.home_big_win_need * 0.20
            - (0.34 if match.home_settled else 0.0)
            - (0.12 if match.home_draw_sufficient else 0.0)
        )
    else:
        raw = (
            match.away_motivation * 0.42
            + match.away_table_pressure * 0.46
            + match.away_survival_pressure * 0.24
            + match.away_big_win_need * 0.20
            - (0.34 if match.away_settled else 0.0)
            - (0.12 if match.away_draw_sufficient else 0.0)
        )
    return _clamp(raw, -1.0, 1.0)


def _side_money_heat(match: MatchInput, side: str) -> float:
    if side == "home":
        return _clamp(match.market.home_money_heat, 0.0, 1.0)
    if side == "away":
        return _clamp(match.market.away_money_heat, 0.0, 1.0)
    return _clamp(match.market.draw_money_heat, 0.0, 1.0)


def _underdog_money_heat(match: MatchInput, favorite: str) -> float:
    if favorite == "home":
        return _clamp(match.market.away_money_heat, 0.0, 1.0)
    if favorite == "away":
        return _clamp(match.market.home_money_heat, 0.0, 1.0)
    return max(
        _clamp(match.market.home_money_heat, 0.0, 1.0),
        _clamp(match.market.away_money_heat, 0.0, 1.0),
    )


def _score_repeat_limit(score: str, slate_size: int) -> int:
    if slate_size <= 3:
        return slate_size
    if score == "2-2":
        return 1
    if score in {"2-1", "1-2", "3-1", "1-3"} and slate_size <= 12:
        return 1
    if score in {"0-0", "1-1"} and slate_size <= 12:
        return 1
    if slate_size <= 7:
        return 1
    if slate_size <= 12:
        return 2
    return max(2, round(slate_size * 0.18))


def _poisson(k: int, lam: float) -> float:
    return math.exp(-lam) * (lam**k) / math.factorial(k)


def _light_candidate(home: int, away: int, edge_ratio: float) -> ScoreCandidate:
    return ScoreCandidate(
        score=f"{home}-{away}",
        home_goals=home,
        away_goals=away,
        total_goals=home + away,
        real_probability=0.0,
        market_probability=0.0,
        edge_ratio=edge_ratio,
        edge_gap=0.0,
        structure_score=0.0,
        cold_bonus=1.0,
        conflict_penalty=0.0,
        final_value=0.0,
        reasons=(),
    )


def _with_extra_reasons(candidate: ScoreCandidate, *extra_reasons: str) -> ScoreCandidate:
    return ScoreCandidate(
        score=candidate.score,
        home_goals=candidate.home_goals,
        away_goals=candidate.away_goals,
        total_goals=candidate.total_goals,
        real_probability=candidate.real_probability,
        market_probability=candidate.market_probability,
        edge_ratio=candidate.edge_ratio,
        edge_gap=candidate.edge_gap,
        structure_score=candidate.structure_score,
        cold_bonus=candidate.cold_bonus,
        conflict_penalty=candidate.conflict_penalty,
        final_value=candidate.final_value,
        reasons=tuple(candidate.reasons + tuple(extra_reasons)),
    )


def _normalize(values: dict) -> dict:
    total = sum(values.values())
    if total <= 0:
        return values
    return {key: value / total for key, value in values.items()}


def _safe_ratio(numerator: float, denominator: float) -> float:
    return _clamp(numerator, 0.25, 2.8) / _clamp(denominator, 0.25, 2.8)


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))
