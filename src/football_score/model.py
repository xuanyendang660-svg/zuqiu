"""Joint negative-binomial / beta-binomial score law and two independent baselines."""

from __future__ import annotations

import math

import numpy as np
from scipy.optimize import minimize, least_squares
from scipy.special import expit, gammaln
from scipy.stats import nbinom, poisson


REQUIRED = (
    "league_avg_goals_pre", "league_home_goals_pre", "league_away_goals_pre",
    "home_recent8_gf", "home_recent8_ga", "away_recent8_gf", "away_recent8_ga",
    "home_elo_pre", "away_elo_pre",
)


def features(row: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Use prior-match rolling values; never fall back silently on a missing input."""
    try:
        z = {k: float(row[k]) for k in REQUIRED}
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"DATA_BLOCKED: missing or invalid prior feature: {exc}") from exc
    if any(not math.isfinite(v) for v in z.values()) or min(z.values()) < 0:
        raise ValueError("DATA_BLOCKED: prior features must be finite and nonnegative")
    total = z["league_avg_goals_pre"]
    home_base = z["league_home_goals_pre"]
    away_base = z["league_away_goals_pre"]
    if min(total, home_base, away_base) <= 0:
        raise ValueError("DATA_BLOCKED: league scoring baselines must be positive")
    home_rate = max(0.15, (z["home_recent8_gf"] + z["away_recent8_ga"]) / 2)
    away_rate = max(0.15, (z["away_recent8_gf"] + z["home_recent8_ga"]) / 2)
    h = math.log(home_rate / home_base)
    a = math.log(away_rate / away_base)
    elo = (z["home_elo_pre"] - z["away_elo_pre"]) / 400
    x_mu = np.array([1, math.log(total / 2.65), h, a, abs(elo)])
    x_p = np.array([1, h - a, elo])
    x_home = np.array([1, math.log(max(.15, z["home_recent8_gf"]) / home_base),
                       math.log(max(.15, z["away_recent8_ga"]) / home_base), elo])
    x_away = np.array([1, math.log(max(.15, z["away_recent8_gf"]) / away_base),
                       math.log(max(.15, z["home_recent8_ga"]) / away_base), -elo])
    return x_mu, x_p, x_home, x_away


def _design(rows: list[dict]):
    block = [features(r) for r in rows]
    return [np.stack([x[i] for x in block]) for i in range(4)]


def _joint_parameters(row: dict, theta: list[float] | np.ndarray):
    x_mu, x_p, _, _ = features(row)
    theta = np.asarray(theta)
    mu = float(row["league_avg_goals_pre"]) * math.exp(float(x_mu @ theta[:5]))
    p = float(expit(math.log(float(row["league_home_goals_pre"]) /
                             float(row["league_away_goals_pre"])) + x_p @ theta[5:8]))
    return mu, p, math.exp(float(theta[8])), math.exp(float(theta[9]))


def fit_joint(rows: list[dict]) -> list[float]:
    if len(rows) < 100:
        raise ValueError("MODEL_UNAVAILABLE: at least 100 earlier matches are needed")
    xm, xp, _, _ = _design(rows)
    h = np.array([int(r["actual_home_goals"]) for r in rows])
    a = np.array(
        [int(r["actual_away_goals"]) for r in rows])
    t = h + a
    baseline = np.array([float(r["league_avg_goals_pre"]) for r in rows])
    logratio = np.log(np.array([float(r["league_home_goals_pre"]) /
                                float(r["league_away_goals_pre"]) for r in rows]))

    def loss(theta):
        mu = np.exp(np.clip(np.log(baseline) + xm @ theta[:5], -4, 4))
        p = np.clip(expit(logratio + xp @ theta[5:8]), 1e-6, 1 - 1e-6)
        kappa, phi = np.exp(theta[8:])
        alpha, beta = p * phi, (1 - p) * phi
        nb = (gammaln(t + kappa) - gammaln(kappa) - gammaln(t + 1)
              + kappa * np.log(kappa / (kappa + mu))
              + t * np.log(mu / (kappa + mu)))
        bb = (gammaln(t + 1) - gammaln(h + 1) - gammaln(a + 1)
              + gammaln(h + alpha) + gammaln(a + beta) - gammaln(t + phi)
              - gammaln(alpha) - gammaln(beta) + gammaln(phi))
        return float(-np.mean(nb + bb) + .0005 * np.sum(theta[:8] ** 2))

    start = np.array([0, 0, .25, .25, 0, 0, .4, .4, math.log(10), math.log(12)])
    bounds = [(-2, 2)] * 8 + [(math.log(.4), math.log(150))] * 2
    result = minimize(loss, start, method="L-BFGS-B", bounds=bounds)
    if not result.success or not math.isfinite(result.fun):
        raise RuntimeError("joint model optimizer failed: " + result.message)
    return result.x.tolist()


def _normalize(scores: dict[tuple[int, int], float]):
    mass = sum(scores.values())
    if mass <= 0:
        raise RuntimeError("empty probability law")
    return {score: value / mass for score, value in scores.items()}


def joint_distribution(row: dict, theta: list[float], tolerance: float = 1e-8):
    mu, p, kappa, phi = _joint_parameters(row, theta)
    if not (0 < mu < 100 and 0 < p < 1 and kappa > 0 and phi > 0):
        raise ValueError("MODEL_UNAVAILABLE: invalid joint parameters")
    alpha, beta = p * phi, (1 - p) * phi
    out = {}
    for t in range(201):
        nb = float(nbinom.pmf(t, kappa, kappa / (kappa + mu)))
        for h in range(t + 1):
            a = t - h
            log_bb = (math.lgamma(t + 1) - math.lgamma(h + 1) - math.lgamma(a + 1)
                      + math.lgamma(h + alpha) + math.lgamma(a + beta)
                      - math.lgamma(t + phi) - math.lgamma(alpha) - math.lgamma(beta)
                      + math.lgamma(phi))
            out[h, a] = nb * math.exp(log_bb)
        if t >= 5 and nbinom.sf(t, kappa, kappa / (kappa + mu)) < tolerance:
            return _normalize(out)
    raise RuntimeError("joint distribution tail did not converge")


def _dc_tau(h: int, a: int, lh: float, la: float, rho: float) -> float:
    if h == 0 and a == 0:
        return 1 - lh * la * rho
    if h == 0 and a == 1:
        return 1 + lh * rho
    if h == 1 and a == 0:
        return 1 + la * rho
    if h == 1 and a == 1:
        return 1 - rho
    return 1.0


def fit_dc(rows: list[dict]) -> list[float]:
    _, _, xh, xa = _design(rows)
    h = np.array([int(r["actual_home_goals"]) for r in rows])
    a = np.array(
        [int(r["actual_away_goals"]) for r in rows])
    bh = np.array([float(r["league_home_goals_pre"]) for r in rows])
    ba = np.array(
        [float(r["league_away_goals_pre"]) for r in rows])
    dates = np.array([np.datetime64(r["date"]) for r in rows])
    age = (np.max(dates) - dates).astype("timedelta64[D]").astype(float)
    weights = np.exp(-math.log(2) * age / 730)

    def loss(theta):
        lh = np.exp(np.clip(np.log(bh) + xh @ theta[:4], -3, math.log(5)))
        la = np.exp(np.clip(np.log(ba) + xa @ theta[4:8], -3, math.log(5)))
        rho = theta[8]
        tau = np.ones(len(h))
        for mask, val in [((h == 0) & (a == 0), 1 - lh * la * rho),
                          ((h == 0) & (a == 1), 1 + lh * rho),
                          ((h == 1) & (a == 0), 1 + la * rho),
                          ((h == 1) & (a == 1), 1 - rho)]:
            tau[mask] = val[mask] if isinstance(val, np.ndarray) else val
        ll = (h * np.log(lh) - lh - gammaln(h + 1) +
              a * np.log(la) - la - gammaln(a + 1) + np.log(tau))
        return float(-np.average(ll, weights=weights) + .0005 * np.sum(theta[:8] ** 2))

    result = minimize(loss, np.zeros(9), method="L-BFGS-B",
                      bounds=[(-2, 2)] * 8 + [(-.03, .03)])
    if not result.success or not math.isfinite(result.fun):
        raise RuntimeError("DC baseline optimizer failed: " + result.message)
    return result.x.tolist()


def poisson_distribution(lh: float, la: float, rho: float = 0.0, tolerance: float = 1e-8):
    out = {}
    for n in range(3, 201):
        if poisson.sf(n, lh) + poisson.sf(n, la) < tolerance:
            break
    else:
        raise RuntimeError("Poisson distribution tail did not converge")
    for h in range(n + 1):
        for a in range(n + 1):
            out[h, a] = (float(poisson.pmf(h, lh) * poisson.pmf(a, la)) *
                         _dc_tau(h, a, lh, la, rho))
    return _normalize(out)


def dc_distribution(row: dict, theta: list[float]):
    _, _, xh, xa = features(row)
    lh = float(row["league_home_goals_pre"]) * math.exp(float(xh @ theta[:4]))
    la = float(row["league_away_goals_pre"]) * math.exp(float(xa @ theta[4:8]))
    return poisson_distribution(lh, la, theta[8])


def devig(odds: list[float]) -> list[float]:
    if any(not math.isfinite(x) or x <= 1 for x in odds):
        raise ValueError("invalid decimal odds")
    inv = [1 / x for x in odds]
    return [x / sum(inv) for x in inv]


def split_line(line: float) -> tuple[float, ...]:
    """Return half-stake component lines, including quarter Asian lines."""
    if not math.isfinite(line) or abs(line * 4 - round(line * 4)) > 1e-8:
        raise ValueError("Asian line must be in quarter-goal increments")
    if round(line * 4) % 2:
        low = math.floor(line * 2) / 2
        return low, low + .5
    return (line,)


def settlement(margin: int, line: float) -> tuple[float, float]:
    """(win stake fraction, loss stake fraction); remainder is returned stake."""
    components = split_line(line)
    wins = sum(margin + component > 0 for component in components) / len(components)
    losses = sum(margin + component < 0 for component in components) / len(components)
    return wins, losses


def fair_cover(scores: dict, line: float, market: str = "total") -> float:
    wins = losses = 0.0
    for (h, a), prob in scores.items():
        margin = h + a if market == "total" else h - a
        win, lose = settlement(margin, -line if market == "total" else line)
        wins += prob * win
        losses += prob * lose
    return wins / (wins + losses)


def _market_target(quotes: dict):
    one = quotes["one_x_two_odds"]
    target = devig([float(one[k]) for k in ("home", "draw", "away")])
    over, under = quotes["over_under_odds"]["over"], quotes["over_under_odds"]["under"]
    target_over = devig([float(over), float(under)])[0]
    line = float(quotes["total_line"])
    split_line(line)
    return target, target_over, line


def market_distribution(quotes: dict):
    target, target_over, line = _market_target(quotes)

    def residual(log_lambdas):
        scores = poisson_distribution(*np.exp(log_lambdas))
        outcomes = [0., 0., 0.]
        for (h, a), prob in scores.items():
            outcomes[0 if h > a else 2 if a > h else 1] += prob
        return [*(outcomes[i] - target[i] for i in range(3)),
                fair_cover(scores, line) - target_over]

    result = least_squares(residual, [math.log(1.4), math.log(1.2)],
                           bounds=([math.log(.08)] * 2, [math.log(6)] * 2),
                           xtol=1e-7, ftol=1e-7, gtol=1e-7)
    if not result.success:
        raise RuntimeError("market score fit failed")
    scores = poisson_distribution(*np.exp(result.x))
    check = None
    if "asian_handicap" in quotes and "asian_home_odds" in quotes:
        check = (fair_cover(scores, float(quotes["asian_handicap"]), "handicap") -
                 devig([float(quotes["asian_home_odds"]),
                        float(quotes["asian_away_odds"])])[0])
    return scores, {"fit_residuals": list(residual(result.x)), "asian_check_residual": check}


def fuse(football: dict, market: dict | None, weight: float):
    if market is None:
        return football
    return _normalize({k: weight * football.get(k, 0) + (1 - weight) * market.get(k, 0)
                       for k in football.keys() | market.keys()})


def top_score(scores: dict):
    # Fixed tie break, with no comfort, popularity, or tail multiplier.
    return min(scores, key=lambda s: (-scores[s], s[0], s[1]))
