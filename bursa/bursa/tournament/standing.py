"""Where you stand, and what it would take to win.

A winner-takes-all contest is not an investing problem. Finishing 4th of 400
pays exactly what finishing 400th pays, so the objective is P(finishing
first), not expected return -- and those call for opposite behaviour. A
steady, well-calibrated 6% is an excellent investing outcome and a certain
loss here.

The useful consequence: **the target is derivable from the entrant count
alone.** You do not need a leaderboard. If 400 people hold roughly
diversified portfolios, the winner is the maximum of 400 draws from that
distribution, and order statistics give you the number.

Everything here is deliberately simple and assumption-visible. It is a
navigation instrument, not a forecast.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy import stats

EULER_MASCHERONI = 0.5772156649015329


@dataclass(frozen=True)
class Contest:
    """The contest's parameters, and your assumptions about the field.

    n_entrants  : how many people are competing.
    field_vol   : the typical entrant's volatility over the FULL contest
        period (not annualised). ~5% for a month of diversified holdings;
        raise it if you think the field is sophisticated, which shrinks your
        edge sharply.
    total_days  : trading days in the contest.
    """
    n_entrants: int
    field_vol: float = 0.05
    total_days: int = 21

    def __post_init__(self):
        if self.n_entrants < 2:
            raise ValueError("a contest needs at least 2 entrants")
        if self.field_vol <= 0:
            raise ValueError("field_vol must be positive")


def expected_winning_score(contest: Contest) -> float:
    """Expected maximum of n draws -- i.e. the score that wins.

    Uses the standard asymptotic approximation for the expected maximum of n
    standard normals, scaled by the field's volatility.
    """
    n = contest.n_entrants
    g = EULER_MASCHERONI
    z = (1 - g) * stats.norm.ppf(1 - 1.0 / n) + g * stats.norm.ppf(
        1 - 1.0 / (n * math.e))
    return float(contest.field_vol * z)


def winning_score_quantiles(contest: Contest,
                            quantiles=(0.25, 0.5, 0.75),
                            trials: int = 20_000,
                            seed: int = 0) -> dict[float, float]:
    """Simulated distribution of the winning score.

    The expectation alone understates how much the target moves run to run.
    """
    rng = np.random.default_rng(seed)
    maxima = rng.normal(0.0, contest.field_vol,
                        size=(trials, contest.n_entrants)).max(axis=1)
    return {q: float(np.quantile(maxima, q)) for q in quantiles}


def prob_win(my_return: float,
             my_vol_remaining: float,
             contest: Contest,
             trials: int = 20_000,
             seed: int = 0) -> float:
    """P(finishing first) from your current position.

    my_return         : where you are now, as a fraction.
    my_vol_remaining  : the volatility you will run over the days that remain.
    """
    rng = np.random.default_rng(seed)
    mine = my_return + rng.normal(0.0, max(my_vol_remaining, 1e-9), trials)
    field = rng.normal(0.0, contest.field_vol,
                       size=(trials, contest.n_entrants - 1)).max(axis=1)
    return float((mine > field).mean())


def required_vol(gap: float, confidence: float = 0.33) -> float:
    """Volatility needed to close `gap` with roughly `confidence` probability.

    Inverts the normal: to land `gap` above your current position with
    probability p, you need sigma = gap / z(1-p).
    """
    if gap <= 0:
        return 0.0
    z = stats.norm.ppf(1.0 - confidence)
    return float(gap / z) if z > 0 else float("inf")


@dataclass
class Standing:
    """A read on the contest at one moment, with a recommendation."""
    my_return: float
    days_left: int
    target: float
    gap: float
    required_vol_remaining: float
    required_daily_vol: float
    p_win_current: float
    p_win_aggressive: float
    verdict: str

    def __str__(self) -> str:
        return (
            f"position {self.my_return:+.2%} | {self.days_left}d left | "
            f"target {self.target:+.2%} | gap {self.gap:+.2%}\n"
            f"  P(win) holding current risk : {self.p_win_current:6.1%}\n"
            f"  P(win) if fully concentrated: {self.p_win_aggressive:6.1%}\n"
            f"  need ~{self.required_vol_remaining:.1%} vol over the remaining "
            f"{self.days_left} days ({self.required_daily_vol:.2%}/day)\n"
            f"  >> {self.verdict}"
        )


def assess(my_return: float,
           days_left: int,
           contest: Contest,
           current_vol_annual: float = 0.20,
           aggressive_vol_annual: float = 0.70) -> Standing:
    """Where you stand and what to do about it.

    current_vol_annual    : annualised vol of what you hold now.
    aggressive_vol_annual : annualised vol you could reach by concentrating
        into one or two high-beta names.
    """
    if days_left < 0:
        raise ValueError("days_left cannot be negative")

    target = expected_winning_score(contest)
    gap = target - my_return

    scale = math.sqrt(max(days_left, 0) / 252.0)
    vol_now = current_vol_annual * scale
    vol_max = aggressive_vol_annual * scale

    need = required_vol(gap)
    need_daily = need / math.sqrt(days_left) if days_left > 0 else float("inf")

    p_now = prob_win(my_return, vol_now, contest)
    p_agg = prob_win(my_return, vol_max, contest)

    if days_left == 0:
        verdict = "contest over"
    elif gap <= 0:
        verdict = ("AHEAD OF TARGET -- cut risk. Protecting a winning "
                   "position is worth more than extending it.")
    elif need <= vol_now:
        verdict = "ON TRACK -- current risk is sufficient. Do not add."
    elif need <= vol_max:
        verdict = (f"BEHIND -- concentrate. Need ~{need:.1%} vol, currently "
                   f"running ~{vol_now:.1%}.")
    else:
        verdict = (f"FAR BEHIND -- the gap needs ~{need:.1%} vol and the most "
                   f"you can reach is ~{vol_max:.1%}. Maximum concentration is "
                   f"the only option with any chance; a sensible portfolio has "
                   f"already lost.")

    return Standing(my_return, days_left, target, gap, need, need_daily,
                    p_now, p_agg, verdict)


def strategy_table(contest: Contest,
                   my_vols=((0.05, "diversified, 15 names"),
                            (0.09, "focused, 5 names"),
                            (0.15, "concentrated, 2-3 names"),
                            (0.25, "one high-beta name")),
                   my_edge: float = 0.0,
                   trials: int = 20_000,
                   seed: int = 0):
    """P(win) by portfolio concentration. The table that sets your sizing.

    Run it once at the start of the contest. It is the argument for holding
    three names instead of fifteen, and it is quantitative rather than
    rhetorical.
    """
    import pandas as pd
    rows = []
    for vol, label in my_vols:
        rows.append({
            "portfolio": label,
            "period_vol": vol,
            "p_win": prob_win(my_edge, vol, contest, trials, seed),
        })
    out = pd.DataFrame(rows)
    out["vs_random"] = out["p_win"] / (1.0 / contest.n_entrants)
    return out.set_index("portfolio")
