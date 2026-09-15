"""Contest simulator. How concentrated should you be to WIN, not to earn?

`standing.py` answers the same question analytically, assuming the field's
scores are normal. They are not. Bursa small-cap returns over a month are
sharply fat-tailed and strongly cross-correlated, and both of those change
the answer -- fat tails because the winning score is further out than a
normal says, correlation because when the market rips the whole field rips
with you and your relative position barely moves.

So this simulates the contest instead, drawing from what actually happened.

## The model

One trial is a whole contest:

1. Pick a real historical window of `horizon` trading days from the panel.
2. Every entrant, including you, picks `k` names from those eligible at the
   START of that window and holds equal-weighted to the end.
3. Score everyone. Count whether you came first.

Repeat some thousands of times. `p_win` is the fraction you won.

Drawing the whole field from the SAME window is the point. It preserves the
market move common to everyone and the correlation structure between names,
which is what makes a contest different from n independent bets. Simulating
each entrant against an independently-drawn market would make winning look
far easier than it is.

## What this does not model

* **Trading during the contest.** Everyone buys once and holds. Real
  entrants will churn, which raises the field's variance and therefore the
  winning score. Treat `p_win` here as an upper bound.
* **Skill in the field.** Opponents pick at random. If any of the other 399
  can actually pick stocks, your odds are worse than this says.
* **Survivorship.** The universe is the Nov-2025 SC list, so every name in
  it survived to 2025. Historical windows therefore contain no company that
  went to zero -- which understates the left tail for everyone, you included.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# Selection rules an entrant can use. Each takes the eligible names and the
# history available at the contest start, and returns an ordering; the entrant
# takes the first k. `random` is the field's default and the honest baseline.
SelectionRule = str
RULES = ("random", "high_vol", "low_vol", "momentum", "reversal")


@dataclass
class ContestConfig:
    n_entrants: int = 400
    horizon: int = 21              # trading days
    my_k: int = 5                  # how many names YOU hold
    field_k: int = 10              # how many the typical opponent holds
    my_rule: SelectionRule = "random"
    field_rule: SelectionRule = "random"
    n_trials: int = 2000
    lookback: int = 60             # bars of history a rule may use
    seed: int = 0
    # A real field is not uniform. Some entrants punt on one stock, most hold
    # a handful, a few hold twenty. Supply {k: weight} to model that; it
    # overrides field_k. This matters more than any other input: your entire
    # advantage is being MORE concentrated than the field, so assuming they
    # are all diversified when they are not overstates P(win) by 20x.
    field_k_mix: dict[int, float] | None = None


@dataclass
class ContestResult:
    p_win: float
    p_top3: float
    my_median_return: float
    my_mean_return: float
    my_p95_return: float
    winning_score_median: float
    n_trials: int
    config: ContestConfig
    my_returns: np.ndarray = field(default_factory=lambda: np.array([]))

    def __str__(self) -> str:
        c = self.config
        return (f"  k={c.my_k:<3} rule={c.my_rule:<10} "
                f"P(win) {self.p_win:6.2%}   P(top3) {self.p_top3:6.2%}   "
                f"median {self.my_median_return:+7.2%}  "
                f"p95 {self.my_p95_return:+7.2%}")


def _window_returns(prices: pd.DataFrame, start_i: int,
                    horizon: int) -> np.ndarray:
    """Total return of every column over [start_i, start_i + horizon]."""
    p0 = prices.iloc[start_i].to_numpy(dtype=float)
    p1 = prices.iloc[start_i + horizon].to_numpy(dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return p1 / p0 - 1.0


def _rank_by_rule(rule: SelectionRule, prices: pd.DataFrame, start_i: int,
                  lookback: int, rng: np.random.Generator) -> np.ndarray:
    """Score each column; higher is picked first. NaN means not pickable.

    Every rule uses ONLY data strictly before `start_i`, which is the bar the
    contest opens on. A rule that peeks at the contest window would make any
    result here meaningless.
    """
    n = prices.shape[1]
    if rule == "random":
        return rng.random(n)

    lo = max(0, start_i - lookback)
    hist = prices.iloc[lo:start_i]
    if len(hist) < 10:
        return rng.random(n)

    rets = hist.pct_change()
    if rule in ("high_vol", "low_vol"):
        v = rets.std().to_numpy(dtype=float)
        return v if rule == "high_vol" else -v
    if rule in ("momentum", "reversal"):
        m = (hist.iloc[-1] / hist.iloc[0] - 1.0).to_numpy(dtype=float)
        return m if rule == "momentum" else -m
    raise ValueError(f"unknown rule {rule!r}")


def _pick(scores: np.ndarray, valid: np.ndarray, k: int,
          rng: np.random.Generator) -> np.ndarray:
    """Indices of the top k valid names, ties broken at random."""
    s = np.where(valid, scores, -np.inf)
    # Jitter so equal scores do not always resolve to the same low indices.
    s = s + rng.random(len(s)) * 1e-9
    k = min(k, int(valid.sum()))
    if k <= 0:
        return np.array([], dtype=int)
    return np.argpartition(-s, k - 1)[:k]


def run_contest(panel: pd.DataFrame,
                cfg: ContestConfig = ContestConfig(),
                price_col: str = "adj_close",
                eligible_col: str = "eligible",
                verbose: bool = False) -> ContestResult:
    """Simulate `n_trials` contests and report how often you finish first."""
    rng = np.random.default_rng(cfg.seed)

    use = panel
    if eligible_col in panel.columns:
        use = panel[panel[eligible_col]]
    prices = use.pivot_table(index="date", columns="ticker",
                             values=price_col, aggfunc="last")
    prices = prices.sort_index()

    n_dates = len(prices)
    first_start = cfg.lookback + 1
    last_start = n_dates - cfg.horizon - 1
    if last_start <= first_start:
        raise ValueError("panel too short for this horizon and lookback")

    my_rets = np.empty(cfg.n_trials)
    win_scores = np.empty(cfg.n_trials)
    wins = 0.0
    top3 = 0.0

    for t in range(cfg.n_trials):
        start_i = int(rng.integers(first_start, last_start))
        fwd = _window_returns(prices, start_i, cfg.horizon)
        # A name is pickable only if it has a price at both ends of the
        # window. Anything else would be a position you could not have held.
        valid = np.isfinite(fwd)
        if valid.sum() < max(cfg.my_k, cfg.field_k) + 5:
            my_rets[t] = np.nan
            win_scores[t] = np.nan
            continue

        my_scores = _rank_by_rule(cfg.my_rule, prices, start_i,
                                  cfg.lookback, rng)
        mine = _pick(my_scores, valid & np.isfinite(my_scores), cfg.my_k, rng)
        my_r = float(np.mean(fwd[mine])) if len(mine) else 0.0

        # The field. Drawn in one batch rather than looped, which matters:
        # 400 entrants x 2000 trials is 800,000 portfolios.
        field_scores = _rank_by_rule(cfg.field_rule, prices, start_i,
                                     cfg.lookback, rng)
        pool = np.flatnonzero(valid & np.isfinite(field_scores))
        if cfg.field_rule != "random":
            # Everyone using the same rule would pick identical portfolios,
            # which is not a contest. Rank the pool, then let each opponent
            # sample from the top half of it.
            order = pool[np.argsort(-field_scores[pool])]
            pool = order[:max(cfg.field_k, len(order) // 2)]
        n_opp = cfg.n_entrants - 1
        if cfg.field_k_mix:
            # Opponents at different concentrations. Each k group is drawn as
            # its own block, then all scores are concatenated.
            ks = np.array(sorted(cfg.field_k_mix), dtype=int)
            w = np.array([cfg.field_k_mix[int(x)] for x in ks], dtype=float)
            w = w / w.sum()
            counts = rng.multinomial(n_opp, w)
            parts = []
            for kk, cnt in zip(ks, counts):
                if cnt == 0:
                    continue
                kk = min(int(kk), len(pool))
                pk = rng.choice(pool, size=(int(cnt), kk), replace=True)
                parts.append(fwd[pk].mean(axis=1))
            field_r = np.concatenate(parts)
        else:
            k = min(cfg.field_k, len(pool))
            picks = rng.choice(pool, size=(n_opp, k), replace=True)
            # replace=True within a row can repeat a name; equal-weighting a
            # duplicate is the same as overweighting it, which is a real thing
            # an entrant can do, so it is left alone.
            field_r = fwd[picks].mean(axis=1)

        my_rets[t] = my_r
        all_scores = np.append(field_r, my_r)
        win_scores[t] = float(np.max(all_scores))

        beaten_by = int((field_r > my_r).sum())
        # Ties are shared, not won. Counting "nobody strictly beat me" as a
        # win reports 50% in a flat market where every entrant scores zero --
        # which is how this was first caught. With m opponents level with you
        # at the top, your share of first place is 1/(m+1).
        tied = int((field_r == my_r).sum())
        if beaten_by == 0:
            wins += 1.0 / (tied + 1)
        if beaten_by < 3:
            top3 += 1.0

    ok = np.isfinite(my_rets)
    n_ok = int(ok.sum())
    res = ContestResult(
        p_win=wins / n_ok if n_ok else float("nan"),
        p_top3=top3 / n_ok if n_ok else float("nan"),
        my_median_return=float(np.nanmedian(my_rets)),
        my_mean_return=float(np.nanmean(my_rets)),
        my_p95_return=float(np.nanpercentile(my_rets[ok], 95)),
        winning_score_median=float(np.nanmedian(win_scores)),
        n_trials=n_ok,
        config=cfg,
        my_returns=my_rets[ok],
    )
    if verbose:
        print(res)
    return res


def concentration_sweep(panel: pd.DataFrame,
                        ks: tuple[int, ...] = (1, 2, 3, 5, 8, 12, 20, 40),
                        base: ContestConfig = ContestConfig(),
                        verbose: bool = True) -> pd.DataFrame:
    """P(win) as a function of how many names you hold. The core table.

    Everything else about the contest is held fixed, so the only thing
    varying is your concentration.
    """
    rows = []
    for k in ks:
        cfg = ContestConfig(**{**base.__dict__, "my_k": k})
        r = run_contest(panel, cfg)
        rows.append({"k": k, "p_win": r.p_win, "p_top3": r.p_top3,
                     "median": r.my_median_return, "mean": r.my_mean_return,
                     "p95": r.my_p95_return,
                     "p_lose_half": float((r.my_returns < -0.5).mean())})
        if verbose:
            print(r)
    return pd.DataFrame(rows).set_index("k")


def rule_sweep(panel: pd.DataFrame,
               rules: tuple[str, ...] = RULES,
               base: ContestConfig = ContestConfig(),
               verbose: bool = True) -> pd.DataFrame:
    """P(win) by selection rule, at a fixed concentration.

    The question this answers is narrow and worth stating: given no ability
    to predict returns, does TILTING the portfolio -- towards volatile names,
    or recent winners -- raise the chance of finishing first? That is a
    different question from whether the tilt raises expected return, and it
    can have a different answer.
    """
    rows = []
    for rule in rules:
        cfg = ContestConfig(**{**base.__dict__, "my_rule": rule})
        r = run_contest(panel, cfg)
        rows.append({"rule": rule, "p_win": r.p_win, "p_top3": r.p_top3,
                     "median": r.my_median_return, "p95": r.my_p95_return})
        if verbose:
            print(r)
    return pd.DataFrame(rows).set_index("rule")
