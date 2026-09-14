"""Cross-sectional ranking strategies.

Structurally identical to LIPS: score every candidate on several factors,
percentile-rank each factor across the cohort, weight, sum, take the top N.
Swap plants for stocks and the machinery is the same -- including the reason
for percentile-ranking, which is that raw factor values live on wildly
different scales and cannot be added until they are all on 0-100.

Every feature here is computed from history up to and including the decision
date, and the harness fills on the following bar. Nothing in this file may
look forward; the tests check that it does not.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------
# Factors. Each returns one value per ticker, computed from `history` only.
# --------------------------------------------------------------------------
def momentum(history: pd.DataFrame, lookback: int = 126, skip: int = 21,
             price_col: str = "adj_close") -> pd.Series:
    """Trailing return over `lookback` bars, excluding the most recent `skip`.

    The skip is not decoration. Momentum and short-term reversal point in
    opposite directions, and including the last month mixes the two into
    something that measures neither.
    """
    px = history.pivot_table(index="date", columns="ticker", values=price_col,
                             aggfunc="last").sort_index()
    if len(px) < lookback + skip:
        return pd.Series(dtype=float)
    recent = px.iloc[-(skip + 1)] if skip > 0 else px.iloc[-1]
    past = px.iloc[-(lookback + skip)]
    return (recent / past - 1.0).replace([np.inf, -np.inf], np.nan).dropna()


def reversal(history: pd.DataFrame, lookback: int = 21,
             price_col: str = "adj_close") -> pd.Series:
    """Negative of the recent return: biggest losers score highest.

    The opposite bet to momentum, at a shorter horizon. Which one works on
    Bursa is an empirical question, which is the point of testing both rather
    than picking the one that sounds more sophisticated.
    """
    px = history.pivot_table(index="date", columns="ticker", values=price_col,
                             aggfunc="last").sort_index()
    if len(px) < lookback + 1:
        return pd.Series(dtype=float)
    ret = px.iloc[-1] / px.iloc[-(lookback + 1)] - 1.0
    return (-ret).replace([np.inf, -np.inf], np.nan).dropna()


def realised_vol(history: pd.DataFrame, lookback: int = 63,
                 price_col: str = "adj_close") -> pd.Series:
    """Annualised standard deviation of daily returns.

    For ordinary investing you would want this LOW. For a winner-takes-all
    tournament you want it high, because only the right tail of the outcome
    distribution wins anything. Same factor, opposite sign of preference --
    which is why it is a factor here and not a filter.
    """
    px = history.pivot_table(index="date", columns="ticker", values=price_col,
                             aggfunc="last").sort_index()
    if len(px) < lookback + 2:
        return pd.Series(dtype=float)
    rets = px.pct_change().iloc[-lookback:]
    return (rets.std(ddof=1) * np.sqrt(252)).replace(
        [np.inf, -np.inf], np.nan).dropna()


def volume_surge(history: pd.DataFrame, short: int = 5,
                 long: int = 63) -> pd.Series:
    """Recent traded value relative to its own longer-run average.

    Normalised per ticker, so it measures unusual activity for that stock
    rather than simply picking the largest companies.
    """
    h = history.copy()
    h["dtv"] = h["close"] * h["volume"]
    dtv = h.pivot_table(index="date", columns="ticker", values="dtv",
                        aggfunc="last").sort_index()
    if len(dtv) < long + 1:
        return pd.Series(dtype=float)
    ratio = dtv.iloc[-short:].mean() / dtv.iloc[-long:].mean()
    return ratio.replace([np.inf, -np.inf], np.nan).dropna()


def low_vol(history: pd.DataFrame, lookback: int = 63,
            price_col: str = "adj_close") -> pd.Series:
    """Realised volatility, negated: calm names score highest."""
    return -realised_vol(history, lookback, price_col)


FACTORS = {
    "momentum": momentum,
    "reversal": reversal,
    "volatility": realised_vol,
    "low_vol": low_vol,
    "volume_surge": volume_surge,
}


# --------------------------------------------------------------------------
# The composite
# --------------------------------------------------------------------------
def percentile_rank(s: pd.Series) -> pd.Series:
    """Rank to 0-100 across the cohort. Exactly the LIPS normalisation.

    Order only: one enormous outlier cannot dominate the composite, which is
    the property that makes the weights mean what they say.
    """
    if s.empty:
        return s
    return s.rank(pct=True, method="average") * 100.0


@dataclass
class CompositeStrategy:
    """A weighted percentile-rank composite over several factors.

    weights : factor name -> weight. Need not sum to 100; they are normalised.
    top_n   : how many names to hold. For a tournament this is deliberately
              small -- concentration is where the win probability comes from.
    """
    weights: dict[str, float]
    top_n: int = 3
    price_col: str = "adj_close"
    name: str = "composite"

    def __post_init__(self):
        unknown = set(self.weights) - set(FACTORS)
        if unknown:
            raise ValueError(f"unknown factor(s): {sorted(unknown)}. "
                             f"Available: {sorted(FACTORS)}")
        if not self.weights:
            raise ValueError("at least one factor is required")
        if any(w < 0 for w in self.weights.values()):
            raise ValueError("weights must be non-negative; negate the factor "
                             "instead (see low_vol)")

    def score(self, history: pd.DataFrame, eligible: list[str]) -> pd.Series:
        """Composite score per eligible ticker, highest is best."""
        if not eligible:
            return pd.Series(dtype=float)

        total_w = sum(self.weights.values())
        parts, used = [], {}

        for fname, w in self.weights.items():
            raw = FACTORS[fname](history, price_col=self.price_col) \
                if fname != "volume_surge" else FACTORS[fname](history)
            raw = raw.reindex(eligible).dropna()
            if raw.empty:
                continue
            used[fname] = len(raw)
            parts.append(percentile_rank(raw) * (w / total_w))

        if not parts:
            return pd.Series(dtype=float)

        # Only score tickers with a value for EVERY contributing factor --
        # otherwise a stock missing the momentum term would be silently
        # advantaged by having fewer ways to score badly.
        combined = pd.concat(parts, axis=1)
        return combined.dropna().sum(axis=1).sort_values(ascending=False)

    def signal(self, history: pd.DataFrame, date: pd.Timestamp,
               eligible: list[str]) -> dict[str, float]:
        """Signal function for the harness: equal weight across the top N."""
        scored = self.score(history, eligible)
        if scored.empty:
            return {}
        return {t: 1.0 for t in scored.head(self.top_n).index}

    def components(self, history: pd.DataFrame,
                   eligible: list[str]) -> pd.DataFrame:
        """Per-factor percentile and weighted contribution, for auditing.

        The direct equivalent of keeping the pr_* columns in LIPS: any score
        can be decomposed into what drove it, instead of being taken on faith.
        """
        total_w = sum(self.weights.values())
        rows = {}
        for fname, w in self.weights.items():
            raw = (FACTORS[fname](history) if fname == "volume_surge"
                   else FACTORS[fname](history, price_col=self.price_col))
            raw = raw.reindex(eligible).dropna()
            if raw.empty:
                continue
            pct = percentile_rank(raw)
            rows[f"raw_{fname}"] = raw
            rows[f"pct_{fname}"] = pct
            rows[f"contrib_{fname}"] = pct * (w / total_w)
        if not rows:
            return pd.DataFrame()
        out = pd.DataFrame(rows)
        contrib = [c for c in out.columns if c.startswith("contrib_")]
        out["score"] = out[contrib].sum(axis=1)
        return out.sort_values("score", ascending=False)


# --------------------------------------------------------------------------
# Presets. Starting points, not recommendations -- test them.
# --------------------------------------------------------------------------
def tournament_preset(top_n: int = 3) -> CompositeStrategy:
    """Weighted for winning a short winner-takes-all contest.

    Volatility dominates because the simulation says concentration and
    variance buy roughly 16 percentage points of win probability while a
    realistic edge buys about 3. The momentum and volume terms are the
    attempt at that 3.
    """
    return CompositeStrategy(
        weights={"volatility": 55, "momentum": 30, "volume_surge": 15},
        top_n=top_n, name="tournament (vol-led)")


def momentum_preset(top_n: int = 5) -> CompositeStrategy:
    """A conventional cross-sectional momentum sleeve, for comparison."""
    return CompositeStrategy(
        weights={"momentum": 70, "volume_surge": 15, "low_vol": 15},
        top_n=top_n, name="momentum")


def reversal_preset(top_n: int = 5) -> CompositeStrategy:
    """The opposite bet. Tested alongside momentum, never instead of it."""
    return CompositeStrategy(
        weights={"reversal": 70, "volume_surge": 15, "low_vol": 15},
        top_n=top_n, name="reversal")


PRESETS = {
    "tournament": tournament_preset,
    "momentum": momentum_preset,
    "reversal": reversal_preset,
}
