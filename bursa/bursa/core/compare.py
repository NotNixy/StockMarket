"""Compare two price panels and report where they disagree.

The validator in core.validate checks whether a panel is *internally
consistent*. It cannot tell you whether the prices are *right*. Only a second,
independent source can do that, and your broker is the one that matters --
their prices are what you would actually trade at.

Real Bursa data has already produced three faults that a single source cannot
detect: an unapplied 2-for-1 split in VITROX, no dividend adjustment at all in
Frontken, and a corrupt bar in CelcomDigi. Those were caught because the checks
looked for them. "Caught what I thought to check for" is not the same as
"correct", and this module closes that gap.

Everything here is pure comparison logic on two DataFrames -- no network, no
broker SDK -- so it is fully testable. The moomoo connection lives in
scripts/compare_moomoo.py and is deliberately thin.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# Two sources rarely agree to the sen. These are the bands beyond which a
# difference stops being rounding and starts being a discrepancy.
CLOSE_TOL = 0.005          # 0.5% on the raw close
ADJ_TOL = 0.02             # 2% on the adjusted close -- adjustment conventions
                           # legitimately differ (Yahoo back-adjusts, moomoo's
                           # QFQ forward-adjusts), so this is deliberately loose
MATERIAL_GAP = 0.10        # a 10%+ divergence is not a convention difference


@dataclass
class Comparison:
    """What the two sources agreed and disagreed on, per ticker."""
    ticker: str
    n_common: int
    n_only_a: int
    n_only_b: int
    close_mismatches: pd.DataFrame = field(default_factory=pd.DataFrame)
    adj_mismatches: pd.DataFrame = field(default_factory=pd.DataFrame)
    max_close_diff: float = 0.0
    max_adj_diff: float = 0.0

    @property
    def clean(self) -> bool:
        return self.close_mismatches.empty and self.adj_mismatches.empty

    @property
    def material(self) -> bool:
        """A divergence too large to be an adjustment convention."""
        return self.max_adj_diff > MATERIAL_GAP or self.max_close_diff > MATERIAL_GAP

    def __str__(self) -> str:
        mark = "OK  " if self.clean else ("!!! " if self.material else "warn")
        return (f"[{mark}] {self.ticker:<12} {self.n_common:>5} common bars  "
                f"close diff max {self.max_close_diff:6.2%}  "
                f"adj diff max {self.max_adj_diff:6.2%}  "
                f"({len(self.close_mismatches)} close, "
                f"{len(self.adj_mismatches)} adj mismatches)")


def compare_ticker(a: pd.DataFrame, b: pd.DataFrame, ticker: str,
                   close_tol: float = CLOSE_TOL,
                   adj_tol: float = ADJ_TOL) -> Comparison:
    """Compare one ticker across two panels, on their overlapping dates.

    Dates present in only one source are counted but not treated as errors --
    holidays, halts and differing history depth all cause them legitimately.
    """
    ta = a[a["ticker"] == ticker][["date", "close", "adj_close"]]
    tb = b[b["ticker"] == ticker][["date", "close", "adj_close"]]

    merged = ta.merge(tb, on="date", how="inner", suffixes=("_a", "_b"))
    only_a = len(ta) - len(merged)
    only_b = len(tb) - len(merged)

    if merged.empty:
        return Comparison(ticker, 0, only_a, only_b)

    merged["close_diff"] = (merged["close_a"] / merged["close_b"] - 1.0).abs()
    merged["adj_diff"] = (merged["adj_close_a"] / merged["adj_close_b"] - 1.0).abs()
    merged = merged.replace([np.inf, -np.inf], np.nan)

    return Comparison(
        ticker=ticker,
        n_common=len(merged),
        n_only_a=only_a,
        n_only_b=only_b,
        close_mismatches=merged[merged["close_diff"] > close_tol].copy(),
        adj_mismatches=merged[merged["adj_diff"] > adj_tol].copy(),
        max_close_diff=float(merged["close_diff"].max(skipna=True) or 0.0),
        max_adj_diff=float(merged["adj_diff"].max(skipna=True) or 0.0),
    )


def compare_panels(a: pd.DataFrame, b: pd.DataFrame,
                   close_tol: float = CLOSE_TOL,
                   adj_tol: float = ADJ_TOL) -> list[Comparison]:
    """Compare every ticker the two panels have in common."""
    shared = sorted(set(a["ticker"]) & set(b["ticker"]))
    return [compare_ticker(a, b, t, close_tol, adj_tol) for t in shared]


def report(comparisons: list[Comparison], name_a: str = "yahoo",
           name_b: str = "moomoo", verbose: bool = True) -> pd.DataFrame:
    """Summarise, and say plainly what the result means.

    A clean comparison does NOT mean the data is correct -- both sources could
    share an upstream feed and the same fault. It means an independent check
    found nothing, which is weaker but still worth having.
    """
    rows = [{
        "ticker": c.ticker, "common_bars": c.n_common,
        f"only_{name_a}": c.n_only_a, f"only_{name_b}": c.n_only_b,
        "max_close_diff": c.max_close_diff, "max_adj_diff": c.max_adj_diff,
        "close_mismatches": len(c.close_mismatches),
        "adj_mismatches": len(c.adj_mismatches),
        "material": c.material,
    } for c in comparisons]
    table = pd.DataFrame(rows)

    if verbose:
        print(f"Comparing {name_a} against {name_b}: "
              f"{len(comparisons)} tickers in common\n")
        for c in comparisons:
            print(f"  {c}")

        bad = [c for c in comparisons if c.material]
        clean = [c for c in comparisons if c.clean]
        print(f"\n  {len(clean)}/{len(comparisons)} agree within tolerance")
        if bad:
            print(f"  {len(bad)} MATERIAL divergence(s) -- these are not "
                  f"adjustment conventions:")
            for c in bad:
                print(f"\n  --- {c.ticker} ---")
                worst = (c.adj_mismatches.nlargest(5, "adj_diff")
                         if len(c.adj_mismatches)
                         else c.close_mismatches.nlargest(5, "close_diff"))
                print(worst.to_string(index=False))
        print(f"\n  Note: agreement is weak evidence, not proof. Both sources "
              f"may share\n  an upstream feed and therefore the same fault.")
    return table


def splits_missing_from(panel: pd.DataFrame, splits: pd.DataFrame,
                        tol: float = 0.05) -> pd.DataFrame:
    """Recorded splits whose price step does NOT appear in the panel.

    If a split is on record but the panel's adjusted series steps by roughly
    the ratio anyway, the adjustment was never applied -- the VITROX case. If
    the adjusted series is smooth across the split date, it was handled.
    """
    if splits.empty:
        return pd.DataFrame(columns=["ticker", "date", "ratio", "adj_step",
                                     "applied"])
    p = panel.sort_values(["ticker", "date"]).copy()
    p["adj_ret"] = p.groupby("ticker", observed=True)["adj_close"].pct_change()

    rows = []
    for _, s in splits.iterrows():
        near = p[(p["ticker"] == s["ticker"])
                 & (p["date"] >= s["date"] - pd.Timedelta(days=45))
                 & (p["date"] <= s["date"] + pd.Timedelta(days=5))]
        if near.empty:
            continue
        step = near.loc[near["adj_ret"].abs().idxmax()] if near["adj_ret"].notna().any() \
            else None
        if step is None:
            continue
        implied = 1.0 / (1.0 + step["adj_ret"]) if step["adj_ret"] < 0 \
            else 1.0 + step["adj_ret"]
        # If the adjusted series stepped by the split ratio, nobody adjusted it.
        unapplied = abs(implied - s["value"]) / s["value"] < tol
        rows.append({"ticker": s["ticker"], "date": s["date"],
                     "ratio": s["value"], "adj_step": step["adj_ret"],
                     "applied": not unapplied})
    return pd.DataFrame(rows)
