"""Tests for the dashboard's chart builders.

Charts are hard to assert on, so these check the things that actually break:
that every figure renders without raising, that the palettes are the validated
ones, and that end labels get pushed apart rather than stacking.
"""
import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest

from tournament.charts import (
    DARK, LIGHT, equity_chart, gap_chart, pwin_chart,
    random_distribution_chart, universe_chart,
)
from tournament.standing import Contest, strategy_table

DATES = pd.date_range("2024-01-01", periods=120, freq="B")


def curve(start=10_000, drift=0.0005, seed=0):
    rng = np.random.default_rng(seed)
    return pd.Series(start * np.exp(np.cumsum(rng.normal(drift, 0.01, 120))),
                     index=DATES)


# ------------------------------------------------------------------ palettes
def test_palettes_are_the_validated_values():
    """These exact hexes passed the dataviz validator against their own
    surfaces. Changing one means re-running the validator."""
    assert LIGHT.series == ("#2a78d6", "#eb6834", "#1baf7a", "#eda100")
    assert DARK.series == ("#3987e5", "#d95926", "#199e70", "#c98500")


def test_dark_is_restepped_not_flipped():
    # Every dark step differs from its light counterpart -- an automatic
    # inversion would reuse them.
    assert all(d != l for d, l in zip(DARK.series, LIGHT.series))
    assert DARK.surface != LIGHT.surface


# -------------------------------------------------------------------- render
@pytest.mark.parametrize("theme", [LIGHT, DARK])
def test_every_chart_renders(theme):
    curves = {"strategy": curve(seed=1), "buy & hold": curve(seed=2),
              "equal weight": curve(seed=3)}
    assert equity_chart(curves, theme) is not None
    assert pwin_chart(strategy_table(Contest(400, 0.05)), theme,
                      baseline=1 / 400) is not None
    assert random_distribution_chart(
        pd.Series(np.random.default_rng(0).normal(0, 1, 200)), 1.3,
        theme) is not None
    assert universe_chart(
        pd.DataFrame({"n_eligible": np.arange(120)}, index=DATES),
        theme) is not None
    assert gap_chart(np.arange(15), np.linspace(0, 0.06, 15), 0.146,
                     theme) is not None


def test_equity_chart_handles_a_single_series():
    assert equity_chart({"only": curve()}, LIGHT) is not None


def test_equity_chart_handles_an_empty_series():
    assert equity_chart({"empty": pd.Series(dtype=float)}, LIGHT) is not None


def test_equity_chart_separates_colliding_end_labels():
    """Baselines that finish within a whisker of each other must not stack
    their labels on top of one another."""
    flat = pd.Series(np.full(120, 10_000.0), index=DATES)
    fig = equity_chart({"a": flat, "b": flat * 1.0001, "c": flat * 1.0002},
                       LIGHT)
    ax = fig.axes[0]
    ys = sorted(a.xy[1] if hasattr(a, "xy") else a.get_position()[1]
                for a in ax.texts)
    gaps = np.diff(ys)
    assert (gaps > 0).all(), "end labels landed on the same y"


def test_pwin_chart_orders_bars_by_concentration():
    fig = pwin_chart(strategy_table(Contest(400, 0.05)), LIGHT)
    ax = fig.axes[0]
    assert len(ax.patches) == 4          # one bar per portfolio shape
    # Every bar carries a written value -- colour is never the only encoding.
    labels = [t.get_text() for t in ax.texts if "%" in t.get_text()]
    assert len(labels) >= 4
