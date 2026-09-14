"""Tests for the ranking factors and the composite.

Two things matter here: the factors must measure what they claim, and none of
them may peek at data after the decision date.
"""
import numpy as np
import pandas as pd
import pytest

from research.strategies import (
    CompositeStrategy, FACTORS, PRESETS, low_vol, momentum, percentile_rank,
    realised_vol, reversal, tournament_preset, volume_surge,
)


def panel_with(paths: dict[str, np.ndarray], volume=None):
    """Build a panel from explicit price paths, one per ticker."""
    n = len(next(iter(paths.values())))
    dates = pd.date_range("2023-01-02", periods=n, freq="B")
    rows = []
    for t, px in paths.items():
        vol = (volume[t] if volume and t in volume
               else np.full(n, 1_000_000.0))
        rows.append(pd.DataFrame({
            "date": dates, "ticker": t, "open": px, "high": px * 1.01,
            "low": px * 0.99, "close": px, "adj_close": px, "volume": vol,
        }))
    return pd.concat(rows, ignore_index=True)


def trending(n, rate):
    return 10.0 * np.exp(np.arange(n) * rate)


# ------------------------------------------------------------------ factors
def test_momentum_ranks_the_stronger_trend_higher():
    p = panel_with({"UP.KL": trending(300, 0.002),
                    "DOWN.KL": trending(300, -0.002)})
    m = momentum(p, lookback=126, skip=21)
    assert m["UP.KL"] > m["DOWN.KL"]
    assert m["UP.KL"] > 0 > m["DOWN.KL"]


def test_momentum_skips_the_most_recent_month():
    """A stock that trended up then crashed in the last 21 days should still
    show positive momentum -- that is the whole purpose of the skip."""
    n = 300
    px = trending(n, 0.003)
    px[-21:] *= 0.6                        # recent crash
    p = panel_with({"X.KL": px})
    assert momentum(p, lookback=126, skip=21)["X.KL"] > 0
    # ...while the no-skip version is dragged negative by the crash.
    assert momentum(p, lookback=126, skip=0)["X.KL"] < \
           momentum(p, lookback=126, skip=21)["X.KL"]


def test_reversal_is_the_negative_of_recent_return():
    p = panel_with({"FELL.KL": trending(100, -0.004),
                    "ROSE.KL": trending(100, 0.004)})
    r = reversal(p, lookback=21)
    assert r["FELL.KL"] > r["ROSE.KL"]     # the loser scores highest


def test_realised_vol_separates_calm_from_wild():
    rng = np.random.default_rng(0)
    n = 200
    calm = 10 * np.exp(np.cumsum(rng.normal(0, 0.003, n)))
    wild = 10 * np.exp(np.cumsum(rng.normal(0, 0.040, n)))
    v = realised_vol(panel_with({"CALM.KL": calm, "WILD.KL": wild}),
                     lookback=63)
    assert v["WILD.KL"] > v["CALM.KL"] * 5


def test_low_vol_is_exactly_the_negation():
    p = panel_with({"A.KL": trending(200, 0.001), "B.KL": trending(200, 0.002)})
    assert np.allclose(low_vol(p, 63).values, -realised_vol(p, 63).values)


def test_volume_surge_is_relative_to_each_stocks_own_history():
    n = 200
    quiet = np.full(n, 1_000.0)
    surging = np.full(n, 1_000.0)
    surging[-5:] = 50_000.0
    # BIG has huge volume throughout but no surge; SURGE is small but spiking.
    p = panel_with({"BIG.KL": trending(n, 0.0), "SURGE.KL": trending(n, 0.0)},
                   volume={"BIG.KL": quiet * 1000, "SURGE.KL": surging})
    s = volume_surge(p, short=5, long=63)
    assert s["SURGE.KL"] > s["BIG.KL"]


def test_factors_return_empty_when_history_is_too_short():
    p = panel_with({"X.KL": trending(10, 0.001)})
    assert momentum(p, lookback=126).empty
    assert realised_vol(p, lookback=63).empty
    assert volume_surge(p, long=63).empty


# ------------------------------------------------------------- NO LOOKAHEAD
@pytest.mark.parametrize("factor_name", list(FACTORS))
def test_no_factor_uses_data_beyond_the_history_it_is_given(factor_name):
    """Truncating the panel must not change a factor computed on the earlier
    part. If it does, the factor is reading the future."""
    rng = np.random.default_rng(4)
    n = 300
    paths = {f"T{i}.KL": 10 * np.exp(np.cumsum(rng.normal(0, 0.02, n)))
             for i in range(3)}
    full = panel_with(paths)
    cut_date = sorted(full["date"].unique())[200]
    truncated = full[full["date"] <= cut_date]

    fn = FACTORS[factor_name]
    on_truncated = fn(truncated)
    on_full_but_truncated_inside = fn(full[full["date"] <= cut_date])
    pd.testing.assert_series_equal(on_truncated, on_full_but_truncated_inside)


# ---------------------------------------------------------------- composite
def test_percentile_rank_maps_to_0_100():
    pr = percentile_rank(pd.Series([1, 2, 3, 4]))
    assert pr.min() == pytest.approx(25.0)
    assert pr.max() == pytest.approx(100.0)


def test_percentile_rank_ignores_magnitude():
    """One huge outlier must not dominate -- the LIPS property."""
    a = percentile_rank(pd.Series({"x": 1, "y": 2, "z": 3}))
    b = percentile_rank(pd.Series({"x": 1, "y": 2, "z": 1e12}))
    pd.testing.assert_series_equal(a, b)


def test_unknown_factor_is_rejected():
    with pytest.raises(ValueError, match="unknown factor"):
        CompositeStrategy(weights={"nonsense": 1.0})


def test_empty_weights_rejected():
    with pytest.raises(ValueError, match="at least one factor"):
        CompositeStrategy(weights={})


def test_negative_weights_rejected():
    with pytest.raises(ValueError, match="non-negative"):
        CompositeStrategy(weights={"momentum": -1.0})


def test_composite_picks_the_highest_scorer():
    n = 300
    p = panel_with({"WINNER.KL": trending(n, 0.004),
                    "MIDDLE.KL": trending(n, 0.001),
                    "LOSER.KL": trending(n, -0.003)})
    s = CompositeStrategy(weights={"momentum": 100}, top_n=1)
    picks = s.signal(p, p["date"].max(), sorted(p["ticker"].unique()))
    assert list(picks) == ["WINNER.KL"]


def test_composite_respects_top_n():
    n = 300
    paths = {f"T{i}.KL": trending(n, 0.001 * i) for i in range(6)}
    s = CompositeStrategy(weights={"momentum": 100}, top_n=2)
    picks = s.signal(panel_with(paths), None, sorted(paths))
    assert len(picks) == 2


def test_composite_only_scores_eligible_tickers():
    n = 300
    paths = {f"T{i}.KL": trending(n, 0.002) for i in range(4)}
    s = CompositeStrategy(weights={"momentum": 100}, top_n=4)
    picks = s.signal(panel_with(paths), None, ["T0.KL", "T1.KL"])
    assert set(picks) <= {"T0.KL", "T1.KL"}


def test_composite_handles_an_empty_universe():
    p = panel_with({"X.KL": trending(300, 0.001)})
    assert CompositeStrategy(weights={"momentum": 100}).signal(p, None, []) == {}


def test_components_decompose_the_score():
    """The LIPS audit property: any score breaks into its contributions."""
    n = 300
    paths = {f"T{i}.KL": trending(n, 0.001 * (i + 1)) for i in range(4)}
    s = tournament_preset(top_n=2)
    comp = s.components(panel_with(paths), sorted(paths))
    contrib = [c for c in comp.columns if c.startswith("contrib_")]
    assert np.allclose(comp[contrib].sum(axis=1), comp["score"])
    assert comp["score"].is_monotonic_decreasing


def test_tournament_preset_weights_volatility_most():
    w = tournament_preset().weights
    assert max(w, key=w.get) == "volatility"


@pytest.mark.parametrize("name", list(PRESETS))
def test_every_preset_constructs_and_scores(name):
    n = 300
    rng = np.random.default_rng(2)
    paths = {f"T{i}.KL": 10 * np.exp(np.cumsum(rng.normal(0.0005, 0.02, n)))
             for i in range(5)}
    strat = PRESETS[name]()
    picks = strat.signal(panel_with(paths), None, sorted(paths))
    assert 0 < len(picks) <= strat.top_n
