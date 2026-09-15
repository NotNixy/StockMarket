"""Tests for the contest simulator.

The controls come first and matter most. A contest simulator that is subtly
wrong produces a confident, plausible, wrong number -- and the whole point of
building it is to size a real position.
"""
import numpy as np
import pandas as pd
import pytest

from tournament.contest import (
    ContestConfig, concentration_sweep, run_contest, rule_sweep,
)


def synthetic_panel(n_tickers=60, n_days=800, seed=0, drift=0.0, vol=0.02):
    """A plain random-walk market. No edge exists in it, by construction."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2020-01-01", periods=n_days, freq="B")
    frames = []
    for i in range(n_tickers):
        r = rng.normal(drift, vol, n_days)
        px = 5.0 * np.exp(np.cumsum(r))
        frames.append(pd.DataFrame({
            "date": dates, "ticker": f"T{i:03d}.KL",
            "adj_close": px, "eligible": True,
        }))
    return pd.concat(frames, ignore_index=True)


# ------------------------------------------------------------------ controls
def test_an_identical_field_gives_you_exactly_your_fair_share():
    """THE negative control.

    If you and all 399 opponents follow the same rule at the same
    concentration, no one has an advantage and P(win) must be 1/400. Any
    simulator that reports more than that is flattering you somewhere -- the
    most likely place being that your portfolio and theirs are not drawn from
    the same distribution.
    """
    panel = synthetic_panel()
    cfg = ContestConfig(n_entrants=400, my_k=10, field_k=10,
                        my_rule="random", field_rule="random",
                        n_trials=4000, seed=1)
    r = run_contest(panel, cfg)
    assert r.p_win == pytest.approx(1 / 400, abs=0.004), (
        f"symmetric contest should give 1/400 = 0.25%, got {r.p_win:.3%}")


def test_a_smaller_field_raises_everyones_odds_proportionally():
    panel = synthetic_panel()
    base = dict(my_k=10, field_k=10, n_trials=4000, seed=2)
    small = run_contest(panel, ContestConfig(n_entrants=10, **base))
    assert small.p_win == pytest.approx(1 / 10, abs=0.02)


def test_a_planted_edge_is_detected():
    """The positive control. One ticker always rises; picking it must win.

    Without this, a simulator that reports 1/400 for everything would pass
    the negative control perfectly while being completely broken.
    """
    panel = synthetic_panel(n_tickers=60, seed=3)
    # Make one name a guaranteed winner and hand it to "momentum".
    m = panel["ticker"] == "T000.KL"
    dates = panel.loc[m, "date"].to_numpy()
    # The drift has to clear the noise it is hiding in, or the control tests
    # nothing. At 2%/day vol, 60-day trailing momentum has sigma ~15%, and the
    # best of 59 random walks reaches ~36%. A plant that only gains 25% over
    # the same window is NOT reliably findable, and the test fails for a
    # reason that has nothing to do with the simulator.
    panel.loc[m, "adj_close"] = 5.0 * np.exp(
        np.linspace(0, 8.0, len(dates)))

    cfg = ContestConfig(n_entrants=400, my_k=1, field_k=10,
                        my_rule="momentum", field_rule="random",
                        n_trials=1500, seed=3)
    r = run_contest(panel, cfg)
    assert r.p_win > 0.5, (
        f"holding a guaranteed winner alone should dominate, got {r.p_win:.1%}")


def test_concentration_beats_diversification_for_winning():
    """The finding the whole module exists to establish.

    With no edge, holding 1 name must beat holding 40 for P(win) -- not
    because it earns more (it does not, in expectation) but because you
    cannot finish first from the middle of the pack. If this ever inverts,
    something is wrong with the simulator, not with tournament theory.
    """
    panel = synthetic_panel(n_tickers=80, seed=4)
    base = ContestConfig(n_entrants=400, field_k=10, n_trials=3000, seed=4)
    sweep = concentration_sweep(panel, ks=(1, 40), base=base, verbose=False)
    assert sweep.loc[1, "p_win"] > sweep.loc[40, "p_win"]


def test_concentration_does_not_raise_expected_return():
    """The other half of the same finding, and the part that costs you.

    Concentration buys win probability by adding variance, not return. The
    mean must be flat across k; if it rises with concentration the simulator
    has an edge baked in somewhere.
    """
    panel = synthetic_panel(n_tickers=80, seed=5)
    base = ContestConfig(n_entrants=400, field_k=10, n_trials=3000, seed=5)
    sweep = concentration_sweep(panel, ks=(1, 5, 40), base=base, verbose=False)
    spread = sweep["mean"].max() - sweep["mean"].min()
    assert spread < 0.02, f"expected return varied with k by {spread:.2%}"
    # And the tail must widen as it concentrates -- that is the mechanism.
    assert sweep.loc[1, "p95"] > sweep.loc[40, "p95"]


# ------------------------------------------------------------------ mechanics
def test_selection_rules_cannot_see_the_contest_window():
    """A rule that peeks at the future would make every number here fiction.

    Checked by flattening all history before the contest opens: if a rule
    were reading forward, momentum would still find the eventual winner.
    """
    panel = synthetic_panel(n_tickers=40, n_days=600, seed=6)
    prices = panel.pivot_table(index="date", columns="ticker",
                               values="adj_close")
    cut = 300
    # Every name identical up to the cut; all information is after it.
    flat = prices.copy()
    flat.iloc[:cut] = 5.0
    long = flat.reset_index().melt(id_vars="date", var_name="ticker",
                                   value_name="adj_close")
    long["eligible"] = True

    # my_k MUST equal field_k here. Holding 1 name against a field holding 10
    # raises P(win) on concentration alone -- that is the whole point of the
    # module -- and would be mistaken for lookahead. Matching concentration
    # isolates the only thing this test is asking about.
    cfg = ContestConfig(n_entrants=400, my_k=10, field_k=10,
                        my_rule="momentum", field_rule="random",
                        n_trials=3000, seed=6, lookback=60, horizon=21)
    r = run_contest(long, cfg)
    # With no information before the open, momentum is a coin: fair share.
    assert r.p_win == pytest.approx(1 / 400, abs=0.005), (
        f"momentum found signal that does not exist pre-open: {r.p_win:.3%}")


def test_every_rule_runs_and_returns_a_probability():
    panel = synthetic_panel(n_tickers=50, seed=7)
    base = ContestConfig(n_entrants=100, my_k=5, n_trials=400, seed=7)
    table = rule_sweep(panel, base=base, verbose=False)
    assert len(table) == 5
    assert table["p_win"].between(0, 1).all()


def test_names_missing_a_price_at_either_end_are_never_picked():
    """A position you could not have held must not appear in a result."""
    panel = synthetic_panel(n_tickers=30, seed=8)
    panel.loc[panel["ticker"] == "T000.KL", "adj_close"] = np.nan
    r = run_contest(panel, ContestConfig(n_entrants=50, my_k=3,
                                         n_trials=300, seed=8))
    assert np.isfinite(r.my_median_return)


def test_a_horizon_longer_than_the_panel_is_refused():
    panel = synthetic_panel(n_tickers=10, n_days=80, seed=9)
    with pytest.raises(ValueError, match="too short"):
        run_contest(panel, ContestConfig(horizon=200, n_trials=10))


def test_a_mixed_field_lands_between_its_extremes():
    """Modelling every opponent at the same concentration is the least
    realistic assumption in the module, and the one P(win) is most sensitive
    to. A mixed field must sit between the uniform fields it is made of."""
    panel = synthetic_panel(n_tickers=80, seed=10)
    base = dict(n_entrants=400, my_k=1, my_rule="random",
                n_trials=3000, seed=10)
    tight = run_contest(panel, ContestConfig(field_k=1, **base)).p_win
    loose = run_contest(panel, ContestConfig(field_k=20, **base)).p_win
    mixed = run_contest(panel, ContestConfig(
        field_k_mix={1: 0.1, 5: 0.5, 20: 0.4}, **base)).p_win
    assert tight < mixed < loose, (
        f"tight {tight:.2%}, mixed {mixed:.2%}, loose {loose:.2%}")


def test_a_field_as_concentrated_as_you_removes_your_whole_advantage():
    """The load-bearing caveat, pinned as a test.

    The advantage is RELATIVE concentration, not concentration. Holding one
    stock against 399 others each holding one stock is a 1-in-400 lottery,
    and no amount of conviction changes that.
    """
    panel = synthetic_panel(n_tickers=80, seed=11)
    r = run_contest(panel, ContestConfig(
        n_entrants=400, my_k=1, field_k=1, my_rule="random",
        field_rule="random", n_trials=4000, seed=11))
    assert r.p_win == pytest.approx(1 / 400, abs=0.005)


def test_the_field_mix_weights_do_not_need_to_sum_to_one():
    panel = synthetic_panel(n_tickers=50, seed=12)
    r = run_contest(panel, ContestConfig(
        n_entrants=100, my_k=1, field_k_mix={2: 3, 10: 7},
        n_trials=500, seed=12))
    assert 0 < r.p_win < 1
