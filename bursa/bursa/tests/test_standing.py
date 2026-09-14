"""Tests for the tournament calculator."""
import numpy as np
import pytest

from tournament.standing import (
    Contest, assess, expected_winning_score, prob_win, required_vol,
    strategy_table, winning_score_quantiles,
)


# ------------------------------------------------------------------- contest
def test_contest_rejects_nonsense():
    with pytest.raises(ValueError):
        Contest(n_entrants=1)
    with pytest.raises(ValueError):
        Contest(n_entrants=100, field_vol=0.0)


def test_winning_score_rises_with_entrants():
    """More people, higher bar -- the order statistic."""
    a = expected_winning_score(Contest(50, 0.05))
    b = expected_winning_score(Contest(150, 0.05))
    c = expected_winning_score(Contest(400, 0.05))
    assert a < b < c


def test_winning_score_scales_with_field_volatility():
    calm = expected_winning_score(Contest(400, 0.05))
    wild = expected_winning_score(Contest(400, 0.10))
    assert wild == pytest.approx(calm * 2, rel=0.01)


def test_expected_winning_score_matches_simulation():
    """The closed form should agree with brute force."""
    c = Contest(400, 0.05)
    analytic = expected_winning_score(c)
    simulated = winning_score_quantiles(c, quantiles=(0.5,))[0.5]
    assert analytic == pytest.approx(simulated, abs=0.01)


def test_four_hundred_entrants_need_about_fifteen_percent():
    # The figure the whole strategy is calibrated against.
    assert expected_winning_score(Contest(400, 0.05)) == pytest.approx(0.146,
                                                                       abs=0.01)


# ----------------------------------------------------------------- prob_win
def test_more_variance_wins_more_often_when_you_have_no_edge():
    """The central, counterintuitive result: in a winner-takes-all contest
    with no skill, volatility IS the strategy."""
    c = Contest(400, 0.05)
    diversified = prob_win(0.0, 0.05, c)
    concentrated = prob_win(0.0, 0.25, c)
    assert concentrated > diversified * 20


def test_diversified_no_edge_is_no_better_than_random():
    c = Contest(400, 0.05)
    assert prob_win(0.0, 0.05, c) == pytest.approx(1 / 400, abs=0.01)


def test_edge_helps_but_far_less_than_concentration():
    """Quantifies the trade-off the whole plan rests on."""
    c = Contest(400, 0.05)
    good_edge_diversified = prob_win(0.05, 0.05, c)   # +5%/mo, 15 names
    no_edge_concentrated = prob_win(0.00, 0.25, c)    # no skill, 1 name
    assert no_edge_concentrated > good_edge_diversified * 5


def test_being_ahead_raises_win_probability():
    c = Contest(400, 0.05)
    assert prob_win(0.20, 0.05, c) > prob_win(0.00, 0.05, c)


def test_prob_win_is_a_probability():
    c = Contest(400, 0.05)
    for r in (-0.5, 0.0, 0.5):
        for v in (0.01, 0.5):
            assert 0.0 <= prob_win(r, v, c) <= 1.0


# ------------------------------------------------------------- required_vol
def test_no_vol_needed_when_already_ahead():
    assert required_vol(-0.05) == 0.0
    assert required_vol(0.0) == 0.0


def test_required_vol_grows_with_the_gap():
    assert required_vol(0.20) > required_vol(0.05)


# ---------------------------------------------------------------- the verdict
def test_leading_late_says_cut_risk():
    s = assess(my_return=0.25, days_left=3, contest=Contest(400, 0.05))
    assert s.gap < 0
    assert "cut risk" in s.verdict.lower()


def test_far_behind_late_says_maximum_concentration():
    s = assess(my_return=0.01, days_left=2, contest=Contest(400, 0.05))
    assert s.gap > 0
    assert "far behind" in s.verdict.lower()


def test_behind_with_time_says_concentrate():
    s = assess(my_return=0.02, days_left=15, contest=Contest(400, 0.05),
               current_vol_annual=0.15, aggressive_vol_annual=1.2)
    assert "concentrate" in s.verdict.lower() or "behind" in s.verdict.lower()


def test_finished_contest_is_reported_as_over():
    s = assess(0.05, 0, Contest(400, 0.05))
    assert s.verdict == "contest over"


def test_negative_days_rejected():
    with pytest.raises(ValueError):
        assess(0.05, -1, Contest(400, 0.05))


def test_aggressive_always_beats_current_when_behind():
    s = assess(my_return=0.0, days_left=10, contest=Contest(400, 0.05),
               current_vol_annual=0.10, aggressive_vol_annual=0.80)
    assert s.p_win_aggressive > s.p_win_current


def test_standing_renders_without_error():
    assert "P(win)" in str(assess(0.03, 10, Contest(400, 0.05)))


# ------------------------------------------------------------ strategy table
def test_strategy_table_orders_by_concentration():
    tbl = strategy_table(Contest(400, 0.05))
    assert tbl["p_win"].is_monotonic_increasing        # more vol, more chance
    assert (tbl["vs_random"] > 0).all()
    # The concentrated row should be dramatically better than random.
    assert tbl["vs_random"].iloc[-1] > 50
