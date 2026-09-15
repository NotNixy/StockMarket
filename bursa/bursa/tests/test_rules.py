"""Tests for the kill criterion.

These pin decisions that are meant to survive contact with a moving screen.
The most important ones are the HOLDs: a rule that quietly starts advising
SWITCH under pressure is worse than no rule, because it carries authority.
"""
import pytest

from tournament.rules import (
    ENDGAME_DAYS, MOMENTUM_GIVEBACK, Action, Situation, decide,
    explain_no_stop_loss,
)


def sit(**kw) -> Situation:
    base = dict(days_left=10, my_return=0.02, position_return=0.02,
                momentum_at_entry=1.0, momentum_now=1.0)
    return Situation(**{**base, **kw})


# ------------------------------------------------------------------ the HOLDs
@pytest.mark.parametrize("loss", [-0.05, -0.15, -0.25, -0.35])
def test_a_drawdown_alone_is_never_a_reason_to_sell(loss):
    """The rule most likely to be overridden, and the costliest to override.

    The 5th percentile of this strategy is about -36%. A fall of that order
    is a draw from the distribution that was chosen deliberately, not new
    information. Selling there realises the loss and forfeits the tail the
    whole plan is playing for.
    """
    v = decide(sit(my_return=loss, position_return=loss))
    assert v.action is Action.HOLD, f"advised {v.action} on a {loss:.0%} draw"


def test_being_far_behind_does_not_trigger_de_risking():
    v = decide(sit(my_return=-0.30, days_left=8))
    assert v.action is Action.HOLD
    assert "no second prize" in v.detail


def test_being_ahead_also_holds_but_for_a_different_reason():
    behind = decide(sit(my_return=-0.10))
    ahead = decide(sit(my_return=0.45))
    assert behind.action is ahead.action is Action.HOLD
    assert behind.reason != ahead.reason


def test_the_endgame_holds_whatever_is_happening():
    for ret in (-0.40, 0.0, 0.60):
        v = decide(sit(days_left=ENDGAME_DAYS, my_return=ret))
        assert v.action is Action.HOLD


# ---------------------------------------------------------------- the SWITCHes
def test_a_halt_outranks_everything_including_the_endgame():
    """A halted stock cannot be exited or compounded. It is the one case
    where speed beats deliberation, so it must win over every other rule."""
    v = decide(sit(halted=True, days_left=1, my_return=0.50))
    assert v.action is Action.SWITCH
    assert "halted" in v.reason


def test_falling_out_of_the_universe_triggers_a_rotation():
    v = decide(sit(still_eligible=False))
    assert v.action is Action.SWITCH
    assert "liquidity" in v.reason


def test_a_broken_thesis_switches_even_when_the_trade_is_profitable():
    """The giveback test is measured against the SIGNAL, not the entry price.

    A stock can be above your entry and still have given back the move that
    caused the rule to buy it. The reason you own it is then gone.
    """
    v = decide(sit(momentum_at_entry=1.0, momentum_now=0.2,
                   position_return=+0.15))
    assert v.action is Action.SWITCH
    assert "momentum" in v.reason


def test_a_partial_giveback_is_tolerated():
    v = decide(sit(momentum_at_entry=1.0,
                   momentum_now=1.0 - (MOMENTUM_GIVEBACK * 0.5)))
    assert v.action is Action.HOLD


def test_the_giveback_threshold_is_where_it_says_it_is():
    just_under = decide(sit(momentum_at_entry=1.0,
                            momentum_now=1.0 - MOMENTUM_GIVEBACK + 0.02))
    just_over = decide(sit(momentum_at_entry=1.0,
                           momentum_now=1.0 - MOMENTUM_GIVEBACK - 0.02))
    assert just_under.action is Action.HOLD
    assert just_over.action is Action.SWITCH


# ------------------------------------------------------------------ mechanics
def test_a_halt_beats_a_broken_thesis():
    """Ordering matters: both apply, and the halt is the urgent one."""
    v = decide(sit(halted=True, momentum_at_entry=1.0, momentum_now=0.0))
    assert "halted" in v.reason


def test_giveback_is_zero_when_entry_momentum_was_not_positive():
    """Guards a divide-by-zero on a name that was not bought on momentum."""
    assert sit(momentum_at_entry=0.0, momentum_now=-0.5).giveback == 0.0


def test_giveback_never_goes_negative_when_momentum_strengthens():
    assert sit(momentum_at_entry=1.0, momentum_now=1.8).giveback == 0.0


def test_every_verdict_carries_a_reason_a_human_can_act_on():
    for s in (sit(), sit(halted=True), sit(still_eligible=False),
              sit(days_left=1), sit(momentum_now=0.0), sit(my_return=-0.4)):
        v = decide(s)
        assert v.reason and len(v.reason) > 10
        assert str(v).startswith(v.action.value)


def test_the_no_stop_loss_note_states_the_condition_it_depends_on():
    """The advice is only correct because the capital is the organisers'.
    Stripping that caveat would turn this into dangerous advice."""
    text = explain_no_stop_loss()
    assert "organisers'" in text
    assert "never run it on your own money" in text.lower()
