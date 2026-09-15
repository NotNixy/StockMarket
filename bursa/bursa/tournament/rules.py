"""The kill criterion. Decided now, while nothing is at stake.

Every rule here exists to be consulted at a moment when you will not want to
consult it. That is the entire design constraint: a rule you write while
watching a position move is not a rule, it is a rationalisation, and it will
always conclude that whatever you already want to do is correct.

So the numbers are fixed here, in code, with reasons attached, and the
dashboard renders the verdict rather than leaving you to weigh it.

## The four situations, and what the maths says

**You are behind, with time left.** Hold. There is no second prize, so
reducing risk converts a low chance of winning into no chance of winning.
The contest simulator is unambiguous: P(win) rises monotonically as
concentration rises, right down to a single name.

**You are ahead, near the end.** Hold. Concentration has done its job.
Switching now pays another round trip (~0.56%) and re-randomises the outcome
for no gain in win probability.

**Your name is halted or suspended.** Switch, if you still can. A halted
stock is not a position, it is a hostage: you cannot exit and you cannot
compound. This is the single case where acting fast matters.

**The thesis broke.** The rule bought a high-momentum name. If momentum has
decisively reversed -- the stock has given back the move that selected it --
the reason you own it is gone. `MOMENTUM_GIVEBACK` is where that is called.

## What is deliberately NOT a reason to sell

A drawdown, on its own. The strategy's median outcome is a small loss and
its 5th percentile is about -36%; a 20% fall is an ordinary draw from the
distribution you chose, not new information. Selling there converts variance
-- the thing you are being paid in -- into a realised loss. The single most
likely way to lose this contest badly is to panic out of a normal drawdown
and then re-enter something else after a cost.

This module encodes that. It will tell you to hold through a 30% fall, and
it is right to, given the money is the organisers'. On your own money none
of this applies.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

# Give back this much of the 60-day move that selected the name, and the
# reason you own it has gone. Not a stop loss -- it is measured against the
# SELECTION signal, not against your entry price.
MOMENTUM_GIVEBACK = 0.60

# Inside this many trading days, switching costs a round trip and re-rolls
# the dice with too little time to recover from a bad roll.
ENDGAME_DAYS = 3

# A round trip on moomoo including a 15bps half-spread. Any switch must be
# worth more than this before it is worth doing.
ROUND_TRIP_PCT = 0.0056


class Action(str, Enum):
    HOLD = "HOLD"
    SWITCH = "SWITCH"
    EXIT = "EXIT"


@dataclass(frozen=True)
class Verdict:
    action: Action
    reason: str
    detail: str = ""

    def __str__(self) -> str:
        return f"{self.action.value}: {self.reason}"


@dataclass(frozen=True)
class Situation:
    """Everything the rules are allowed to look at.

    Deliberately small. A rule that can see more finds more reasons.
    """
    days_left: int
    my_return: float               # contest-to-date, whole account
    position_return: float         # the holding, from your entry
    target_return: float = 0.29    # the score that wins, from the simulator
    halted: bool = False
    still_eligible: bool = True    # passes the liquidity/price screen today
    momentum_at_entry: float = 0.0
    momentum_now: float = 0.0

    @property
    def giveback(self) -> float:
        """Fraction of the selecting move that has been handed back."""
        if self.momentum_at_entry <= 0:
            return 0.0
        lost = self.momentum_at_entry - self.momentum_now
        return max(0.0, lost / self.momentum_at_entry)


def decide(s: Situation) -> Verdict:
    """What to do. Ordered so the urgent, irreversible cases come first."""
    # 1. Halted. The only true emergency: you cannot exit a halted stock, so
    #    if it is trading again and you can move, that decision is now.
    if s.halted:
        return Verdict(
            Action.SWITCH,
            "the position is halted or suspended",
            "A halted stock cannot be exited or compounded. Move to the next "
            "eligible name the moment it trades. This is the one case where "
            "speed matters more than deliberation.")

    # 2. Dropped out of the tradable universe -- usually liquidity collapsing,
    #    which means the exit you are counting on may not be there later.
    if not s.still_eligible:
        return Verdict(
            Action.SWITCH,
            "the name no longer passes the liquidity screen",
            "Thin names are exitable until suddenly they are not. Rotate to "
            "the top-ranked name that still passes.")

    # 3. Endgame. Concentration has either worked or not; churning cannot help.
    if s.days_left <= ENDGAME_DAYS:
        return Verdict(
            Action.HOLD,
            f"{s.days_left} day(s) left — too late for a switch to pay",
            f"A round trip costs ~{ROUND_TRIP_PCT:.2%} and re-randomises the "
            f"outcome with no time to recover from a bad draw. Whatever you "
            f"hold, hold it.")

    # 4. The thesis, not the price. Momentum bought this name; if the move
    #    that selected it is mostly gone, the reason to own it is gone --
    #    regardless of whether you are up or down on the trade.
    if s.giveback >= MOMENTUM_GIVEBACK:
        return Verdict(
            Action.SWITCH,
            f"momentum has given back {s.giveback:.0%} of the selecting move",
            "This is not a stop loss. It is measured against the SIGNAL, not "
            "your entry price: the rule bought a stock that had just moved, "
            "and it no longer has. Rotate to the current top rank.")

    # 5. Behind. The tempting moment, and the one where the maths is clearest.
    if s.my_return < s.target_return:
        gap = s.target_return - s.my_return
        return Verdict(
            Action.HOLD,
            f"behind by {gap:.1%} with {s.days_left} days — stay concentrated",
            "There is no second prize. De-risking here converts a small "
            "chance of winning into none. A drawdown is not information: the "
            "5th percentile of this strategy is about -36%, so a fall of that "
            "order is an ordinary draw from the distribution you chose.")

    # 6. Ahead. Also hold -- for a different reason.
    return Verdict(
        Action.HOLD,
        f"ahead of the winning score with {s.days_left} days left",
        "Concentration has done its job. Switching pays a round trip and "
        "re-rolls the dice for no gain in win probability.")


def explain_no_stop_loss() -> str:
    """Why there is no stop loss, in one paragraph, for the dashboard.

    Written out because this is the rule people override, and overriding it
    is the most likely way to lose badly.
    """
    return (
        "**There is no stop loss, on purpose.** This strategy is paid in "
        "variance: its median outcome is a small loss and its 95th percentile "
        "is +43%. A 20-30% fall is an ordinary draw, not new information, and "
        "cutting there realises the loss while giving up the tail you are "
        "playing for. The rules above sell for *structural* reasons — halted, "
        "illiquid, thesis broken — never because the price fell. "
        "**This is correct only because the capital is the organisers'. "
        "Never run it on your own money.**")
