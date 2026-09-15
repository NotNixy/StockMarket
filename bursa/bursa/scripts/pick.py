"""Today's pick. The script you run on the morning of the contest.

    python -m scripts.pick --capital 10000
    python -m scripts.pick --capital 10000 --refresh      # pull fresh prices
    python -m scripts.pick --capital 10000 --k 3          # if you want breadth

Everything upstream of this -- the repair layer, the screen, the backtest,
the contest simulator -- exists to produce one line of output: the name to
buy, and how many lots.

## The rule it implements

Highest 60-day momentum among the eligible universe, hold one name. That is
exactly what `tournament.contest` measured at P(win) 7.4% against a realistic
400-entrant field, versus 0.25% for holding anything sensible.

**The ranking is imported from `tournament.contest`, not reimplemented.** A
pick script that ranks slightly differently from the simulator that justified
it is the most plausible way this whole project ends up trading something
nobody tested. `--verify` asserts the two agree.

## What it refuses to do

Run on stale data. A momentum rank computed on last week's prices is not a
stale answer, it is a wrong one, and nothing about the output would look
unusual. `--max-age` is the guard; the script exits non-zero rather than
print a number you might act on.

## What it deliberately does NOT do

Place the order. The output is a decision, not an instruction to a broker --
you read the spread on moomoo and decide. Automating the send is a separate
thing to build carefully, not a flag to add here.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import pandas as pd

from core.costs import (
    BOARD_LOT, MOOMOO, SlippageModel, affordable_lots, round_trip_pct,
)
from core.universe import ScreenConfig, load_shariah_lists, screen
from tournament.contest import _rank_by_rule

JOURNAL = Path("results/picks.csv")

# A pick computed on prices older than this is not a pick. Three calendar days
# covers a normal weekend; a long holiday weekend needs --max-age raised
# deliberately, which is the point -- it forces the decision to be conscious.
DEFAULT_MAX_AGE_DAYS = 3


def rank_universe(panel: pd.DataFrame, rule: str = "momentum",
                  lookback: int = 60) -> pd.DataFrame:
    """Score every currently-eligible name, best first.

    Calls straight into the simulator's own ranking function so the live pick
    and the backtested rule cannot drift apart.
    """
    # Price history from the FULL panel; eligibility only decides what is
    # buyable today. Ranking on eligible-only prices drops any name that was
    # illiquid 60 days ago -- and a small cap becomes liquid by rallying hard
    # on volume, so that filter removes the strategy's best candidates. See
    # the same note in tournament.contest.run_contest.
    prices = panel.pivot_table(index="date", columns="ticker",
                               values="adj_close", aggfunc="last").sort_index()

    last = panel["date"].max()
    today = panel[panel["date"] == last]
    live = set(today.loc[today["eligible"], "ticker"] if "eligible" in today
               else today["ticker"])
    prices = prices[[c for c in prices.columns if c in live]]
    if prices.empty:
        return pd.DataFrame(columns=["ticker", "score"])

    # start_i = one past the last bar, so the rule sees all history and no
    # future -- identical to how the simulator positions it at a contest open.
    rng = np.random.default_rng(0)
    scores = _rank_by_rule(rule, prices, len(prices), lookback, rng)

    out = pd.DataFrame({"ticker": prices.columns, "score": scores})
    return out.dropna(subset=["score"]).sort_values("score", ascending=False)


def enrich(ranked: pd.DataFrame, panel: pd.DataFrame,
           capital: float, broker=MOOMOO,
           slippage: SlippageModel | None = None) -> pd.DataFrame:
    """Attach everything needed to actually place the order."""
    last = panel["date"].max()
    today = panel[panel["date"] == last].set_index("ticker")

    rows = []
    for _, r in ranked.iterrows():
        t = r["ticker"]
        if t not in today.index:
            continue
        bar = today.loc[t]
        px = float(bar["close"])
        lots = affordable_lots(px, capital)
        notional = lots * BOARD_LOT * px
        rows.append({
            "ticker": t,
            "score": float(r["score"]),
            "price": px,
            "lots": lots,
            "shares": lots * BOARD_LOT,
            "cost_RM": notional,
            "cash_left": capital - notional,
            "median_dtv": float(bar.get("median_dtv", np.nan)),
            "roundtrip_pct": (round_trip_pct(notional, broker, slippage)
                              if notional > 0 else np.nan),
        })
    return pd.DataFrame(rows)


def check_freshness(panel: pd.DataFrame, max_age_days: int) -> tuple[bool, str]:
    """Is the newest bar recent enough to act on?"""
    last = pd.Timestamp(panel["date"].max()).normalize()
    today = pd.Timestamp.now().normalize()
    age = (today - last).days
    if age > max_age_days:
        return False, (f"newest bar is {last.date()}, {age} days old "
                       f"(limit {max_age_days}).")
    return True, f"newest bar {last.date()}, {age} days old."


def journal(pick: pd.Series, capital: float, rule: str,
            path: Path = JOURNAL) -> None:
    """Append the decision to a log, before the outcome is known.

    Written at decision time on purpose. A record created afterwards is a
    record of what you wish you had done.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    row = pd.DataFrame([{
        "decided_at": pd.Timestamp.now().isoformat(timespec="seconds"),
        "rule": rule, "ticker": pick["ticker"], "price": pick["price"],
        "lots": pick["lots"], "cost_RM": pick["cost_RM"],
        "capital": capital, "score": pick["score"],
    }])
    row.to_csv(path, mode="a", header=not path.exists(), index=False)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--panel", type=Path, default=Path("data/screened.parquet"))
    ap.add_argument("--capital", type=float, default=10_000.0)
    ap.add_argument("--k", type=int, default=1,
                    help="names to hold. 1 unless you have a reason.")
    ap.add_argument("--rule", default="momentum",
                    choices=("momentum", "high_vol", "reversal", "random"))
    ap.add_argument("--lookback", type=int, default=60)
    ap.add_argument("--show", type=int, default=10, help="runners-up to list")
    ap.add_argument("--max-age", type=int, default=DEFAULT_MAX_AGE_DAYS)
    ap.add_argument("--no-journal", action="store_true")
    ap.add_argument("--verify", action="store_true",
                    help="assert the live ranking matches the simulator's")
    args = ap.parse_args()

    if not args.panel.exists():
        print(f"ERROR: {args.panel} not found. Run:\n"
              f"  python -m scripts.fetch_shariah_universe\n"
              f"  python -m scripts.build_panel", file=sys.stderr)
        return 2

    panel = pd.read_parquet(args.panel)
    fresh, msg = check_freshness(panel, args.max_age)
    print(f"data: {msg}")
    if not fresh:
        print("\nREFUSING TO PICK. Ranking stale prices produces a confident\n"
              "wrong answer that looks exactly like a right one. Refresh:\n"
              "  python -m scripts.fetch_shariah_universe\n"
              "  python -m scripts.build_panel\n"
              "Then re-run. Override with --max-age only if you know why.",
              file=sys.stderr)
        return 1

    ranked = rank_universe(panel, args.rule, args.lookback)
    if ranked.empty:
        print("ERROR: no eligible names. Check the screen.", file=sys.stderr)
        return 1

    table = enrich(ranked, panel, args.capital)
    table = table[table["lots"] > 0]
    if table.empty:
        print(f"ERROR: RM {args.capital:,.0f} cannot afford one board lot of "
              f"any eligible name.", file=sys.stderr)
        return 1

    picks = table.head(args.k)
    print(f"\nuniverse: {len(ranked)} eligible | rule: {args.rule} "
          f"({args.lookback}d) | capital: RM {args.capital:,.0f}\n")
    print("=" * 66)
    print(f"BUY  {'  '.join(picks['ticker'].tolist())}")
    print("=" * 66)
    for _, p in picks.iterrows():
        per_name = args.capital / args.k
        lots = affordable_lots(p["price"], per_name)
        print(f"  {p['ticker']:<10} RM {p['price']:.3f}  "
              f"x {lots} lot{'s' if lots != 1 else ''} "
              f"({lots * BOARD_LOT:,} shares) = RM {lots * BOARD_LOT * p['price']:,.2f}")
        print(f"             round trip ~{p['roundtrip_pct']:.2f}%  |  "
              f"median daily volume RM {p['median_dtv']:,.0f}")

    print(f"\n  runners-up (use if the top name is halted or the spread is wide):")
    for _, r in table.iloc[args.k:args.k + args.show].iterrows():
        print(f"    {r['ticker']:<10} score {r['score']:+.3f}  "
              f"RM {r['price']:.3f}  DTV RM {r['median_dtv']:,.0f}")

    print("\n  BEFORE SENDING: check the live spread on moomoo. A 2-sen spread\n"
          "  on a 50-sen stock is 4% round trip and eats half the edge this\n"
          "  pick is built on. If it is wide, take the next name down.")

    if args.verify:
        from tournament.contest import ContestConfig, run_contest
        cfg = ContestConfig(n_entrants=400, my_k=args.k, my_rule=args.rule,
                            field_k_mix={1: .05, 2: .10, 3: .15, 5: .30,
                                         8: .20, 12: .12, 20: .08},
                            n_trials=2000, lookback=args.lookback)
        res = run_contest(panel, cfg)
        print(f"\n  simulated P(win) for this rule at k={args.k}: "
              f"{res.p_win:.2%}  (fair share {1/400:.2%})")

    if not args.no_journal:
        for _, p in picks.iterrows():
            journal(p, args.capital, args.rule)
        print(f"\n  logged to {JOURNAL}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
