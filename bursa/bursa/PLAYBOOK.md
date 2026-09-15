# Contest playbook

400 entrants. ~21 trading days. Highest portfolio wins. Capital provided by
the organisers.

That last fact is what licenses everything below. When the downside is "I
don't win" rather than "I lose my savings", maximising variance is not
recklessness — it is the correct play. **None of this is how you would
invest your own money, and none of it should be reused on your own money.**

---

## The answer

**Hold one stock. Pick it by 60-day momentum. Do not diversify.**

Against a realistic field, simulated over ten years of actual Bursa windows:

| Your book | Rule | P(win) | vs. fair share | Median | Mean | 95th pct |
|---|---|---|---|---|---|---|
| **1 name** | **momentum** | **7.40%** | **30×** | −1.48% | +1.52% | +43.26% |
| 1 name | high volatility | 6.47% | 26× | −4.26% | −2.02% | +33.85% |
| 1 name | random | 2.53% | 10× | +0.00% | +0.63% | +22.65% |
| 2 names | momentum | 3.38% | 14× | −0.21% | +0.95% | +31.12% |
| 3 names | momentum | 1.70% | 7× | +0.33% | +0.87% | +25.00% |
| 5 names | momentum | 0.26% | 1× | +0.38% | +0.68% | +17.65% |

Fair share — what you get for turning up and holding anything sensible — is
1/400 = 0.25%. Five names at any rule is indistinguishable from it.

The median winning score is **+29%** over the contest. Not +6%, not +15%.
You cannot reach it from a diversified book.

---

## Why momentum, when the backtest said momentum has no edge

Both are true, and the tension is the most useful thing in this document.

The walk-forward found momentum's deflated Sharpe was **0.12** — its ranking
ability is indistinguishable from noise. That has not changed.

What momentum does reliably is **select high-variance names**. The top
60-day mover is, by construction, a stock that just moved a long way, and
such stocks have wide forward distributions. Look at the median column: at
k=1, momentum's median outcome is **worse** than random's (−1.48% vs 0.00%)
while its 95th percentile is nearly double (+43% vs +23%).

So momentum is not being used here as a forecast. It is being used as a
variance selector, and it is a better one than picking on volatility
directly (7.40% vs 6.47%) because it selects for recent *upward* moves and
inherits their right skew.

This is why the strategy is worth running for a contest and worthless for
an account.

---

## The assumption everything rests on

**Your edge is being more concentrated than the field, not being
concentrated.** P(win) holding one name, as the field changes:

| Field holds | P(win) | vs. fair |
|---|---|---|
| 20 names each | 26.2% | 105× |
| 10 names each | 20.7% | 83× |
| 5 names each | 13.9% | 55× |
| 3 names each | 8.3% | 33× |
| 2 names each | 5.4% | 22× |
| **1 name each** | **0.68%** | **2.7×** |

If every entrant reasons this way, the advantage evaporates completely — it
becomes a 400-way lottery. The 7.40% headline assumes a mixed field
(5% punting on one name, most holding 3–8, some holding 20), which is how
amateur fields usually look. It is a **guess**, and it is the input P(win)
is most sensitive to. If the contest attracts sophisticated entrants,
revise down hard.

---

## What it costs

At k=1 with momentum, over the contest:

- **Median outcome: −1.48%.** The typical result is a small loss.
- **P(losing more than 30%): 8.3%**
- **P(losing more than 50%): 1.4%**
- **You do not win 92.6% of the time.**

The mean is +1.52%, and essentially all of it lives in the right tail.

---

## Where this is optimistic

1. **Survivorship.** The universe is the November 2025 SC list, so every
   name in it survived to 2025. No historical window contains a company that
   was suspended, delisted, or went to zero. Single-stock left-tail risk —
   fraud, PN17, a halt you cannot trade out of — is therefore **absent from
   the simulation entirely.** Real P(catastrophe) on one small-cap name over
   a month is not the 1.4% above.
2. **The field does not trade.** Every simulated opponent buys once and
   holds. Real entrants will churn, which raises their variance, which
   raises the winning score and lowers your odds.
3. **The field has no skill.** Opponents pick at random. Any genuine skill
   in the field makes these numbers worse.
4. **Backfilled Shariah compliance.** One SC release applied backwards over
   ten years — lookahead, optimistic, size unknown.
5. **Price return, not total return.** 227 of 865 tickers have
   `adj_close == close` throughout and the data cannot say whether that means
   "never paid a dividend" or "never adjusted one".

---

## Execution

**Selection.** One name, highest 60-day momentum among the eligible set —
Shariah-compliant, median daily traded value ≥ RM 500k, price RM 0.20–50.
Re-run `scripts/run_contest.py` the morning the contest opens; the ranking
moves.

**Liquidity check before committing.** At RM 500k median DTV your whole
position is a small fraction of a day's volume, so entry is fine. Verify the
spread by eye on moomoo before sending — a 2-sen spread on a 50-sen stock is
4% round trip and eats a quarter of your edge.

**Costs.** One round trip, moomoo: ~0.56% including 15bps half-spread. At
k=1 this is negligible. Do not churn — every rebalance is another 0.56% and
the simulation assumes you trade twice, total.

**When to deviate.** If you are ahead near the end, concentration has done
its job and holding is correct. If you are far behind with days left, there
is no second prize — the only losing move is to de-risk.

---

## Reproducing this

```
python -m scripts.build_panel          # raw -> repaired, validated panel
python -m scripts.run_contest          # concentration + rule sweeps
python -m scripts.run_walkforward      # the honest strategy evaluation
python -m scripts.compare_baselines    # strategy vs doing nothing
python -m pytest tests/ -q             # 294 tests
```

The contest simulator's controls are in `tests/test_contest.py`. The two
that matter: a symmetric field returns exactly 1/400, and a field as
concentrated as you removes the entire advantage.
