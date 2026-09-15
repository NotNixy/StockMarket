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
| **1 name** | **momentum** | **~8%** | **~32x** | -1.7% | +1.5% | +45% |
| 1 name | high volatility | ~8% | ~32x | -5.7% | -2.0% | +46% |
| 1 name | random | ~2.3% | ~9x | -0.4% | +0.6% | +20% |
| 2 names | momentum | 3.4% | 14x | -0.2% | +1.0% | +31% |
| 3 names | momentum | 1.7% | 7x | +0.3% | +0.9% | +25% |
| 5 names | momentum | 0.3% | 1x | +0.4% | +0.7% | +18% |

Fair share -- what you get for turning up and holding anything sensible -- is
1/400 = 0.25%. Five names at any rule is indistinguishable from it.

The median winning score is **+29%** over the contest. Not +6%, not +15%.
You cannot reach it from a diversified book.

### Two numbers, and only one of them is trustworthy

An earlier version of this document said 7.40%, to two decimals. That was one
sample reported as if it were a measurement. Re-running the comparison inside
six consecutive slices of history (`scripts/rule_stability.py`) separates
what holds from what does not:

| period | random | momentum | high_vol | reversal | best |
|---|---|---|---|---|---|
| 2016-17 | 1.98% | 6.55% | 3.33% | 2.53% | momentum |
| 2018-19 | 2.55% | 8.80% | 4.10% | 5.12% | momentum |
| 2020-21 | 1.57% | 19.34% | 7.82% | 2.30% | momentum |
| 2022-23 | 1.42% | 8.38% | 6.91% | 5.82% | momentum |
| 2024-25 | 2.39% | 6.48% | 10.03% | 2.08% | high_vol |
| 2026 | 2.93% | 2.57% | 0.13% | 4.22% | reversal |

Measured on the rebuilt universe: six SC releases, 952 tickers, 67% of
eligible bars under point-in-time compliance rather than backfilled.

**Concentration is the edge, and it is stable.** Random picking at k=1 stays
between 1.5% and 2.9% in every single period -- about 9x fair share -- with
no selection skill whatsoever. That number you can lean on.

**The tilt adds more, but its size is not knowable in advance.** Momentum has
the best mean (8.7%) and wins four of six periods, but it ranges from 2.6% to
19.3% and loses outright in two. Quote it as **5-12%**, never as a single
figure.

**momentum and high_vol are the same idea.** Both buy names that have just
moved a long way, and such names have wide forward distributions. Choosing
between them on their mean difference would be fitting noise.

**low_vol is the control that confirms the mechanism**: 0.58% mean, worse
than random in every period. Picking calm stocks is how you reliably lose a
tournament.

The earlier version of this table was built on one SC release backfilled
across ten years. Rebuilding on six releases — 67% real compliance, plus 87
companies that were compliant once and are absent from the 2025 list —
*strengthened* momentum rather than weakening it (4 wins, up from 3). The
result was not a lookahead artefact.

### A warning about short samples

Restricted to the ten months where Shariah compliance is real rather than
backfilled, reversal won at 11.9% and high_vol collapsed to 0.03%. Read
alone, that looks like proof the whole finding was a lookahead artefact.
Read against the table above, it is one regime out of six -- roughly nine
independent months, which is nothing. Neither number means much without the
other, which is the entire reason the per-period view exists.

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
is most sensitive to — more than the choice of rule. If the contest attracts
sophisticated entrants, revise down hard.

---

## What it costs

At k=1 with momentum, over the contest:

- **Median outcome: about −1.7%.** The typical result is a small loss.
- **P(losing more than 30%): ~8%**
- **P(losing more than 50%): ~1.4%**
- **You do not win roughly 92% of the time.**

The mean is about +1.5%, and essentially all of it lives in the right tail.

---

## Where this is optimistic

1. **Survivorship — now partially fixed, and the remainder is measured.**
   Six SC releases name 991 distinct companies, of which 136 were compliant
   once and are gone from the 2025 list. Price history was recovered for 87
   of them; **52 returned HTTP 404 from Yahoo**, which is what a delisted
   Bursa company looks like from this data source. So the universe still
   cannot see roughly 52 companies that died, and single-stock left-tail risk
   — fraud, PN17, a halt you cannot trade out of — remains understated. The
   1.4% ruin figure is a floor, not an estimate.
2. **The field does not trade.** Every simulated opponent buys once and
   holds. Real entrants will churn, which raises their variance, which
   raises the winning score and lowers your odds.
3. **The field has no skill.** Opponents pick at random. Any genuine skill
   in the field makes these numbers worse.
4. **Backfilled Shariah compliance — mostly fixed.** Six releases now cover
   November 2020 onward, so 67% of eligible bars use real point-in-time
   membership. Only 2016-09 to 2020-11 is still backfilled, and
   `scripts/build_panel.py` prints the split on every run. Earlier releases
   would close the rest.
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
python -m scripts.fetch_sc_lists       # SC releases -> point-in-time compliance
python -m scripts.fetch_shariah_universe  # prices for every listed name
python -m scripts.build_panel          # repair, validate, screen
python -m scripts.run_contest          # concentration + rule sweeps
python -m scripts.run_walkforward      # the honest strategy evaluation
python -m scripts.compare_baselines    # strategy vs doing nothing
python -m scripts.rule_stability       # does the same rule win twice?
python -m scripts.pick --capital 10000 # today's name, with lot sizing
python -m pytest tests/ -q             # 390 tests
```

The contest simulator's controls are in `tests/test_contest.py`. The two
that matter: a symmetric field returns exactly 1/400, and a field as
concentrated as you removes the entire advantage.
