# bursa

A ranking and backtesting system for Bursa Malaysia equities, built for a
one-month trading competition and intended to outlive it.

## What this is

Two things sharing one data layer:

- **`research/`** — an honest backtest harness. Its job is to tell you whether
  a strategy works, including when the answer is no.
- **`tournament/`** — position tracking and the standing calculator for a
  winner-takes-all competition, where the objective is *probability of
  finishing first*, not expected return. Those require opposite strategies.

## Design decisions

**The execution layer is swappable.** Everything upstream of it is pure
computation whose output is *target positions*, not buy/sell instructions.
The same signal code drives three backends — `simulator` (backtest), `paper`
(moomoo's simulated account), and `live` — so a backtest and a live run
cannot silently compute different things.

```
data → features → model → signal → sizing → EXECUTION → reconcile → journal
                                                ↑
                                  simulator | paper | live
```

**`costs.py` and `baselines.py` are isolated files.** Those are the two
things anyone is tempted to quietly weaken when results disappoint. On their
own, they are hard to fudge without noticing.

**Results are committed alongside code.** Every backtest run is written to
`results/` dated, and committed with the code that produced it. The git
history is therefore a record of how many strategy variants were tried —
which is a required input to any honest multiple-testing correction, and
not something anyone remembers accurately by month four.

**The universe is point-in-time.** Liquidity screens and Shariah-compliance
status are evaluated *as of each date*, never once at the end. Using today's
list against historical data is lookahead bias, and structurally the same
error as survivorship bias.

## Costs, because they decide more than the strategy does

Round-trip cost as a percentage of position value (statutory charges only):

| Trade size | moomoo (0%) | 0.1%, min RM8 | min RM30 |
|---:|---:|---:|---:|
| RM 500 | 0.46% | 3.92% | 13.42% |
| RM 1,000 | 0.26% | 1.99% | 6.74% |
| RM 10,000 | 0.26% | 0.48% | 0.91% |

Add a realistic 15bps half-spread and moomoo's round trip becomes **0.56%**.
That is the hurdle a signal must clear before it has earned anything.

Notes: stamp duty is RM 1 per RM 1,000 *or part thereof* — it rounds up, so a
RM 500 trade pays RM 1. SST applies to the brokerage component only. Bursa
trades in board lots of 100 shares, which is why a small account cannot hold
many names.

## Layout

```
core/
  costs.py       Bursa fee model. Pure functions, no I/O.
  loader.py      yfinance fetch + Parquet cache, incremental.
  universe.py    point-in-time liquidity + Shariah screen.
  validate.py    ten integrity checks, PAIP-style.
research/
  metrics.py       Sharpe, drawdown, and the DEFLATED Sharpe.
  backtest.py      the harness. Signal at t, fill at t+1, costs on turnover.
  baselines.py     buy & hold, equal weight, random entry at matched exposure.
  strategies.py    percentile-rank composite. Same shape as LIPS.
  harness_check.py negative + positive controls. Run this first.
tournament/
  standing.py      target score, gap, variance required, risk verdict.
  charts.py        chart builders. Validated palettes, light and dark.
  app.py           the Streamlit dashboard.
results/           dated run outputs, committed
tests/
```

## The dashboard

```bash
streamlit run tournament/app.py
```

Deploying to Streamlit Community Cloud: the entrypoint is
`tournament/app.py`, and `tournament/requirements.txt` exists so the install
works regardless of how deeply this repo sits inside the GitHub project.
Cloud only looks beside the entrypoint or at the repository root -- a
requirements file anywhere else is ignored, and the app boots with nothing but
streamlit installed.

Opens on `localhost:8501`. It starts on **synthetic data** so it runs before
any Bursa data exists — every panel is live, the numbers are invented, and a
banner says so until you populate `data/raw/`.

Four tabs, in the order you would use them:

- **Tournament** — the daily decision. Where you stand, the score that wins,
  P(win) at current versus concentrated risk, and a verdict: add risk, hold, or
  cut. The target comes from the entrant count via order statistics, so no
  leaderboard is needed.
- **Backtest** — the strategy against buy-and-hold, equal-weight, and the
  random-entry distribution at matched exposure.
- **Universe** — how many names were tradable over time.
- **Data health** — the ten integrity checks.

Chart colours were validated with a CVD/contrast checker against both
surfaces; the dark steps are the same hues re-stepped for a dark ground, not
an automatic flip. Worst adjacent colourblind separation sits in the band that
requires secondary encoding, so every series is direct-labelled as well as
listed in the legend. Don't strip those labels.

## Validate the harness before trusting it

```bash
python -m research.harness_check
```

Three controls, because a backtest engine that reports everything as
profitable is useless and so is one that reports everything as worthless:

- **Negative** — synthetic random walks, no edge exists. The engine must not
  find one. If it does, it leaks future information.
- **Positive** — the same data with a momentum edge deliberately planted. The
  engine must find it. Without this, the negative result proves nothing: an
  engine that earns nothing on anything would also pass it.
- **Deflation** — take the best of 100 worthless strategies and confirm the
  deflated Sharpe is unimpressed.

Current output on the deflation control:

```
best Sharpe found across 100 random strategies : +1.30
P(true SR > 0) if you pretend it was 1 trial   : 0.988
P(true SR > 0) accounting for all 100 trials   : 0.386
```

That gap is the entire argument for committing `results/` to git. The trial
count has to come from somewhere honest, and nobody remembers it by month four.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pytest -q
```

Fetching data:

```python
from core.loader import fetch_many, load_panel, add_returns

fetch_many(["1155", "5225", "6888"])       # .KL suffix added automatically
panel = add_returns(load_panel())
```

## Things that are deliberately not here

**Credentials.** Never committed, and `.gitignore` is written before
`git init` rather than after — once a broker credential is in git history it
is there permanently.

**Raw data pulls.** Bulky and re-downloadable. The hand-made artefacts that
*are* committed: the Shariah-compliant list parsed from the SC, and the
screened universe definition.

## Status

- [x] `core/costs.py` — 19 tests
- [x] `core/loader.py` — 13 tests
- [x] `core/universe.py` — 14 tests
- [x] `core/validate.py` — 21 tests
- [x] `research/metrics.py` — 20 tests
- [x] `research/backtest.py` — 15 tests
- [x] `research/baselines.py` — 13 tests
- [x] `research/strategies.py` — 26 tests
- [x] `tournament/standing.py` — 20 tests
- [x] `tournament/charts.py` + `app.py` — 8 tests, dashboard runs
- [ ] real Bursa data: ticker list, SC Shariah list parsed to CSV
- [ ] execution backends: paper, live

171 tests passing. Run `pytest -q`.

### The tests worth reading

Most are routine. These four are the ones holding the thing up:

- `test_signal_never_sees_future_bars` — the signal receives history up to the
  decision date and no further.
- `test_fills_happen_on_the_bar_after_the_signal` — paired with
  `test_a_genuine_oracle_does_make_money`, so passing cannot mean "the engine
  simply earns nothing".
- `test_median_traded_value_excludes_the_current_bar` — a stock that trades
  once, hugely, on day 150 must not be eligible on day 150.
- `test_deflated_sharpe_rejects_a_lucky_coin_flip` — best-of-200 random walks
  must not pass as a discovery.

### The Shariah list

`core/universe.py` expects a CSV of the SC's published releases — one row per
(release, ticker), each release a complete snapshot:

```
list_date,ticker
2025-05-30,1155
2025-05-30,5225
2025-11-28,1155
```

Compliance on any date resolves to the most recent release on or before it.
A date earlier than the first release yields an **empty** universe rather than
falling back to the oldest list — an empty universe is a loud failure, and a
silent fallback would be lookahead.
