from __future__ import annotations

import sys
from pathlib import Path

# Streamlit puts the SCRIPT's directory on sys.path, not the working directory,
# so `core` and `research` are invisible when launched as
# `streamlit run tournament/app.py`. Put the repo root on the path first.
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Bursa", layout="wide",
                   initial_sidebar_state="expanded")


# --------------------------------------------------------------------------
# Imports, guarded
# --------------------------------------------------------------------------
# Streamlit Community Cloud REDACTS the message of any uncaught exception:
#
#   ImportError: This app has encountered an error. The original error message
#   is redacted to prevent data leaks.
#
# For an ImportError the message is the entire diagnosis -- "cannot import
# name X from Y" versus "No module named matplotlib" are different problems
# with different fixes, and the redacted version distinguishes them not at
# all. A whole debugging session went into an error that says nothing.
#
# Catching it here and rendering our OWN string defeats the redaction: the
# text below is authored by this file, not by the exception handler, so
# Streamlit passes it through. Everything shown is repository structure and
# package versions -- no data, which is what the redaction exists to protect.
def _fail(title: str, exc: Exception) -> None:
    st.error(f"**{title}**\n\n`{type(exc).__name__}: {exc}`")
    st.markdown(
        "Most likely causes, in the order worth checking:\n\n"
        "1. **The deployed build is stale.** If this repo looks correct on "
        "GitHub, the running container is on an older commit. Manage app "
        "-> Reboot. This is the usual one.\n"
        "2. **A package is missing.** `requirements.txt` must sit in the "
        "repo root or beside this file; Cloud reads nowhere else.\n"
        "3. **A name genuinely moved** between modules.")

    with st.expander("Diagnostics"):
        st.write("**Repo root on sys.path:**", str(_ROOT))
        st.write("**Root contents:**",
                 sorted(p.name for p in _ROOT.iterdir()) if _ROOT.exists()
                 else "MISSING")
        pkg = _ROOT / "tournament"
        st.write("**tournament/ contents:**",
                 sorted(p.name for p in pkg.iterdir()) if pkg.exists()
                 else "MISSING")
        # What the module ACTUALLY exports, which is the answer whenever the
        # error is "cannot import name".
        try:
            import tournament.charts as _c
            st.write("**tournament.charts exports:**",
                     sorted(n for n in dir(_c) if not n.startswith("_")))
            st.write("**loaded from:**", getattr(_c, "__file__", "?"))
        except Exception as e:            # noqa: BLE001 - diagnostic path
            st.write("**tournament.charts failed to import:**",
                     f"{type(e).__name__}: {e}")
        for mod in ("matplotlib", "scipy", "pandas", "numpy", "pyarrow"):
            try:
                st.write(f"**{mod}:**", __import__(mod).__version__)
            except Exception as e:        # noqa: BLE001 - diagnostic path
                st.write(f"**{mod}:**", f"NOT AVAILABLE ({type(e).__name__})")
        st.write("**sys.path:**", sys.path)
    st.stop()


try:
    from core.costs import MOOMOO, SlippageModel, round_trip_pct
    from core.feed import get_feed
    from core.quotes import (
        Position, Quote, market_status, standing,
    )
    from core.validate import validate
    from research.backtest import BacktestConfig, run_backtest
    from research.baselines import (
        percentile_vs_random, run_buy_and_hold, run_equal_weight,
        run_random_trials,
    )
    from research.baserates import (
        DEFAULT_HORIZON, base_rates, current_bucket,
    )
    from research.harness_check import synth_panel
    from research.strategies import PRESETS
    from research.walkforward import walk_forward
    from tournament.charts import (
        DARK, LIGHT, equity_chart, gap_chart, pwin_chart,
        random_distribution_chart, universe_chart, walkforward_chart,
    )
    from tournament.rules import (
        Action, Situation, decide, explain_no_stop_loss,
    )
    from tournament.standing import Contest, assess, strategy_table
except ImportError as exc:
    _fail("The app could not import its own modules.", exc)

DATA_DIR = Path("data/raw")
SCREENED = Path("data/screened.parquet")


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_data(use_real: bool):
    """The screened panel if it exists, then the raw cache, else synthetic.

    The screened file is preferred and it matters. This used to load the raw
    cache and set `eligible = True` on every row, which silently switched the
    entire screen off: the Backtest, Walk-forward and Universe tabs were
    running over all 865 fetched tickers -- including names that fail the
    Shariah, liquidity and price filters -- while the Universe tab cheerfully
    reported everything as eligible. Every number on real data was computed
    over a universe you could not actually have traded.

    `data/screened.parquet` carries the real per-row `eligible` flag from
    core.universe.screen(). Build it with:

        python -m scripts.build_panel
    """
    if use_real and SCREENED.exists():
        return pd.read_parquet(SCREENED), True

    if use_real and DATA_DIR.exists() and any(DATA_DIR.glob("*.parquet")):
        from core.loader import load_panel
        panel = load_panel(raw_dir=DATA_DIR)
        # No screen available. Marking everything eligible is a LIE that this
        # branch is honest about: the flag is set so the tabs render, and
        # `screened` comes back False so the UI can say so.
        panel["eligible"] = True
        return panel, "unscreened"
    return synth_panel(n_tickers=40, n_days=750, seed=1), False


@st.cache_data(show_spinner=False)
def run_everything(panel: pd.DataFrame, preset: str, top_n: int,
                   n_random: int):
    cfg = BacktestConfig(capital=10_000, rebalance="ME", broker=MOOMOO,
                         slippage=SlippageModel(15.0, 0.0), max_positions=top_n)
    strat = PRESETS[preset](top_n=top_n)
    res = run_backtest(panel, strat.signal, cfg)
    bh = run_buy_and_hold(panel, cfg)
    ew = run_equal_weight(panel, cfg)
    randoms = run_random_trials(panel, top_n, cfg, n_trials=n_random, seed=0)
    return strat, res, bh, ew, randoms, cfg


# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------
st.sidebar.title("Bursa")

dark = st.sidebar.toggle("Dark charts", value=False)
theme = DARK if dark else LIGHT

st.sidebar.subheader("Contest")
n_entrants = st.sidebar.number_input("Entrants", 2, 5000, 400, step=10)
total_days = st.sidebar.number_input("Contest length (trading days)", 1, 250, 21)
field_vol = st.sidebar.slider(
    "Assumed field volatility", 0.02, 0.30, 0.05, 0.01,
    help="Volatility of a typical entrant over the whole contest. ~5% if the "
         "field holds diversified portfolios. Raise it if you think other "
         "entrants are also concentrating -- your edge shrinks sharply.")
my_return = st.sidebar.slider("My return so far", -0.50, 1.00, 0.03, 0.01)
days_left = st.sidebar.number_input("Trading days remaining", 0, 250, 15)

st.sidebar.subheader("Strategy")
preset = st.sidebar.selectbox("Preset", list(PRESETS), index=0)
top_n = st.sidebar.slider("Positions held", 1, 10, 3)
n_random = st.sidebar.select_slider("Random baseline trials",
                                    [20, 40, 60, 100], value=40)

st.sidebar.subheader("Data")
use_real = st.sidebar.checkbox("Use cached Bursa data if present", value=True)

panel, is_real = load_data(use_real)
contest = Contest(n_entrants=int(n_entrants), field_vol=float(field_vol),
                  total_days=int(total_days))

if is_real == "unscreened":
    # The dangerous middle state: real prices, no screen. Louder than the
    # synthetic banner, because invented numbers are obviously invented and
    # these are not -- they look exactly like results.
    st.error(
        "**Real prices, NO SCREEN.** Every name in the raw cache is being "
        "treated as tradable, including ones that fail the Shariah, "
        "liquidity and price filters. Results below are computed over a "
        "universe you could not have traded. Build the screened panel with "
        "`python -m scripts.build_panel`.", icon="🚨")
elif not is_real:
    st.warning(
        "**Synthetic data.** Every figure below is generated from random "
        "walks, not Bursa. The dashboard is live; the numbers are invented. "
        f"Populate `{DATA_DIR}` with `core.loader.fetch_many()` to replace them.",
        icon="⚠️")
else:
    st.caption(f"Screened panel: {panel['ticker'].nunique()} tickers, "
               f"{len(panel):,} bars, "
               f"{panel['date'].min().date()} to {panel['date'].max().date()}.")

tab_l, tab_r, tab_t, tab_b, tab_w, tab_u, tab_d = st.tabs(
    ["Live", "Stock report", "Tournament", "Backtest", "Walk-forward",
     "Universe", "Data health"])


# --------------------------------------------------------------------------
# Stock report
# --------------------------------------------------------------------------
# The honest form of "should I buy this": not a direction, a distribution.
# See research.baserates for why the buckets are cross-sectional and why no
# p-value appears anywhere on this tab.
with tab_r:
    st.subheader("What happened to stocks that looked like this one")

    if is_real is not True:
        st.warning("Base rates need the screened panel. Build it with "
                   "`python -m scripts.build_panel`.", icon="⚠️")
    else:
        rc1, rc2, rc3 = st.columns([2, 1, 1])
        live_names = sorted(
            panel.loc[(panel["date"] == panel["date"].max())
                      & panel["eligible"], "ticker"].unique())
        r_ticker = rc1.selectbox("Ticker", live_names,
                                 index=0 if live_names else None)
        r_factor = rc2.selectbox("Factor",
                                 ["momentum", "volatility", "reversal"])
        r_horizon = rc3.number_input("Horizon (trading days)", 5, 120,
                                     DEFAULT_HORIZON)

        @st.cache_data(show_spinner="computing base rates...")
        def _rates(_p, factor, horizon):
            return base_rates(_p, factor, horizon)

        if r_ticker:
            pos = current_bucket(panel, r_ticker, r_factor)
            if not pos["found"]:
                st.error(pos["reason"])
            else:
                stats = _rates(panel, r_factor, int(r_horizon))
                mine = next((x for x in stats
                             if x.bucket == pos["bucket"]), None)

                m1, m2, m3 = st.columns(3)
                m1.metric("Rank",
                          f"{pos['rank']} of {pos['n_universe']}")
                m2.metric(f"{r_factor} decile", f"{pos['bucket']} of 10")
                m3.metric(f"{r_factor}", f"{pos['factor_value']:+.1%}")

                if mine is None:
                    st.warning("No history for this decile.")
                else:
                    st.markdown(f"**Decile {pos['bucket']} over the next "
                                f"{int(r_horizon)} trading days:**")
                    b1, b2, b3, b4 = st.columns(4)
                    b1.metric("Median", f"{mine.median:+.2%}")
                    b2.metric("5th – 95th",
                              f"{mine.p05:+.0%} … {mine.p95:+.0%}")
                    b3.metric("Ended higher", f"{mine.p_positive:.0%}")
                    b4.metric("Lost >30%", f"{mine.p_loss_30:.1%}")

                    # The spread IS the finding. Stating it beside the median
                    # stops the median being read as an expectation.
                    st.info(
                        f"The median is {mine.median:+.2%} and the range is "
                        f"{mine.p05:+.0%} to {mine.p95:+.0%}. **The range is "
                        f"the finding**, not the median — this is a "
                        f"distribution, not a forecast.", icon="📊")

                    table = pd.DataFrame([{
                        "decile": x.bucket, "n": x.n,
                        "independent periods": round(x.independent_periods),
                        "median": x.median, "5th": x.p05, "95th": x.p95,
                        "ended higher": x.p_positive,
                        "lost >30%": x.p_loss_30,
                    } for x in stats]).set_index("decile")
                    st.dataframe(
                        table.style.format({
                            "median": "{:+.2%}", "5th": "{:+.1%}",
                            "95th": "{:+.1%}", "ended higher": "{:.1%}",
                            "lost >30%": "{:.1%}", "n": "{:,}"}),
                        use_container_width=True)

                    st.caption(
                        f"{mine.n:,} observations behind decile "
                        f"{pos['bucket']}, but only ~"
                        f"{mine.independent_periods:.0f} independent "
                        f"{int(r_horizon)}-day periods — daily windows "
                        f"overlap by {int(r_horizon) - 1} of {int(r_horizon)} "
                        f"days. Roughly 52 delisted companies are missing "
                        f"from the panel, so the loss figures are floors.")


# --------------------------------------------------------------------------
# Live
# --------------------------------------------------------------------------
# Delayed quotes, labelled as delayed. See core.quotes for why this is not and
# cannot be a real-time feed on a deployed dashboard, and why the strategy
# does not need one.
with tab_l:
    st.subheader("Where you stand, right now")

    lc1, lc2, lc3, lc4 = st.columns([2, 1, 1, 1])
    live_ticker = lc1.text_input("Ticker", value="0270.KL",
                                 help="The name you are holding.")
    live_shares = lc2.number_input("Shares", min_value=0, value=6000, step=100)
    live_entry = lc3.number_input("Entry price (RM)", min_value=0.0,
                                  value=1.660, step=0.005, format="%.3f")
    live_capital = lc4.number_input("Starting capital (RM)", min_value=1.0,
                                    value=10_000.0, step=500.0)

    state = market_status()
    st.caption(f"Bursa is **{state.value}** "
               f"(09:00–12:30 and 14:30–17:00 Malaysia time).")

    if st.button("Refresh quote", type="primary"):
        st.cache_data.clear()

    # Real-time via moomoo's OpenD when this is running on your machine and
    # the gateway is up; Yahoo's delayed feed otherwise. The panel states
    # which one it got -- a fallback nobody notices is the dangerous kind.
    @st.cache_data(ttl=30, show_spinner="fetching quote...")
    def _live(t: str):
        feed = get_feed()
        q = feed.quote(t)
        # Quote is frozen; return plain fields so Streamlit can cache it.
        return {"price": q.price, "prev": q.prev_close, "at": q.quoted_at,
                "stale": q.staleness(), "err": q.error,
                "high": q.day_high, "low": q.day_low, "vol": q.volume,
                "feed": feed.name, "realtime": feed.realtime,
                "feed_desc": feed.describe()}

    info = _live(live_ticker.strip().upper())
    if info["realtime"]:
        st.success(f"Feed: **{info['feed']}** — real-time.", icon="🟢")
    else:
        st.warning(f"Feed: **{info['feed']}**. {info['feed_desc']}", icon="🟡")

    if info["err"] or info["price"] is None:
        st.error(f"No quote for {live_ticker}: {info['err'] or 'no data'}")
    else:
        pos = Position(live_ticker.strip().upper(), int(live_shares),
                       float(live_entry))
        qq = Quote(pos.ticker, info["price"], info["prev"], info["high"],
                   info["low"], info["vol"], info["at"])
        stand = standing(pos, qq, capital=float(live_capital))

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Last", f"RM {info['price']:.3f}",
                  f"{qq.change_pct:+.2%}" if qq.change_pct is not None else None)
        m2.metric("Position", f"RM {stand['position_value']:,.0f}")
        m3.metric("Contest return", f"{stand['return_pct']:+.2%}")
        m4.metric("Gap to winning score", f"{stand['gap_pct']:+.2%}",
                  help="Against +29%, the median winning score the contest "
                       "simulator produced over real Bursa windows. It is the "
                       "bar the field sets, not a forecast of your return.")

        # The delay is stated every time, not buried in a tooltip. A quote
        # presented as live is how you price an order against a market that
        # moved on fifteen minutes ago.
        st.caption(f"Quoted {info['at']:%Y-%m-%d %H:%M} MYT — {info['stale']}")
        # The verdict, rendered where the temptation is: next to a moving
        # price. Rules consulted only when you feel like consulting them are
        # not rules.
        st.divider()
        st.subheader("What the rule says")

        rc1, rc2, rc3 = st.columns(3)
        r_days = rc1.number_input("Trading days left", 0, 60, 15)
        r_halted = rc2.checkbox("Halted / suspended today")
        r_elig = rc3.checkbox("Still passes the screen", value=True)
        mc1, mc2 = st.columns(2)
        mom_entry = mc1.number_input(
            "60d momentum when you bought", value=1.20, step=0.05,
            help="From the pick script's score column on the day you entered.")
        mom_now = mc2.number_input("60d momentum now", value=1.20, step=0.05)

        verdict = decide(Situation(
            days_left=int(r_days),
            my_return=float(stand["return_pct"] or 0.0),
            position_return=float(pos.pnl_pct(qq) or 0.0),
            halted=bool(r_halted), still_eligible=bool(r_elig),
            momentum_at_entry=float(mom_entry), momentum_now=float(mom_now)))

        if verdict.action is Action.HOLD:
            st.success(f"**{verdict.action.value}** — {verdict.reason}",
                       icon="✋")
        else:
            st.warning(f"**{verdict.action.value}** — {verdict.reason}",
                       icon="🔁")
        st.caption(verdict.detail)
        with st.expander("Why there is no stop loss"):
            st.markdown(explain_no_stop_loss())

        if not info["realtime"]:
            st.info(
                "This is Yahoo's **delayed** KLSE data (~15 min) and the panel "
                "never claims otherwise. **Read the live spread in moomoo "
                "before you send an order.** To get real-time here, run this "
                "dashboard on your own machine with OpenD started: "
                "`python -m scripts.check_feed` verifies it.",
                icon="ℹ️")


# --------------------------------------------------------------------------
# Tournament
# --------------------------------------------------------------------------
with tab_t:
    s = assess(my_return, int(days_left), contest)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("My return", f"{s.my_return:+.1%}")
    c2.metric("Score that wins", f"{s.target:+.1%}",
              help="Expected maximum of n entrants -- derived from the entrant "
                   "count, not from a leaderboard.")
    c3.metric("Gap", f"{s.gap:+.1%}",
              delta="ahead" if s.gap <= 0 else "behind",
              delta_color="normal" if s.gap <= 0 else "inverse")
    c4.metric("P(win) at current risk", f"{s.p_win_current:.1%}",
              help=f"Rises to {s.p_win_aggressive:.1%} fully concentrated.")

    if s.gap <= 0:
        st.success(f"**{s.verdict}**")
    elif "FAR BEHIND" in s.verdict:
        st.error(f"**{s.verdict}**")
    else:
        st.info(f"**{s.verdict}**")

    st.pyplot(pwin_chart(strategy_table(contest), theme,
                         baseline=1 / contest.n_entrants))

    st.caption(
        "Read the gap between the top and bottom bars. Concentration is worth "
        "far more than skill over a contest this short: a genuinely excellent "
        "+5%/month edge, played with a diversified portfolio, loses to holding "
        "three names with no edge at all."
    )

    with st.expander("Required volatility, and the arithmetic behind it"):
        st.write(
            f"- Need roughly **{s.required_vol_remaining:.1%}** volatility over "
            f"the remaining {s.days_left} days "
            f"(**{s.required_daily_vol:.2%}/day**)\n"
            f"- P(win) holding current risk: **{s.p_win_current:.1%}**\n"
            f"- P(win) fully concentrated: **{s.p_win_aggressive:.1%}**\n"
            f"- Random chance: **{1 / contest.n_entrants:.2%}**")
        st.caption(
            "The target is the expected maximum of "
            f"{contest.n_entrants} draws at {contest.field_vol:.0%} volatility. "
            "It needs no leaderboard -- only the entrant count and an "
            "assumption about how the field behaves.")


# --------------------------------------------------------------------------
# Backtest
# --------------------------------------------------------------------------
with tab_b:
    strat, res, bh, ew, randoms, cfg = run_everything(
        panel, preset, int(top_n), int(n_random))

    perf = res.performance(strat.name, n_trials=1)
    pct = percentile_vs_random(perf.sharpe, randoms["sharpe"])

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total return", f"{perf.total_return:+.1%}")
    c2.metric("Sharpe", f"{perf.sharpe:+.2f}")
    c3.metric("Max drawdown", f"{perf.max_drawdown:.1%}")
    c4.metric("vs random", f"{pct:.0f}th pctile",
              help="Percentile within the random-entry distribution at matched "
                   "exposure. Below ~90 and the signal is not doing much.")

    st.pyplot(equity_chart({
        strat.name: res.equity,
        "buy & hold": bh.equity,
        "equal weight": ew.equity,
    }, theme))

    st.pyplot(random_distribution_chart(randoms["sharpe"], perf.sharpe, theme))

    st.subheader("Against every baseline")
    table = pd.DataFrame([
        perf.as_row(),
        bh.performance("buy & hold").as_row(),
        ew.performance("equal weight, rebalanced").as_row(),
    ]).set_index("strategy")
    st.dataframe(
        table.style.format({
            "total_return": "{:+.2%}", "cagr": "{:+.2%}", "vol": "{:.2%}",
            "sharpe": "{:+.2f}", "sortino": "{:+.2f}", "max_dd": "{:.2%}",
            "calmar": "{:+.2f}", "hit_rate": "{:.1%}", "turnover": "{:.2f}",
            "dsr": "{:.3f}",
        }), use_container_width=True)

    c1, c2 = st.columns(2)
    c1.metric("Cost drag", f"{res.cost_drag:.2%}",
              help="Total trading costs as a fraction of starting capital.")
    c2.metric("Round trip on RM 1,000",
              f"{round_trip_pct(1000, cfg.broker, cfg.slippage):.2f}%")

    st.caption(
        "`dsr` is the deflated Sharpe -- the probability the true Sharpe "
        "exceeds zero once the number of variants tested is accounted for. It "
        "shows only when more than one trial is declared. Reporting a Sharpe "
        "without it, after trying fifty configurations, is the single most "
        "common way a backtest misleads its author."
    )


# --------------------------------------------------------------------------
# Walk-forward -- the only tab whose numbers are out-of-sample
# --------------------------------------------------------------------------
@st.cache_data(show_spinner="Running walk-forward...")
def run_wf(panel: pd.DataFrame, top_n: int, n_folds: int):
    cfg = BacktestConfig(capital=10_000, rebalance="ME", broker=MOOMOO,
                         slippage=SlippageModel(15.0, 0.0), max_positions=top_n)
    strategies = {n: (lambda n=n: PRESETS[n](top_n=top_n)) for n in PRESETS}
    return walk_forward(panel, strategies, cfg, n_folds=n_folds,
                        train_months=24, test_months=6, expanding=True,
                        holdout_months=12, verbose=False)


with tab_w:
    st.caption(
        "Every other tab reports numbers from data the strategy was chosen on. "
        "These are not: at each fold the best strategy on the training window "
        "is picked, then measured on months it has never seen."
    )
    n_folds = st.slider("Folds", 3, 8, 6)
    try:
        wf = run_wf(panel, int(top_n), int(n_folds))
    except Exception as e:
        st.error(f"Not enough history to walk forward: {e}")
        wf = None

    if wf and wf.folds:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Mean in-sample SR", f"{wf.mean_is:+.2f}")
        c2.metric("Mean out-of-sample SR", f"{wf.mean_oos:+.2f}")
        c3.metric("Survival ratio", f"{wf.degradation:+.2f}",
                  help="OOS divided by IS. Below ~0.5 means most of the "
                       "apparent edge was fitted rather than found.")
        c4.metric("Stitched OOS Sharpe", f"{wf.stitched_sharpe:+.2f}",
                  help="The equity curve you would actually have lived "
                       "through, choosing as you went.")

        st.pyplot(walkforward_chart(wf.folds, theme))

        st.subheader("Fold by fold")
        st.dataframe(pd.DataFrame([{
            "fold": f.index,
            "train ends": f.train_end.date(),
            "test window": f"{f.test_start.date()} → {f.test_end.date()}",
            "chosen": f.chosen,
            "IS Sharpe": f.is_sharpe,
            "OOS Sharpe": f.oos_sharpe,
            "OOS return": f.oos_return,
            "OOS max DD": f.oos_maxdd,
        } for f in wf.folds]).style.format({
            "IS Sharpe": "{:+.2f}", "OOS Sharpe": "{:+.2f}",
            "OOS return": "{:+.2%}", "OOS max DD": "{:.2%}",
        }), use_container_width=True)

        if wf.holdout_start is not None:
            st.info(
                f"**Holdout from {wf.holdout_start.date()} has not been "
                f"touched.** Evaluate it once, at the very end, with "
                f"`research.walkforward.evaluate_holdout` — and count that as "
                f"a trial. Re-running it while adjusting the strategy turns it "
                f"into another training set.", icon="🔒")

        chosen = [f.chosen for f in wf.folds]
        if len(set(chosen)) == 1:
            st.warning(
                f"**{chosen[0]} was chosen in every fold**, so the selection "
                f"step never actually selected anything. This is a test of "
                f"that one strategy, not of a process for picking between "
                f"them.", icon="⚠️")


# --------------------------------------------------------------------------
# Universe
# --------------------------------------------------------------------------
with tab_u:
    if is_real is not True:
        st.info("This panel marks every name eligible — either it is "
                "synthetic, or it is the raw cache with no screen applied. "
                "The real screen needs the SC Shariah list and a liquidity "
                "history; build it with `python -m scripts.build_panel`.",
                icon="ℹ️")

    daily = (panel.groupby("date", observed=True)
                  .agg(n_listed=("ticker", "nunique"),
                       n_eligible=("eligible", "sum")))
    st.pyplot(universe_chart(daily, theme))

    c1, c2, c3 = st.columns(3)
    c1.metric("Tickers in panel", f"{panel['ticker'].nunique()}")
    c2.metric("Eligible today", f"{int(daily['n_eligible'].iloc[-1])}")
    c3.metric("Bars", f"{len(panel):,}")

    st.caption(
        "A universe that collapses to a handful of names for part of the "
        "period explains a lot of apparent alpha. Check this before trusting "
        "any backtest above."
    )


# --------------------------------------------------------------------------
# Data health
# --------------------------------------------------------------------------
with tab_d:
    results = validate(panel, verbose=False)
    n_pass = sum(r.passed for r in results)

    st.metric("Checks passed", f"{n_pass}/{len(results)}")

    for r in results:
        if r.passed:
            st.success(f"**{r.name}** — {r.message}", icon="✅")
        else:
            with st.expander(f"⚠️  **{r.name}** — {r.message}", expanded=False):
                if not r.sample.empty:
                    st.dataframe(r.sample, use_container_width=True)

    st.caption(
        "`extreme moves` and `calendar gaps` are expected to flag on a real "
        "panel — genuine 40% days and real suspensions both exist. They want "
        "reading, not fixing. Everything else failing means the data is broken."
    )
