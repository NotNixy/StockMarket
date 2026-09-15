"""Tests for the base-rate engine.

The two that matter are the lookahead test and the cross-sectional bucketing
test. Both failure modes produce a study that looks strongly predictive and
is measuring its own construction.
"""
import numpy as np
import pandas as pd
import pytest

from research.baserates import (
    MIN_BUCKET_OBS, add_factor, base_rates, bucket_cross_sectionally,
    current_bucket, forward_returns, summarise_buckets,
)


def panel(n_tickers=40, n_days=400, seed=0, drift=0.0, vol=0.02):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2022-01-03", periods=n_days, freq="B")
    frames = []
    for i in range(n_tickers):
        px = 5.0 * np.exp(np.cumsum(rng.normal(drift, vol, n_days)))
        frames.append(pd.DataFrame({
            "date": dates, "ticker": f"T{i:03d}.KL",
            "adj_close": px, "eligible": True}))
    return pd.concat(frames, ignore_index=True)


# ------------------------------------------------------------------ lookahead
def test_the_forward_return_starts_at_the_next_bar_not_this_one():
    """Entry at t+1. Measuring from t hands the strategy an overnight gap it
    could never have captured -- the most common way a study like this
    invents an edge out of nothing."""
    dates = pd.date_range("2024-01-01", periods=10, freq="B")
    p = pd.DataFrame({"date": dates, "ticker": "A.KL",
                      "adj_close": [10, 20, 21, 22, 23, 24, 25, 26, 27, 28.0]})
    f = forward_returns(p, horizon=2)
    # Row 0: entry at bar 1 (20), exit at bar 3 (22) -> +10%.
    # If it measured from bar 0 (10) it would read +120%, capturing a jump
    # that happened before any order could exist.
    assert f["fwd_return"].iloc[0] == pytest.approx(0.10)


def test_a_factor_cannot_see_the_return_it_is_being_scored_against():
    """The end-to-end version: on a pure random walk, no bucket may show a
    reliable edge. If any does, information is leaking backwards."""
    stats = base_rates(panel(n_tickers=60, n_days=500, seed=1),
                       factor="momentum", horizon=21, lookback=60)
    medians = [s.median for s in stats if not s.thin]
    assert medians, "no bucket had enough observations to judge"
    # A leak shows up as a monotone ramp across deciles with a wide spread.
    assert max(medians) - min(medians) < 0.05, (
        f"decile medians span {max(medians) - min(medians):.1%} on a random "
        f"walk -- something is leaking")


# ------------------------------------------------------- cross-sectional rule
def test_buckets_are_formed_within_a_date_not_across_history():
    """The load-bearing choice.

    Built here: a market that trends hard upward, so LATE dates have high
    momentum and EARLY dates low. Ranked against history, every late name
    would land in the top bucket and the bucket would just mean "2023".
    Ranked within each date, both buckets must contain every date.
    """
    p = panel(n_tickers=30, n_days=400, seed=2, drift=0.004, vol=0.01)
    b = bucket_cross_sectionally(forward_returns(add_factor(p)), n_buckets=5)
    spans = b.groupby("bucket", observed=True)["date"].agg(["min", "max"])
    # Every bucket must be populated across the whole sample, not clustered.
    overall = (b["date"].min(), b["date"].max())
    for bucket, row in spans.iterrows():
        assert row["min"] - overall[0] < pd.Timedelta(days=120), (
            f"bucket {bucket} only appears late -- ranking is not "
            f"cross-sectional")


def test_every_date_contributes_to_every_bucket_roughly_equally():
    p = panel(n_tickers=40, n_days=300, seed=3)
    b = bucket_cross_sectionally(forward_returns(add_factor(p)), n_buckets=10)
    per_date = b.groupby(["date", "bucket"], observed=True).size().unstack()
    # Within a date, deciles should hold roughly equal counts.
    spread = (per_date.max(axis=1) - per_date.min(axis=1)).median()
    assert spread <= 1, f"deciles are uneven within a date (spread {spread})"


def test_a_date_with_too_few_names_is_dropped_not_bucketed():
    """Ten deciles from eight stocks is noise wearing a decile's clothes."""
    p = panel(n_tickers=6, n_days=300, seed=4)
    b = bucket_cross_sectionally(forward_returns(add_factor(p)), n_buckets=10)
    assert b.empty


# -------------------------------------------------------------- honest sample
def test_the_overlap_in_daily_windows_is_reported_not_hidden():
    """200,000 overlapping rows are not 200,000 independent facts.

    Consecutive 21-day windows sampled daily share 20 of 21 days. The row
    count invites a t-statistic that would be meaningless, so the honest
    denominator has to travel with the result.
    """
    stats = base_rates(panel(n_tickers=40, n_days=400, seed=5), horizon=21)
    s = stats[0]
    assert s.independent_periods < s.n / 10, (
        "independent_periods should be far smaller than the row count")
    assert s.independent_periods > 0


def test_a_thin_bucket_is_flagged():
    tiny = pd.DataFrame({
        "date": pd.date_range("2024-01-01", periods=5),
        "ticker": [f"T{i}" for i in range(5)],
        "bucket": [1] * 5, "fwd_return": [0.01, -0.02, 0.03, 0.0, 0.05]})
    assert summarise_buckets(tiny)[0].thin


def test_a_full_bucket_is_not_flagged():
    n = MIN_BUCKET_OBS + 50
    full = pd.DataFrame({
        "date": pd.date_range("2024-01-01", periods=n, freq="B"),
        "ticker": "A.KL", "bucket": 1,
        "fwd_return": np.linspace(-0.2, 0.2, n)})
    assert not summarise_buckets(full)[0].thin


# ------------------------------------------------------------ planted signal
def test_a_real_factor_edge_is_detected():
    """Positive control. Without it, a module that reported "no edge" for
    everything would pass every test above while being broken."""
    rng = np.random.default_rng(6)
    dates = pd.date_range("2022-01-03", periods=500, freq="B")
    frames = []
    for i in range(40):
        # Half the names trend, half chop. The trending ones will hold high
        # momentum AND go on rising, so the top decile must outperform.
        drift = 0.0025 if i < 20 else -0.0015
        px = 5.0 * np.exp(np.cumsum(rng.normal(drift, 0.015, len(dates))))
        frames.append(pd.DataFrame({"date": dates, "ticker": f"T{i:03d}.KL",
                                    "adj_close": px, "eligible": True}))
    stats = base_rates(pd.concat(frames, ignore_index=True), horizon=21)
    top = [s for s in stats if s.bucket == max(x.bucket for x in stats)][0]
    bottom = [s for s in stats if s.bucket == 1][0]
    assert top.median > bottom.median, "planted momentum edge not detected"


# ------------------------------------------------------------ current bucket
def test_a_tickers_rank_is_reported_out_of_the_live_universe():
    p = panel(n_tickers=50, n_days=300, seed=7)
    res = current_bucket(p, "T000.KL")
    assert res["found"]
    assert 1 <= res["rank"] <= res["n_universe"]
    assert res["n_universe"] == 50
    assert 1 <= res["bucket"] <= 10


def test_the_best_ranked_name_lands_in_the_top_bucket():
    """Bucket numbering must match bucket_cross_sectionally, or the report
    would look up the wrong row of the base-rate table and quote a
    distribution belonging to the opposite decile."""
    p = panel(n_tickers=50, n_days=300, seed=8)
    scored = add_factor(p)
    last = scored["date"].max()
    best = (scored[scored["date"] == last]
            .sort_values("factor", ascending=False)["ticker"].iloc[0])
    res = current_bucket(p, best)
    assert res["rank"] == 1
    assert res["bucket"] == 10
    assert res["percentile"] == pytest.approx(1.0, abs=0.02)


def test_an_ineligible_ticker_is_reported_as_such_not_silently_ranked():
    p = panel(n_tickers=30, n_days=300, seed=9)
    p.loc[p["ticker"] == "T000.KL", "eligible"] = False
    res = current_bucket(p, "T000.KL")
    assert not res["found"]
    assert "not eligible" in res["reason"]


def test_an_unknown_ticker_does_not_raise():
    assert not current_bucket(panel(seed=10), "NOPE.KL")["found"]


# ------------------------------------------------------------ the report CLI
def test_ticker_normalisation_accepts_every_spelling():
    from scripts.report import normalise
    for given in ("0270", "0270.KL", " 0270.kl ", "0270.Kl"):
        assert normalise(given) == "0270.KL"


def test_eligibility_report_names_the_condition_that_failed():
    """A verdict of 'not tradable' is useless without saying which screen
    rejected it -- illiquid is a different problem from non-compliant."""
    from scripts.report import eligibility_report

    d = pd.Timestamp("2026-01-05")
    p = pd.DataFrame([{
        "date": d, "ticker": "A.KL", "close": 1.50, "adj_close": 1.50,
        "eligible": False, "shariah": True, "liquid": False,
        "priced": True, "seasoned": True,
        "median_dtv": 12_000.0, "bars_available": 900,
    }])
    checks = eligibility_report(p, "A.KL", d)
    by_label = {label: ok for label, ok, _ in checks}
    assert by_label["Shariah-compliant"] is True
    assert by_label["liquid enough"] is False
    # The failing row must carry the number that caused it.
    detail = next(dt for label, _, dt in checks if label == "liquid enough")
    assert "12,000" in detail


def test_a_ticker_absent_from_the_panel_is_reported_not_crashed():
    from scripts.report import eligibility_report
    d = pd.Timestamp("2026-01-05")
    p = pd.DataFrame([{"date": d, "ticker": "A.KL", "close": 1.0,
                       "adj_close": 1.0, "eligible": True}])
    checks = eligibility_report(p, "ZZZ.KL", d)
    assert checks[0][1] is False
    assert "no bar" in checks[0][2]


def test_the_decile_lookup_and_the_bucket_table_use_the_same_numbering():
    """If current_bucket numbered deciles the other way round, the report
    would quote the distribution of the OPPOSITE decile -- confidently, and
    with no error anywhere. The two must be checked against each other."""
    p = panel(n_tickers=60, n_days=400, seed=11)
    scored = add_factor(p)
    last = scored["date"].max()
    today = scored[scored["date"] == last].sort_values("factor",
                                                       ascending=False)

    best = current_bucket(p, today["ticker"].iloc[0])
    worst = current_bucket(p, today["ticker"].iloc[-1])
    assert best["bucket"] == 10 and worst["bucket"] == 1

    # And the table's own deciles must run the same way: bucket 10 holds the
    # highest factor values.
    b = bucket_cross_sectionally(forward_returns(scored), n_buckets=10)
    means = b.groupby("bucket", observed=True)["factor"].mean()
    assert means.loc[10] > means.loc[1]
