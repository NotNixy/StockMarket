"""Chart builders for the dashboard.

Kept out of app.py so they can be tested without launching Streamlit.

Palette note: both light and dark steps were validated with the dataviz
validator against their own surfaces -- the dark column is the same hues
re-stepped for a dark ground, not an automatic flip. Worst adjacent CVD
separation is 9.1 (light) / 8.4 (dark), which sits in the band that REQUIRES
secondary encoding, so every series carries a direct label at its line end as
well as a legend entry. Do not remove those labels.
"""
from __future__ import annotations

from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Theme:
    surface: str
    ink: str
    ink2: str
    grid: str
    series: tuple[str, ...]


LIGHT = Theme(
    surface="#ffffff", ink="#0b1a24", ink2="#46606f", grid="#dbe4ea",
    series=("#2a78d6", "#eb6834", "#1baf7a", "#eda100"),
)
DARK = Theme(
    surface="#0e1117", ink="#e6edf3", ink2="#9aa7b2", grid="#2a3038",
    series=("#3987e5", "#d95926", "#199e70", "#c98500"),
)


def _style(ax, theme: Theme, ylabel: str = "", xlabel: str = ""):
    """Recessive axes, no chartjunk."""
    ax.set_facecolor(theme.surface)
    ax.figure.set_facecolor(theme.surface)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(theme.grid)
    ax.tick_params(colors=theme.ink2, labelsize=8, length=0)
    ax.grid(axis="y", color=theme.grid, lw=0.7, alpha=0.7, zorder=0)
    ax.set_axisbelow(True)
    if ylabel:
        ax.set_ylabel(ylabel, color=theme.ink2, fontsize=8.5)
    if xlabel:
        ax.set_xlabel(xlabel, color=theme.ink2, fontsize=8.5)


def equity_chart(curves: dict[str, pd.Series], theme: Theme = LIGHT,
                 title: str = "Equity curve"):
    """Strategy against its baselines. One y-axis, always.

    Series are direct-labelled at the line end as well as listed in the
    legend -- identity never rests on colour alone.
    """
    fig, ax = plt.subplots(figsize=(9, 4.2), dpi=120)
    _style(ax, theme, ylabel="Portfolio value (RM)")

    ends = []
    for i, (name, s) in enumerate(curves.items()):
        colour = theme.series[i % len(theme.series)]
        lw = 2.2 if i == 0 else 1.5          # the strategy reads heaviest
        ax.plot(s.index, s.values, color=colour, lw=lw, zorder=3 + i,
                label=name, solid_capstyle="round")
        if len(s):
            ends.append([float(s.iloc[-1]), name, colour, i == 0, s.index[-1]])

    # Baselines often finish within a whisker of each other, which stacks the
    # end labels on top of one another. Push them apart vertically; the line
    # still says where the value is, the label only needs to be readable.
    if ends:
        lo, hi = ax.get_ylim()
        min_gap = (hi - lo) * 0.045
        ends.sort(key=lambda e: e[0])
        for j in range(1, len(ends)):
            if ends[j][0] - ends[j - 1][0] < min_gap:
                ends[j][0] = ends[j - 1][0] + min_gap
        for y, name, colour, is_strategy, x in ends:
            ax.annotate(name, (x, y), xytext=(6, 0),
                        textcoords="offset points", color=colour,
                        fontsize=8, va="center",
                        fontweight="600" if is_strategy else "normal")

    ax.set_title(title, color=theme.ink, fontsize=12, fontweight="bold",
                 loc="left", pad=10)
    leg = ax.legend(frameon=False, fontsize=8, loc="upper left", ncol=2)
    for t in leg.get_texts():
        t.set_color(theme.ink2)
    ax.margins(x=0.14)                        # room for the end labels
    fig.tight_layout()
    return fig


def pwin_chart(table: pd.DataFrame, theme: Theme = LIGHT,
               baseline: float | None = None):
    """P(win) by portfolio concentration. The table that sets your sizing."""
    fig, ax = plt.subplots(figsize=(9, 3.4), dpi=120)
    _style(ax, theme)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color=theme.grid, lw=0.7, alpha=0.7)

    labels = list(table.index)[::-1]
    values = list(table["p_win"] * 100)[::-1]
    colours = [theme.series[0]] * len(labels)
    colours[-1] = theme.series[1]             # the most concentrated stands out

    bars = ax.barh(labels, values, height=0.6, color=colours, zorder=3)
    for b, v in zip(bars, values):
        ax.text(v + max(values) * 0.015, b.get_y() + b.get_height() / 2,
                f"{v:.1f}%", va="center", fontsize=9, color=theme.ink,
                fontweight="600")

    if baseline is not None:
        ax.axvline(baseline * 100, color=theme.ink2, lw=1.1, ls="--", zorder=4)
        ax.text(baseline * 100, len(labels) - 0.35, " random chance",
                fontsize=8, color=theme.ink2, va="bottom")

    ax.set_title("Probability of finishing first", color=theme.ink,
                 fontsize=12, fontweight="bold", loc="left", pad=10)
    ax.set_xlabel("P(win), %", color=theme.ink2, fontsize=8.5)
    ax.set_xlim(0, max(values) * 1.18)
    fig.tight_layout()
    return fig


def random_distribution_chart(random_sharpes: pd.Series,
                              strategy_sharpe: float,
                              theme: Theme = LIGHT):
    """Where the strategy sits inside the random-entry distribution.

    The honest version of "my backtest had a Sharpe of 0.8".
    """
    fig, ax = plt.subplots(figsize=(9, 3.4), dpi=120)
    _style(ax, theme, ylabel="Random trials")

    ax.hist(random_sharpes, bins=24, color=theme.series[0], alpha=0.75,
            zorder=3, edgecolor=theme.surface, linewidth=1.2)
    ax.axvline(strategy_sharpe, color=theme.series[1], lw=2.4, zorder=5)

    pct = (random_sharpes < strategy_sharpe).mean() * 100
    ax.annotate(f"strategy {strategy_sharpe:+.2f}  ({pct:.0f}th pctile)",
                (strategy_sharpe, ax.get_ylim()[1] * 0.92),
                xytext=(8, 0), textcoords="offset points",
                color=theme.series[1], fontsize=9, fontweight="600")

    ax.set_title("Strategy vs random entry at matched exposure",
                 color=theme.ink, fontsize=12, fontweight="bold",
                 loc="left", pad=10)
    ax.set_xlabel("Sharpe ratio", color=theme.ink2, fontsize=8.5)
    fig.tight_layout()
    return fig


def universe_chart(summary: pd.DataFrame, theme: Theme = LIGHT):
    """How many names were tradable over time.

    Single series, so no legend -- the title names it. A universe that
    collapses to three names for a year explains a lot of apparent alpha.
    """
    fig, ax = plt.subplots(figsize=(9, 3.0), dpi=120)
    _style(ax, theme, ylabel="Eligible names")
    ax.plot(summary.index, summary["n_eligible"], color=theme.series[0],
            lw=2.0, zorder=3)
    ax.fill_between(summary.index, summary["n_eligible"], color=theme.series[0],
                    alpha=0.12, zorder=2)
    ax.set_title("Tradable universe over time", color=theme.ink, fontsize=12,
                 fontweight="bold", loc="left", pad=10)
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    return fig


def gap_chart(days: np.ndarray, my_path: np.ndarray, target: float,
              theme: Theme = LIGHT):
    """Your contest position against the score that wins it."""
    fig, ax = plt.subplots(figsize=(9, 3.2), dpi=120)
    _style(ax, theme, ylabel="Return")

    ax.axhline(target * 100, color=theme.series[1], lw=2.0, ls="--", zorder=4)
    ax.annotate(f"target to win  {target:+.1%}", (days[-1], target * 100),
                xytext=(-4, 6), textcoords="offset points", ha="right",
                color=theme.series[1], fontsize=9, fontweight="600")

    ax.plot(days, my_path * 100, color=theme.series[0], lw=2.2, zorder=5)
    ax.annotate("you", (days[-1], my_path[-1] * 100), xytext=(6, 0),
                textcoords="offset points", color=theme.series[0],
                fontsize=9, fontweight="600", va="center")

    ax.axhline(0, color=theme.grid, lw=1.0, zorder=1)
    ax.set_title("Position against the winning score", color=theme.ink,
                 fontsize=12, fontweight="bold", loc="left", pad=10)
    ax.set_xlabel("Contest day", color=theme.ink2, fontsize=8.5)
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:+.0f}%")
    ax.margins(x=0.08)
    fig.tight_layout()
    return fig
