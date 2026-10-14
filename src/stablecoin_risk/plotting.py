"""Figure builders with one consistent house style.

Every function returns a ``matplotlib.figure.Figure`` and never calls
``plt.show()``, so the same code drives notebooks, the CLI and tests.
"""

from __future__ import annotations

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd

from stablecoin_risk import config

# Categorical palette (colour-blind safe, Okabe-Ito based).
PALETTE = {
    "USDC": "#0072B2",
    "USDT": "#009E73",
    "USD": "#555555",
    "accent": "#D55E00",
    "neutral": "#999999",
    "extra1": "#CC79A7",
    "extra2": "#E69F00",
}
QUOTE_COLORS = {"BTCUSD": PALETTE["USD"], "BTCUSDC": PALETTE["USDC"], "BTCUSDT": PALETTE["USDT"],
                "USDCUSD": PALETTE["USDC"], "USDTUSD": PALETTE["USDT"]}
SCENARIO_COLORS = {"Baseline": "#333333", "A: Liquidity run": PALETTE["extra2"],
                   "B: Flash crash": PALETTE["extra1"], "C: Credit contagion": PALETTE["accent"]}


def apply_style() -> None:
    plt.rcParams.update({
        "figure.dpi": 110, "savefig.dpi": 200, "savefig.bbox": "tight",
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.alpha": 0.25, "grid.linestyle": "--",
        "axes.titleweight": "bold", "axes.titlesize": 12, "axes.labelsize": 10,
        "legend.frameon": False, "legend.fontsize": 9, "font.size": 10,
    })


apply_style()


def mark_svb_events(ax: plt.Axes, labels: bool = True) -> None:
    """Shade the crisis window and draw the SVB / de-peg / re-peg markers."""
    ax.axvspan(pd.Timestamp(config.CRISIS_SHADE[0], tz="UTC"), pd.Timestamp(config.CRISIS_SHADE[1], tz="UTC"),
               color=PALETTE["accent"], alpha=0.08, lw=0)
    for name, ts in config.EVENTS.items():
        t = pd.Timestamp(ts, tz="UTC")
        ax.axvline(t, color=PALETTE["accent"], lw=0.9, ls="--", alpha=0.8)
        if labels:
            ax.text(t, 0.98, f" {name}", transform=ax.get_xaxis_transform(), rotation=90, va="top", ha="right",
                    fontsize=8, color=PALETTE["accent"])


def _svb_xaxis(ax: plt.Axes) -> None:
    ax.set_xlim(pd.Timestamp(config.SVB_WINDOW[0], tz="UTC"),
                pd.Timestamp(config.SVB_WINDOW[1], tz="UTC") + pd.Timedelta(days=1))
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))


def _color_for(key: str, i: int) -> str:
    pair = key.rsplit("_", 1)[-1]
    cycle = [PALETTE[k] for k in ("USDC", "USDT", "USD", "extra1", "extra2", "neutral")]
    return QUOTE_COLORS.get(pair, cycle[i % len(cycle)])


def plot_svb_panels(panels: dict[str, dict[str, pd.Series]], ylabel: str, resample: str | None = None,
                    ylim: tuple[float, float] | None = None, legend_loc: str = "upper left") -> plt.Figure:
    """Stacked SVB-window time-series panels: ``{panel title: {label: series}}``."""
    fig, axes = plt.subplots(len(panels), 1, figsize=(12, 3.6 * len(panels)), sharex=True, squeeze=False)
    for ax, (title, series) in zip(axes[:, 0], panels.items(), strict=False):
        for i, (label, s) in enumerate(series.items()):
            s = s.resample(resample).mean() if resample else s
            ls = "--" if label.startswith(("kraken", "coinbase")) else "-"
            ax.plot(s.index, s.to_numpy(), lw=1.1, ls=ls, label=label, color=_color_for(label, i))
        mark_svb_events(ax)
        ax.set_title(title, loc="left")
        ax.set_ylabel(ylabel)
        if ylim:
            ax.set_ylim(*ylim)
        ax.legend(loc=legend_loc, ncol=min(3, len(series)))
    _svb_xaxis(axes[-1, 0])
    fig.tight_layout()
    return fig


def plot_stress_test(cushions: pd.DataFrame, title: str) -> plt.Figure:
    """Equity cushion under each scenario with the insolvency line (Figures 6-7)."""
    fig, ax = plt.subplots(figsize=(12, 5))
    for col in cushions:
        ax.plot(cushions.index, cushions[col] / 1e9, lw=1.6 if col == "Baseline" else 1.1,
                color=SCENARIO_COLORS.get(col), label=col)
    ax.axhline(0, color="black", lw=1)
    ax.axvspan(pd.Timestamp(config.SVB_STRESS_WINDOW[0]), pd.Timestamp(config.SVB_STRESS_WINDOW[1]),
               color=PALETTE["accent"], alpha=0.12, lw=0, label="SVB window")
    ax.fill_between(cushions.index, 0, min(0, cushions.min().min() / 1e9), color="red", alpha=0.04, lw=0)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"${v:,.1f}bn"))
    ax.set_title(title, loc="left")
    ax.set_ylabel("Equity cushion")
    ax.legend(loc="upper left", ncol=3)
    fig.tight_layout()
    return fig


def plot_reserve_evolution(q: pd.DataFrame) -> plt.Figure:
    """Quarter-end assets vs tokens issued (bars) and equity ratio (lines) - Figure 4."""
    fig, ax = plt.subplots(figsize=(12, 5))
    x = range(len(q))
    w = 0.2
    bars = [("tether_assets", "Tether total assets", "#1b7837", -1.5), ("usdt_issued", "USDT issued", "#a6dba0", -0.5),
            ("circle_assets", "Circle total assets", "#08519c", 0.5), ("usdc_issued", "USDC issued", "#9ecae1", 1.5)]
    for col, label, color, off in bars:
        ax.bar([i + off * w for i in x], q[col] / 1e9, width=w, color=color, label=label)
    ax.set_ylabel("USD billions")
    ax.set_xticks(list(x), [d.strftime("%b-%y") for d in q.index], rotation=90)
    ax2 = ax.twinx()
    ax2.plot(list(x), q["tether_equity_pct"] * 100, "o-", color=PALETTE["USDT"], label="Tether equity % (rhs)")
    ax2.plot(list(x), q["circle_equity_pct"] * 100, "o-", color=PALETTE["USDC"], label="Circle equity % (rhs)")
    ax2.set_ylabel("Equity / total assets (%)")
    ax2.grid(False)
    ax2.spines["right"].set_visible(True)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper left", ncol=2)
    ax.set_title("Issuer balance sheets: assets, liabilities and equity buffer", loc="left")
    fig.tight_layout()
    return fig


def plot_reserve_composition(tether: pd.DataFrame, circle: pd.DataFrame) -> plt.Figure:
    """Side-by-side 100% stacked bars of quarter-end reserve mix - Figure 5."""
    colors = ["#08306b", "#6a51a3", "#c51b8a", "#fb6a4a", "#fdae6b"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), sharey=True)
    for ax, (name, comp) in zip(axes, [("Tether", tether), ("Circle", circle)], strict=False):
        bottom = pd.Series(0.0, index=comp.index)
        labels = [f"{d.year} Q{d.quarter}" for d in comp.index]
        for color, col in zip(colors, comp.columns, strict=False):
            ax.bar(labels, comp[col] * 100, bottom=bottom * 100, color=color, label=col, width=0.75)
            for i, (v, b) in enumerate(zip(comp[col], bottom, strict=False)):
                if v >= 0.06:
                    ax.text(i, (b + v / 2) * 100, f"{v:.0%}", ha="center", va="center", fontsize=7, color="white")
            bottom += comp[col]
        ax.set_title(f"{name} reserve composition", loc="left")
        ax.tick_params(axis="x", rotation=90)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2)
        ax.grid(False)
    axes[0].set_ylabel("Share of reserves (%)")
    fig.tight_layout()
    return fig


def plot_connectedness(tsi: pd.Series, events: dict[str, str] | None = None) -> plt.Figure:
    """Dynamic total spillover index from the TVP-VAR."""
    events = events or {"SVB / USDC de-peg": "2023-03-10", "GENIUS Act signed": "2025-07-18"}
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(tsi.index, tsi, color=PALETTE["USDC"], lw=1.2)
    ax.axhline(tsi.mean(), color=PALETTE["neutral"], ls=":", lw=1, label=f"Average TSI = {tsi.mean():.1f}%")
    for name, d in events.items():
        ax.axvline(pd.Timestamp(d), color=PALETTE["accent"], ls="--", lw=0.9)
        ax.text(pd.Timestamp(d), 0.98, f" {name}", transform=ax.get_xaxis_transform(), rotation=90,
                va="top", ha="right", fontsize=8, color=PALETTE["accent"])
    ax.set_ylabel("Total spillover index (%)")
    ax.set_title("Dynamic volatility connectedness (TVP-VAR, H = 10)", loc="left")
    ax.legend(loc="lower left")
    fig.tight_layout()
    return fig


def plot_net_spillovers(table: pd.DataFrame) -> plt.Figure:
    """Horizontal bar chart of NET directional spillovers (TO - FROM)."""
    net = table.loc["NET"].drop("FROM others").sort_values()
    fig, ax = plt.subplots(figsize=(7, 4))
    colors = [PALETTE["accent"] if v > 0 else PALETTE["neutral"] for v in net]
    ax.barh(net.index, net.to_numpy(), color=colors)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("Net spillover (pp of forecast-error variance)")
    ax.set_title("Net transmitters (>0) vs receivers (<0)", loc="left")
    fig.tight_layout()
    return fig
