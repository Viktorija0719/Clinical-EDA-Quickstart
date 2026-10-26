"""Plot helpers with one consistent style (RSU red for the diabetes group, slate for the rest)."""
from __future__ import annotations

from typing import Iterable, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

RED, SLATE, INK = "#8E001C", "#6B7B8C", "#1F1F1F"
GROUP_COLORS = {0: SLATE, 1: RED}
GROUP_LABELS = {0: "No diabetes", 1: "Diabetes"}


def set_style() -> None:
    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False,
                         "axes.titleweight": "bold", "axes.titlelocation": "left",
                         "figure.dpi": 110, "savefig.dpi": 200, "savefig.bbox": "tight"})


def hist_qq(x: pd.Series, label: str, axes=None, color: str = SLATE):
    """Histogram and normal Q-Q plot side by side for one variable."""
    x = pd.to_numeric(x, errors="coerce").dropna()
    if axes is None:
        _, axes = plt.subplots(1, 2, figsize=(9, 3.2))
    sns.histplot(x, bins=30, color=color, ax=axes[0])
    axes[0].set(xlabel=label, ylabel="Count", title=f"{label}: skew {x.skew():.2f}")
    (osm, osr), (slope, icpt, _) = stats.probplot(x, dist="norm")
    axes[1].scatter(osm, osr, s=8, color=color, alpha=0.7)
    axes[1].plot(osm, slope * np.asarray(osm) + icpt, color=INK, lw=1)
    axes[1].set(xlabel="Theoretical normal quantiles", ylabel=label, title="Normal Q-Q plot")
    return axes


def box_grid(df: pd.DataFrame, cols: Iterable[str], ncols: int = 4, height: float = 3.0):
    """One box plot per numeric column, each on its own scale."""
    cols = list(cols)
    nrows = int(np.ceil(len(cols) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, height * nrows), squeeze=False)
    for ax, col in zip(axes.flat, cols):
        sns.boxplot(y=df[col].dropna(), ax=ax, color="#E3E6EA", fliersize=3)
        ax.set(title=col, ylabel="")
    for ax in list(axes.flat)[len(cols):]:
        ax.axis("off")
    fig.tight_layout()
    return fig


def group_box_strip(df: pd.DataFrame, y: str, group: str = "diabetes", ax=None, ylabel: Optional[str] = None):
    """Box plot plus every individual point, with n per group in the labels."""
    ax = ax or plt.subplots(figsize=(4.5, 4))[1]
    d = df.dropna(subset=[y, group])
    order = sorted(d[group].unique())
    sns.boxplot(data=d, x=group, y=y, order=order, showfliers=False, color="white", ax=ax,
                linecolor=INK)
    sns.stripplot(data=d, x=group, y=y, order=order, hue=group, palette=GROUP_COLORS,
                  alpha=0.45, size=3.5, jitter=0.18, legend=False, ax=ax)
    ax.set_xticks(range(len(order)),
                  [f"{GROUP_LABELS.get(g, g)}\nn = {(d[group] == g).sum()}" for g in order])
    ax.set(xlabel="", ylabel=ylabel or y)
    return ax


def bar_counts(df: pd.DataFrame, col: str, ax=None, order: Optional[list] = None):
    """Bar chart of category counts labelled n/N (%), with N = non-missing values."""
    ax = ax or plt.subplots(figsize=(4, 3))[1]
    counts = df[col].value_counts().reindex(order) if order else df[col].value_counts()
    total = int(df[col].notna().sum())
    ax.bar(counts.index.astype(str), counts.values, color=SLATE)
    for i, v in enumerate(counts.values):
        ax.text(i, v, f"{v}/{total}\n({100 * v / total:.0f}%)", ha="center", va="bottom", fontsize=8)
    ax.set(title=col, ylabel="Count")
    ax.margins(y=0.25)
    return ax
