"""Descriptive summaries and a Table 1 builder.

Design choice: the analyst states how each continuous variable is summarised
("mean" -> mean (SD), "median" -> median (Q1 to Q3)). The choice is made from the
histogram, Q-Q plot, skewness and clinical convention, and written down in the notebook.
It is not made automatically from a normality-test p-value, because that p-value
depends mostly on the sample size.
"""
from __future__ import annotations

from typing import Dict, Iterable, Optional

import numpy as np
import pandas as pd


def fmt_n_pct(n: int, den: int, digits: int = 0) -> str:
    """'n/N (p%)', with N = non-missing values in that group."""
    if den == 0:
        return "NA"
    return f"{int(n)}/{int(den)} ({100.0 * n / den:.{digits}f}%)"


def fmt_continuous(x: pd.Series, how: str = "mean", digits: int = 1) -> str:
    x = pd.to_numeric(x, errors="coerce").dropna()
    if x.empty:
        return "NA"
    if how == "mean":
        return f"{x.mean():.{digits}f} ({x.std(ddof=1):.{digits}f})"
    if how == "median":
        q1, med, q3 = x.quantile([0.25, 0.5, 0.75])
        return f"{med:.{digits}f} ({q1:.{digits}f} to {q3:.{digits}f})"
    raise ValueError("how must be 'mean' or 'median'")


def shape_table(df: pd.DataFrame, cols: Iterable[str], by: Optional[str] = None) -> pd.DataFrame:
    """n, missing, mean, SD, median, IQR and skewness: the numbers that guide the summary choice."""
    def one(x: pd.Series) -> dict:
        x = pd.to_numeric(x, errors="coerce")
        q1, q3 = x.quantile([0.25, 0.75])
        return {"n": int(x.notna().sum()), "missing": int(x.isna().sum()), "mean": x.mean(),
                "sd": x.std(), "median": x.median(), "q1": q1, "q3": q3, "skew": x.skew()}
    if by is None:
        return pd.DataFrame({c: one(df[c]) for c in cols}).T.round(2)
    parts = {(c, g): one(sub[c]) for c in cols for g, sub in df.groupby(by)}
    return pd.DataFrame(parts).T.rename_axis(["variable", by]).round(2)


def build_table1(
    df: pd.DataFrame,
    group_col: str,
    continuous: Dict[str, str],
    categorical: Dict[str, Optional[list]],
    labels: Optional[Dict[str, str]] = None,
    group_labels: Optional[Dict] = None,
    digits: int = 1,
) -> pd.DataFrame:
    """Descriptive Table 1: Overall, one column per group, and missing counts.

    continuous  : {column: "mean" | "median"}
    categorical : {column: [levels to show] or None for all levels}
    No p-values: baseline differences belong in the analysis, with effect estimates.
    """
    labels = labels or {}
    groups = list(pd.Series(df[group_col].dropna().unique()).sort_values())
    names = {g: (group_labels or {}).get(g, str(g)) for g in groups}
    cols = ["Overall"] + [names[g] for g in groups] + ["Missing"]
    subsets = [("Overall", df)] + [(names[g], df[df[group_col] == g]) for g in groups]

    rows, index = [], []
    rows.append(dict({n: str(len(d)) for n, d in subsets}, Missing=""))
    index.append(("N", ""))

    for col, how in continuous.items():
        stat = "mean (SD)" if how == "mean" else "median (IQR)"
        r = {n: fmt_continuous(d[col], how, digits) for n, d in subsets}
        r["Missing"] = str(int(df[col].isna().sum()))
        rows.append(r)
        index.append((f"{labels.get(col, col)}, {stat}", ""))

    for col, levels in categorical.items():
        lv = levels if levels is not None else list(pd.Series(df[col].dropna().unique()).sort_values())
        for i, level in enumerate(lv):
            r = {n: fmt_n_pct((d[col] == level).sum(), d[col].notna().sum()) for n, d in subsets}
            r["Missing"] = str(int(df[col].isna().sum())) if i == 0 else ""
            rows.append(r)
            index.append((f"{labels.get(col, col)}, n/N (%)", str(level)))

    out = pd.DataFrame(rows, columns=cols)
    out.index = pd.MultiIndex.from_tuples(index, names=["Characteristic", "Level"])
    return out
