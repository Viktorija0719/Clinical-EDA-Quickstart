"""Exploring missing data and simulating missingness mechanisms.

Design choice: these tools describe *patterns*. A pattern can be consistent with
MCAR, MAR or MNAR, but observed data cannot prove the mechanism; in particular MAR and
MNAR cannot be told apart from the data alone. Clinical knowledge of how the data were
collected decides which assumption is plausible.
"""
from __future__ import annotations

from typing import Iterable, Optional

import numpy as np
import pandas as pd


def missing_summary(df: pd.DataFrame) -> pd.DataFrame:
    n = df.isna().sum()
    return (pd.DataFrame({"n_missing": n, "pct_missing": (100 * n / len(df)).round(1)})
            .sort_values("n_missing", ascending=False))


def missing_by(df: pd.DataFrame, target: str, by: str, bins: Optional[list] = None) -> pd.DataFrame:
    """% of rows with `target` missing, within each level (or bin) of `by`."""
    key = pd.cut(df[by], bins) if bins is not None else df[by]
    g = df[target].isna().groupby(key, observed=True)
    return pd.DataFrame({"n": g.size(), "pct_missing": (100 * g.mean()).round(1)})


def compare_by_missingness(df: pd.DataFrame, target: str, others: Iterable[str]) -> pd.DataFrame:
    """Compare other variables between rows where `target` is missing vs observed.

    Uses the standardized mean difference (SMD) rather than a p-value: with many rows,
    tiny differences give small p-values; with few rows, important ones do not.
    |SMD| above about 0.1 is a common flag for a meaningful imbalance.
    Numeric columns only (binary 0/1 columns work too).
    """
    miss = df[target].isna()
    rows = []
    for col in others:
        x = pd.to_numeric(df[col], errors="coerce")
        a, b = x[miss].dropna(), x[~miss].dropna()
        pooled = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
        rows.append({"variable": col, "mean_if_missing": a.mean(), "mean_if_observed": b.mean(),
                     "smd": (a.mean() - b.mean()) / pooled if pooled > 0 else np.nan,
                     "n_missing": len(a), "n_observed": len(b)})
    return pd.DataFrame(rows).round(3)


def simulate_missing(df: pd.DataFrame, target: str, mechanism: str, rate: float = 0.3,
                     driver: Optional[str] = None, seed: int = 0) -> pd.DataFrame:
    """Return a copy of df with values of `target` set to missing.

    MCAR: every row has the same probability `rate`.
    MAR : probability rises with the rank of an observed `driver` column (e.g. age).
    MNAR: probability rises with the rank of `target` itself (high values go missing).
    The average missing fraction is close to `rate` in all three cases.
    """
    rng = np.random.default_rng(seed)
    out = df.copy()
    if mechanism == "MCAR":
        p = np.full(len(df), rate)
    elif mechanism in ("MAR", "MNAR"):
        source = driver if mechanism == "MAR" else target
        if source is None:
            raise ValueError("MAR needs a driver column")
        r = df[source].rank(pct=True).to_numpy()
        p = np.clip(2 * rate * r, 0, 1)        # mean of 2*rate*U(0,1) = rate
    else:
        raise ValueError("mechanism must be MCAR, MAR or MNAR")
    out.loc[rng.random(len(df)) < p, target] = np.nan
    return out
