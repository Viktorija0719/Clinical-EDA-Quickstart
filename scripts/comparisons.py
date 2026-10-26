"""Group comparisons that report an effect estimate with a confidence interval.

Design choice: each function returns the size of the difference and its uncertainty,
with the p-value as one column among several. The analyst chooses the effect measure
for each variable (difference in means, ratio of geometric means for right-skewed
variables, difference in proportions) from its distribution and clinical meaning.
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests


def _two_groups(df: pd.DataFrame, outcome: str, group: str, g1, g0):
    y1 = pd.to_numeric(df.loc[df[group] == g1, outcome], errors="coerce").dropna()
    y0 = pd.to_numeric(df.loc[df[group] == g0, outcome], errors="coerce").dropna()
    return y1, y0


def mean_difference(df, outcome, group, g1=1, g0=0, conf=0.95) -> dict:
    """Difference in means (g1 minus g0), Welch t-test CI and p-value."""
    y1, y0 = _two_groups(df, outcome, group, g1, g0)
    res = stats.ttest_ind(y1, y0, equal_var=False)
    ci = res.confidence_interval(conf)
    return {"variable": outcome, "measure": "difference in means",
            "estimate": y1.mean() - y0.mean(), "ci_low": ci.low, "ci_high": ci.high,
            "p_value": res.pvalue, "n1": len(y1), "n0": len(y0)}


def geometric_mean_ratio(df, outcome, group, g1=1, g0=0, conf=0.95) -> dict:
    """Ratio of geometric means (g1 / g0) for positive, right-skewed variables.

    Welch t-test on the log scale, back-transformed. A ratio of 1.25 means values are
    25% higher in g1 on a typical (geometric mean) patient.
    """
    y1, y0 = _two_groups(df, outcome, group, g1, g0)
    if (y1 <= 0).any() or (y0 <= 0).any():
        raise ValueError(f"{outcome} has values <= 0; a log scale is not defined")
    r = mean_difference(pd.DataFrame({"y": np.log(pd.concat([y1, y0])),
                                      "g": [g1] * len(y1) + [g0] * len(y0)}),
                        "y", "g", g1, g0, conf)
    return {"variable": outcome, "measure": "ratio of geometric means",
            "estimate": np.exp(r["estimate"]), "ci_low": np.exp(r["ci_low"]),
            "ci_high": np.exp(r["ci_high"]), "p_value": r["p_value"], "n1": r["n1"], "n0": r["n0"]}


def proportion_difference(df, outcome, level, group, g1=1, g0=0, conf=0.95) -> dict:
    """Difference in the proportion with outcome == level (g1 minus g0), in percentage points.

    Wald confidence interval (adequate for moderate counts); p-value from the chi-square
    test, or Fisher's exact test when an expected count is below 5.
    """
    d = df[[outcome, group]].dropna()
    d = d[d[group].isin([g1, g0])]
    hit = d[outcome] == level
    n1, n0 = (d[group] == g1).sum(), (d[group] == g0).sum()
    p1, p0 = hit[d[group] == g1].mean(), hit[d[group] == g0].mean()
    se = np.sqrt(p1 * (1 - p1) / n1 + p0 * (1 - p0) / n0)
    z = stats.norm.ppf(0.5 + conf / 2)
    table = pd.crosstab(hit, d[group])
    chi2, p, _, expected = stats.chi2_contingency(table, correction=False)
    if (expected < 5).any():
        p = stats.fisher_exact(table.values).pvalue
    diff = (p1 - p0) * 100
    shown = int(level) if isinstance(level, float) and float(level).is_integer() else level
    return {"variable": f"{outcome} = {shown}", "measure": "difference in % (points)",
            "estimate": diff, "ci_low": diff - z * se * 100, "ci_high": diff + z * se * 100,
            "p_value": p, "n1": int(n1), "n0": int(n0)}


def compare_groups(df: pd.DataFrame, group: str, specs: Dict[str, str], g1=1, g0=0,
                   adjust: str = "fdr_bh") -> pd.DataFrame:
    """Compare many variables between two groups and adjust p-values for multiplicity.

    specs: {column: "mean" | "log" | "prop:<level>"}
    adjust: any statsmodels multipletests method ('fdr_bh' = Benjamini-Hochberg, 'holm').
    """
    rows = []
    for col, how in specs.items():
        if how == "mean":
            rows.append(mean_difference(df, col, group, g1, g0))
        elif how == "log":
            rows.append(geometric_mean_ratio(df, col, group, g1, g0))
        elif how.startswith("prop:"):
            level = how.split(":", 1)[1]
            if pd.api.types.is_numeric_dtype(df[col]):
                level = float(level)          # "prop:1" on a 0/1 column means the value 1
            rows.append(proportion_difference(df, col, level, group, g1, g0))
        else:
            raise ValueError(f"Unknown spec {how!r} for {col}")
    out = pd.DataFrame(rows)
    out[f"p_{adjust}"] = multipletests(out["p_value"], method=adjust)[1]
    return out


def adjusted_effect(df: pd.DataFrame, formula: str, term: str, conf: float = 0.95) -> dict:
    """Estimate, CI and p-value for one term of an OLS model, e.g. 'sbp ~ diabetes + age'."""
    import statsmodels.formula.api as smf
    fit = smf.ols(formula, data=df).fit()
    lo, hi = fit.conf_int(alpha=1 - conf).loc[term]
    return {"model": formula, "estimate": fit.params[term], "ci_low": lo, "ci_high": hi,
            "p_value": fit.pvalues[term], "n": int(fit.nobs)}
