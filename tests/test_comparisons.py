import numpy as np
import pandas as pd
from scipy import stats

from scripts.comparisons import (compare_groups, geometric_mean_ratio, mean_difference,
                                 proportion_difference)

rng = np.random.default_rng(1)
D = pd.DataFrame({"g": [1] * 60 + [0] * 80,
                  "y": np.r_[rng.normal(130, 15, 60), rng.normal(120, 15, 80)],
                  "z": np.r_[rng.lognormal(1, 0.8, 60), rng.lognormal(0.8, 0.8, 80)],
                  "c": rng.choice(["a", "b"], 140)})


def test_mean_difference_matches_scipy_welch():
    r = mean_difference(D, "y", "g")
    ref = stats.ttest_ind(D.y[D.g == 1], D.y[D.g == 0], equal_var=False)
    assert np.isclose(r["p_value"], ref.pvalue)
    assert r["ci_low"] < r["estimate"] < r["ci_high"]


def test_geometric_mean_ratio_positive():
    r = geometric_mean_ratio(D, "z", "g")
    assert r["estimate"] > 0 and r["ci_low"] < r["estimate"] < r["ci_high"]


def test_proportion_difference_range():
    r = proportion_difference(D, "c", "a", "g")
    assert -100 <= r["estimate"] <= 100


def test_compare_groups_adds_adjusted_p():
    out = compare_groups(D, "g", {"y": "mean", "z": "log", "c": "prop:a"})
    assert "p_fdr_bh" in out and (out["p_fdr_bh"] >= out["p_value"] - 1e-12).all()


def test_prop_spec_on_numeric_column():
    d = pd.DataFrame({"g": [1, 1, 0, 0], "s": [1, 1, 0, 1]})
    out = compare_groups(d, "g", {"s": "prop:1"})
    assert np.isclose(out.loc[0, "estimate"], 50.0)
