import numpy as np
import pandas as pd

from scripts.summaries import build_table1, fmt_continuous, fmt_n_pct


def test_fmt_n_pct():
    assert fmt_n_pct(1, 4) == "1/4 (25%)"
    assert fmt_n_pct(0, 0) == "NA"


def test_fmt_continuous():
    x = pd.Series([1, 2, 3, 4, np.nan])
    assert fmt_continuous(x, "mean") == "2.5 (1.3)"
    assert fmt_continuous(x, "median") == "2.5 (1.8 to 3.2)"


def test_build_table1_columns_and_missing():
    d = pd.DataFrame({"g": [0, 0, 1, 1], "age": [50, 60, 70, np.nan],
                      "sex": ["female", "male", "female", np.nan]})
    t = build_table1(d, "g", {"age": "mean"}, {"sex": ["female"]}, group_labels={0: "No", 1: "Yes"})
    assert list(t.columns) == ["Overall", "No", "Yes", "Missing"]
    assert t.loc[("sex, n/N (%)", "female"), "Yes"] == "1/1 (100%)"
    assert t.loc[("age, mean (SD)", ""), "Missing"] == "1"
