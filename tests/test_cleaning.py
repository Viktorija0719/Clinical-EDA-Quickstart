import numpy as np
import pandas as pd

from scripts.cleaning import (DecisionLog, load_visits, parse_censored, plausibility_flags,
                              standardise_labels, visit_order_problems)
from conftest import ROOT


def test_load_visits_reads_missing_codes_and_dates():
    df = load_visits(ROOT / "data" / "clinic_visits.csv")
    assert len(df) == 550
    assert pd.api.types.is_datetime64_any_dtype(df["visit_date"])
    assert not (df["hdl"] == -99).any()


def test_standardise_labels():
    s = pd.Series(["F", "Female", "Male ", "male", np.nan])
    out = standardise_labels(s, {"f": "female", "m": "male"})
    assert out.tolist()[:4] == ["female", "female", "male", "male"]
    assert pd.isna(out.iloc[4])


def test_parse_censored_half_limit_and_flag():
    vals, flag = parse_censored(pd.Series(["<0.3", "1.5", np.nan]))
    assert vals.iloc[0] == 0.15 and vals.iloc[1] == 1.5 and np.isnan(vals.iloc[2])
    assert flag.tolist() == [True, False, False]


def test_plausibility_flags():
    d = pd.DataFrame({"systolic_bp": [120, 80, np.nan], "diastolic_bp": [80, 120, 70],
                      "height_cm": [170, 17.2, 160], "weight_kg": [70, 70, 70],
                      "cholesterol_total": [200, 5.2, np.nan]})
    f = plausibility_flags(d)
    assert f["diastolic_bp >= systolic_bp"].tolist() == [False, True, False]
    assert f["height_cm outside 120-220"].tolist() == [False, True, False]
    assert f["cholesterol_total < 20 (mmol/L?)"].tolist() == [False, True, False]


def test_visit_order_problems():
    d = pd.DataFrame({"patient_id": ["a", "a", "b"], "visit": [1, 2, 1],
                      "visit_date": pd.to_datetime(["2023-05-01", "2023-01-01", "2023-02-01"])})
    assert visit_order_problems(d).tolist() == [False, True, False]


def test_decision_log():
    log = DecisionLog()
    log.add("clean", "duplicates", 4, "removed")
    assert log.to_frame().shape == (1, 4)
