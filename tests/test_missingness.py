import numpy as np
import pandas as pd

from scripts.missingness import compare_by_missingness, missing_summary, simulate_missing

rng = np.random.default_rng(0)
D = pd.DataFrame({"age": rng.normal(60, 10, 4000), "sbp": rng.normal(125, 15, 4000)})


def test_simulated_rates_close_to_target():
    for mech, drv in [("MCAR", None), ("MAR", "age"), ("MNAR", None)]:
        out = simulate_missing(D, "sbp", mech, rate=0.3, driver=drv, seed=1)
        assert abs(out["sbp"].isna().mean() - 0.3) < 0.03


def test_mnar_removes_high_values():
    out = simulate_missing(D, "sbp", "MNAR", rate=0.3, seed=1)
    assert out["sbp"].mean() < D["sbp"].mean()


def test_summaries_run():
    out = simulate_missing(D, "sbp", "MAR", rate=0.3, driver="age", seed=1)
    assert missing_summary(out).loc["sbp", "n_missing"] > 0
    smd = compare_by_missingness(out, "sbp", ["age"]).loc[0, "smd"]
    assert smd > 0.1          # older patients are more often missing under this MAR
