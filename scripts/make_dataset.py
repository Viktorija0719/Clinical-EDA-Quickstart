"""Generate data/clinic_visits.csv, the synthetic dataset used in the tutorial and the slides.

One row = one clinic visit. About 30% of patients have a follow-up visit.
No real patients are included. Data problems are planted on purpose so that the
cleaning steps have something to find:
  (i) inconsistent sex labels, (ii) cholesterol in mmol/L at site C during 2023,
  (iii) exact duplicate records, (iv) a follow-up dated before baseline,
  (v) impossible blood pressure and height values, (vi) CRP below the detection limit,
  (vii) -99 codes for "not recorded", (viii) missingness that depends on site and age.

Run from the repository root:  python -m scripts.make_dataset
"""
from pathlib import Path

import numpy as np
import pandas as pd

rng = np.random.default_rng(2026)
N = 420

pid = [f"P{i:04d}" for i in range(1, N + 1)]
site = rng.choice(["A", "B", "C"], size=N, p=[0.45, 0.35, 0.20])
age = np.clip(rng.normal(58, 10, N), 35, 85).round(0)
sex = rng.choice(["female", "male"], size=N)
height = np.where(sex == "male", rng.normal(178, 7, N), rng.normal(165, 6, N)).round(1)
bmi_true = np.clip(rng.normal(27.5, 4.2, N), 17, 45)
weight = (bmi_true * (height / 100) ** 2).round(1)
smoking = rng.choice(["never", "former", "current"], size=N, p=[0.5, 0.3, 0.2])

# diabetes depends on age and BMI (age is the confounder for SBP)
lin = -3.3 + 0.065 * (age - 40) + 0.13 * (bmi_true - 22)
diabetes = (rng.random(N) < 1 / (1 + np.exp(-lin))).astype(int)
statin = (rng.random(N) < 0.25 + 0.35 * diabetes).astype(int)

sbp = 108 + 0.62 * (age - 40) + 0.6 * (bmi_true - 25) + 5.0 * diabetes + rng.normal(0, 13, N)
dbp = 60 + 0.35 * (sbp - 110) + rng.normal(0, 7, N)
chol = rng.normal(205, 38, N) - 25 * statin
hdl = np.clip(rng.normal(52, 13, N) - 4 * diabetes + np.where(sex == "female", 7, 0), 20, 110)
crp = np.exp(rng.normal(np.log(1.8) + 0.35 * diabetes + 0.03 * (bmi_true - 27), 0.95, N))
ccs = rng.choice([0, 1, 2, 3, 4], size=N, p=[0.45, 0.25, 0.17, 0.09, 0.04])

start = pd.Timestamp("2022-01-10")
d1 = start + pd.to_timedelta(rng.integers(0, 800, N), unit="D")

base = pd.DataFrame({
    "patient_id": pid, "visit": 1, "visit_date": d1, "site": site, "age": age, "sex": sex,
    "height_cm": height, "weight_kg": weight, "systolic_bp": sbp.round(0), "diastolic_bp": dbp.round(0),
    "cholesterol_total": chol.round(0), "hdl": hdl.round(0), "crp": crp.round(2),
    "smoking_status": smoking, "diabetes": diabetes, "statin": statin, "ccs_class": ccs,
})

# follow-up visits for ~30% of patients (legitimate repeat measurements)
fu = base.sample(frac=0.3, random_state=7).copy()
gap = rng.integers(150, 400, len(fu))
fu["visit"] = 2
fu["visit_date"] = fu["visit_date"] + pd.to_timedelta(gap, unit="D")
fu["age"] = fu["age"] + (gap > 365)
fu["weight_kg"] = (fu["weight_kg"] + rng.normal(-0.5, 2, len(fu))).round(1)
fu["systolic_bp"] = (fu["systolic_bp"] - 3 + rng.normal(0, 8, len(fu))).round(0)
fu["diastolic_bp"] = (fu["diastolic_bp"] - 1 + rng.normal(0, 5, len(fu))).round(0)
fu["cholesterol_total"] = (fu["cholesterol_total"] - 10 + rng.normal(0, 15, len(fu))).round(0)
fu["hdl"] = (fu["hdl"] + rng.normal(0, 4, len(fu))).round(0)
fu["crp"] = (fu["crp"] * np.exp(rng.normal(0, 0.4, len(fu)))).round(2)

df = pd.concat([base, fu]).sort_values(["patient_id", "visit"]).reset_index(drop=True)
n = len(df)

# ---- missingness ----
# MAR: lipid panel not run at site B during early 2022 (depends on site and date)
lip = (df["site"] == "B") & (df["visit_date"] < "2022-09-01") & (rng.random(n) < 0.8)
df.loc[lip, ["cholesterol_total", "hdl"]] = np.nan
# MAR: CRP ordered less often for younger patients
p_crp = np.clip(0.55 - 0.013 * (df["age"] - 40), 0.03, 0.6)
df.loc[rng.random(n) < p_crp, "crp"] = np.nan
# roughly random gaps in weight and height
df.loc[rng.random(n) < 0.06, "weight_kg"] = np.nan
df.loc[rng.random(n) < 0.04, "height_cm"] = np.nan
df.loc[rng.random(n) < 0.03, "smoking_status"] = np.nan

df["visit_date"] = df["visit_date"].dt.strftime("%Y-%m-%d")
df = df.astype(object)

# ---- planted data problems ----
# coded missing values (-99 = not recorded)
for col, k in [("hdl", 4), ("weight_kg", 3)]:
    idx = df.index[df[col].notna()].to_series().sample(k, random_state=k).values
    df.loc[idx, col] = -99
# inconsistent sex labels
idx = df.index[df["sex"] == "female"].to_series().sample(6, random_state=1).values
df.loc[idx[:4], "sex"] = "F"
df.loc[idx[4:], "sex"] = "Female"
idx = df.index[df["sex"] == "male"].to_series().sample(3, random_state=2).values
df.loc[idx, "sex"] = "Male "
# unit mismatch: site C reported total cholesterol in mmol/L during 2023
um = (df["site"] == "C") & (df["visit_date"].str.startswith("2023")) & df["cholesterol_total"].notna()
df.loc[um, "cholesterol_total"] = (df.loc[um, "cholesterol_total"].astype(float) / 38.67).round(1)
# CRP below the assay detection limit (0.3 mg/L) recorded as text
low = df["crp"].notna() & (df["crp"].astype(float) < 0.3)
df["crp"] = df["crp"].astype(object)
df.loc[low, "crp"] = "<0.3"
# impossible values / combinations
df.loc[12, "height_cm"] = 17.2          # typo for 172
df.loc[57, ["systolic_bp", "diastolic_bp"]] = [78, 121]   # columns swapped at entry
df.loc[233, ["systolic_bp", "diastolic_bp"]] = [82, 128]
df.loc[301, "systolic_bp"] = 212        # extreme but plausible
# a follow-up visit dated before baseline
v2 = df.index[df["visit"] == 2][5]
p = df.loc[v2, "patient_id"]
d1 = df.loc[(df["patient_id"] == p) & (df["visit"] == 1), "visit_date"].iloc[0]
df.loc[v2, "visit_date"] = (pd.Timestamp(d1) - pd.Timedelta(days=40)).strftime("%Y-%m-%d")
# exact duplicate records (double data entry)
dups = df.sample(4, random_state=11)
df = pd.concat([df, dups]).sort_values(["patient_id", "visit"]).reset_index(drop=True)

def tidy(v):
    if isinstance(v, float) and not np.isnan(v) and float(v).is_integer():
        return int(v)
    return v

for c in ["visit", "age", "systolic_bp", "diastolic_bp", "cholesterol_total", "hdl", "diabetes", "statin", "ccs_class", "weight_kg"]:
    df[c] = pd.Series([tidy(v) for v in df[c]], dtype=object, index=df.index)

OUT = Path(__file__).resolve().parents[1] / "data" / "clinic_visits.csv"
df.to_csv(OUT, index=False)
print(df.shape, df["patient_id"].nunique())

