"""Loading, cleaning and clinical plausibility checks.

Design choice: checks return boolean *flags*; they never drop rows. Every change to the
data is made explicitly in the notebook and recorded in a DecisionLog, so the final
report can state what was changed, why, and how many rows were affected.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple

import numpy as np
import pandas as pd

# Codes that mean "not recorded" in clinic_visits.csv (see data/data_dictionary.csv)
MISSING_CODES = ["", "NA", -99]


def load_visits(path: str | Path) -> pd.DataFrame:
    """Read the visit-level CSV with missing-value codes and dates handled at load time."""
    return pd.read_csv(path, na_values=MISSING_CODES, parse_dates=["visit_date"])


def standardise_labels(s: pd.Series, mapping: Optional[Dict[str, str]] = None) -> pd.Series:
    """Trim whitespace, lower-case, then map abbreviations (e.g. {'f': 'female'}).

    Missing values stay missing.
    """
    out = s.astype("string").str.strip().str.lower()
    if mapping:
        out = out.replace(mapping)
    return out.astype(object).where(out.notna(), np.nan)


def parse_censored(
    s: pd.Series, substitute: str | float = "half"
) -> Tuple[pd.Series, pd.Series]:
    """Convert values such as '<0.3' (below a detection limit) to numbers.

    Returns (numeric_values, below_limit_flag). ``substitute`` is either 'half'
    (limit / 2, a common documented convention), 'limit', or a number. The flag keeps
    the information so that a sensitivity analysis can treat these values differently.
    """
    text = s.astype("string").str.strip()
    below = text.str.startswith("<").fillna(False).astype(bool)
    limit = pd.to_numeric(text.str.lstrip("<"), errors="coerce")
    values = pd.to_numeric(text, errors="coerce")
    if substitute == "half":
        fill = limit / 2
    elif substitute == "limit":
        fill = limit
    else:
        fill = pd.Series(float(substitute), index=s.index)
    values = values.where(~below, fill)
    return values.astype(float), below


Rule = Callable[[pd.DataFrame], pd.Series]


def _outside(col: str, lo: float, hi: float) -> Rule:
    return lambda d: d[col].notna() & ~d[col].between(lo, hi)


# Clinical plausibility rules. Ranges are deliberately wide: they catch impossible or
# almost certainly mistyped values, not unusual-but-real patients.
DEFAULT_RULES: Dict[str, Rule] = {
    "systolic_bp outside 60-260 mmHg": _outside("systolic_bp", 60, 260),
    "diastolic_bp outside 30-150 mmHg": _outside("diastolic_bp", 30, 150),
    "diastolic_bp >= systolic_bp": lambda d: d["diastolic_bp"] >= d["systolic_bp"],
    "height_cm outside 120-220": _outside("height_cm", 120, 220),
    "weight_kg outside 30-250": _outside("weight_kg", 30, 250),
    "cholesterol_total < 20 (mmol/L?)": lambda d: d["cholesterol_total"] < 20,
}


def plausibility_flags(df: pd.DataFrame, rules: Optional[Dict[str, Rule]] = None) -> pd.DataFrame:
    """One boolean column per rule; True = the row breaks that rule."""
    rules = rules or DEFAULT_RULES
    return pd.DataFrame({name: rule(df).fillna(False).astype(bool) for name, rule in rules.items()},
                        index=df.index)


def visit_order_problems(df: pd.DataFrame, id_col: str = "patient_id",
                         visit_col: str = "visit", date_col: str = "visit_date") -> pd.Series:
    """True where a visit is dated before the previous visit of the same patient."""
    d = df.sort_values([id_col, visit_col])
    prev = d.groupby(id_col)[date_col].shift(1)
    return (d[date_col] < prev).reindex(df.index).fillna(False).astype(bool)


@dataclass
class DecisionLog:
    """Record of every cleaning and analysis decision, for the EDA report."""
    rows: list = field(default_factory=list)

    def add(self, step: str, issue: str, n_rows: int, decision: str) -> None:
        self.rows.append({"step": step, "issue": issue, "rows_affected": int(n_rows),
                          "decision": decision})

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.rows, columns=["step", "issue", "rows_affected", "decision"])
