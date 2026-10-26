# Quickstart: Clinical Data Exploratory Data Analysis (EDA)

A hands-on tutorial for exploring clinical tabular data in Python, from a written study question to a
documented EDA report. It accompanies the slide deck in `slides/` and follows the same steps.

## What the tutorial covers

| Step | Notebook section | Main idea |
|---|---|---|
| 0 | Study question and data dictionary | Define population, comparison, outcome and the unit of analysis first |
| 1 | Load and first look | Declare missing-value codes and dates at load time; check structure before statistics |
| 2 | Clean and clinical checks | Labels, types, duplicates versus repeat visits, unit mismatches, impossible values, detection limits |
| 3 | Explore | Summaries chosen by type and shape; histograms, Q-Q plots, group and relationship plots |
| 4 | Outliers | Flag statistically, judge clinically, check sensitivity |
| 5 | Missing data | Patterns and plausible reasons; a simulation of MCAR, MAR and MNAR; multiple imputation with pooling |
| 6 | Features | Derived variables, encoding by variable type, scaling versus transformation |
| 7A | Statistical inference | Effect estimates with confidence intervals, multiplicity, confounding |
| 7B | Predictive modelling | Leakage-safe pipeline with patient-grouped cross-validation |
| 8 | Report | Table 1, saved figures and a decision log |

## Repository structure

```text
eda_tutorial.ipynb   Main tutorial notebook (executed, with outputs)
scripts/             Reusable functions used by the notebook
  cleaning.py          loading, label standardisation, censored values, plausibility rules, decision log
  summaries.py         shape summaries and a Table 1 builder (n/N (%), mean (SD) or median (IQR), missing)
  comparisons.py       effect estimates with CIs (means, geometric-mean ratios, proportions), multiplicity
  missingness.py       missingness summaries, SMD comparison, MCAR/MAR/MNAR simulation
  plots.py             consistent plot helpers
  make_dataset.py      regenerates data/clinic_visits.csv exactly
data/
  clinic_visits.csv    synthetic dataset (550 visits, 420 patients, 3 sites)
  data_dictionary.csv  variable types, units, codes and missing-value codes
outputs/             Files written by the notebook: table1.csv, decision_log.csv, figures/
slides/              Quickstart_Clinical_EDA.pptx
tests/               Unit tests for scripts/ (pytest)
requirements.txt     Python dependencies (versions the notebook was run with)
```

## Dataset

`data/clinic_visits.csv` is **synthetic**. It contains no real patients. It imitates a cardiovascular risk
clinic: one row per visit, about 30% of patients with a follow-up visit. Problems are planted on purpose so
that the cleaning steps have something to find: inconsistent sex labels, cholesterol in mmol/L at one site,
exact duplicate records, a follow-up dated before baseline, impossible blood pressure and height values,
CRP below the detection limit, `-99` codes for "not recorded", and missingness that depends on site and age.

Regenerate it from the repository root with:

```bash
python3 -m scripts.make_dataset
```

The same workflow applies to public datasets, for example the Kaggle
[cardiovascular disease](https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset) or
[heart failure clinical records](https://www.kaggle.com/datasets/andrewmvd/heart-failure-clinical-data) datasets.

## Installation and setup

Clone or download the repository, then work from its root folder.

**Linux / macOS**

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
```

**Windows (PowerShell)**

```powershell
py -m venv .venv
.venv\Scripts\activate
py -m pip install --upgrade pip
py -m pip install -r requirements.txt
```

Optional: register the environment as a named Jupyter kernel, useful when several environments exist.

```bash
python3 -m ipykernel install --user --name clinical-eda --display-name "Python (clinical-eda)"
```

## Run the tutorial

```bash
python3 -m jupyter lab
```

Open `eda_tutorial.ipynb` and run the cells from top to bottom (or *Run > Run All Cells*). The notebook writes
`outputs/table1.csv`, `outputs/decision_log.csv` and three figures to `outputs/figures/`.

The optional automated profile report uses `fg-data-profiling`, the renamed `ydata-profiling` package (the old
name no longer receives updates). Set `RUN_PROFILE = True` in section 1 to build `outputs/eda_report.html`.

Run the unit tests with:

```bash
python3 -m pytest
```

## Design decisions

* **Flags, not silent deletion.** Cleaning functions return flags. Every change is made explicitly in the notebook
  and recorded in a decision log with the number of rows affected, because the report must state what changed and why.
* **No normality-test gate.** The summary (mean (SD) or median (IQR)) and the effect measure are chosen by the analyst
  from the histogram, Q-Q plot, skewness, group sizes and study design. A Shapiro-Wilk p-value depends mostly on the
  sample size: section 3.5 shows the same variable "passing" at n = 30 and "failing" at n = 419.
* **Effect estimates first.** Comparisons report the size of the difference with a 95% confidence interval; p-values
  are adjusted for the number of tests.
* **Missing data by goal.** Inference uses complete cases only when justified, otherwise multiple imputation with
  pooling; mean imputation is avoided because it shrinks variance and weakens correlations even under MCAR.
  Prediction fits imputation inside the cross-validated pipeline.
* **Patient-level splitting.** Visits from one patient never appear in both training and test folds.

## Resources by task

| Task | Resource |
|---|---|
| Beginner Python | [Software Carpentry: Programming with Python](https://swcarpentry.github.io/python-novice-inflammation/) (patient data example) |
| Beginner R and EDA | [R for Data Science, 2nd edition](https://r4ds.hadley.nz/) |
| Loading and manipulating tables | [pandas: Getting started tutorials](https://pandas.pydata.org/docs/getting_started/intro_tutorials/index.html) |
| Plotting distributions | [seaborn: Visualizing distributions of data](https://seaborn.pydata.org/tutorial/distributions.html) |
| Practising visualization | [Kaggle Learn: Data Visualization](https://www.kaggle.com/learn/data-visualization) |
| Automated profiling | [YData Profiling: Quickstart](https://docs.profiling.ydata.ai/latest/getting-started/quickstart/) |
| Clinical summary tables (R) | [gtsummary: tbl_summary() tutorial](https://www.danieldsjoberg.com/gtsummary/articles/tbl_summary.html) |
| Missing-data reasoning | [van Buuren: Flexible Imputation of Missing Data](https://stefvanbuuren.name/fimd/) |
| Multiple-imputation analysis (R) | [mice: impute, with(), pool() workflow](https://amices.org/mice/) |
| Transition to machine learning | [scikit-learn: Common pitfalls and data leakage](https://scikit-learn.org/stable/common_pitfalls.html) |
| p-values and effect sizes | [ASA statement on statistical significance and p-values](https://www.amstat.org/asa/files/pdfs/p-valuestatements.pdf) |

**Main textbooks:**
[Data Science for Health Informatists](https://book.datascience.appliedhealthinformatics.com/docs/intro) (start with Ch. 2 and 3) ·
[Python for Health Data Science](https://www.pythonhealthdatascience.com/content/front_page.html) (start with the pandas chapter and the stroke data wrangling exercise) ·
[Data Analysis in Medicine and Health using R](https://bookdown.org/drki_musa/dataanalysis/) (start with Ch. 4 to 6)
