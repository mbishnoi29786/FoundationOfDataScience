# Assessment 3 — Objective 2 Toolkit

## Why we built this
Our team set out to answer a simple question for Objective 2: **does rat activity change how boldly bats land?**  
`assesment3_obj2.py` is the playbook we wrote to make that call. It cleans the two Learnline CSVs,
aligns landings with 30‑minute rat windows, tests the patterns we discussed in studio, and leaves an audit trail of charts and tables we can drop straight into the report.

---

## How we approached the question
We agreed on three checkpoints before trusting any conclusions:

1. **Context checks** – confirm that rat pressure, season, and time-since-arrival behave sensibly (histograms, ECDFs, monthly lines).
2. **Group comparisons** – quantify how risk shifts when rats are around and across seasons. We chose Wilson confidence intervals to keep proportions honest even with small bins.
3. **Model cross-checks** – fit a ridge-penalised logistic regression (plus window-level binomial GLM) so any claims about pressure × season interactions rest on more than one lens.

Every figure or CSV in `outputs/` supports one of those checkpoints and is labelled accordingly.

---

## Running the toolkit
```bash
python3 -m venv .venv        # optional virtual environment
source .venv/bin/activate
pip install -r requirements.txt
python assesment3_obj2.py
```

Drop `dataset1.csv` and `dataset2.csv` into this folder before running.  
All refreshed outputs land in `outputs/`—rerunning will overwrite the previous run.

---

## What the outputs say (and why they exist)

| Artefact | What we looked for | Why it matters |
| --- | --- | --- |
| `fig_risk_by_rat_present.png` | Risk rates when rats are absent vs present, Wilson CIs, χ² p-value, and the difference CI. | Validates the headline hypothesis that sparked the project. |
| `fig_risk_by_season.png` | Seasonal bars with CIs plus χ². | Checks the Objective 2 factor the brief asked us to explore. |
| `fig_risk_vs_rat_pressure_deciles*.png` | Ordered bins (with sparse top deciles merged) overall and by season. | Keeps each bar above n≈20 so the Wilson CIs stay trustworthy. |
| `fig_risk_by_*_quartile.png` | Rat-pressure and time-since-arrival quartiles. | Provides coarser groupings for the report appendix and flags overlap in CIs. |
| `fig_heatmap_risk_season_pressure.png` | Season × pressure grid. | Quick read on interaction hotspots before trusting model coefficients. |
| `fig_lowess_risk_vs_rat_pressure_by_season.png` | LOWESS risk curve with 95 % bands for each season. | Lets us discuss the continuous trend without arguing about bin edges. |
| `fig_predicted_risk_by_rat_pressure_by_season.png` | Penalised-logit predictions with other covariates fixed at medians. | Shows how the interaction plays out in probability space. |
| `fig_predicted_risk_by_hours_after_sunset.png` | Same model but varying hours after sunset. | Checks whether timing, not just rat pressure, nudges risk up or down. |
| `fig_logit_forest.png` | Penalised logit odds ratios with 95 % CIs. | Summarises which predictors are pulling risk up or down. |
| `fig_ecdf_rat_pressure_by_season.png` | ECDF curves for pressure split by season. | Confirms whether seasons really face different exposure levels. |
| `fig_hexbin_landings_vs_rat_minutes.png` | Hexbin with log colour scale + negative-binomial fit. | Handles overplotting and reports the slope/CIs from an over-dispersion-aware model. |
| `fig_rat_pressure_hist.png`, `fig_window_monthly_*.png` | Histogram with quantile markers and monthly sanity checks. | Shows the landmarks we use in the rest of the analysis. |

All related CSVs (`risk_by_*.csv`, `chi_crosstab_*.csv`, `logit_odds_ratios.csv`, `glm_binomial_or.csv`, etc.) carry the same story with exact numbers, plus sample size (`n`) and confidence bounds for transparency.

---

## How the analysis evolved
- The **rat-present comparison** drove us to add Wilson intervals—the normal approximation was too optimistic in sparse bins.  
- The **χ² tests** frequently came up in feedback, so the script prints them into `results_summary_obj2.txt` and annotates the plots.  
- Pressure deciles were occasionally thin at the top, so we now merge sparse upper bins and add a LOWESS view to keep the story stable.  
- Over-dispersion showed up in the landings scatter, so we swapped the straight line for a negative-binomial GLM and report its slope/CIs.  
- The logistic is now ridge-penalised; we keep the forest plot plus new marginal-effect curves to make the coefficients easier to explain.  
- Window-level weighting via the **binomial GLM** still acts as a sensitivity check so we know the interaction pattern is not a sampling quirk.

The summary text stitches these pieces together so the marker sees the same narrative we discussed: higher rat pressure generally nudges risk upward, season changes the slope a little, and overlapping CIs keep the tone cautious.

---

## Repository layout
```
Assesment3/
├─ assesment3_obj2.py        # Main script 
├─ dataset1.csv              
├─ dataset2.csv              
├─ outputs/                  # Auto-created: charts, tables, summaries
├─ README.md                 # This document
└─ requirements.txt          # Locked dependency versions
```

---

## Requirements (same set we used in venv)
```
numpy==1.26.4
pandas==2.2.2
matplotlib==3.8.4
seaborn==0.13.2
scikit-learn==1.4.2
statsmodels==0.14.2
scipy==1.13.1
patsy==0.5.6
```

No extra packages are needed beyond the course standard stack.

---

If you extend the work—say by swapping in a different Objective 2 factor—follow the same checkpoints so future reviewers can trace the story from raw data through to the final claim.
