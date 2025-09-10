# HIT140 – Foundations of Data Science
## Assessment 2 (Objective 1 / Investigation A)

### What we’re doing
We want to check if **bats behave more cautiously** (take less “risk”) **when rats are active**.  
If bats treat rats like predators, then during periods of higher rat activity—or right after rats arrive—**bats should avoid risky behaviour more often**.

---

## Data we must use (both files)
- **`dataset1.csv`** – Bat landings (each row is a landing)  
  Columns typically include: `start_time`, `risk` (0/1), `seconds_after_rat_arrival`, `season`, `hours_after_sunset`, …
- **`dataset2.csv`** – Rat activity in **30-minute windows**  
  Columns typically include: `time`, `rat_minutes`, `rat_arrival_number`, `food_availability`, …

> Times are **day/month/year** (e.g., `26/12/2017 16:13`). We parse with that exact format to avoid day-month mixups.

---

## Step by step plan for analysis
1) **Load both CSVs** and **parse date/time** (day-first with `"%d/%m/%Y %H:%M"`).  
2) **Align** each bat landing to the **nearest 30-minute rat window** using a nearest-time join with **±15 min tolerance**.  
3) **Create features**:
   - `rat_pressure = rat_minutes / 30`  → share of the window with rats present (0–1)
   - `time_since_rat_min = seconds_after_rat_arrival / 60` → seconds to minutes
4) **Descriptives**:
   - Mean **risk** by **rat_pressure quartiles** (low → high)  
   - Mean **risk** by **time_since_rat_min quartiles** (early → late)  
   - Save small tables + bar charts (matplotlib).
5) **Inferential tests**:
   - **Chi-square**: compare risk in **Low vs High** rat_pressure (median split)  
   - **Logistic regression** (binary):  
     `risk ~ rat_pressure + time_since_rat_min + food_availability + C(season)`  
     *(Add `hours_after_sunset` only if there’s sufficient non-missing data.)*
6) **Save everything**: merged CSV, result tables, figures (PNG), odds ratios, and a short text summary.

---

## How to run (VS Code / terminal)
Install once (preferably in a virtual environment):
```bash
pip install -r requirements.txt
```