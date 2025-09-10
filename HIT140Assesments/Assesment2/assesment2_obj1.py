"""
HIT140 – Foundations of Data Science
Assessment 2 (Investigation A) — Bats and Rats

Goal of investigation:
Check if bats act more cautiously (take less “risk”) when rats are active.
If rats are treated like predators, higher rat activity and time just after rats arrive
should be linked with lower bat risk-taking.

What we do:
1) Load dataset1 (bat landings) & dataset2 (30-min rat windows)
2) Parse times (day-first) and align each landing to the nearest window (±15 min)
3) Create features:
    - rat_pressure = rat_minutes / 30
    - time_since_rat_min = seconds_after_rat_arrival / 60
4) Descriptives: mean risk by quartiles of rat_pressure and time_since_rat_min (+ charts)
5) Chi-square: compare risk in Low vs High rat_pressure (median split)
6) Logistic regression: risk ~ rat_pressure + time_since_rat_min + food_availability + C(season)
    + hours_after_sunset (if available and enough data)
7) Save: merged CSV, small tables, figures (PNG), odds ratios, and a text summary
"""

# ---------- CONSTANTS Configuration ----------
DATASET1_PATH = "./dataset1.csv"    # landing-level data
DATASET2_PATH = "./dataset2.csv"    # 30-min windows data
OUTDIR        = "./outputs"         # results folder
TOLERANCE_MIN = 15                  # nearest-time matching tolerance in minutes
# ======================================================================

from pathlib import Path
from typing import Dict, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------- Step 1: Load data & parse times (day-first to avoid confusion) ----------
def load_data(p1: Path, p2: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Read both CSV files and convert their time columns to real timestamps.

    Why this matters:
    If we leave times as plain text, we can’t do “nearest time” matching. By parsing them
    as timestamps, we can line up bat landings to rat windows accurately.

    Note on date format:
    Our times are day-first (DD/MM/YYYY HH:MM), e.g., 26/12/2017 16:13.
    We tell pandas the exact format so it doesn’t mistake “12/10/2017” as Dec 10 vs Oct 12.
    """

    df1 = pd.read_csv(p1)
    df2 = pd.read_csv(p2)

    # dataset1: landing timestamp (e.g., "26/12/2017 16:13")
    if "start_time" in df1.columns:
        df1["start_time"] = pd.to_datetime(df1["start_time"], format="%d/%m/%Y %H:%M", errors="coerce")

    # dataset2: window start time (same format)
    if "time" in df2.columns:
        df2["time"] = pd.to_datetime(df2["time"], format="%d/%m/%Y %H:%M", errors="coerce")

    return df1, df2

# Helper to prefer left-side column if merge created *_x / *_y
def unify_column(df: pd.DataFrame, base: str) -> pd.Series:
    """
    Small helper: after merging two tables, pandas sometimes appends _x (left) and _y (right)
    to column names if both tables had a column with the same name.

    We prefer the left version (usually the bat-landing table), then the plain name,
    and finally the right version. If none exist, return a column of NaN so the code won’t crash.

    This keeps our downstream code clean and predictable.
    """
    if f"{base}_x" in df.columns:
        return df[f"{base}_x"]
    if base in df.columns:
        return df[base]
    if f"{base}_y" in df.columns:
        return df[f"{base}_y"]
    return pd.Series([np.nan] * len(df), index=df.index)

# ---------- Step 2: Align landings to nearest 30-min window (± TOLERANCE_MIN) ----------
def align_asof(df1: pd.DataFrame, df2: pd.DataFrame, tolerance_min: int = 15) -> pd.DataFrame:
    """
    Core idea:
    A bat landing happens at an exact timestamp (e.g., 16:28:05).
    Rat activity is summarised over 30-minute windows (e.g., a window starting at 16:13).
    We “join by nearest time” so each landing inherits the rat info from the closest window,
    as long as that window is within ±tolerance minutes.

    Why not a normal merge?
    Because the timestamps almost never match exactly. merge_asof is built for this use-case.
    """
    if "start_time" not in df1.columns or "time" not in df2.columns:
        raise ValueError("Expected 'start_time' in dataset1 and 'time' in dataset2.")

    # merge_asof requires both sides to be sorted by time (otherwise it refuses to run)
    dfa = df1.dropna(subset=["start_time"]).sort_values("start_time").copy()
    dfb = df2.dropna(subset=["time"]).sort_values("time").copy()

    merged = pd.merge_asof(
        dfa, dfb,
        left_on="start_time", right_on="time",  # choose the closest window (could be before or after)
        direction="nearest", tolerance=pd.Timedelta(f"{tolerance_min}min")  # only if within ±tolerance
    )

    # If both datasets carried hours_after_sunset (rare), pick the left one
    if ("hours_after_sunset" in merged.columns or
        "hours_after_sunset_x" in merged.columns or
        "hours_after_sunset_y" in merged.columns):
        merged["hours_after_sunset_merged"] = pd.to_numeric(
            unify_column(merged, "hours_after_sunset"), errors="coerce"
        )

    # Features from the matched window
    # 1) How much of the 30-min window had rats present? (0 = none, 1 = all 30 minutes)
    if "rat_minutes" in merged.columns:
        merged["rat_pressure"] = pd.to_numeric(merged["rat_minutes"], errors="coerce") / 30.0

    # 2) How long after the rat arrived did the bat land? (in minutes, easier to read than seconds)
    if "seconds_after_rat_arrival" in merged.columns:
        merged["time_since_rat_min"] = pd.to_numeric(
            merged["seconds_after_rat_arrival"], errors="coerce"
        ) / 60.0

    # 3) Was there any rat arrival in that window at all? Nice binary flag for quick summaries.
    if "rat_arrival_number" in merged.columns:
        merged["rat_present"] = (
            pd.to_numeric(merged["rat_arrival_number"], errors="coerce") > 0
        ).astype("Int64")

    # Ensured numeric types for columns we’ll model on (avoids “object” dtype headaches)
    if "risk" in merged.columns:
        merged["risk"] = pd.to_numeric(merged["risk"], errors="coerce")

    # Prefer the unified hours_after_sunset if we made it above
    merged["hours_after_sunset"] = (
        merged["hours_after_sunset_merged"]
        if "hours_after_sunset_merged" in merged.columns
        else pd.to_numeric(merged.get("hours_after_sunset"), errors="coerce")
    )

    for c in ["food_availability", "time_since_rat_min", "rat_pressure"]:
        if c in merged.columns:
            merged[c] = pd.to_numeric(merged[c], errors="coerce")

    return merged

# ---------- Step 3: Descriptives (tables + simple bar charts) ----------
def descriptives(merged: pd.DataFrame, outdir: Path) -> Dict[str, pd.DataFrame]:
    """
    The goal here is to “just look” before we test:
    • Do we see lower risk when rat_pressure is high?
    • Do we see lower risk right after rats arrive?

    We use quartiles (four equal-sized groups) so we don’t need to pick arbitrary cut-offs.
    """

    outputs: Dict[str, pd.DataFrame] = {}

    # A) Risk by rat_pressure quartile
    if {"risk", "rat_pressure"}.issubset(merged.columns):
        rp = merged[["risk", "rat_pressure"]].dropna().copy()
        if not rp.empty and rp["rat_pressure"].nunique() > 1:
            # pd.qcut splits the data into 4 groups with roughly equal counts
            rp["rp_bin"] = pd.qcut(rp["rat_pressure"], 4, duplicates="drop")
            tbl = rp.groupby("rp_bin", observed=True)["risk"].mean().reset_index(name="risk_rate")
            tbl.to_csv(outdir / "risk_by_rat_pressure_quartile.csv", index=False)
            outputs["risk_by_rat_pressure_quartile"] = tbl

            plt.figure()
            x = tbl["rp_bin"].astype(str)
            y = tbl["risk_rate"]
            plt.bar(x, y)  
            for i, v in enumerate(y):
                plt.text(i, float(v), f"{v:.3f}", ha="center", va="bottom", fontsize=9)
            plt.title("Risk-taking by rat-pressure quartile")
            plt.xlabel("Rat-pressure quartile (low → high)")
            plt.ylabel("Mean risk (proportion)")
            plt.xticks(rotation=30, ha="right")
            plt.tight_layout()
            plt.savefig(outdir / "fig_risk_by_rat_pressure_quartile.png", dpi=150)
            plt.close()

    # B) Risk by time_since_rat_min quartile
    if {"risk", "time_since_rat_min"}.issubset(merged.columns):
        ts = merged[["risk", "time_since_rat_min"]].dropna().copy()
        if not ts.empty and ts["time_since_rat_min"].nunique() > 1:
            ts["ts_bin"] = pd.qcut(ts["time_since_rat_min"], 4, duplicates="drop")
            tbl2 = ts.groupby("ts_bin", observed=True)["risk"].mean().reset_index(name="risk_rate")
            tbl2.to_csv(outdir / "risk_by_time_since_rat_quartile.csv", index=False)
            outputs["risk_by_time_since_rat_quartile"] = tbl2

            plt.figure()
            x2 = tbl2["ts_bin"].astype(str)
            y2 = tbl2["risk_rate"]
            plt.bar(x2, y2)
            for i, v in enumerate(y):
                plt.text(i, float(v), f"{v:.3f}", ha="center", va="bottom", fontsize=9)
            plt.title("Risk-taking by time-since-rat-arrival quartile")
            plt.xlabel("Time since rat arrival (minutes), quartiles")
            plt.ylabel("Mean risk (proportion)")
            plt.xticks(rotation=30, ha="right")
            plt.tight_layout()
            plt.savefig(outdir / "fig_risk_by_time_since_rat_quartile.png", dpi=150)
            plt.close()

    # C) Risk when any rat present vs none
    if {"risk","rat_present"}.issubset(merged.columns):
        rp_tbl = (merged[["risk","rat_present"]]
                .dropna()
                .groupby("rat_present", observed=True)["risk"]
                .mean()
                .reset_index(name="risk_rate"))
        rp_tbl["label"] = rp_tbl["rat_present"].astype(int).map({0: "No rat present", 1: "Rat present"})
        rp_tbl.to_csv(outdir / "risk_by_rat_present.csv", index=False)

        plt.figure()
        x3 = rp_tbl["label"].astype(str)
        y3 = rp_tbl["risk_rate"]
        plt.bar(x3, y3)
        for i, v in enumerate(y3):
            plt.text(i, float(v), f"{v:.3f}", ha="center", va="bottom", fontsize=9)
        plt.title("Risk-taking with vs without rats present")
        plt.xlabel("")
        plt.ylabel("Mean risk (proportion)")
        plt.tight_layout()
        plt.savefig(outdir / "fig_risk_by_rat_present.png", dpi=150)
        plt.close()

        # D) Risk vs rat_pressure (deciles)
    if {"risk","rat_pressure"}.issubset(merged.columns):
        d = merged[["risk","rat_pressure"]].dropna().copy()
        if d["rat_pressure"].nunique() > 1:
            d["decile"] = pd.qcut(d["rat_pressure"], 10, labels=False, duplicates="drop")
            g = d.groupby("decile", observed=True)["risk"].mean().reset_index()
            g.to_csv(outdir / "risk_by_rat_pressure_deciles.csv", index=False)

            plt.figure()
            plt.plot(g["decile"], g["risk"], marker="o")
            for i, v in enumerate(g["risk"]):
                plt.text(g["decile"][i], float(v), f"{v:.3f}", ha="center", va="bottom", fontsize=8)
            plt.title("Risk-taking across rat-pressure deciles")
            plt.xlabel("Rat-pressure decile (low → high)")
            plt.ylabel("Mean risk (proportion)")
            plt.tight_layout()
            plt.savefig(outdir / "fig_risk_vs_rat_pressure_deciles.png", dpi=150)
            plt.close()

    # F) Distribution of rat_pressure
    if "rat_pressure" in merged.columns:
        vals = merged["rat_pressure"].dropna()
        if len(vals) > 0:
            plt.figure()
            plt.hist(vals, bins=20)
            plt.axvline(vals.median(), linestyle="--")  # median reference
            plt.title("Distribution of rat-pressure")
            plt.xlabel("Rat-pressure (share of 30-min window with rats)")
            plt.ylabel("Count of landings")
            plt.tight_layout()
            plt.savefig(outdir / "fig_rat_pressure_hist.png", dpi=150)
            plt.close()

            
    return outputs

# ---------- Step 4: Chi-square (High vs Low rat activity) ----------
def chi_square(merged: pd.DataFrame, outdir: Path) -> dict:
    """
    “Is risk different when rat activity is high vs low?”

    We take rat_pressure, split it at the median into two groups: Low and High.
    Then we make a 2×2 table: rows = Low/High rat activity, columns = risk 0/1.
    The chi-square test tells us whether the difference we see is bigger than we’d
    expect by random chance (p-value).
    """
    from scipy.stats import chi2_contingency
    result = {}

    if {"risk", "rat_pressure"}.issubset(merged.columns):
        tmp = merged[["risk", "rat_pressure"]].dropna().copy()
        if len(tmp) >= 10 and tmp["rat_pressure"].nunique() > 1:
            med = tmp["rat_pressure"].median()
            tmp["rp_high"] = (tmp["rat_pressure"] >= med).astype(int)
            ct = pd.crosstab(tmp["rp_high"], tmp["risk"])
            if ct.shape == (2, 2):
                chi2, p, dof, _ = chi2_contingency(ct)
                result = {"chi2": float(chi2), "p_value": float(p), "dof": int(dof)}
                ct.to_csv(outdir / "chi_crosstab_highlow_rat_pressure_vs_risk.csv")
    return result

# ---------- Step 5: Logistic regression (binary risk) ----------
def logistic_model(merged: pd.DataFrame, outdir: Path) -> str:
    """
    Our outcome “risk” is 0/1. Logistic regression is the standard model for binary outcomes.
    It tells us how the odds of “risk = 1” change with rat_pressure and other factors.

    Model we try:
        risk ~ rat_pressure + time_since_rat_min + food_availability + C(season)
    If we have enough values, we also include hours_after_sunset (or we skip it to keep rows).
    """
    import statsmodels.formula.api as smf

    preds = []
    if "rat_pressure" in merged.columns:       preds.append("rat_pressure")
    if "time_since_rat_min" in merged.columns: preds.append("time_since_rat_min")
    if "food_availability" in merged.columns:  preds.append("food_availability")
    if "season" in merged.columns:             preds.append("C(season)")
    # Add only if enough data; otherwise model drops many rows
    if "hours_after_sunset" in merged.columns and merged["hours_after_sunset"].notna().sum() > 50:
        preds.append("hours_after_sunset")

    if "risk" not in merged.columns or not preds:
        return "[Model skipped: missing columns]"

    # Build a clean modelling table with just the columns we need, dropping rows with NA
    cols_needed = ["risk"] + [p.replace("C(", "").replace(")", "") for p in preds]
    model_df = merged[[c for c in cols_needed if c in merged.columns]].dropna()

    # Need at least some variation in outcome and enough rows to fit a model
    if model_df["risk"].nunique() < 2 or len(model_df) < 50:
        return "[Model skipped: insufficient variation / rows]"

    formula = "risk ~ " + " + ".join(preds)

    try:
        logit = smf.logit(formula=formula, data=model_df).fit(disp=False)
        summary_text = logit.summary2().as_text()

        conf = logit.conf_int()
        or_table = pd.DataFrame({
            "term": logit.params.index,
            "odds_ratio": np.exp(logit.params.values),
            "ci_lower": np.exp(conf[0].values),
            "ci_upper": np.exp(conf[1].values),
            "p_value": logit.pvalues.values
        })
        (outdir / "logit_odds_ratios.csv").write_text(or_table.to_csv(index=False))
        return summary_text

    except Exception as e:
        # If the model struggles (e.g., perfect separation), we don’t crash the whole script.
        return f"[Model failed: {e}]"

# ---------- Step 6: Short text summary ----------
def write_text_report(outdir: Path, dsc: dict, chi: dict, logit_summary: str) -> Path:
    lines = []
    lines.append("HIT140 – Assessment 2 (Investigation A) — Results Summary")
    lines.append("=" * 62)

    # Descriptive highlights: one line per chart
    if "risk_by_rat_pressure_quartile" in dsc:
        tbl = dsc["risk_by_rat_pressure_quartile"]
        rates = ", ".join(f"{r:.3f}" for r in tbl["risk_rate"].tolist())
        lines.append(f"\nRisk by rat-pressure quartile (low→high): {rates}")
    if "risk_by_time_since_rat_quartile" in dsc:
        tbl2 = dsc["risk_by_time_since_rat_quartile"]
        rates2 = ", ".join(f"{r:.3f}" for r in tbl2["risk_rate"].tolist())
        lines.append(f"Risk by time-since-rat-arrival quartile (early→late): {rates2}")

    # Chi-square results
    lines.append("\nChi-square (High vs Low rat_pressure)")
    if chi:
        lines.append(f"  chi2={chi['chi2']:.3f}, p={chi['p_value']:.4f}, dof={chi['dof']}")
    else:
        lines.append("  [Not computed]")

    # Logistic regression results
    lines.append("\nLogistic regression (risk ~ rat_pressure + time_since_rat_min + "
                    "food_availability + C(season) [+ hours_after_sunset if available])")
    lines.append(logit_summary if logit_summary else "[No model]")

    path = outdir / "results_summary.txt"
    path.write_text("\n".join(lines))
    return path

# ---------- Main ----------
def main():
    outdir = Path(OUTDIR); outdir.mkdir(parents=True, exist_ok=True)

    df1, df2 = load_data(Path(DATASET1_PATH), Path(DATASET2_PATH))
    merged = align_asof(df1, df2, tolerance_min=TOLERANCE_MIN)

    # Save merged dataset
    merged.to_csv(outdir / "bat_rat_merged_asof.csv", index=False)

    # Descriptives + figures
    dsc = descriptives(merged, outdir)

    # Chi-square test
    chi = chi_square(merged, outdir)

    # Logistic regression (if feasible)
    logit_summary = logistic_model(merged, outdir)

    # Text summary
    report_path = write_text_report(outdir, dsc, chi, logit_summary)

    # Console summary
    print(f"Rows: merged={len(merged)} | d1={len(df1)} | d2={len(df2)}")
    if chi:
        print(f"Chi-square: chi2={chi['chi2']:.3f}, p={chi['p_value']:.4f}, dof={chi['dof']}")
    print(f"Report: {report_path}")
    print(f"Saved outputs to: {outdir.resolve()}")

if __name__ == "__main__":
    main()
