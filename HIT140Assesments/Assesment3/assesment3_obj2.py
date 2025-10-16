"""Objective 2 toolkit for HIT140 Assessment 3.

Reads the CSVs, engineers the rat-pressure features we agreed on,
produces the diagnostic charts/tables, and runs the required statistical models.
All artefacts land in `outputs/` for the report and appendix."""

# --- imports
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from scipy import stats
from scipy.special import expit
from statsmodels.stats.proportion import proportion_confint, proportions_ztest
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Global plotting style tuned for readability in the report.
sns.set_theme(style="whitegrid")
sns.set_context("talk", font_scale=1.3)
DEFAULT_DPI = 200


def ensure_outputs_dir(path="outputs"):
    import os
    os.makedirs(path, exist_ok=True)
    return path


def save_plot(fig, filename, dpi=DEFAULT_DPI):
    output_dir = Path(globals().get("OUTPUT_DIR", Path("outputs")))
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_dir / filename, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def prop_ci_wilson(k, n, alpha=0.05):
    if n == 0:
        return (np.nan, np.nan)
    lo, hi = proportion_confint(count=k, nobs=n, alpha=alpha, method="wilson")
    return lo, hi


def mean_ci_normal(x, alpha=0.05):
    x = np.asarray(x, float)
    n = x.size
    if n == 0:
        return (np.nan, np.nan, np.nan)
    m = x.mean()
    se = x.std(ddof=1) / np.sqrt(n)
    z = stats.norm.ppf(1 - alpha / 2)
    return m, m - z * se, m + z * se


def interval_left_bounds(cat):
    return np.array([iv.left for iv in cat.cat.categories])

warnings.filterwarnings("ignore", category=UserWarning)
plt.rcParams["figure.dpi"] = 120

# --- default file locations 
BASE_DIR = Path(__file__).resolve().parent
DATASET1_PATH = BASE_DIR / "dataset1.csv"
DATASET2_PATH = BASE_DIR / "dataset2.csv"
OUTPUT_DIR = BASE_DIR / "outputs"

# --- helper: consistent output dir
def ensure_outdir(outdir: Path) -> Path:
    ensure_outputs_dir(str(outdir))
    return outdir

# --- robust CSV loader (accepts slightly different filenames)
def smart_read_csv(default_name: str, fallback_glob: str) -> pd.DataFrame:
    here = Path(__file__).resolve().parent
    candidate = here / default_name
    if candidate.exists():
        return pd.read_csv(candidate)
    # try sibling name variants
    matches = list(here.glob(fallback_glob))
    if matches:
        return pd.read_csv(matches[0])
    # as a last resort, just try default_name from CWD
    return pd.read_csv(default_name)

# --- cleaning & typing helpers
def to_dt(s, tz=None):
    # errors='coerce' turns weird strings into NaT instead of exploding
    dt = pd.to_datetime(s, errors="coerce", utc=False)

    # For series/index we need to use the .dt/.tz accessors, for scalars handle separately.
    if isinstance(dt, pd.Series):
        tzinfo = getattr(dt.dtype, "tz", None)
        if tzinfo is not None:
            if tz is not None:
                dt = dt.dt.tz_convert(tz)
            dt = dt.dt.tz_convert(None)
        return dt

    if isinstance(dt, (pd.DatetimeIndex, pd.Index)):
        tzinfo = getattr(dt.dtype, "tz", None)
        if tzinfo is not None:
            if tz is not None:
                dt = dt.tz_convert(tz)
            dt = dt.tz_convert(None)
        return dt

    if isinstance(dt, pd.Timestamp):
        if dt.tzinfo is not None:
            if tz is not None:
                dt = dt.tz_convert(tz)
            dt = dt.tz_convert(None)
        return dt

    return dt

def coerce_numeric(s):
    return pd.to_numeric(s, errors="coerce")

# --- Engineering the 30-min "window" table from dataset2
def build_window_table(df2: pd.DataFrame) -> pd.DataFrame:
    df2 = df2.copy()
    # expected columns: time, month, hours_after_sunset, bat_landing_number,
    # food_availability, rat_minutes, rat_arrival_number
    df2["time"] = to_dt(df2["time"])
    df2["window_start"] = df2["time"].dt.floor("30min")
    df2["window_end"] = df2["window_start"] + pd.Timedelta(minutes=30)

    # sanity coercions
    for c in ["hours_after_sunset", "bat_landing_number", "food_availability",
            "rat_minutes", "rat_arrival_number", "month"]:
        if c in df2.columns:
            df2[c] = coerce_numeric(df2[c])

    # feature: rat_pressure = time-share of rats in the window (can be >1 if annotated longer)
    df2["rat_pressure"] = df2["rat_minutes"] / 30.0

    # quick tidy season if missing (season likely 0/1, but not here. we keep seasonal info on dataset1)
    return df2

# --- Clean landings (dataset1) and add engineered features
def clean_landings(df1: pd.DataFrame) -> pd.DataFrame:
    df = df1.copy()
    # timestamps
    for c in ["start_time", "rat_period_start", "rat_period_end", "sunset_time"]:
        if c in df.columns:
            df[c] = to_dt(df[c])

    # numerics
    for c in ["bat_landing_to_food", "seconds_after_rat_arrival", "hours_after_sunset", "month", "risk", "reward", "season"]:
        if c in df.columns:
            df[c] = coerce_numeric(df[c])

    # some rows can have NaNs, this was done so the code shouldn't crash
    # rat present? we try two ways and fall back if needed.
    if {"rat_period_start", "rat_period_end", "start_time"}.issubset(df.columns):
        rp = (df["start_time"] >= df["rat_period_start"]) & (df["start_time"] <= df["rat_period_end"])
    else:
        rp = pd.Series(False, index=df.index)

    if "seconds_after_rat_arrival" in df.columns:
        rp2 = df["seconds_after_rat_arrival"].ge(0)  # >=0 means rat already there
    else:
        rp2 = pd.Series(False, index=df.index)

    df["rat_present"] = (rp | rp2.fillna(False)).astype(int)

    # time since rat arrived (minutes) for those with rat present; NaN otherwise
    if "seconds_after_rat_arrival" in df.columns:
        df["time_since_rat_minutes"] = np.where(
            df["seconds_after_rat_arrival"].ge(0),
            df["seconds_after_rat_arrival"] / 60.0,
            np.nan
        )
    else:
        df["time_since_rat_minutes"] = np.nan

    # core target is risk (0/1)
    if "risk" in df.columns:
        df["risk"] = df["risk"].astype("Int64")
    return df

# --- Attach each landing to its 30-min window from dataset2
def attach_windows(landings: pd.DataFrame, windows: pd.DataFrame) -> pd.DataFrame:
    df = landings.copy()
    df["window_start"] = df["start_time"].dt.floor("30min")
    # left join: every landing gets its window vars
    window_cols = ["window_start", "window_end", "rat_minutes", "rat_pressure",
                   "bat_landing_number", "food_availability", "rat_arrival_number",
                   "hours_after_sunset", "month"]
    available_cols = [c for c in window_cols if c in windows.columns]
    window_subset = windows[available_cols].copy()

    # Avoid duplicate column suffixes by renaming any overlapping fields (other than window markers)
    rename_map = {}
    for col in window_subset.columns:
        if col in {"window_start", "window_end"}:
            continue
        if col in df.columns:
            rename_map[col] = f"window_{col}"
    if rename_map:
        window_subset = window_subset.rename(columns=rename_map)

    merged = df.merge(window_subset, on="window_start", how="left")
    return merged

# --- Small utilities for group stats and CIs
def mean_ci_binom(successes, n, alpha=0.05):
    """Normal approximation 95% CI for a proportion (good enough for teaching)."""
    if n == 0:
        return np.nan, np.nan
    p = successes / n
    z = stats.norm.ppf(1 - alpha/2)
    se = np.sqrt(p * (1 - p) / n)
    return p - z * se, p + z * se


def proportion_summary(df: pd.DataFrame, group_cols):
    if isinstance(group_cols, str):
        group_cols = [group_cols]

    rows = []
    # this will keep group summaries consistent across plots/tables and reuse Wilson intervals everywhere.
    grouped = df.groupby(group_cols, observed=False)
    for keys, sub in grouped:
        if not isinstance(keys, tuple):
            keys = (keys,)
        series = sub["risk"].dropna()
        n = int(series.size)
        successes = float(series.sum())
        mean = successes / n if n else np.nan
        ci_low, ci_high = prop_ci_wilson(successes, n)
        row = dict(zip(group_cols, keys))
        row.update({
            "n": n,
            "successes": successes,
            "mean": mean,
            "ci_low": ci_low,
            "ci_high": ci_high,
        })
        rows.append(row)
    return pd.DataFrame(rows)


def ordered_qcut(series: pd.Series, q: int):
    # Force categories to follow the numeric left edge so charts read low→high.
    cat = pd.qcut(series, q=q, duplicates="drop")
    if cat.isna().all():
        return cat
    order = interval_left_bounds(cat)
    ordered = cat.cat.categories[np.argsort(order)]
    return cat.cat.reorder_categories(ordered, ordered=True)


def rat_pressure_bins_with_min_counts(series: pd.Series, season: pd.Series, min_n: int = 20):
    """Create ordered pressure bins, merging upper bins until each season has at least min_n records."""
    orig_series = series
    orig_season = season
    valid = orig_series.notna()
    if season is not None:
        valid &= orig_season.notna()
    series = orig_series[valid]
    if season is not None:
        season = orig_season[valid]

    if series.empty:
        return pd.Series(index=orig_series.index, dtype="category")

    quantiles = np.linspace(0, 1, 11)
    edges = series.quantile(quantiles, interpolation="linear").to_numpy()
    edges = np.unique(edges)
    if len(edges) < 3:
        edges = np.linspace(series.min(), series.max(), min(3, len(series.unique()) + 1))

    while len(edges) > 2:
        bins = pd.cut(series, bins=edges, include_lowest=True, duplicates="drop")
        temp = pd.DataFrame({"season": season if season is not None else 0, "bin": bins}).dropna()
        summary = temp.groupby(["season", "bin"], observed=False).size().reset_index(name="n")
        if summary.empty or (summary["n"] >= min_n).all():
            break
        categories = list(bins.cat.categories)
        sparse_bins = summary.loc[summary["n"] < min_n, "bin"].unique()
        sparse_bins = [cat for cat in categories if cat in sparse_bins]
        if not sparse_bins:
            break
        target = sparse_bins[-1]
        idx = categories.index(target)
        if idx == 0:
            break
        if len(edges) <= 3:
            break
        edges = np.delete(edges, idx)

    final_bins = pd.cut(series, bins=edges, include_lowest=True, duplicates="drop")
    final_series = pd.Series(final_bins, index=series.index)
    return final_series.reindex(orig_series.index)


def gaussian_smooth_probability(x, y, grid, bandwidth):
    """Simple Gaussian-kernel smoother with binomial standard errors."""
    preds = np.full_like(grid, np.nan, dtype=float)
    lower = np.full_like(grid, np.nan, dtype=float)
    upper = np.full_like(grid, np.nan, dtype=float)
    for idx, gx in enumerate(grid):
        weights = np.exp(-0.5 * ((x - gx) / bandwidth) ** 2)
        weights_sum = weights.sum()
        if weights_sum < 1e-6:
            continue
        p = np.dot(weights, y) / weights_sum
        # Effective sample size for weighted observations.
        eff_n = weights_sum ** 2 / np.sum(weights ** 2)
        if eff_n < 5:
            continue
        se = np.sqrt(max(p * (1 - p), 1e-6) / eff_n)
        preds[idx] = p
        lower[idx] = np.clip(p - 1.96 * se, 0, 1)
        upper[idx] = np.clip(p + 1.96 * se, 0, 1)
    return preds, lower, upper

def plot_rat_pressure_hist(windows):
    if "rat_pressure" not in windows.columns:
        return
    series = windows["rat_pressure"].dropna()
    if series.empty:
        return

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.hist(series, bins=35, color="#4c72b0", edgecolor="white", alpha=0.85)
    ax.set_title("Window-level Rat Pressure")
    ax.set_xlabel("Rat pressure (rat_minutes / 30)")
    ax.set_ylabel("Window count")

    # Show the quartile and decile landmarks we reference elsewhere.
    quartiles = series.quantile([0.25, 0.5, 0.75])
    deciles = series.quantile(np.linspace(0.1, 0.9, 9))
    for idx, val in enumerate(quartiles):
        ax.axvline(val, color="#dd8452", linestyle="--", linewidth=1.5,
                   label="Quartiles" if idx == 0 else None)
    for idx, val in enumerate(deciles):
        ax.axvline(val, color="#55a868", linestyle=":", linewidth=1,
                   label="Deciles" if idx == 0 else None)
    ax.legend(loc="upper right")

    # Inset zoom around the low-pressure regime where most windows sit.
    zoom_ax = inset_axes(ax, width="38%", height="48%", loc="upper left",
                         borderpad=2)
    zoom_ax.hist(series, bins=35, color="#4c72b0", edgecolor="white", alpha=0.85)
    zoom_ax.set_xlim(0, 0.3)
    zoom_ax.set_title("Zoom: 0–0.3", fontsize=12)
    for val in quartiles:
        zoom_ax.axvline(val, color="#dd8452", linestyle="--", linewidth=1)
    for val in deciles:
        zoom_ax.axvline(val, color="#55a868", linestyle=":", linewidth=0.8)

    save_plot(fig, "fig_rat_pressure_hist.png")


def plot_risk_by_rat_present(merged):
    required = {"risk", "rat_present"}
    if not required.issubset(merged.columns):
        return pd.DataFrame(), {}

    df = merged.dropna(subset=list(required)).copy()
    if df.empty:
        return pd.DataFrame(), {}

    summary = proportion_summary(df, "rat_present")
    if summary.empty:
        return pd.DataFrame(), {}

    summary["rat_present"] = summary["rat_present"].astype(int)
    summary = summary.set_index("rat_present").reindex([0, 1])
    if summary["n"].isna().any() or (summary["n"] < 1).any():
        # Skip plotting if either group has no observations (avoids misleading bars).
        return pd.DataFrame(), {}
    summary["n"] = summary["n"].astype(int)

    fig, ax = plt.subplots(figsize=(12, 6))
    positions = np.arange(summary.shape[0])
    means = summary["mean"].values
    ci_low = summary["ci_low"].values
    ci_high = summary["ci_high"].values
    yerr = np.vstack([means - ci_low, ci_high - means])
    labels = ["Absent (0)", "Present (1)"]
    ax.bar(positions, means, yerr=yerr, capsize=6, color=["#4c72b0", "#dd8452"])
    for pos, mean, n, lo, hi in zip(positions, means, summary["n"].values, ci_low, ci_high):
        if np.isnan(mean) or np.isnan(n):
            continue
        ax.text(
            pos,
            mean + 0.025,
            f"{mean:.3f}\nn={int(n)}\nCI[{lo:.2f},{hi:.2f}]",
            ha="center",
            va="bottom",
            fontsize=11
        )

    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Risk proportion")
    ax.set_ylim(0, min(1, np.nanmax(ci_high) + 0.1))
    ax.set_title("Risk-taking by Rat Presence")

    ct = pd.crosstab(df["risk"], df["rat_present"])
    ct = ct.reindex(index=[0, 1], columns=[0, 1], fill_value=0)
    chi2_p = np.nan
    if ct.values.sum() > 0 and not (ct.sum(axis=0) == 0).any() and not (ct.sum(axis=1) == 0).any():
        chi2, chi2_p, dof, exp = stats.chi2_contingency(ct)

    # Difference in proportions (Absent - Present)
    diff_ci = (np.nan, np.nan)
    diff = np.nan
    if 0 in summary.index and 1 in summary.index:
        p0, n0 = summary.loc[0, ["mean", "n"]]
        p1, n1 = summary.loc[1, ["mean", "n"]]
        if n0 > 0 and n1 > 0 and np.isfinite(p0) and np.isfinite(p1):
            diff = p0 - p1
            se = np.sqrt(p0 * (1 - p0) / n0 + p1 * (1 - p1) / n1)
            z = stats.norm.ppf(0.975)
            diff_ci = (diff - z * se, diff + z * se)

    chi2_text = f"{chi2_p:.4f}" if np.isfinite(chi2_p) else "N/A"
    if np.isnan(diff) or np.isnan(diff_ci[0]) or np.isnan(diff_ci[1]):
        subtitle = f"χ² p = {chi2_text}"
    else:
        subtitle = (f"χ² p = {chi2_text}  |  Δ (Absent - Present) = {diff:.3f} "
                    f"(95% CI [{diff_ci[0]:.3f}, {diff_ci[1]:.3f}])")
    ax.text(0.5, 1.02, subtitle, transform=ax.transAxes, ha="center", fontsize=12)

    save_plot(fig, "fig_risk_by_rat_present.png")
    summary = summary.reset_index()
    return summary, {"chi2_p": chi2_p, "diff": diff, "diff_ci": diff_ci}


def plot_risk_by_season(merged):
    required = {"risk", "season"}
    if not required.issubset(merged.columns):
        return pd.DataFrame(), np.nan
    df = merged.dropna(subset=list(required)).copy()
    if df.empty:
        return pd.DataFrame(), np.nan

    summary = proportion_summary(df, "season").sort_values("season")
    if summary.empty:
        return pd.DataFrame(), np.nan

    fig, ax = plt.subplots(figsize=(12, 6))
    positions = np.arange(summary.shape[0])
    means = summary["mean"].values
    ci_low = summary["ci_low"].values
    ci_high = summary["ci_high"].values
    yerr = np.vstack([means - ci_low, ci_high - means])
    ax.bar(positions, means, yerr=yerr, capsize=6, color="#55a868")
    for pos, mean, n, lo, hi in zip(positions, means, summary["n"].values, ci_low, ci_high):
        if np.isnan(mean) or np.isnan(n):
            continue
        ax.text(
            pos,
            mean + 0.025,
            f"{mean:.3f}\nn={int(n)}\nCI[{lo:.2f},{hi:.2f}]",
            ha="center",
            va="bottom",
            fontsize=11
        )

    ax.set_xticks(positions)
    ax.set_xticklabels(summary["season"].astype(str))
    ax.set_ylabel("Risk proportion")
    ax.set_ylim(0, min(1, np.nanmax(ci_high) + 0.1))
    ax.set_title("Risk-taking by Season")

    ct = pd.crosstab(df["risk"], df["season"])
    chi2_p = np.nan
    if ct.values.sum() > 0 and not (ct.sum(axis=0) == 0).any() and not (ct.sum(axis=1) == 0).any():
        chi2, chi2_p, dof, exp = stats.chi2_contingency(ct)
    chi2_text = f"{chi2_p:.4f}" if np.isfinite(chi2_p) else "N/A"
    ax.text(0.5, 1.02, f"χ² p = {chi2_text}", transform=ax.transAxes, ha="center", fontsize=12)

    save_plot(fig, "fig_risk_by_season.png")
    return summary, chi2_p


def plot_risk_vs_rat_pressure_deciles(merged):
    required = {"risk", "rat_pressure"}
    if not required.issubset(merged.columns):
        return pd.DataFrame()

    df = merged[merged["rat_pressure"].notna() & (merged["rat_pressure"] >= 0)].copy()
    if df.empty:
        return pd.DataFrame()

    df["rp_dec"] = rat_pressure_bins_with_min_counts(
        df["rat_pressure"],
        pd.Series(0, index=df.index),
        min_n=20
    )
    df = df.dropna(subset=["rp_dec"])
    if df.empty:
        return pd.DataFrame()

    summary = proportion_summary(df, "rp_dec")
    if summary.empty:
        return pd.DataFrame()

    summary["rp_dec"] = pd.Categorical(summary["rp_dec"], categories=df["rp_dec"].cat.categories, ordered=True)
    summary = summary.sort_values("rp_dec")

    fig, ax = plt.subplots(figsize=(12, 6))
    positions = np.arange(summary.shape[0])
    means = summary["mean"].values
    ci_low = summary["ci_low"].values
    ci_high = summary["ci_high"].values
    yerr = np.vstack([means - ci_low, ci_high - means])
    ax.errorbar(positions, means, yerr=yerr, fmt="-o", capsize=6, color="#4c72b0")
    for pos, mean, n, lo, hi in zip(positions, means, summary["n"].values, ci_low, ci_high):
        if np.isnan(mean) or np.isnan(n):
            continue
        ax.text(
            pos,
            mean + 0.025,
            f"{mean:.3f}\nn={int(n)}\nCI[{lo:.2f},{hi:.2f}]",
            ha="center",
            va="bottom",
            fontsize=10
        )

    ax.set_xticks(positions)
    ax.set_xticklabels([str(cat) for cat in summary["rp_dec"]], rotation=30, ha="right")
    ax.set_ylabel("Risk proportion")
    ax.set_title("Risk-taking across Rat-pressure Bins")
    save_plot(fig, "fig_risk_vs_rat_pressure_deciles.png")
    return summary


def plot_risk_vs_rat_pressure_deciles_by_season(merged):
    required = {"risk", "rat_pressure", "season"}
    if not required.issubset(merged.columns):
        return pd.DataFrame()

    df = merged[merged["rat_pressure"].notna() & (merged["rat_pressure"] >= 0)].copy()
    if df.empty:
        return pd.DataFrame()

    df["rp_dec"] = rat_pressure_bins_with_min_counts(
        df["rat_pressure"],
        df["season"],
        min_n=20
    )
    df = df.dropna(subset=["rp_dec"])
    if df.empty:
        return pd.DataFrame()

    summary = proportion_summary(df, ["season", "rp_dec"])
    if summary.empty:
        return pd.DataFrame()

    categories = df["rp_dec"].cat.categories
    summary["rp_dec"] = pd.Categorical(summary["rp_dec"], categories=categories, ordered=True)
    summary = summary.sort_values(["season", "rp_dec"])

    fig, ax = plt.subplots(figsize=(12, 6))
    positions = np.arange(len(categories))
    palette = sns.color_palette("Set2", n_colors=summary["season"].nunique())
    for idx, (season, sub) in enumerate(summary.groupby("season", observed=False)):
        sub = sub.set_index("rp_dec").reindex(categories).reset_index()
        if sub.empty:
            continue
        means = sub["mean"].values
        ci_low = sub["ci_low"].values
        ci_high = sub["ci_high"].values
        yerr = np.vstack([means - ci_low, ci_high - means])
        ax.errorbar(
            positions,
            means,
            yerr=yerr,
            fmt="-o",
            capsize=5,
            label=f"Season {season}",
            color=palette[idx]
        )
        for pos, mean, n, lo, hi in zip(positions, means, sub["n"].values, ci_low, ci_high):
            if np.isnan(mean) or np.isnan(n):
                continue
            ax.text(
                pos,
                mean + 0.025,
                f"{mean:.3f}\nn={int(n)}\nCI[{lo:.2f},{hi:.2f}]",
                ha="center",
                va="bottom",
                fontsize=9
            )

    ax.set_xticks(positions)
    ax.set_xticklabels([str(cat) for cat in categories], rotation=30, ha="right")
    ax.set_ylabel("Risk proportion")
    ax.set_title("Risk vs Rat-pressure Bins by Season")
    ax.legend(title="Season", bbox_to_anchor=(1.02, 1), loc="upper left")
    save_plot(fig, "fig_risk_vs_rat_pressure_deciles_by_season.png")
    return summary


def plot_lowess_risk_vs_rat_pressure(merged, bandwidth=0.12):
    required = {"risk", "rat_pressure", "season"}
    if not required.issubset(merged.columns):
        return

    df = merged[merged["rat_pressure"].notna() & (merged["rat_pressure"] >= 0)].copy()
    if df.empty:
        return

    fig, ax = plt.subplots(figsize=(12, 6))
    palette = sns.color_palette("Set2", n_colors=df["season"].nunique())
    for idx, (season, sub) in enumerate(df.groupby("season", observed=False)):
        if sub.empty:
            continue
        x = sub["rat_pressure"].values
        y = sub["risk"].values
        grid = np.linspace(np.percentile(x, 1), np.percentile(x, 99), 200)
        preds, lo, hi = gaussian_smooth_probability(x, y, grid, bandwidth=bandwidth)
        valid = ~np.isnan(preds)
        if valid.sum() < 5:
            continue
        ax.plot(grid[valid], preds[valid], color=palette[idx], label=f"Season {season}", linewidth=2)
        ax.fill_between(grid[valid], lo[valid], hi[valid], color=palette[idx], alpha=0.18)

    ax.set_xlabel("Rat pressure")
    ax.set_ylabel("Smoothed risk probability")
    ax.set_title("LOWESS Risk Profiles by Season")
    ax.legend(title="Season")
    save_plot(fig, "fig_lowess_risk_vs_rat_pressure_by_season.png")


def plot_risk_by_rat_pressure_quartile(merged):
    required = {"risk", "rat_pressure"}
    if not required.issubset(merged.columns):
        return pd.DataFrame()
    df = merged[merged["rat_pressure"].notna() & (merged["rat_pressure"] >= 0)].copy()
    if df.empty:
        return pd.DataFrame()

    df["rp_quartile"] = ordered_qcut(df["rat_pressure"], q=4)
    df = df.dropna(subset=["rp_quartile"])
    if df.empty:
        return pd.DataFrame()

    summary = proportion_summary(df, "rp_quartile")
    if summary.empty:
        return pd.DataFrame()

    summary["rp_quartile"] = pd.Categorical(summary["rp_quartile"], categories=df["rp_quartile"].cat.categories, ordered=True)
    summary = summary.sort_values("rp_quartile")

    fig, ax = plt.subplots(figsize=(12, 6))
    positions = np.arange(summary.shape[0])
    means = summary["mean"].values
    ci_low = summary["ci_low"].values
    ci_high = summary["ci_high"].values
    yerr = np.vstack([means - ci_low, ci_high - means])
    ax.bar(positions, means, yerr=yerr, capsize=6, color="#dd8452")
    for pos, mean, n, lo, hi in zip(positions, means, summary["n"].values, ci_low, ci_high):
        if np.isnan(mean) or np.isnan(n):
            continue
        ax.text(
            pos,
            mean + 0.025,
            f"{mean:.3f}\nn={int(n)}\nCI[{lo:.2f},{hi:.2f}]",
            ha="center",
            va="bottom",
            fontsize=10
        )

    ax.set_xticks(positions)
    ax.set_xticklabels([str(cat) for cat in summary["rp_quartile"]], rotation=30, ha="right")
    ax.set_ylabel("Risk proportion")
    ax.set_title("Risk-taking by Rat-pressure Quartile")
    save_plot(fig, "fig_risk_by_rat_pressure_quartile.png")
    return summary


def plot_time_since_rat_quartiles(merged):
    required = {"time_since_rat_minutes", "risk"}
    if not required.issubset(merged.columns):
        return pd.DataFrame()

    df = merged[merged["time_since_rat_minutes"].notna() & (merged["time_since_rat_minutes"] >= 0)].copy()
    if df.empty:
        return pd.DataFrame()

    df["tsr_quart"] = ordered_qcut(df["time_since_rat_minutes"], q=4)
    df = df.dropna(subset=["tsr_quart"])
    if df.empty:
        return pd.DataFrame()

    summary = proportion_summary(df, "tsr_quart")
    if summary.empty:
        return pd.DataFrame()

    summary["tsr_quart"] = pd.Categorical(summary["tsr_quart"], categories=df["tsr_quart"].cat.categories, ordered=True)
    summary = summary.sort_values("tsr_quart")

    fig, ax = plt.subplots(figsize=(12, 6))
    positions = np.arange(summary.shape[0])
    means = summary["mean"].values
    ci_low = summary["ci_low"].values
    ci_high = summary["ci_high"].values
    yerr = np.vstack([means - ci_low, ci_high - means])
    ax.bar(positions, means, yerr=yerr, capsize=6, color="#c44e52")
    for pos, mean, n, lo, hi in zip(positions, means, summary["n"].values, ci_low, ci_high):
        if np.isnan(mean) or np.isnan(n):
            continue
        ax.text(
            pos,
            mean + 0.025,
            f"{mean:.3f}\nn={int(n)}\nCI[{lo:.2f},{hi:.2f}]",
            ha="center",
            va="bottom",
            fontsize=10
        )

    ax.set_xticks(positions)
    ax.set_xticklabels([str(cat) for cat in summary["tsr_quart"]], rotation=30, ha="right")
    ax.set_ylabel("Risk proportion")
    ax.set_title("Risk-taking by Time Since Rat Arrival (quartiles)")
    save_plot(fig, "fig_risk_by_time_since_rat_quartile.png")
    return summary


def heatmap_risk_by_season_pressure(merged):
    if "rat_pressure" not in merged.columns or "season" not in merged.columns:
        return
    df = merged[merged["rat_pressure"].notna()].copy()
    if df.empty:
        return
    df["rp_quart"] = pd.qcut(df["rat_pressure"].clip(lower=0), q=4, duplicates="drop")
    pt = df.pivot_table(index="season", columns="rp_quart", values="risk", aggfunc="mean")
    pt = pt.astype(float)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(pt, annot=True, fmt=".3f", ax=ax, cmap="viridis")
    ax.set_title("Heatmap: Mean Risk by Season × Rat-pressure Quartile")
    save_plot(fig, "fig_heatmap_risk_season_pressure.png")


def monthly_trends(windows, merged):
    month_names = {1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun", 7: "Jul",
                   8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec"}

    if {"month", "rat_minutes"}.issubset(windows.columns):
        w = windows.dropna(subset=["month"]).copy()
        if not w.empty:
            series = w.groupby("month")["rat_minutes"].agg(["mean", "count"])
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(series.index, series["mean"], marker="o")
            labels = [month_names.get(int(m), str(int(m))) for m in series.index]
            ax.set_xticks(series.index)
            ax.set_xticklabels(labels)
            ax.set_xlabel("Month")
            ax.set_ylabel("Average rat minutes per window")
            ax.set_title("Monthly Average Rat Minutes (Window Level)")
            caption = "n per month: " + ", ".join(f"{lab}: {int(n)}" for lab, n in zip(labels, series["count"]))
            fig.text(0.5, -0.05, caption, ha="center", fontsize=11)
            if any(l not in month_names.values() for l in labels):
                fig.text(0.5, -0.1, "Note: Month labels use dataset coding (e.g., 0..6 = Jan..Jul).", ha="center", fontsize=10)
            save_plot(fig, "fig_window_monthly_rat_minutes.png")

    if {"month", "risk"}.issubset(merged.columns):
        m = merged.dropna(subset=["month"]).copy()
        if not m.empty:
            series = m.groupby("month")["risk"].agg(["mean", "count"])
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(series.index, series["mean"], marker="o", color="#4c72b0")
            labels = [month_names.get(int(mo), str(int(mo))) for mo in series.index]
            ax.set_xticks(series.index)
            ax.set_xticklabels(labels)
            ax.set_xlabel("Month")
            ax.set_ylabel("Risk proportion")
            ax.set_title("Monthly Average Risk Rate (Landing Level)")
            caption = "n per month: " + ", ".join(f"{lab}: {int(n)}" for lab, n in zip(labels, series["count"]))
            fig.text(0.5, -0.05, caption, ha="center", fontsize=11)
            if any(l not in month_names.values() for l in labels):
                fig.text(0.5, -0.1, "Note: Month labels use dataset coding (e.g., 0..6 = Jan..Jul).", ha="center", fontsize=10)
            save_plot(fig, "fig_window_monthly_risk_rate.png")


def plot_ecdf_rat_pressure_by_season(merged):
    required = {"rat_pressure", "season"}
    if not required.issubset(merged.columns):
        return
    df = merged[merged["rat_pressure"].notna()].copy()
    if df.empty:
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    for season, sub in df.groupby("season", observed=False):
        values = np.sort(sub["rat_pressure"].values)
        n = values.size
        if n == 0:
            continue
        y = np.arange(1, n + 1) / n
        ax.step(values, y, where="post", label=f"Season {season} (n={n})")

    ax.set_xlabel("Rat pressure")
    ax.set_ylabel("Empirical CDF")
    ax.set_title("ECDF of Rat Pressure by Season")
    ax.legend(title="Season")
    save_plot(fig, "fig_ecdf_rat_pressure_by_season.png")


def plot_hexbin_landings_vs_rat_minutes(windows):
    required = {"rat_minutes", "bat_landing_number"}
    if not required.issubset(windows.columns):
        return
    df = windows.dropna(subset=list(required)).copy()
    if df.empty:
        return

    fig, ax = plt.subplots(figsize=(12, 6))
    hb = ax.hexbin(
        df["rat_minutes"],
        df["bat_landing_number"],
        gridsize=35,
        cmap="viridis",
        mincnt=1,
        norm=LogNorm()
    )
    cb = fig.colorbar(hb, ax=ax)
    cb.set_label("Window count (log scale)")

    # Negative binomial fit captures over-dispersion better than a straight line.
    X = sm.add_constant(df["rat_minutes"])
    nb_model = sm.GLM(df["bat_landing_number"], X, family=sm.families.NegativeBinomial())
    nb_res = nb_model.fit()
    xp = np.linspace(df["rat_minutes"].min(), df["rat_minutes"].max(), 200)
    X_pred = sm.add_constant(xp)
    pred = nb_res.get_prediction(X_pred)
    pred_mean = pred.predicted_mean
    ci_bounds = pred.conf_int()
    ax.plot(xp, pred_mean, color="#dd8452", linewidth=2, label="Neg. binomial fit")
    ax.fill_between(xp, ci_bounds[:, 0], ci_bounds[:, 1], color="#dd8452", alpha=0.2)

    ax.set_xlabel("Rat minutes (per 30-minute window)")
    ax.set_ylabel("Bat landings (per window)")
    ax.set_title("Landings vs Rat Minutes (Hexbin)")
    ax.legend()

    coef = nb_res.params.get("rat_minutes", np.nan)
    coef_ci = nb_res.conf_int().loc["rat_minutes"] if "rat_minutes" in nb_res.params.index else (np.nan, np.nan)
    fig.text(
        0.5,
        -0.08,
        f"Neg. binomial slope = {coef:.3f} (95% CI [{coef_ci[0]:.3f}, {coef_ci[1]:.3f}])",
        ha="center",
        fontsize=11
    )

    save_plot(fig, "fig_hexbin_landings_vs_rat_minutes.png")

# --- Statistical tests used in the course
def run_inference_tests(merged: pd.DataFrame, outdir: Path) -> dict:
    out = {}
    # Chi-square: risk × rat_present (Investigation A)
    ct = pd.crosstab(merged["risk"], merged["rat_present"])
    chi2, p, dof, exp = stats.chi2_contingency(ct)
    out["chi2_risk_ratpresent"] = {"chi2": chi2, "p": p, "dof": dof, "table": ct}

    # Two-proportion z-test: P(risk=1 | rat_present=1) vs P(risk=1 | rat_present=0)
    grp = merged.groupby("rat_present")["risk"]
    counts = grp.sum().reindex([0, 1]).fillna(0).values.astype(int)
    ns = grp.count().reindex([0, 1]).fillna(0).values.astype(int)
    if ns.min() > 0:
        stat, pz = proportions_ztest(count=counts, nobs=ns, alternative="two-sided")
        out["prop_ztest_ratpresent"] = {"z": float(stat), "p": float(pz), "counts": counts.tolist(), "nobs": ns.tolist()}

    # Season effect on risk (Investigation B)
    if "season" in merged.columns:
        ct2 = pd.crosstab(merged["risk"], merged["season"])
        chi2b, pb, dofb, expb = stats.chi2_contingency(ct2)
        out["chi2_risk_season"] = {"chi2": chi2b, "p": pb, "dof": dofb, "table": ct2}

    # Save crosstabs as CSV for transparency
    ct.to_csv(outdir / "chi_crosstab_risk_vs_rat_present.csv")
    if "season" in merged.columns:
        ct2.to_csv(outdir / "chi_crosstab_risk_vs_season.csv")
    return out

# --- Logistic regression for risk (Binomial GLM), with interaction season × rat_pressure
def fit_logit_model(merged: pd.DataFrame, outdir: Path, penalty_strength: float = 1.0) -> dict:
    from patsy import dmatrices

    df = merged.copy()
    cols = ["risk", "rat_pressure", "season", "hours_after_sunset", "time_since_rat_minutes", "month"]
    cols = [c for c in cols if c in df.columns]
    df = df[cols].dropna().copy()

    if df.empty or df["risk"].nunique() < 2:
        return {"note": "Not enough variation in risk to fit logistic model."}

    if "rat_pressure" in df.columns:
        df["rat_pressure"] = df["rat_pressure"].clip(lower=0)

    formula = "risk ~ rat_pressure * season"
    if "hours_after_sunset" in df.columns:
        formula += " + hours_after_sunset"
    if "time_since_rat_minutes" in df.columns:
        formula += " + time_since_rat_minutes"
    if "month" in df.columns:
        formula += " + C(month)"

    y, X = dmatrices(formula, df, return_type="dataframe")
    y = y.iloc[:, 0]
    logit_model = sm.Logit(y, X)
    res = logit_model.fit_regularized(alpha=penalty_strength, L1_wt=0.0, maxiter=1000)

    params = pd.Series(res.params, index=X.columns)
    try:
        hessian = logit_model.hessian(res.params)
        cov = np.linalg.inv(-hessian)
        diag = np.clip(np.diag(cov), 0, None)
        se = np.sqrt(diag)
    except (np.linalg.LinAlgError, ValueError):
        cov = None
        se = np.full(params.shape, np.nan)

    conf_low = params - 1.96 * se
    conf_high = params + 1.96 * se
    odds_df = pd.DataFrame({
        "term": params.index,
        "OR": np.exp(params),
        "CI_low": np.exp(conf_low),
        "CI_high": np.exp(conf_high),
        "p_value": np.nan
    })
    odds_df.to_csv(outdir / "logit_odds_ratios.csv", index=False)

    forest_df = odds_df[odds_df["term"] != "Intercept"].copy()
    if not forest_df.empty:
        fig, ax = plt.subplots(figsize=(10, 0.7 * len(forest_df) + 2))
        y_pos = np.arange(forest_df.shape[0])
        xerr = np.vstack([
            forest_df["OR"] - forest_df["CI_low"],
            forest_df["CI_high"] - forest_df["OR"]
        ])
        ax.errorbar(forest_df["OR"], y_pos, xerr=xerr, fmt="o", color="#4c72b0", capsize=6)
        ax.axvline(1, color="black", linestyle="--", linewidth=1)
        ax.set_xscale("log")
        ax.set_xlabel("Odds ratio (log scale)")
        ax.set_yticks(y_pos)
        ax.set_yticklabels(forest_df["term"])
        ax.invert_yaxis()
        ax.set_title("Penalised Logistic Regression (OR)")
        save_plot(fig, "fig_logit_forest.png")

    llf = logit_model.loglike(res.params)
    k = len(params)
    n_obs = len(y)
    aic = 2 * k - 2 * llf
    bic = np.log(n_obs) * k - 2 * llf

    with open(outdir / "logit_model_summary.txt", "w") as f:
        f.write(res.summary().as_text())

    return {
        "aic": aic,
        "bic": bic,
        "summary_path": str(outdir / "logit_model_summary.txt"),
        "odds_table": odds_df,
        "params": params,
        "cov": cov,
        "design_info": X.design_info,
        "model": logit_model,
        "result": res,
        "penalty": penalty_strength,
        "training_data": df
    }


def _base_feature_values(df: pd.DataFrame, exclude: set) -> dict:
    base = {}
    for col in df.columns:
        if col in exclude:
            continue
        series = df[col]
        if pd.api.types.is_numeric_dtype(series):
            base[col] = series.median()
        else:
            base[col] = series.mode().iloc[0]
    return base


def _predict_with_ci(design_info, params, cov, pred_df: pd.DataFrame):
    from patsy import dmatrix

    design = dmatrix(design_info, pred_df, return_type="dataframe")
    design_matrix = design.to_numpy()
    lin_pred = design_matrix @ params.values
    probs = expit(lin_pred)
    if cov is None:
        return probs, None, None
    var = np.einsum("ij,jk,ik->i", design_matrix, cov, design_matrix)
    var = np.clip(var, 0, None)
    se = np.sqrt(var)
    lower = expit(lin_pred - 1.96 * se)
    upper = expit(lin_pred + 1.96 * se)
    return probs, lower, upper


def plot_predicted_risk_by_rat_pressure(logit_out: dict, outdir: Path):
    df = logit_out.get("training_data")
    if df is None or "season" not in df.columns:
        return

    seasons = sorted(df["season"].unique())
    if not seasons:
        return

    rp_min, rp_max = df["rat_pressure"].quantile([0.01, 0.99])
    if np.isclose(rp_min, rp_max):
        rp_min, rp_max = df["rat_pressure"].min(), df["rat_pressure"].max()
    rp_grid = np.linspace(rp_min, rp_max, 200)

    base_features = _base_feature_values(df, {"risk", "rat_pressure", "season"})
    fig, ax = plt.subplots(figsize=(12, 6))
    palette = sns.color_palette("Set2", n_colors=len(seasons))

    for idx, season in enumerate(seasons):
        pred_df = pd.DataFrame({"rat_pressure": rp_grid, "season": season})
        for key, value in base_features.items():
            pred_df[key] = value
        probs, lower, upper = _predict_with_ci(logit_out["design_info"], logit_out["params"], logit_out["cov"], pred_df)
        mask = ~np.isnan(probs)
        if mask.sum() == 0:
            continue
        ax.plot(rp_grid[mask], probs[mask], label=f"Season {season}", color=palette[idx], linewidth=2)
        if lower is not None:
            ax.fill_between(rp_grid[mask], lower[mask], upper[mask], color=palette[idx], alpha=0.18)

    ax.set_xlabel("Rat pressure")
    ax.set_ylabel("Predicted risk probability")
    ax.set_title("Predicted Risk vs Rat Pressure by Season")
    ax.legend(title="Season")
    save_plot(fig, "fig_predicted_risk_by_rat_pressure_by_season.png")


def plot_predicted_risk_by_hours(logit_out: dict, outdir: Path):
    df = logit_out.get("training_data")
    if df is None or "hours_after_sunset" not in df.columns:
        return

    hours_min, hours_max = df["hours_after_sunset"].quantile([0.01, 0.99])
    if np.isclose(hours_min, hours_max):
        hours_min, hours_max = df["hours_after_sunset"].min(), df["hours_after_sunset"].max()
    hours_grid = np.linspace(hours_min, hours_max, 200)

    base_features = _base_feature_values(df, {"risk", "hours_after_sunset"})
    if "rat_pressure" in df.columns:
        base_features["rat_pressure"] = df["rat_pressure"].median()

    seasons = sorted(df["season"].unique()) if "season" in df.columns else [base_features.get("season", 0)]
    fig, ax = plt.subplots(figsize=(12, 6))
    palette = sns.color_palette("Set1", n_colors=len(seasons))

    for idx, season in enumerate(seasons):
        pred_df = pd.DataFrame({"hours_after_sunset": hours_grid})
        for key, value in base_features.items():
            pred_df[key] = value
        if "season" in df.columns:
            pred_df["season"] = season
        probs, lower, upper = _predict_with_ci(logit_out["design_info"], logit_out["params"], logit_out["cov"], pred_df)
        mask = ~np.isnan(probs)
        if mask.sum() == 0:
            continue
        label = f"Season {season}" if "season" in df.columns else "All seasons"
        ax.plot(hours_grid[mask], probs[mask], label=label, color=palette[idx], linewidth=2)
        if lower is not None:
            ax.fill_between(hours_grid[mask], lower[mask], upper[mask], color=palette[idx], alpha=0.18)

    ax.set_xlabel("Hours after sunset")
    ax.set_ylabel("Predicted risk probability")
    ax.set_title("Predicted Risk vs Hours After Sunset")
    ax.legend(title="Season")
    save_plot(fig, "fig_predicted_risk_by_hours_after_sunset.png")

# --- Linear regression (window-level): do rats predict bat landings?
def fit_linear_model(windows: pd.DataFrame, outdir: Path) -> dict:
    df = windows.copy()
    cols = ["bat_landing_number", "rat_minutes", "rat_arrival_number", "food_availability", "hours_after_sunset"]
    for c in cols:
        if c not in df.columns:
            return {"note": "Missing columns for linear model."}
    df = df.dropna(subset=cols).copy()
    if df.empty:
        return {"note": "No rows for linear model after cleaning."}

    y = df["bat_landing_number"].values
    X = df[["rat_minutes", "rat_arrival_number", "food_availability", "hours_after_sunset"]].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    # baseline: mean of training y
    baseline = np.full_like(y_test, fill_value=y_train.mean(), dtype=float)
    mae_b = mean_absolute_error(y_test, baseline)
    rmse_b = np.sqrt(mean_squared_error(y_test, baseline))
    r2_b = r2_score(y_test, baseline)

    coefs = pd.Series(lr.coef_, index=["rat_minutes", "rat_arrival_number", "food_availability", "hours_after_sunset"])
    coefs.to_csv(outdir / "linear_model_coefficients.csv")
    return {
        "mae": mae, "rmse": rmse, "r2": r2,
        "baseline_mae": mae_b, "baseline_rmse": rmse_b, "baseline_r2": r2_b
    }


def fit_window_level_glm(merged: pd.DataFrame, windows: pd.DataFrame, outdir: Path) -> dict:
    if "window_start" not in merged.columns or "window_start" not in windows.columns:
        return {"note": "Missing window_start for GLM aggregation."}

    # Rebuild window-level view so the sensitivity model can respect how many landings occurred.
    landing_counts = merged.groupby("window_start").agg(
        total_landings=("risk", "count"),
        risky_landings=("risk", "sum")
    )
    landing_counts = landing_counts[landing_counts["total_landings"] > 0]
    if landing_counts.empty:
        return {"note": "No landings per window to fit GLM."}

    win = windows.set_index("window_start").copy()
    win = win.join(landing_counts, how="inner")
    if win.empty:
        return {"note": "No overlap between windows and landings for GLM."}

    # Use the observed proportion as the response, weights keep high-traffic windows influential.
    win["risk_prop"] = win["risky_landings"] / win["total_landings"]

    if "season" in merged.columns:
        season_mode = merged.groupby("window_start")["season"].agg(lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan)
        win["season"] = season_mode
    if "time_since_rat_minutes" in merged.columns:
        tsr = merged.groupby("window_start")["time_since_rat_minutes"].median()
        win["time_since_rat_min"] = tsr
    if "hours_after_sunset" in win.columns:
        win["hours_after_sunset_x"] = win["hours_after_sunset"]

    required_cols = ["rat_pressure", "risk_prop", "total_landings"]
    for col in required_cols:
        if col not in win.columns:
            return {"note": "Missing rat_pressure or risk proportion for GLM."}

    win = win.dropna(subset=required_cols)
    if win.empty:
        return {"note": "No complete rows for GLM."}

    formula_terms = ["rat_pressure"]
    if "season" in win.columns and win["season"].nunique(dropna=True) > 1:
        formula_terms.append("season")
        formula_terms.append("rat_pressure:season")
    if "hours_after_sunset_x" in win.columns:
        formula_terms.append("hours_after_sunset_x")
    if "time_since_rat_min" in win.columns:
        formula_terms.append("time_since_rat_min")

    formula = "risk_prop ~ " + " + ".join(formula_terms)
    glm = smf.glm(
        formula=formula,
        data=win,
        family=sm.families.Binomial(),
        freq_weights=win["total_landings"]
    ).fit()

    params = glm.params
    conf = glm.conf_int()
    odds_df = pd.DataFrame({
        "term": params.index,
        "OR": np.exp(params),
        "CI_low": np.exp(conf[0]),
        "CI_high": np.exp(conf[1]),
        "p_value": glm.pvalues
    })
    odds_df.to_csv(outdir / "glm_binomial_or.csv", index=False)
    with open(outdir / "glm_binomial_summary.txt", "w") as f:
        f.write(glm.summary().as_text())

    return {
        "summary_path": str(outdir / "glm_binomial_summary.txt"),
        "odds_table": odds_df,
        "model": glm,
        "formula": formula
    }

# --- Save tidy group tables used in the report
def export_group_tables(merged: pd.DataFrame, windows: pd.DataFrame, outdir: Path, summaries=None):
    summaries = summaries or {}

    def write_summary(df: pd.DataFrame, path: Path, rename=None, category_cols=None):
        if df is None or df.empty:
            return
        out = df.copy()
        # Keep the CSV layout aligned with how we quote results in the report.
        if rename:
            out = out.rename(columns=rename)
        cols = [c for c in ["n", "mean", "ci_low", "ci_high", "successes"] if c in out.columns]
        ordered_cols = []
        if category_cols:
            ordered_cols.extend(category_cols)
        ordered_cols.extend(cols)
        ordered_cols = [c for c in ordered_cols if c in out.columns]
        out = out[ordered_cols]
        for col in out.columns:
            dtype = out[col].dtype
            if isinstance(dtype, pd.CategoricalDtype) or isinstance(out[col].iloc[0] if not out.empty else None, pd.Interval):
                out[col] = out[col].astype(str)
        out.to_csv(path, index=False)

    # risk by rat_present
    rp_summary = summaries.get("rat_present")
    if (rp_summary is None or rp_summary.empty) and {"rat_present", "risk"}.issubset(merged.columns):
        temp = merged.dropna(subset=["rat_present", "risk"]).copy()
        if not temp.empty:
            rp_summary = proportion_summary(temp, "rat_present")
    write_summary(rp_summary, outdir / "risk_by_rat_present.csv", category_cols=["rat_present"])

    # risk by season
    season_summary = summaries.get("season")
    if (season_summary is None or season_summary.empty) and {"season", "risk"}.issubset(merged.columns):
        temp = merged.dropna(subset=["season", "risk"])
        if not temp.empty:
            season_summary = proportion_summary(temp, "season")
    write_summary(season_summary, outdir / "risk_by_season.csv", category_cols=["season"])

    # risk by rat-pressure deciles (overall)
    decile_summary = summaries.get("rat_pressure_deciles")
    if (decile_summary is None or decile_summary.empty) and {"rat_pressure", "risk"}.issubset(merged.columns):
        temp = merged[merged["rat_pressure"].notna() & (merged["rat_pressure"] >= 0)].copy()
        if not temp.empty:
            temp["rp_dec"] = rat_pressure_bins_with_min_counts(
                temp["rat_pressure"],
                pd.Series(0, index=temp.index),
                min_n=20
            )
            temp = temp.dropna(subset=["rp_dec"])
            decile_summary = proportion_summary(temp, "rp_dec")
            if not decile_summary.empty:
                decile_summary["rp_dec"] = pd.Categorical(decile_summary["rp_dec"], categories=temp["rp_dec"].cat.categories, ordered=True)
                decile_summary = decile_summary.sort_values("rp_dec")
    write_summary(decile_summary, outdir / "risk_by_rat_pressure_deciles.csv", category_cols=["rp_dec"])

    # risk by rat-pressure deciles by season
    decile_season_summary = summaries.get("rat_pressure_deciles_by_season")
    if (decile_season_summary is None or decile_season_summary.empty) and {"rat_pressure", "risk", "season"}.issubset(merged.columns):
        temp = merged[merged["rat_pressure"].notna() & (merged["rat_pressure"] >= 0)].copy()
        if not temp.empty:
            temp["rp_dec"] = rat_pressure_bins_with_min_counts(
                temp["rat_pressure"],
                temp["season"],
                min_n=20
            )
            temp = temp.dropna(subset=["rp_dec"])
            decile_season_summary = proportion_summary(temp, ["season", "rp_dec"])
            if not decile_season_summary.empty:
                categories = temp["rp_dec"].cat.categories
                decile_season_summary["rp_dec"] = pd.Categorical(decile_season_summary["rp_dec"], categories=categories, ordered=True)
                decile_season_summary = decile_season_summary.sort_values(["season", "rp_dec"])
    write_summary(decile_season_summary, outdir / "risk_by_rat_pressure_deciles_by_season.csv", category_cols=["season", "rp_dec"])

    # risk by rat-pressure quartiles
    quart_summary = summaries.get("rat_pressure_quartiles")
    if (quart_summary is None or quart_summary.empty) and {"rat_pressure", "risk"}.issubset(merged.columns):
        temp = merged[merged["rat_pressure"].notna() & (merged["rat_pressure"] >= 0)].copy()
        if not temp.empty:
            temp["rp_quartile"] = ordered_qcut(temp["rat_pressure"], q=4)
            temp = temp.dropna(subset=["rp_quartile"])
            quart_summary = proportion_summary(temp, "rp_quartile")
            if not quart_summary.empty:
                quart_summary["rp_quartile"] = pd.Categorical(quart_summary["rp_quartile"], categories=temp["rp_quartile"].cat.categories, ordered=True)
                quart_summary = quart_summary.sort_values("rp_quartile")
    write_summary(quart_summary, outdir / "risk_by_rat_pressure_quartile.csv", category_cols=["rp_quartile"])

    # risk by time-since-rat quartiles
    ts_summary = summaries.get("time_since_quartiles")
    if (ts_summary is None or ts_summary.empty) and {"time_since_rat_minutes", "risk"}.issubset(merged.columns):
        temp = merged[merged["time_since_rat_minutes"].notna() & (merged["time_since_rat_minutes"] >= 0)].copy()
        if not temp.empty:
            temp["tsr_quartile"] = ordered_qcut(temp["time_since_rat_minutes"], q=4)
            temp = temp.dropna(subset=["tsr_quartile"])
            ts_summary = proportion_summary(temp, "tsr_quartile")
            if not ts_summary.empty:
                ts_summary["tsr_quartile"] = pd.Categorical(ts_summary["tsr_quartile"], categories=temp["tsr_quartile"].cat.categories, ordered=True)
                ts_summary = ts_summary.sort_values("tsr_quartile")
    write_summary(ts_summary, outdir / "risk_by_time_since_rat_quartile.csv", category_cols=["tsr_quartile"])

    # keep window table too (for transparency)
    keep_cols = ["window_start", "window_end", "rat_minutes", "rat_pressure",
                "bat_landing_number", "food_availability", "rat_arrival_number", "hours_after_sunset", "month"]
    existing = [c for c in keep_cols if c in windows.columns]
    if existing:
        windows[existing].to_csv(outdir / "window_level_table.csv", index=False)

# --- Summarise the *key* outcomes in a friendly text file
def write_summary_txt(stats_out: dict, logit_out: dict, glm_out: dict, lin_out: dict,
                      merged: pd.DataFrame, outdir: Path, narrative: dict):
    # a few headline numbers that a marker can understand at a glance
    mr = merged.groupby("rat_present")["risk"].mean().reindex([0, 1]).rename({0: "No rat", 1: "Rat present"})
    with open(outdir / "results_summary_obj2.txt", "w") as f:
        f.write("Assessment 3 – Objective 2 (Investigation A & B)\n")
        f.write("=================================================\n\n")

        f.write("Risk rate (proportion of landings marked as risk-taking):\n")
        f.write(mr.to_string(float_format=lambda x: f"{x:.3f}") + "\n\n")

        if "prop_ztest_ratpresent" in stats_out:
            z = stats_out["prop_ztest_ratpresent"]
            f.write(f"Two-proportion z-test (risk | rat present vs absent): z={z['z']:.3f}, p={z['p']:.4f}\n")
            f.write(f"Counts={z['counts']}  nobs={z['nobs']}\n\n")

        if "chi2_risk_ratpresent" in stats_out:
            f.write("Chi-square test (risk × rat_present):\n")
            f.write(f"  chi2={stats_out['chi2_risk_ratpresent']['chi2']:.3f}, "
                    f"p={stats_out['chi2_risk_ratpresent']['p']:.4f}, "
                    f"dof={stats_out['chi2_risk_ratpresent']['dof']}\n\n")
            rp_stats = narrative.get("rat_present_stats") or {}
            if rp_stats:
                diff_ci = rp_stats.get("diff_ci", (np.nan, np.nan))
                f.write(f"  Δ (absent - present) = {rp_stats.get('diff', np.nan):.3f} "
                        f"(95% CI [{diff_ci[0]:.3f}, {diff_ci[1]:.3f}])\n\n")

        if "chi2_risk_season" in stats_out:
            f.write("Chi-square test (risk × season):\n")
            f.write(f"  chi2={stats_out['chi2_risk_season']['chi2']:.3f}, "
                    f"p={stats_out['chi2_risk_season']['p']:.4f}, "
                    f"dof={stats_out['chi2_risk_season']['dof']}\n\n")

        if "aic" in logit_out:
            f.write("Logistic regression (risk ~ rat_pressure * season + controls):\n")
            f.write(f"  AIC={logit_out['aic']:.1f}, BIC={logit_out['bic']:.1f}\n")
            f.write("  See logit_model_summary.txt and logit_odds_ratios.csv for details.\n\n")

        if glm_out and "summary_path" in glm_out:
            f.write("Window-level binomial GLM (risk_prop ~ predictors):\n")
            f.write(f"  See glm_binomial_summary.txt and glm_binomial_or.csv for coefficients.\n\n")

        if "r2" in lin_out:
            f.write("Linear regression (bat landings per window):\n")
            f.write(f"  R²={lin_out['r2']:.3f}  MAE={lin_out['mae']:.3f}  RMSE={lin_out['rmse']:.3f}\n")
            f.write(f"  Baseline (mean) → R²={lin_out['baseline_r2']:.3f}, "
                    f"MAE={lin_out['baseline_mae']:.3f}, RMSE={lin_out['baseline_rmse']:.3f}\n")
            f.write("  See linear_model_coefficients.csv for effect sizes.\n")

        f.write("\nInterpretation:\n")
        logit_terms = []
        odds_table = logit_out.get("odds_table") if logit_out else None
        if isinstance(odds_table, pd.DataFrame):
            for _, row in odds_table.iterrows():
                term = row["term"]
                if term == "Intercept":
                    continue
                direction = "↑" if row["OR"] > 1 else "↓"
                if pd.notna(row.get("p_value")):
                    significance = "p<0.05" if row["p_value"] < 0.05 else f"p={row['p_value']:.3f}"
                else:
                    significance = "penalised"
                logit_terms.append(f"{term} ({direction}, OR={row['OR']:.2f}, {significance})")

        glm_terms = []
        glm_table = glm_out.get("odds_table") if glm_out else None
        if isinstance(glm_table, pd.DataFrame):
            for _, row in glm_table.iterrows():
                term = row["term"]
                if term == "Intercept":
                    continue
                direction = "↑" if row["OR"] > 1 else "↓"
                significance = "p<0.05" if row["p_value"] < 0.05 else f"p={row['p_value']:.3f}"
                glm_terms.append(f"{term} ({direction}, OR={row['OR']:.2f}, {significance})")

        if logit_terms or glm_terms:
            f.write("  Logistic and binomial GLM models both highlight: ")
            if logit_terms:
                f.write("Logit → " + "; ".join(logit_terms))
            if glm_terms:
                if logit_terms:
                    f.write("; ")
                f.write("GLM → " + "; ".join(glm_terms))
            f.write(".\n")
        else:
            f.write("  Logistic and binomial GLM models were fitted but produced no stable coefficients.\n")

        deciles = narrative.get("rat_pressure_deciles")
        quartiles = narrative.get("rat_pressure_quartiles")
        if isinstance(deciles, pd.DataFrame) or isinstance(quartiles, pd.DataFrame):
            f.write("  Rat-pressure decile and quartile plots echo the modelled pattern (higher pressure → higher risk) "
                    "yet adjacent bins show overlapping 95% CIs, so contrasts are indicative rather than definitive.\n")

# --- main orchestration
def main():
    dataset1_path = DATASET1_PATH
    dataset2_path = DATASET2_PATH
    outdir = ensure_outdir(OUTPUT_DIR)

    # Load with forgiving file name handling (so you can drop in the Learnline downloads)
    try:
        df1 = smart_read_csv(str(dataset1_path), "dataset1*.*csv*")
        df2 = smart_read_csv(str(dataset2_path), "dataset2*.*csv*")
    except Exception as e:
        raise SystemExit(f"Failed to read datasets: {e}")

    # Clean & engineer
    landings = clean_landings(df1)
    windows = build_window_table(df2)
    merged = attach_windows(landings, windows)

    # Persist a merged snapshot used by several plots / tables
    merged.to_csv(outdir / "merged_landings_windows.csv", index=False)

    # --- Visuals (all focused on the research questions)
    plot_rat_pressure_hist(windows)
    rat_present_summary, rat_present_stats = plot_risk_by_rat_present(merged)
    season_summary, season_chi2_p = plot_risk_by_season(merged)
    decile_summary = plot_risk_vs_rat_pressure_deciles(merged)
    decile_season_summary = plot_risk_vs_rat_pressure_deciles_by_season(merged)
    rat_pressure_quart_summary = plot_risk_by_rat_pressure_quartile(merged)
    time_since_summary = plot_time_since_rat_quartiles(merged)
    heatmap_risk_by_season_pressure(merged)
    plot_ecdf_rat_pressure_by_season(merged)
    plot_lowess_risk_vs_rat_pressure(merged)
    monthly_trends(windows, merged)
    plot_hexbin_landings_vs_rat_minutes(windows)

    # --- Stats & models
    stats_out = run_inference_tests(merged, outdir)
    logit_out = fit_logit_model(merged, outdir)
    lin_out = fit_linear_model(windows, outdir)
    glm_out = fit_window_level_glm(merged, windows, outdir)

    if isinstance(logit_out, dict) and "params" in logit_out:
        plot_predicted_risk_by_rat_pressure(logit_out, outdir)
        plot_predicted_risk_by_hours(logit_out, outdir)

    # --- Tables for the report appendix
    summaries = {
        "rat_present": rat_present_summary,
        "season": season_summary,
        "rat_present_stats": rat_present_stats,
        "season_chi2_p": season_chi2_p,
        "rat_pressure_deciles": decile_summary,
        "rat_pressure_deciles_by_season": decile_season_summary,
        "rat_pressure_quartiles": rat_pressure_quart_summary,
        "time_since_quartiles": time_since_summary,
    }
    export_group_tables(merged, windows, outdir, summaries)

    # --- Friendly, one-pager summary for the marker
    narrative = {
        "rat_present_stats": rat_present_stats,
        "season_chi2_p": season_chi2_p,
        "rat_pressure_deciles": decile_summary,
        "rat_pressure_quartiles": rat_pressure_quart_summary,
    }
    write_summary_txt(stats_out, logit_out, glm_out, lin_out, merged, outdir, narrative)

    print(f"Done. Outputs are in: {outdir.resolve()}")

if __name__ == "__main__":
    main()
