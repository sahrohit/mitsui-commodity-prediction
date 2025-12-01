import sys
from pathlib import Path
from typing import Tuple, Optional
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------
# User-configurable settings
# -----------------------------
SUBMISSIONS = [
    # ("outputs/naive.csv", "Naive"),
    # ("outputs/arima.csv", "ARIMA"),
    # ("outputs/submission_baseline_full_lgbm.csv", "Baseline LGBM"),
    ("outputs/submission_advanced_domain_rolling_lgbm.csv", "Advanced LGBM"),
    # ("/mnt/data/another_submission.csv", "My Model V2"),
]

TEST_ACTUALS_FILE = Path("dataset/test.csv")
TRAIN_LABELS_FILE = Path("dataset/train_labels.csv")
TARGET_MAP_FILE = Path("dataset/target_pairs.csv")

TARGET = "target_0"
# TARGET = "target_19"
# TARGET = "target_31"
# TARGET = "target_44"

BULK_TARGETS = [
    # "target_31",
    # "target_1",
    # "target_2",
]

# -----------------------------
# Helpers
# -----------------------------
def load_submission(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.reset_index(drop=True)
    return df


def pick_actuals(test_path: Path, train_labels_path: Path) -> Tuple[pd.DataFrame, str]:
    """
    Prefer sample_test.csv only if it contains target_* columns. Otherwise fall back to sample_train_labels.csv.
    Returns: (df_actuals, source_name)
    """
    def has_target_cols(df: pd.DataFrame) -> bool:
        return any(str(c).startswith("target_") for c in df.columns)

    # Try test first
    if test_path.exists():
        try:
            df = pd.read_csv(test_path)
            df = df.reset_index(drop=True)
            if has_target_cols(df):
                return df, "sample_test.csv"
        except Exception:
            pass

    # Fallback to train labels
    if train_labels_path.exists():
        df = pd.read_csv(train_labels_path)
        df = df.reset_index(drop=True)
        if has_target_cols(df):
            return df, "sample_train_labels.csv"

    raise FileNotFoundError("No actuals with target_* columns found. Provide sample_test.csv or sample_train_labels.csv with targets.")


def load_target_map(path: Path) -> Optional[pd.DataFrame]:
    if path.exists():
        try:
            m = pd.read_csv(path)
            return m
        except Exception:
            return None
    return None

def friendly_name(target: str, mapping_df: Optional[pd.DataFrame]) -> str:
    if mapping_df is None:
        return target
    # mapping_df columns: ['target','lag','pair']
    try:
        row = mapping_df.loc[mapping_df['target'] == target]
        if not row.empty:
            pair = str(row.iloc[0]['pair'])
            return f"{target} — {pair}"
    except Exception:
        pass
    return target

def get_candidate_targets(sub_df: pd.DataFrame, actuals_df: pd.DataFrame) -> list:
    sub_targets = [c for c in sub_df.columns if c.startswith("target_")]
    act_targets = [c for c in actuals_df.columns if c.startswith("target_")]
    # common to both so we can compare
    common = sorted(set(sub_targets).intersection(act_targets), key=lambda x: int(x.split("_")[1]))
    return common

def build_time_index(df: pd.DataFrame) -> pd.Index:
    if "date_id" in df.columns:
        return pd.Index(df["date_id"], name="date_id")
    # fallback: simple integer index as a pseudo-date axis
    return pd.RangeIndex(start=0, stop=len(df), step=1, name="t")


def align_actuals_preds(actuals: pd.DataFrame, preds: pd.DataFrame, target: str) -> Tuple[pd.Series, pd.Series, pd.Index]:
    """
    Alignment priority:
    1) If both have date_id, align on the intersection of date_id.
    2) Else if lengths equal, align by index.
    3) Else align predictions to the tail of actuals (last len(preds) rows).
    """
    # Case 1: align on date_id if available in both
    if "date_id" in actuals.columns and "date_id" in preds.columns:
        a = actuals[["date_id", target]].dropna(subset=[target]).copy()
        p = preds[["date_id", target]].dropna(subset=[target]).copy()
        merged = pd.merge(a, p, on="date_id", how="inner", suffixes=("_actual", "_pred"))
        merged = merged.sort_values("date_id")
        idx = pd.Index(merged["date_id"], name="date_id")
        return merged[f"{target}_actual"].rename("actual").set_axis(idx), merged[f"{target}_pred"].rename("predicted").set_axis(idx), idx

    # Case 2: same length -> 1:1 index
    if len(actuals) == len(preds):
        idx = build_time_index(actuals)
        a = pd.Series(actuals[target].values, index=idx, name="actual")
        p = pd.Series(preds[target].values, index=idx, name="predicted")
        return a, p, idx

    # Case 3: tail align predictions against actuals (common in TS forecasting)
    n = min(len(preds), len(actuals))
    act_idx = build_time_index(actuals)
    # take the tail of actuals to match predictions length
    a_tail = pd.Series(actuals[target].values[-n:], index=act_idx[-n:], name="actual")
    # use same index for predictions for overlay
    p_tail = pd.Series(preds[target].values[-n:], index=a_tail.index, name="predicted")
    return a_tail, p_tail, a_tail.index
def plot_actual_vs_pred(actual: pd.Series, pred: pd.Series, target_label: str, outdir: Path) -> Path:
    outdir.mkdir(parents=True, exist_ok=True)
    outpath = outdir / f"plot_{target_label}.png"

    plt.figure(figsize=(10, 5))
    plt.plot(actual.index, actual.values, label="Actual")
    # Plot predictions only where they exist (index-matched to the tail of actual)
    plt.plot(pred.index, pred.values, label="Predicted")
    plt.title(f"Actual vs Predicted — {target_label}")
    plt.xlabel(actual.index.name if actual.index.name else "time")
    plt.ylabel("value")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.show()
    return outpath
def load_submissions(entries: list) -> list[tuple[str, pd.DataFrame]]:
    loaded = []
    for path, label in entries:
        p = Path(path)
        if p.exists():
            try:
                df = pd.read_csv(p).reset_index(drop=True)
                loaded.append((label, df))
            except Exception as e:
                print(f"Skipping {path} ({label}): {e}")
        else:
            print(f"Missing submission file: {path} ({label})")
    if not loaded:
        raise FileNotFoundError("No valid submission files loaded. Please check SUBMISSIONS paths.")
    return loaded

def get_common_targets_for_average(actuals_df: pd.DataFrame, subs: list[tuple[str, pd.DataFrame]]) -> list[str]:
    act_targets = {c for c in actuals_df.columns if str(c).startswith('target_')}
    if not act_targets:
        return []
    common = act_targets.copy()
    for _, sub_df in subs:
        sub_targets = {c for c in sub_df.columns if str(c).startswith('target_')}
        common = common.intersection(sub_targets)
    # sort by index number
    return sorted(common, key=lambda x: int(x.split('_')[1]))

def plot_multi_preds_for_target(actual: pd.Series, preds_dict: dict[str, pd.Series], target_label: str, outdir: Path) -> Path:
    outdir.mkdir(parents=True, exist_ok=True)
    outpath = outdir / f"plot_multi_{target_label}.png"
    plt.figure(figsize=(10, 5))
    plt.plot(actual.index, actual.values, label="Actual")
    for label, series in preds_dict.items():
        plt.plot(series.index, series.values, label=label)  # one line per submission
    plt.title(f"Actual vs Predicted (Multiple) — {target_label}")
    plt.xlabel(actual.index.name if actual.index.name else "time")
    plt.ylabel("value")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.show()
    return outpath

def plot_average_chart(actuals_df: pd.DataFrame, subs: list[tuple[str, pd.DataFrame]], outdir: Path) -> Path:
    """
    Compute row-wise mean across all common targets for Actual vs each submission's Predicted,
    align by date_id if present, else by length/tail like before, and plot on a single chart.
    """
    outdir.mkdir(parents=True, exist_ok=True)
    outpath = outdir / "plot_average_across_targets.png"

    common_targets = get_common_targets_for_average(actuals_df, subs)
    if not common_targets:
        raise ValueError("No common target_* columns between actuals and submissions for averaging.")

    # Build an 'actual mean' series first
    # We'll use build_time_index for the index.
    idx_actual = build_time_index(actuals_df)
    actual_mean = pd.Series(actuals_df[common_targets].mean(axis=1, skipna=True).values, index=idx_actual, name="actual_mean")

    # For each submission, compute predicted mean and align to actual_mean by our rules
    pred_means = {}
    for label, sub_df in subs:
        # Compute predicted mean over the same targets
        # If a submission is missing some targets, intersect for safety
        have = [t for t in common_targets if t in sub_df.columns]
        sub_mean = sub_df[have].mean(axis=1, skipna=True)

        # Build pseudo-frames to reuse align logic
        A = pd.DataFrame({"__actual__": actual_mean.values})
        A.index = actual_mean.index
        if actual_mean.index.name == "date_id":
            A = A.reset_index()
        P = pd.DataFrame({"__pred__": sub_mean.values})
        if "date_id" in actuals_df.columns:
            # give P a matching date_id if available length-wise
            # align via case 2 or 3; case 1 applies only when both have date_id
            pass

        # Reuse align logic by renaming to target-like temp names
        A_tmp = A.copy()
        if "date_id" not in A_tmp.columns and actual_mean.index.name:
            A_tmp = A_tmp.reset_index()
        A_tmp.rename(columns={"__actual__": "target_tmp"}, inplace=True)
        P_tmp = pd.DataFrame({"target_tmp": sub_mean}).reset_index(drop=True)

        a_series, p_series, _ = align_actuals_preds(A_tmp, P_tmp, "target_tmp")
        pred_means[label] = p_series.rename(label)
        actual_mean_aligned = a_series  # grab the most recent one for x-axis reference

    # Plot
    plt.figure(figsize=(10, 5))
    plt.plot(actual_mean_aligned.index, actual_mean_aligned.values, label="Actual (mean)")
    for label, s in pred_means.items():
        plt.plot(s.index, s.values, label=f"{label} (mean)")
    plt.title("Average across targets — Actual vs Predicted")
    plt.xlabel(actual_mean_aligned.index.name if actual_mean_aligned.index.name else "time")
    plt.ylabel("value")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.show()
    return outpath

def load_submissions(entries: list) -> list[tuple[str, pd.DataFrame]]:
    loaded = []
    for path, label in entries:
        p = Path(path)
        if p.exists():
            try:
                df = pd.read_csv(p).reset_index(drop=True)
                loaded.append((label, df))
            except Exception as e:
                print(f"Skipping {path} ({label}): {e}")
        else:
            print(f"Missing submission file: {path} ({label})")
    if not loaded:
        raise FileNotFoundError("No valid submission files loaded. Please check SUBMISSIONS paths.")
    return loaded

def get_common_targets_for_average(actuals_df: pd.DataFrame, subs: list[tuple[str, pd.DataFrame]]) -> list[str]:
    act_targets = {c for c in actuals_df.columns if str(c).startswith('target_')}
    if not act_targets:
        return []
    common = act_targets.copy()
    for _, sub_df in subs:
        sub_targets = {c for c in sub_df.columns if str(c).startswith('target_')}
        common = common.intersection(sub_targets)
    return sorted(common, key=lambda x: int(x.split('_')[1]))

def plot_multi_preds_for_target(actual: pd.Series, preds_dict: dict[str, pd.Series], target_label: str, outdir: Path) -> Path:
    outdir.mkdir(parents=True, exist_ok=True)
    outpath = outdir / f"plot_multi_{target_label}.png"
    plt.figure(figsize=(10, 5))
    plt.plot(actual.index, actual.values, label="Actual")
    for label, series in preds_dict.items():
        plt.plot(series.index, series.values, label=label)
    plt.title(f"Actual vs Predicted (Multiple) — {target_label}")
    plt.xlabel(actual.index.name if actual.index.name else "time")
    plt.ylabel("value")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.show()
    return outpath

def plot_average_chart(actuals_df: pd.DataFrame, subs: list[tuple[str, pd.DataFrame]], outdir: Path) -> Path:
    """Compute row-wise mean across all common targets for Actual vs each submission's Predicted."""
    outdir.mkdir(parents=True, exist_ok=True)
    outpath = outdir / "plot_average_across_targets.png"

    common_targets = get_common_targets_for_average(actuals_df, subs)
    if not common_targets:
        raise ValueError("No common target_* columns between actuals and submissions for averaging.")

    idx_actual = build_time_index(actuals_df)
    actual_mean = pd.Series(actuals_df[common_targets].mean(axis=1, skipna=True).values, index=idx_actual, name="actual_mean")

    pred_means = {}
    actual_mean_aligned = actual_mean
    for label, sub_df in subs:
        have = [t for t in common_targets if t in sub_df.columns]
        sub_mean = sub_df[have].mean(axis=1, skipna=True)

        A_tmp = pd.DataFrame({"target_tmp": actual_mean.values})
        if actual_mean.index.name:
            A_tmp[actual_mean.index.name] = actual_mean.index
            A_tmp = A_tmp[[actual_mean.index.name, "target_tmp"]]

        P_tmp = pd.DataFrame({"target_tmp": sub_mean}).reset_index(drop=True)

        a_series, p_series, _ = align_actuals_preds(A_tmp, P_tmp, "target_tmp")
        pred_means[label] = p_series.rename(label)
        actual_mean_aligned = a_series

    plt.figure(figsize=(10, 5))
    plt.plot(actual_mean_aligned.index, actual_mean_aligned.values, label="Actual (mean)")
    for label, s in pred_means.items():
        plt.plot(s.index, s.values, label=f"{label} (mean)")
    plt.title("Average across targets — Actual vs Predicted")
    plt.xlabel(actual_mean_aligned.index.name if actual_mean_aligned.index.name else "time")
    plt.ylabel("value")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.show()
    return outpath


def main():
    # Load files
    subs = load_submissions(SUBMISSIONS)

    actuals_df, source_name = pick_actuals(TEST_ACTUALS_FILE, TRAIN_LABELS_FILE)
    mapping_df = load_target_map(TARGET_MAP_FILE)
    # Build candidate list from actuals and the first submission
    first_label, first_sub = subs[0]
    candidates = get_candidate_targets(first_sub, actuals_df)
    print("\nAvailable targets (comment/uncomment for reuse):")
    for t in candidates[:50]:
        print(" ", t)
    if len(candidates) > 50:
        print(" ... ({} total)".format(len(candidates)))

    if TARGET not in candidates:
        print(f"Requested {TARGET} not in available targets; using {candidates[0]} instead.")
        chosen = candidates[0]
    else:
        chosen = TARGET

    # Align for each submission and overlay
    actual, _, _ = align_actuals_preds(actuals_df, first_sub, chosen)  # use first to get index
    preds_dict = {}
    for label, sub_df in subs:
        _, pred, _ = align_actuals_preds(actuals_df, sub_df, chosen)
        preds_dict[label] = pred

    label = friendly_name(chosen, mapping_df)
    multi_out = plot_multi_preds_for_target(actual, preds_dict, label, Path("outputs/plots"))
    print(f"Saved multi-submission plot to: {multi_out} (actuals from {source_name})")

    # Average chart across targets
    avg_out = plot_average_chart(actuals_df, subs, Path("outputs/plots"))
    print(f"Saved average-across-targets plot to: {avg_out}")

    # If you want to bulk-generate for many targets, uncomment below:
    # for t in BULK_TARGETS:
    #     if t in candidates:
    #         a, p, _ = align_actuals_preds(actuals_df, sub_df, t)
    #         lbl = friendly_name(t, mapping_df)
    #         plot_actual_vs_pred(a, p, lbl, Path("/mnt/data/plots"))

if __name__ == "__main__":
    main()