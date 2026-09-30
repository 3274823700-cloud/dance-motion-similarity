# -*- coding: utf-8 -*-
"""
Phase 3.4A - Score Architecture Calibration with Block-wise Holdout Validation

Goal
----
Decide which RAW metrics should play which role before designing the final 0-100 score.

Inputs from previous validated experiments:
1) phase3_calibration_distribution/pair_summary.csv
   - DTW pose distance
   - DTW timing/path diagnostics
2) phase3_smoothing_diagnostic/smoothing_pair_results.csv
   - 3-frame-median joint-angle error

Validation design
-----------------
The 18 D/E pairs naturally form three independent music/dancer blocks:

Block A: D1-D3, E1-E3  (mBR0)
Block B: D4-D6, E4-E6  (mBR2)
Block C: D7-D9, E7-E9  (mBR4)

For each fold:
- hold out ONE entire block for testing;
- calibrate feature scales using only the other TWO blocks;
- test on the unseen block.

This avoids fitting and testing a score mapping on the exact same pair set.

Important
---------
- This script does NOT create the final 0-100 score.
- It does NOT change joint weights.
- It does NOT change DTW.
- Timing is evaluated as a diagnostic candidate, but should not automatically
  become part of overall pose similarity because timing has a different meaning.
"""

import os
import csv
import math
import numpy as np
import matplotlib.pyplot as plt


# ======================================================================
# Paths
# ======================================================================
CALIBRATION_CSV_CANDIDATES = [
    os.path.join("phase3_calibration_distribution", "pair_summary.csv"),
    "pair_summary.csv",
]

SMOOTHING_CSV_CANDIDATES = [
    os.path.join("phase3_smoothing_diagnostic", "smoothing_pair_results.csv"),
    "smoothing_pair_results.csv",
]

OUT_DIR = "phase3_score_architecture"


# ======================================================================
# Candidate score architectures
# All features are ERROR measures: larger = less similar.
# Weights are intentionally simple and fixed; they are NOT optimized.
# ======================================================================
CANDIDATES = {
    "Pose only": {
        "dtw_pose_distance": 1.0,
    },
    "Angle only": {
        "smoothed_angle_error_deg": 1.0,
    },
    "Pose70 + Angle30": {
        "dtw_pose_distance": 0.70,
        "smoothed_angle_error_deg": 0.30,
    },
    "Pose50 + Angle50": {
        "dtw_pose_distance": 0.50,
        "smoothed_angle_error_deg": 0.50,
    },

    # Diagnostic only: timing is semantically different from pose quality.
    "Timing only [diagnostic]": {
        "mean_relative_time_deviation": 1.0,
    },
    "Pose40 + Angle40 + Timing20 [diagnostic]": {
        "dtw_pose_distance": 0.40,
        "smoothed_angle_error_deg": 0.40,
        "mean_relative_time_deviation": 0.20,
    },
}


# ======================================================================
# Helpers
# ======================================================================
def find_existing(candidates):
    for path in candidates:
        if os.path.isfile(path):
            return path
    raise IOError(
        "Required CSV not found. Tried:\n  " + "\n  ".join(candidates)
    )


def read_csv_dict(path):
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def pair_block(pair_id):
    """Map D1/E1... to the three independent calibration blocks."""
    n = int(pair_id[1:])
    if 1 <= n <= 3:
        return "A"
    if 4 <= n <= 6:
        return "B"
    if 7 <= n <= 9:
        return "C"
    raise ValueError("Unexpected pair ID: %s" % pair_id)


def median(values):
    return float(np.median(np.asarray(values, dtype=np.float64)))


def mean(values):
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def std(values):
    x = np.asarray(values, dtype=np.float64)
    return float(np.std(x, ddof=1)) if len(x) > 1 else 0.0


def auc_binary(labels, scores):
    """
    Manual ROC AUC.
    labels: 0 for D, 1 for E
    score: larger means more likely E / larger error.
    AUC = P(score_E > score_D) + 0.5 P(tie)
    """
    d = [s for y, s in zip(labels, scores) if y == 0]
    e = [s for y, s in zip(labels, scores) if y == 1]

    if not d or not e:
        return float("nan")

    wins = 0.0
    total = 0

    for se in e:
        for sd in d:
            total += 1
            if se > sd:
                wins += 1.0
            elif se == sd:
                wins += 0.5

    return wins / total


def balanced_accuracy(labels, predictions):
    d_ok = []
    e_ok = []

    for y, p in zip(labels, predictions):
        if y == 0:
            d_ok.append(1.0 if p == 0 else 0.0)
        else:
            e_ok.append(1.0 if p == 1 else 0.0)

    tnr = mean(d_ok) if d_ok else float("nan")
    tpr = mean(e_ok) if e_ok else float("nan")
    return 0.5 * (tnr + tpr)


def summarize(values):
    x = np.asarray(values, dtype=np.float64)
    return {
        "count": len(x),
        "mean": float(np.mean(x)),
        "median": float(np.median(x)),
        "min": float(np.min(x)),
        "max": float(np.max(x)),
        "std": float(np.std(x, ddof=1)) if len(x) > 1 else 0.0,
    }


# ======================================================================
# Load and merge previous experiment outputs
# ======================================================================
def load_feature_table():
    calibration_path = find_existing(CALIBRATION_CSV_CANDIDATES)
    smoothing_path = find_existing(SMOOTHING_CSV_CANDIDATES)

    calibration_rows = read_csv_dict(calibration_path)
    smoothing_rows = read_csv_dict(smoothing_path)

    cal = {r["pair_id"]: r for r in calibration_rows}
    smo = {r["pair_id"]: r for r in smoothing_rows}

    expected = (
        ["D%d" % i for i in range(1, 10)] +
        ["E%d" % i for i in range(1, 10)]
    )

    rows = []

    for pair_id in expected:
        if pair_id not in cal:
            raise KeyError("Missing %s in %s" % (pair_id, calibration_path))
        if pair_id not in smo:
            raise KeyError("Missing %s in %s" % (pair_id, smoothing_path))

        c = cal[pair_id]
        s = smo[pair_id]

        if c["group"] != s["group"]:
            raise ValueError("Group mismatch for %s" % pair_id)

        rows.append({
            "pair_id": pair_id,
            "group": c["group"],
            "block": pair_block(pair_id),
            "dtw_pose_distance": float(c["dtw_pose_distance"]),
            "smoothed_angle_error_deg": float(s["smoothed_angle_error_deg"]),
            "mean_relative_time_deviation": float(
                c["mean_relative_time_deviation"]
            ),
            "non_diagonal_ratio": float(c["non_diagonal_ratio"]),
        })

    return rows, calibration_path, smoothing_path


# ======================================================================
# Train-only normalization
# ======================================================================
def train_feature_anchor(train_rows, feature):
    """
    Use class medians from TRAINING BLOCKS ONLY.

    normalized error:
        D training median -> 0
        E training median -> 1

    Values are NOT clipped so held-out distribution shift remains visible.
    """
    d = [r[feature] for r in train_rows if r["group"] == "D"]
    e = [r[feature] for r in train_rows if r["group"] == "E"]

    d_med = median(d)
    e_med = median(e)
    scale = e_med - d_med

    if abs(scale) < 1e-12:
        raise ValueError(
            "Feature %s has near-zero D/E training-median difference." % feature
        )

    return d_med, e_med, scale


def candidate_index(rows, weights, anchors):
    values = []

    for r in rows:
        index = 0.0

        for feature, weight in weights.items():
            d_med, e_med, scale = anchors[feature]
            normalized_error = (r[feature] - d_med) / scale
            index += weight * normalized_error

        values.append(float(index))

    return values


# ======================================================================
# Cross-validation
# ======================================================================
def run_block_cv(rows):
    fold_rows = []

    all_features = sorted({
        feature
        for weights in CANDIDATES.values()
        for feature in weights.keys()
    })

    for holdout in ("A", "B", "C"):
        train = [r for r in rows if r["block"] != holdout]
        test = [r for r in rows if r["block"] == holdout]

        anchors = {
            feature: train_feature_anchor(train, feature)
            for feature in all_features
        }

        labels = [0 if r["group"] == "D" else 1 for r in test]

        for candidate_name, weights in CANDIDATES.items():
            scores = candidate_index(test, weights, anchors)

            # Because train D median maps to 0 and train E median maps to 1,
            # 0.5 is the natural fixed midpoint threshold.
            threshold = 0.5
            predictions = [1 if s >= threshold else 0 for s in scores]

            auc = auc_binary(labels, scores)
            bal_acc = balanced_accuracy(labels, predictions)

            d_scores = [s for y, s in zip(labels, scores) if y == 0]
            e_scores = [s for y, s in zip(labels, scores) if y == 1]

            fold_rows.append({
                "holdout_block": holdout,
                "candidate": candidate_name,
                "test_auc": auc,
                "test_balanced_accuracy": bal_acc,
                "test_D_mean": mean(d_scores),
                "test_D_median": median(d_scores),
                "test_D_min": min(d_scores),
                "test_D_max": max(d_scores),
                "test_E_mean": mean(e_scores),
                "test_E_median": median(e_scores),
                "test_E_min": min(e_scores),
                "test_E_max": max(e_scores),
                "test_gap_minE_minus_maxD": min(e_scores) - max(d_scores),
            })

    return fold_rows


def summarize_candidates(fold_rows):
    summary = []

    for candidate_name in CANDIDATES.keys():
        rows = [r for r in fold_rows if r["candidate"] == candidate_name]

        aucs = [r["test_auc"] for r in rows]
        bals = [r["test_balanced_accuracy"] for r in rows]
        gaps = [r["test_gap_minE_minus_maxD"] for r in rows]

        summary.append({
            "candidate": candidate_name,
            "mean_test_auc": mean(aucs),
            "min_test_auc": min(aucs),
            "mean_test_balanced_accuracy": mean(bals),
            "min_test_balanced_accuracy": min(bals),
            "mean_test_gap": mean(gaps),
            "min_test_gap": min(gaps),
        })

    return summary


# ======================================================================
# Raw feature statistics by held-out block
# ======================================================================
def raw_block_statistics(rows):
    result = []

    for block in ("A", "B", "C"):
        for group in ("D", "E"):
            subset = [
                r for r in rows
                if r["block"] == block and r["group"] == group
            ]

            for feature in (
                "dtw_pose_distance",
                "smoothed_angle_error_deg",
                "mean_relative_time_deviation",
                "non_diagonal_ratio",
            ):
                s = summarize([r[feature] for r in subset])
                result.append({
                    "block": block,
                    "group": group,
                    "feature": feature,
                    "count": s["count"],
                    "mean": s["mean"],
                    "median": s["median"],
                    "min": s["min"],
                    "max": s["max"],
                    "std": s["std"],
                })

    return result


# ======================================================================
# CSV output
# ======================================================================
def write_dict_csv(path, rows):
    if not rows:
        return

    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)


# ======================================================================
# Plots
# ======================================================================
def plot_pose_angle(rows, out_path):
    plt.figure(figsize=(8, 6))

    for group, marker in (("D", "o"), ("E", "x")):
        subset = [r for r in rows if r["group"] == group]
        x = [r["dtw_pose_distance"] for r in subset]
        y = [r["smoothed_angle_error_deg"] for r in subset]

        plt.scatter(x, y, marker=marker, label=group)

        for r in subset:
            plt.annotate(
                r["pair_id"],
                (r["dtw_pose_distance"], r["smoothed_angle_error_deg"]),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=8,
            )

    plt.xlabel("DTW Pose Distance")
    plt.ylabel("Smoothed Weighted Angle Error (deg)")
    plt.title("Phase 3.4A: Pose Distance vs Smoothed Angle Error")
    plt.legend()
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def plot_candidate_accuracy(summary_rows, out_path):
    names = [r["candidate"] for r in summary_rows]
    values = [r["mean_test_balanced_accuracy"] for r in summary_rows]

    plt.figure(figsize=(10, 6))
    x = np.arange(len(names))
    plt.bar(x, values)
    plt.xticks(x, names, rotation=25, ha="right")
    plt.ylim(0.0, 1.05)
    plt.ylabel("Mean Held-out Balanced Accuracy")
    plt.title("Phase 3.4A: Block-wise Holdout Validation")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


# ======================================================================
# Console report
# ======================================================================
def print_feature_ranges(rows):
    print("\n" + "=" * 88)
    print("RAW FEATURE RANGES")
    print("=" * 88)

    for feature, label in (
        ("dtw_pose_distance", "Pose distance"),
        ("smoothed_angle_error_deg", "Smoothed angle error"),
        ("mean_relative_time_deviation", "Mean relative-time deviation"),
    ):
        print("\n%s" % label)

        for group in ("D", "E"):
            x = [r[feature] for r in rows if r["group"] == group]
            s = summarize(x)
            print(
                "  %s: mean %.6f | median %.6f | min %.6f | max %.6f | std %.6f"
                % (
                    group,
                    s["mean"], s["median"],
                    s["min"], s["max"], s["std"],
                )
            )


def print_cv(fold_rows, summary_rows):
    print("\n" + "=" * 88)
    print("BLOCK-WISE HOLDOUT VALIDATION")
    print("=" * 88)

    for row in fold_rows:
        print(
            "%s | %-42s | AUC %.3f | BalAcc %.3f | gap %+.3f"
            % (
                row["holdout_block"],
                row["candidate"],
                row["test_auc"],
                row["test_balanced_accuracy"],
                row["test_gap_minE_minus_maxD"],
            )
        )

    print("\n" + "=" * 88)
    print("CANDIDATE SUMMARY")
    print("=" * 88)

    for row in summary_rows:
        print(
            "%-42s | mean AUC %.3f | mean BalAcc %.3f | min BalAcc %.3f | mean gap %+.3f"
            % (
                row["candidate"],
                row["mean_test_auc"],
                row["mean_test_balanced_accuracy"],
                row["min_test_balanced_accuracy"],
                row["mean_test_gap"],
            )
        )

    eligible = [
        r for r in summary_rows
        if "[diagnostic]" not in r["candidate"]
    ]

    best = sorted(
        eligible,
        key=lambda r: (
            r["mean_test_balanced_accuracy"],
            r["min_test_balanced_accuracy"],
            r["mean_test_gap"],
        ),
        reverse=True,
    )[0]

    print("\nBest non-timing candidate in this calibration:")
    print("  %s" % best["candidate"])
    print(
        "  mean held-out balanced accuracy = %.3f"
        % best["mean_test_balanced_accuracy"]
    )

    print("\nImportant interpretation:")
    print("  - Timing candidates are diagnostic only; do NOT automatically put timing into overall pose similarity.")
    print("  - AUC measures ordering; balanced accuracy also tests whether the training-calibrated scale transfers to a new block.")
    print("  - This phase selects score ARCHITECTURE only. It does not define the final 0-100 mapping.")


# ======================================================================
# Main
# ======================================================================
def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    rows, calibration_path, smoothing_path = load_feature_table()

    print("=" * 88)
    print("Phase 3.4A - Score Architecture Calibration")
    print("=" * 88)
    print("Calibration CSV :", calibration_path)
    print("Smoothing CSV   :", smoothing_path)
    print("Pairs           :", len(rows))
    print("Validation      : leave-one-block-out (A / B / C)")
    print("Final 0-100     : NOT designed in this phase")

    print_feature_ranges(rows)

    fold_rows = run_block_cv(rows)
    summary_rows = summarize_candidates(fold_rows)
    block_rows = raw_block_statistics(rows)

    print_cv(fold_rows, summary_rows)

    feature_csv = os.path.join(OUT_DIR, "score_feature_table.csv")
    fold_csv = os.path.join(OUT_DIR, "architecture_cv_folds.csv")
    summary_csv = os.path.join(OUT_DIR, "architecture_summary.csv")
    block_csv = os.path.join(OUT_DIR, "raw_feature_block_statistics.csv")
    scatter_png = os.path.join(OUT_DIR, "pose_vs_angle_scatter.png")
    accuracy_png = os.path.join(OUT_DIR, "candidate_balanced_accuracy.png")

    write_dict_csv(feature_csv, rows)
    write_dict_csv(fold_csv, fold_rows)
    write_dict_csv(summary_csv, summary_rows)
    write_dict_csv(block_csv, block_rows)

    plot_pose_angle(rows, scatter_png)
    plot_candidate_accuracy(summary_rows, accuracy_png)

    print("\nSaved:")
    print(" ", feature_csv)
    print(" ", fold_csv)
    print(" ", summary_csv)
    print(" ", block_csv)
    print(" ", scatter_png)
    print(" ", accuracy_png)


if __name__ == "__main__":
    main()
