# -*- coding: utf-8 -*-
"""
Phase 3.4B - Unseen Choreography Holdout Validation

Purpose
-------
Validate the score architecture selected in Phase 3.4A on NEW choreography IDs.

Calibration set already used:
    ch01, ch02, ch03

This script validates on:
    ch04, ch05, ch06

Reserved for later:
    ch07, ch08, ch09, ch10

The tested architecture is:
    Overall motion similarity -> DTW Pose Distance
    Joint feedback            -> 3-frame-median joint-angle errors
    Timing feedback           -> DTW timing diagnostics

Important
---------
- The Pose-only architecture is NOT re-selected here.
- The calibration threshold is learned ONLY from the previous ch01-ch03 set.
- ch04-ch06 are used only as unseen validation.
- No final 0-100 mapping is designed in this script.
- Joint weights and DTW are unchanged.
"""

import os
import csv
import importlib.util
from itertools import combinations
import numpy as np


# ======================================================================
# Paths
# ======================================================================
V11_FILE = "Dance_Motion_Similarity_Evaluation_V1.1_final.py"
VIDEO_DIR = os.path.join("data", "aistpp", "videos")
OFFICIAL_CSV = os.path.join(VIDEO_DIR, "refined_10M_all_video_url.csv")
CALIBRATION_PAIR_CSV = os.path.join(
    "phase3_calibration_distribution", "pair_summary.csv"
)
CALIBRATION_SMOOTH_CSV = os.path.join(
    "phase3_smoothing_diagnostic", "smoothing_pair_results.csv"
)
OUT_DIR = "phase3_unseen_choreo_validation"

SMOOTH_RADIUS = 1


# ======================================================================
# Unseen validation videos: ch04-ch06
# ======================================================================
VIDEO_KEYS = {
    # Block A: d04 vs d05, mBR0
    "d04_mBR0_ch04": "gBR_sBM_c01_d04_mBR0_ch04.mp4",
    "d04_mBR0_ch05": "gBR_sBM_c01_d04_mBR0_ch05.mp4",
    "d04_mBR0_ch06": "gBR_sBM_c01_d04_mBR0_ch06.mp4",
    "d05_mBR0_ch04": "gBR_sBM_c01_d05_mBR0_ch04.mp4",
    "d05_mBR0_ch05": "gBR_sBM_c01_d05_mBR0_ch05.mp4",
    "d05_mBR0_ch06": "gBR_sBM_c01_d05_mBR0_ch06.mp4",

    # Block B: d04 vs d06, mBR2
    "d04_mBR2_ch04": "gBR_sBM_c01_d04_mBR2_ch04.mp4",
    "d04_mBR2_ch05": "gBR_sBM_c01_d04_mBR2_ch05.mp4",
    "d04_mBR2_ch06": "gBR_sBM_c01_d04_mBR2_ch06.mp4",
    "d06_mBR2_ch04": "gBR_sBM_c01_d06_mBR2_ch04.mp4",
    "d06_mBR2_ch05": "gBR_sBM_c01_d06_mBR2_ch05.mp4",
    "d06_mBR2_ch06": "gBR_sBM_c01_d06_mBR2_ch06.mp4",

    # Block C: d05 vs d06, mBR4
    "d05_mBR4_ch04": "gBR_sBM_c01_d05_mBR4_ch04.mp4",
    "d05_mBR4_ch05": "gBR_sBM_c01_d05_mBR4_ch05.mp4",
    "d05_mBR4_ch06": "gBR_sBM_c01_d05_mBR4_ch06.mp4",
    "d06_mBR4_ch04": "gBR_sBM_c01_d06_mBR4_ch04.mp4",
    "d06_mBR4_ch05": "gBR_sBM_c01_d06_mBR4_ch05.mp4",
    "d06_mBR4_ch06": "gBR_sBM_c01_d06_mBR4_ch06.mp4",
}

# D = different dancer, same music + same choreography
D_PAIRS = [
    ("VD1", "A", "d04_mBR0_ch04", "d05_mBR0_ch04"),
    ("VD2", "A", "d04_mBR0_ch05", "d05_mBR0_ch05"),
    ("VD3", "A", "d04_mBR0_ch06", "d05_mBR0_ch06"),

    ("VD4", "B", "d04_mBR2_ch04", "d06_mBR2_ch04"),
    ("VD5", "B", "d04_mBR2_ch05", "d06_mBR2_ch05"),
    ("VD6", "B", "d04_mBR2_ch06", "d06_mBR2_ch06"),

    ("VD7", "C", "d05_mBR4_ch04", "d06_mBR4_ch04"),
    ("VD8", "C", "d05_mBR4_ch05", "d06_mBR4_ch05"),
    ("VD9", "C", "d05_mBR4_ch06", "d06_mBR4_ch06"),
]

# E = same dancer + same music, different choreography
E_PAIRS = [
    ("VE1", "A", "d04_mBR0_ch04", "d04_mBR0_ch05"),
    ("VE2", "A", "d04_mBR0_ch04", "d04_mBR0_ch06"),
    ("VE3", "A", "d04_mBR0_ch05", "d04_mBR0_ch06"),

    ("VE4", "B", "d06_mBR2_ch04", "d06_mBR2_ch05"),
    ("VE5", "B", "d06_mBR2_ch04", "d06_mBR2_ch06"),
    ("VE6", "B", "d06_mBR2_ch05", "d06_mBR2_ch06"),

    ("VE7", "C", "d05_mBR4_ch04", "d05_mBR4_ch05"),
    ("VE8", "C", "d05_mBR4_ch04", "d05_mBR4_ch06"),
    ("VE9", "C", "d05_mBR4_ch05", "d05_mBR4_ch06"),
]


# ======================================================================
# Utility
# ======================================================================
def load_v11():
    path = os.path.abspath(V11_FILE)
    if not os.path.isfile(path):
        raise IOError("V1.1 file not found: %s" % path)

    spec = importlib.util.spec_from_file_location("dance_v11", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_csv_dict(path):
    if not os.path.isfile(path):
        raise IOError("CSV not found: %s" % path)

    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def load_official_filenames():
    if not os.path.isfile(OFFICIAL_CSV):
        return None

    names = set()

    with open(OFFICIAL_CSV, "r", encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            url = line.split(",")[0].strip().strip('"')
            name = os.path.basename(url.split("?")[0])
            if name:
                names.add(name)

    return names


def validate_inputs():
    official = load_official_filenames()

    if official is not None:
        invalid = [
            name for name in VIDEO_KEYS.values()
            if name not in official
        ]

        if invalid:
            print("\n[ERROR] These filenames are not in refined_10M_all_video_url.csv:")
            for name in invalid:
                print(" ", name)
            return False

        print(
            "Official CSV validation: all %d unseen-validation filenames are valid."
            % len(VIDEO_KEYS)
        )
    else:
        print("[Warning] Official refined CSV not found; CSV validation skipped.")

    missing = [
        os.path.join(VIDEO_DIR, name)
        for name in VIDEO_KEYS.values()
        if not os.path.isfile(os.path.join(VIDEO_DIR, name))
    ]

    if missing:
        print("\n[ERROR] Missing local validation videos:")
        for path in missing:
            print(" ", path)
        return False

    print(
        "Local file validation : all %d unseen-validation videos are present."
        % len(VIDEO_KEYS)
    )

    return True


def load_prepare(v11, filename):
    path = os.path.join(VIDEO_DIR, filename)

    kps, ids, fps, total = v11.get_or_extract(
        path, v11.NET_RESOLUTION, v11.FRAME_STEP
    )

    kps_i = v11.fill_missing(kps)
    norm, conf = v11.normalize_frames(kps_i)

    raw_angles = v11.build_joint_angle_sequence(norm, conf)
    smooth_angles = v11.median_smooth_angle_sequence(
        raw_angles, SMOOTH_RADIUS
    )

    return {
        "filename": filename,
        "norm": norm,
        "conf": conf,
        "angles": smooth_angles,
        "sampled_frames": len(norm),
        "original_frames": total,
    }


def path_statistics(path, n_ref, n_test):
    diagonal = 0
    ref_only = 0
    test_only = 0

    for k in range(1, len(path)):
        ia0, ib0 = path[k - 1]
        ia1, ib1 = path[k]

        da = ia1 - ia0
        db = ib1 - ib0

        if da == 1 and db == 1:
            diagonal += 1
        elif da == 1 and db == 0:
            ref_only += 1
        elif da == 0 and db == 1:
            test_only += 1

    relative_dev = []

    for ia, ib in path:
        ta = ia / float(max(n_ref - 1, 1))
        tb = ib / float(max(n_test - 1, 1))
        relative_dev.append(abs(ta - tb))

    steps = max(len(path) - 1, 1)

    return {
        "path_length": len(path),
        "diagonal_steps": diagonal,
        "ref_only_steps": ref_only,
        "test_only_steps": test_only,
        "non_diagonal_ratio": (ref_only + test_only) / float(steps),
        "mean_relative_time_deviation": float(np.mean(relative_dev)),
        "max_relative_time_deviation": float(np.max(relative_dev)),
    }


def describe(values):
    x = np.asarray(values, dtype=np.float64)

    return {
        "count": len(x),
        "mean": float(np.mean(x)),
        "median": float(np.median(x)),
        "min": float(np.min(x)),
        "max": float(np.max(x)),
        "std": float(np.std(x, ddof=1)) if len(x) > 1 else 0.0,
    }


def auc_binary(labels, scores):
    d = [s for y, s in zip(labels, scores) if y == 0]
    e = [s for y, s in zip(labels, scores) if y == 1]

    wins = 0.0
    total = 0

    for se in e:
        for sd in d:
            total += 1
            if se > sd:
                wins += 1.0
            elif se == sd:
                wins += 0.5

    return wins / total if total else float("nan")


def balanced_accuracy(labels, predictions):
    d_correct = []
    e_correct = []

    for y, p in zip(labels, predictions):
        if y == 0:
            d_correct.append(1.0 if p == y else 0.0)
        else:
            e_correct.append(1.0 if p == y else 0.0)

    return 0.5 * (
        float(np.mean(d_correct)) +
        float(np.mean(e_correct))
    )


# ======================================================================
# Learn thresholds ONLY from ch01-ch03 calibration data
# ======================================================================
def get_calibration_thresholds():
    pair_rows = read_csv_dict(CALIBRATION_PAIR_CSV)
    smooth_rows = read_csv_dict(CALIBRATION_SMOOTH_CSV)

    d_pose = [
        float(r["dtw_pose_distance"])
        for r in pair_rows if r["group"] == "D"
    ]
    e_pose = [
        float(r["dtw_pose_distance"])
        for r in pair_rows if r["group"] == "E"
    ]

    # Pose had a clean gap in the calibration set.
    pose_gap = min(e_pose) - max(d_pose)

    if pose_gap > 0:
        pose_threshold = 0.5 * (max(d_pose) + min(e_pose))
        pose_threshold_method = "midpoint of calibration max(D) and min(E)"
    else:
        pose_threshold = 0.5 * (
            np.median(d_pose) + np.median(e_pose)
        )
        pose_threshold_method = "midpoint of calibration class medians"

    d_angle = [
        float(r["smoothed_angle_error_deg"])
        for r in smooth_rows if r["group"] == "D"
    ]
    e_angle = [
        float(r["smoothed_angle_error_deg"])
        for r in smooth_rows if r["group"] == "E"
    ]

    # Angle overlaps, so use class-median midpoint only as comparison diagnostic.
    angle_threshold = 0.5 * (
        np.median(d_angle) + np.median(e_angle)
    )

    d_time = [
        float(r["mean_relative_time_deviation"])
        for r in pair_rows if r["group"] == "D"
    ]
    e_time = [
        float(r["mean_relative_time_deviation"])
        for r in pair_rows if r["group"] == "E"
    ]

    time_threshold = 0.5 * (
        np.median(d_time) + np.median(e_time)
    )

    return {
        "pose_threshold": float(pose_threshold),
        "pose_threshold_method": pose_threshold_method,
        "calibration_pose_gap": float(pose_gap),
        "angle_threshold": float(angle_threshold),
        "time_threshold": float(time_threshold),
        "calibration_D_pose": describe(d_pose),
        "calibration_E_pose": describe(e_pose),
    }


# ======================================================================
# Evaluate one validation pair
# ======================================================================
def evaluate_pair(v11, pair_id, block, group, ref_key, test_key, seq):
    a = seq[ref_key]
    b = seq[test_key]

    cost = v11.build_cost_matrix(
        a["norm"], a["conf"],
        b["norm"], b["conf"]
    )

    path, pose_dist, win = v11.dtw_align(
        cost, v11.DTW_WINDOW_RATIO
    )

    joint_err = v11.per_joint_path_mean_errors(
        a["angles"], b["angles"], path
    )

    angle_error = v11.weighted_overall_from_joint_errors(
        joint_err
    )

    stats = path_statistics(
        path, len(a["norm"]), len(b["norm"])
    )

    row = {
        "pair_id": pair_id,
        "block": block,
        "group": group,
        "reference": a["filename"],
        "test": b["filename"],
        "dtw_pose_distance": float(pose_dist),
        "smoothed_angle_error_deg": float(angle_error),
        "mean_relative_time_deviation":
            stats["mean_relative_time_deviation"],
        "non_diagonal_ratio":
            stats["non_diagonal_ratio"],
        "path_length":
            stats["path_length"],
    }

    print("\n%s [%s / block %s]" % (pair_id, group, block))
    print("  Pose distance          : %.6f" % row["dtw_pose_distance"])
    print("  Smoothed angle error   : %.3f deg" % row["smoothed_angle_error_deg"])
    print("  Mean relative-time dev.: %.6f" % row["mean_relative_time_deviation"])

    return row


# ======================================================================
# Validation report
# ======================================================================
def evaluate_metric(rows, metric, threshold):
    labels = [
        0 if r["group"] == "D" else 1
        for r in rows
    ]
    scores = [r[metric] for r in rows]
    preds = [1 if s >= threshold else 0 for s in scores]

    d = [
        r[metric] for r in rows
        if r["group"] == "D"
    ]
    e = [
        r[metric] for r in rows
        if r["group"] == "E"
    ]

    return {
        "auc": auc_binary(labels, scores),
        "balanced_accuracy":
            balanced_accuracy(labels, preds),
        "gap": min(e) - max(d),
        "D": describe(d),
        "E": describe(e),
    }


def write_csv(path, rows):
    if not rows:
        return

    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        wr = csv.DictWriter(
            f, fieldnames=list(rows[0].keys())
        )
        wr.writeheader()
        wr.writerows(rows)


# ======================================================================
# Main
# ======================================================================
def main():
    v11 = load_v11()
    os.makedirs(OUT_DIR, exist_ok=True)

    print("=" * 90)
    print("Phase 3.4B - Unseen Choreography Holdout Validation")
    print("=" * 90)
    print("Calibration choreography : ch01-ch03")
    print("Validation choreography  : ch04-ch06")
    print("Reserved                 : ch07-ch10")
    print("Overall architecture     : Pose Distance")
    print("Angle feedback           : 3-frame median")
    print("Final 0-100 mapping      : NOT designed yet")

    if not validate_inputs():
        return

    thresholds = get_calibration_thresholds()

    print("\nCalibration-only thresholds")
    print(
        "  Pose threshold  : %.6f (%s)"
        % (
            thresholds["pose_threshold"],
            thresholds["pose_threshold_method"],
        )
    )
    print(
        "  Calibration pose gap min(E)-max(D): %.6f"
        % thresholds["calibration_pose_gap"]
    )
    print(
        "  Angle diagnostic threshold: %.3f deg"
        % thresholds["angle_threshold"]
    )
    print(
        "  Timing diagnostic threshold: %.6f"
        % thresholds["time_threshold"]
    )

    # Load/extract the 18 unseen videos.
    seq = {}

    for i, (key, filename) in enumerate(
        VIDEO_KEYS.items(), start=1
    ):
        print(
            "\n[%d/%d] Loading %s"
            % (i, len(VIDEO_KEYS), filename)
        )
        seq[key] = load_prepare(v11, filename)
        print(
            "  Sampled frames: %d / original %d"
            % (
                seq[key]["sampled_frames"],
                seq[key]["original_frames"],
            )
        )

    pairs = [
        (pid, block, "D", a, b)
        for pid, block, a, b in D_PAIRS
    ]
    pairs += [
        (pid, block, "E", a, b)
        for pid, block, a, b in E_PAIRS
    ]

    rows = []

    for pid, block, group, a, b in pairs:
        rows.append(
            evaluate_pair(
                v11, pid, block, group, a, b, seq
            )
        )

    pose_result = evaluate_metric(
        rows,
        "dtw_pose_distance",
        thresholds["pose_threshold"],
    )

    angle_result = evaluate_metric(
        rows,
        "smoothed_angle_error_deg",
        thresholds["angle_threshold"],
    )

    timing_result = evaluate_metric(
        rows,
        "mean_relative_time_deviation",
        thresholds["time_threshold"],
    )

    print("\n" + "=" * 90)
    print("UNSEEN VALIDATION SUMMARY")
    print("=" * 90)

    print("\nPOSE DISTANCE -- selected overall architecture")
    print(
        "  D: mean %.6f | min %.6f | max %.6f"
        % (
            pose_result["D"]["mean"],
            pose_result["D"]["min"],
            pose_result["D"]["max"],
        )
    )
    print(
        "  E: mean %.6f | min %.6f | max %.6f"
        % (
            pose_result["E"]["mean"],
            pose_result["E"]["min"],
            pose_result["E"]["max"],
        )
    )
    print(
        "  Validation gap min(E)-max(D): %+.6f"
        % pose_result["gap"]
    )
    print(
        "  AUC: %.3f"
        % pose_result["auc"]
    )
    print(
        "  Balanced accuracy with calibration threshold %.6f: %.3f"
        % (
            thresholds["pose_threshold"],
            pose_result["balanced_accuracy"],
        )
    )

    print("\nSMOOTHED ANGLE ERROR -- feedback diagnostic")
    print(
        "  Validation gap min(E)-max(D): %+.3f deg"
        % angle_result["gap"]
    )
    print(
        "  AUC: %.3f | balanced accuracy: %.3f"
        % (
            angle_result["auc"],
            angle_result["balanced_accuracy"],
        )
    )

    print("\nTIMING -- separate diagnostic")
    print(
        "  Validation gap min(E)-max(D): %+.6f"
        % timing_result["gap"]
    )
    print(
        "  AUC: %.3f | balanced accuracy: %.3f"
        % (
            timing_result["auc"],
            timing_result["balanced_accuracy"],
        )
    )

    summary_rows = [
        {
            "metric": "dtw_pose_distance",
            "role": "overall similarity candidate",
            "calibration_threshold":
                thresholds["pose_threshold"],
            "validation_auc":
                pose_result["auc"],
            "validation_balanced_accuracy":
                pose_result["balanced_accuracy"],
            "validation_gap_minE_minus_maxD":
                pose_result["gap"],
            "validation_D_mean":
                pose_result["D"]["mean"],
            "validation_E_mean":
                pose_result["E"]["mean"],
        },
        {
            "metric": "smoothed_angle_error_deg",
            "role": "joint/body-part feedback",
            "calibration_threshold":
                thresholds["angle_threshold"],
            "validation_auc":
                angle_result["auc"],
            "validation_balanced_accuracy":
                angle_result["balanced_accuracy"],
            "validation_gap_minE_minus_maxD":
                angle_result["gap"],
            "validation_D_mean":
                angle_result["D"]["mean"],
            "validation_E_mean":
                angle_result["E"]["mean"],
        },
        {
            "metric": "mean_relative_time_deviation",
            "role": "timing feedback",
            "calibration_threshold":
                thresholds["time_threshold"],
            "validation_auc":
                timing_result["auc"],
            "validation_balanced_accuracy":
                timing_result["balanced_accuracy"],
            "validation_gap_minE_minus_maxD":
                timing_result["gap"],
            "validation_D_mean":
                timing_result["D"]["mean"],
            "validation_E_mean":
                timing_result["E"]["mean"],
        },
    ]

    pair_csv = os.path.join(
        OUT_DIR, "unseen_validation_pair_results.csv"
    )
    summary_csv = os.path.join(
        OUT_DIR, "unseen_validation_summary.csv"
    )

    write_csv(pair_csv, rows)
    write_csv(summary_csv, summary_rows)

    print("\nInterpretation rule:")
    print(
        "  If Pose Distance keeps strong ordering and the calibration threshold transfers well"
    )
    print(
        "  to ch04-ch06, the Pose-only overall architecture is supported on unseen choreography."
    )
    print(
        "  ch07-ch10 remain untouched and can later serve as a final test/demo reserve."
    )

    print("\nSaved:")
    print(" ", pair_csv)
    print(" ", summary_csv)


if __name__ == "__main__":
    main()
