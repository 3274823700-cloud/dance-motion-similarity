# -*- coding: utf-8 -*-
"""
Phase 3.2C - Small Distribution Calibration (CSV-validated version)

All selected videos are chosen from refined_10M_all_video_url.csv.

D group = different dancer + same music + same choreography
E group = same dancer + same music + different choreography

No legacy 0-100 score, no smoothing, no joint-weight change, no DTW rewrite.
All core algorithms are imported from Dance_Motion_Similarity_Evaluation_V1.1_final.py.
"""

import os
import csv
import time
import importlib.util
import numpy as np


# ======================================================================
# Configuration
# ======================================================================
V11_FILE = "Dance_Motion_Similarity_Evaluation_V1.1_final.py"
VIDEO_DIR = os.path.join("data", "aistpp", "videos")
OFFICIAL_CSV = os.path.join(VIDEO_DIR, "refined_10M_all_video_url.csv")
OUT_DIR = "phase3_calibration_distribution"

# ----------------------------------------------------------------------
# Explicitly selected VALID videos from refined_10M_all_video_url.csv.
#
# Three matched dancer-pair blocks:
#   Block A: d04 vs d05, music mBR0
#   Block B: d04 vs d06, music mBR2
#   Block C: d05 vs d06, music mBR4
#
# For each block use ch01/ch02/ch03.
# This avoids assuming that one Music ID exists for all three dancers.
# ----------------------------------------------------------------------
VIDEO_KEYS = {
    # Block A: d04 / d05, mBR0
    "d04_mBR0_ch01": "gBR_sBM_c01_d04_mBR0_ch01.mp4",
    "d04_mBR0_ch02": "gBR_sBM_c01_d04_mBR0_ch02.mp4",
    "d04_mBR0_ch03": "gBR_sBM_c01_d04_mBR0_ch03.mp4",
    "d05_mBR0_ch01": "gBR_sBM_c01_d05_mBR0_ch01.mp4",
    "d05_mBR0_ch02": "gBR_sBM_c01_d05_mBR0_ch02.mp4",
    "d05_mBR0_ch03": "gBR_sBM_c01_d05_mBR0_ch03.mp4",

    # Block B: d04 / d06, mBR2
    "d04_mBR2_ch01": "gBR_sBM_c01_d04_mBR2_ch01.mp4",
    "d04_mBR2_ch02": "gBR_sBM_c01_d04_mBR2_ch02.mp4",
    "d04_mBR2_ch03": "gBR_sBM_c01_d04_mBR2_ch03.mp4",
    "d06_mBR2_ch01": "gBR_sBM_c01_d06_mBR2_ch01.mp4",
    "d06_mBR2_ch02": "gBR_sBM_c01_d06_mBR2_ch02.mp4",
    "d06_mBR2_ch03": "gBR_sBM_c01_d06_mBR2_ch03.mp4",

    # Block C: d05 / d06, mBR4
    "d05_mBR4_ch01": "gBR_sBM_c01_d05_mBR4_ch01.mp4",
    "d05_mBR4_ch02": "gBR_sBM_c01_d05_mBR4_ch02.mp4",
    "d05_mBR4_ch03": "gBR_sBM_c01_d05_mBR4_ch03.mp4",
    "d06_mBR4_ch01": "gBR_sBM_c01_d06_mBR4_ch01.mp4",
    "d06_mBR4_ch02": "gBR_sBM_c01_d06_mBR4_ch02.mp4",
    "d06_mBR4_ch03": "gBR_sBM_c01_d06_mBR4_ch03.mp4",
}

# D: only dancer changes.
D_PAIRS = [
    ("D1", "d04_mBR0_ch01", "d05_mBR0_ch01"),
    ("D2", "d04_mBR0_ch02", "d05_mBR0_ch02"),
    ("D3", "d04_mBR0_ch03", "d05_mBR0_ch03"),

    ("D4", "d04_mBR2_ch01", "d06_mBR2_ch01"),
    ("D5", "d04_mBR2_ch02", "d06_mBR2_ch02"),
    ("D6", "d04_mBR2_ch03", "d06_mBR2_ch03"),

    ("D7", "d05_mBR4_ch01", "d06_mBR4_ch01"),
    ("D8", "d05_mBR4_ch02", "d06_mBR4_ch02"),
    ("D9", "d05_mBR4_ch03", "d06_mBR4_ch03"),
]

# E: only choreography changes.
# Music distribution is also mBR0 / mBR2 / mBR4, matching the D group.
E_PAIRS = [
    ("E1", "d04_mBR0_ch01", "d04_mBR0_ch02"),
    ("E2", "d04_mBR0_ch01", "d04_mBR0_ch03"),
    ("E3", "d04_mBR0_ch02", "d04_mBR0_ch03"),

    ("E4", "d06_mBR2_ch01", "d06_mBR2_ch02"),
    ("E5", "d06_mBR2_ch01", "d06_mBR2_ch03"),
    ("E6", "d06_mBR2_ch02", "d06_mBR2_ch03"),

    ("E7", "d05_mBR4_ch01", "d05_mBR4_ch02"),
    ("E8", "d05_mBR4_ch01", "d05_mBR4_ch03"),
    ("E9", "d05_mBR4_ch02", "d05_mBR4_ch03"),
]


# ======================================================================
# Load official V1.1
# ======================================================================
def load_v11():
    path = os.path.abspath(V11_FILE)
    if not os.path.isfile(path):
        raise IOError("V1.1 file not found: %s" % path)

    spec = importlib.util.spec_from_file_location("dance_v11", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ======================================================================
# Validate selected filenames against official refined CSV
# ======================================================================
def load_official_filenames(csv_path):
    if not os.path.isfile(csv_path):
        return None

    names = set()
    with open(csv_path, "r", encoding="utf-8-sig") as f:
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
    official = load_official_filenames(OFFICIAL_CSV)

    if official is None:
        print("[Warning] Official CSV not found: %s" % OFFICIAL_CSV)
        print("          Filename validation against CSV will be skipped.")
    else:
        invalid = [name for name in VIDEO_KEYS.values() if name not in official]
        if invalid:
            print("\n[ERROR] The following selected filenames are NOT in refined_10M_all_video_url.csv:")
            for name in invalid:
                print(" ", name)
            return False
        print("Official CSV validation: all %d selected filenames are valid." % len(VIDEO_KEYS))

    missing = []
    for name in VIDEO_KEYS.values():
        path = os.path.join(VIDEO_DIR, name)
        if not os.path.isfile(path):
            missing.append(path)

    if missing:
        print("\n[ERROR] Required local videos are missing:")
        for path in missing:
            print(" ", path)
        return False

    print("Local file validation : all %d selected videos are present." % len(VIDEO_KEYS))
    return True


# ======================================================================
# Sequence preparation
# ======================================================================
def load_prepare(v11, filename):
    path = os.path.join(VIDEO_DIR, filename)
    kps, ids, fps, total = v11.get_or_extract(
        path, v11.NET_RESOLUTION, v11.FRAME_STEP
    )
    kps_i = v11.fill_missing(kps)
    norm, conf = v11.normalize_frames(kps_i)

    return {
        "filename": filename,
        "kps": kps,
        "ids": ids,
        "fps": fps,
        "total": total,
        "norm": norm,
        "conf": conf,
    }


# ======================================================================
# DTW path diagnostics
# ======================================================================
def path_statistics(path, n_ref, n_test):
    diagonal = 0
    ref_only = 0
    test_only = 0

    for k in range(1, len(path)):
        ia0, ib0 = path[k - 1]
        ia1, ib1 = path[k]
        da, db = ia1 - ia0, ib1 - ib0

        if da == 1 and db == 1:
            diagonal += 1
        elif da == 1 and db == 0:
            ref_only += 1
        elif da == 0 and db == 1:
            test_only += 1

    steps = max(len(path) - 1, 1)

    relative_dev = []
    for ia, ib in path:
        ta = ia / float(max(n_ref - 1, 1))
        tb = ib / float(max(n_test - 1, 1))
        relative_dev.append(abs(ta - tb))

    return {
        "path_length": len(path),
        "diagonal_steps": diagonal,
        "ref_only_steps": ref_only,
        "test_only_steps": test_only,
        "non_diagonal_ratio": (ref_only + test_only) / float(steps),
        "mean_relative_time_deviation": float(np.mean(relative_dev)),
        "max_relative_time_deviation": float(np.max(relative_dev)),
    }


# ======================================================================
# Evaluate one pair
# ======================================================================
def evaluate_pair(v11, pair_id, group, ref_key, test_key, ref, test):
    t0 = time.time()

    cost = v11.build_cost_matrix(
        ref["norm"], ref["conf"], test["norm"], test["conf"]
    )
    path, dtw_dist, win = v11.dtw_align(cost, v11.DTW_WINDOW_RATIO)

    overall_err, joint_err, coverage, detail = v11.evaluate_alignment(
        ref["norm"], ref["conf"], test["norm"], test["conf"], path
    )

    stats = path_statistics(path, len(ref["norm"]), len(test["norm"]))

    row = {
        "pair_id": pair_id,
        "group": group,
        "reference_key": ref_key,
        "test_key": test_key,
        "reference_file": ref["filename"],
        "test_file": test["filename"],
        "reference_frames": len(ref["norm"]),
        "test_frames": len(test["norm"]),
        "dtw_pose_distance": float(dtw_dist),
        "weighted_angle_error_deg": float(overall_err),
        "mean_joint_coverage": float(np.mean(coverage)),
        "min_joint_coverage": float(np.min(coverage)),
        "dtw_window": win,
        "runtime_sec": time.time() - t0,
    }
    row.update(stats)

    print("\n%s [%s]" % (pair_id, group))
    print("  %s" % ref["filename"])
    print("  vs")
    print("  %s" % test["filename"])
    print("  DTW pose distance         : %.6f" % dtw_dist)
    print("  Weighted angle error      : %.3f deg" % overall_err)
    print("  Non-diagonal DTW ratio    : %.2f%%" % (100.0 * stats["non_diagonal_ratio"]))
    print("  Mean relative-time dev.   : %.6f" % stats["mean_relative_time_deviation"])

    return row, joint_err, coverage


# ======================================================================
# Statistics / outputs
# ======================================================================
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


def save_pair_summary(rows, path):
    fields = [
        "pair_id", "group",
        "reference_key", "test_key",
        "reference_file", "test_file",
        "reference_frames", "test_frames",
        "dtw_pose_distance", "weighted_angle_error_deg",
        "mean_joint_coverage", "min_joint_coverage",
        "path_length", "diagonal_steps", "ref_only_steps", "test_only_steps",
        "non_diagonal_ratio",
        "mean_relative_time_deviation", "max_relative_time_deviation",
        "dtw_window", "runtime_sec",
    ]

    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        wr = csv.DictWriter(f, fieldnames=fields)
        wr.writeheader()
        wr.writerows(rows)


def save_per_joint(v11, items, path):
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        wr = csv.writer(f)
        wr.writerow([
            "Pair ID", "Group", "Reference", "Test",
            "Joint", "Angle Error (deg)", "Coverage", "Weight"
        ])

        for row, joint_err, coverage in items:
            for k, (name, v, e1, e2, weight) in enumerate(v11.JOINT_DEFS):
                wr.writerow([
                    row["pair_id"], row["group"],
                    row["reference_file"], row["test_file"],
                    name,
                    "" if not np.isfinite(joint_err[k]) else "%.6f" % joint_err[k],
                    "%.6f" % coverage[k],
                    weight,
                ])


def save_group_statistics(rows, path):
    metrics = [
        "dtw_pose_distance",
        "weighted_angle_error_deg",
        "non_diagonal_ratio",
        "mean_relative_time_deviation",
    ]

    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        wr = csv.writer(f)
        wr.writerow(["Group", "Metric", "Count", "Mean", "Median", "Min", "Max", "Std"])

        for group in ("D", "E"):
            group_rows = [r for r in rows if r["group"] == group]
            for metric in metrics:
                s = describe([r[metric] for r in group_rows])
                wr.writerow([
                    group, metric, s["count"],
                    "%.6f" % s["mean"], "%.6f" % s["median"],
                    "%.6f" % s["min"], "%.6f" % s["max"],
                    "%.6f" % s["std"],
                ])


def save_joint_group_statistics(v11, items, path):
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        wr = csv.writer(f)
        wr.writerow([
            "Group", "Joint", "Count",
            "Mean Error (deg)", "Median Error (deg)",
            "Min Error (deg)", "Max Error (deg)", "Std Error (deg)"
        ])

        for group in ("D", "E"):
            group_items = [item for item in items if item[0]["group"] == group]

            for k, (name, v, e1, e2, weight) in enumerate(v11.JOINT_DEFS):
                values = [
                    item[1][k] for item in group_items
                    if np.isfinite(item[1][k])
                ]
                if not values:
                    continue

                s = describe(values)
                wr.writerow([
                    group, name, s["count"],
                    "%.6f" % s["mean"], "%.6f" % s["median"],
                    "%.6f" % s["min"], "%.6f" % s["max"],
                    "%.6f" % s["std"],
                ])


def print_summary(rows):
    print("\n" + "=" * 86)
    print("D / E DISTRIBUTION SUMMARY")
    print("=" * 86)

    for group in ("D", "E"):
        group_rows = [r for r in rows if r["group"] == group]
        pose = describe([r["dtw_pose_distance"] for r in group_rows])
        angle = describe([r["weighted_angle_error_deg"] for r in group_rows])

        print("\nGroup %s (%d pairs)" % (group, len(group_rows)))
        print("  Pose distance : mean %.4f | median %.4f | min %.4f | max %.4f | std %.4f" % (
            pose["mean"], pose["median"], pose["min"], pose["max"], pose["std"]
        ))
        print("  Angle error   : mean %.2f | median %.2f | min %.2f | max %.2f | std %.2f deg" % (
            angle["mean"], angle["median"], angle["min"], angle["max"], angle["std"]
        ))

    d = [r for r in rows if r["group"] == "D"]
    e = [r for r in rows if r["group"] == "E"]

    angle_gap = min(r["weighted_angle_error_deg"] for r in e) - max(r["weighted_angle_error_deg"] for r in d)
    pose_gap = min(r["dtw_pose_distance"] for r in e) - max(r["dtw_pose_distance"] for r in d)

    print("\nObserved D/E separation")
    print("  Angle gap = min(E) - max(D) = %.3f deg" % angle_gap)
    print("  Pose gap  = min(E) - max(D) = %.6f" % pose_gap)
    print("  Positive gap = complete separation in this sample; negative gap = overlap.")
    print("  This is calibration evidence, NOT a final threshold.")


# ======================================================================
# Main
# ======================================================================
def main():
    v11 = load_v11()
    os.makedirs(OUT_DIR, exist_ok=True)

    print("=" * 86)
    print("Phase 3.2C - Small Distribution Calibration (CSV-validated)")
    print("=" * 86)
    print("V1.1 file       :", os.path.abspath(V11_FILE))
    print("Video directory :", os.path.abspath(VIDEO_DIR))
    print("Official CSV    :", os.path.abspath(OFFICIAL_CSV))
    print("Net resolution  :", v11.NET_RESOLUTION)
    print("Frame step      :", v11.FRAME_STEP)
    print("DTW window ratio:", v11.DTW_WINDOW_RATIO)
    print("Legacy score    : NOT USED")

    if not validate_inputs():
        return

    # Load/extract each selected video once.
    sequences = {}
    total = len(VIDEO_KEYS)

    for idx, (key, filename) in enumerate(VIDEO_KEYS.items(), start=1):
        print("\n[%d/%d] Loading %s" % (idx, total, filename))
        sequences[key] = load_prepare(v11, filename)
        print("  Sampled frames: %d / original %d" % (
            len(sequences[key]["norm"]), sequences[key]["total"]
        ))

    pairs = [(pid, "D", a, b) for pid, a, b in D_PAIRS]
    pairs += [(pid, "E", a, b) for pid, a, b in E_PAIRS]

    print("\nPair matrix: %d D + %d E = %d total pairs" % (
        len(D_PAIRS), len(E_PAIRS), len(pairs)
    ))

    rows = []
    joint_items = []

    for pair_id, group, ref_key, test_key in pairs:
        row, joint_err, coverage = evaluate_pair(
            v11, pair_id, group, ref_key, test_key,
            sequences[ref_key], sequences[test_key]
        )
        rows.append(row)
        joint_items.append((row, joint_err, coverage))

    pair_csv = os.path.join(OUT_DIR, "pair_summary.csv")
    joint_csv = os.path.join(OUT_DIR, "per_joint_results.csv")
    group_csv = os.path.join(OUT_DIR, "group_statistics.csv")
    joint_group_csv = os.path.join(OUT_DIR, "joint_group_statistics.csv")

    save_pair_summary(rows, pair_csv)
    save_per_joint(v11, joint_items, joint_csv)
    save_group_statistics(rows, group_csv)
    save_joint_group_statistics(v11, joint_items, joint_group_csv)

    print_summary(rows)

    print("\nSaved:")
    print(" ", pair_csv)
    print(" ", joint_csv)
    print(" ", group_csv)
    print(" ", joint_group_csv)


if __name__ == "__main__":
    main()
