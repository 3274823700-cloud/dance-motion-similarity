# -*- coding: utf-8 -*-
"""
Phase 3.3B - DTW Repeated-Frame Weighting Diagnostic

Purpose
-------
The current V1.1 angle score averages every (reference_frame, test_frame) pair
on the DTW path. When DTW repeats one reference/test frame, that frame can
therefore contribute more than once.

This script keeps the SAME:
- videos
- OpenPose JSON
- normalization
- pose cost
- DTW path
- joint weights

It compares, on the 3-frame-median angle sequences:
1) PATH_WEIGHTED:
   Current behavior. Every DTW path pair has equal weight.
2) REF_BALANCED:
   First average all matches belonging to each reference frame, then average
   reference frames. Every reference frame has equal weight.
3) TEST_BALANCED:
   Same idea from the test-video side.
4) SYMMETRIC_BALANCED:
   Average REF_BALANCED and TEST_BALANCED joint errors.

This is a diagnostic only. It does not change the official V1.1 program.
"""

import os
import csv
import importlib.util
from collections import defaultdict
import numpy as np

V11_FILE = "Dance_Motion_Similarity_Evaluation_V1.1_final.py"
VIDEO_DIR = os.path.join("data", "aistpp", "videos")
OUT_DIR = "phase3_path_weighting_diagnostic"
SMOOTH_RADIUS = 1

VIDEO_KEYS = {
    "d04_mBR0_ch01": "gBR_sBM_c01_d04_mBR0_ch01.mp4",
    "d04_mBR0_ch02": "gBR_sBM_c01_d04_mBR0_ch02.mp4",
    "d04_mBR0_ch03": "gBR_sBM_c01_d04_mBR0_ch03.mp4",
    "d05_mBR0_ch01": "gBR_sBM_c01_d05_mBR0_ch01.mp4",
    "d05_mBR0_ch02": "gBR_sBM_c01_d05_mBR0_ch02.mp4",
    "d05_mBR0_ch03": "gBR_sBM_c01_d05_mBR0_ch03.mp4",

    "d04_mBR2_ch01": "gBR_sBM_c01_d04_mBR2_ch01.mp4",
    "d04_mBR2_ch02": "gBR_sBM_c01_d04_mBR2_ch02.mp4",
    "d04_mBR2_ch03": "gBR_sBM_c01_d04_mBR2_ch03.mp4",
    "d06_mBR2_ch01": "gBR_sBM_c01_d06_mBR2_ch01.mp4",
    "d06_mBR2_ch02": "gBR_sBM_c01_d06_mBR2_ch02.mp4",
    "d06_mBR2_ch03": "gBR_sBM_c01_d06_mBR2_ch03.mp4",

    "d05_mBR4_ch01": "gBR_sBM_c01_d05_mBR4_ch01.mp4",
    "d05_mBR4_ch02": "gBR_sBM_c01_d05_mBR4_ch02.mp4",
    "d05_mBR4_ch03": "gBR_sBM_c01_d05_mBR4_ch03.mp4",
    "d06_mBR4_ch01": "gBR_sBM_c01_d06_mBR4_ch01.mp4",
    "d06_mBR4_ch02": "gBR_sBM_c01_d06_mBR4_ch02.mp4",
    "d06_mBR4_ch03": "gBR_sBM_c01_d06_mBR4_ch03.mp4",
}

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


def load_v11():
    path = os.path.abspath(V11_FILE)
    spec = importlib.util.spec_from_file_location("dance_v11", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_prepare(v11, filename):
    path = os.path.join(VIDEO_DIR, filename)
    if not os.path.isfile(path):
        raise IOError("Video not found: %s" % path)

    kps, ids, fps, total = v11.get_or_extract(
        path, v11.NET_RESOLUTION, v11.FRAME_STEP
    )
    kps_i = v11.fill_missing(kps)
    norm, conf = v11.normalize_frames(kps_i)

    raw = v11.build_joint_angle_sequence(norm, conf)
    smooth = v11.median_smooth_angle_sequence(raw, SMOOTH_RADIUS)

    return {
        "filename": filename,
        "norm": norm,
        "conf": conf,
        "angles": smooth,
    }


def grouped_joint_errors(v11, angles_a, angles_b, path):
    """
    Return 4 per-joint vectors:
      path_weighted, ref_balanced, test_balanced, symmetric_balanced
    """
    J = len(v11.JOINT_DEFS)
    path_weighted = np.full(J, np.nan)
    ref_balanced = np.full(J, np.nan)
    test_balanced = np.full(J, np.nan)
    symmetric_balanced = np.full(J, np.nan)

    for k, (name, vertex, e1, e2, weight) in enumerate(v11.JOINT_DEFS):
        circular = (e2 == -1)

        all_values = []
        by_ref = defaultdict(list)
        by_test = defaultdict(list)

        for ia, ib in path:
            a = angles_a[ia, k]
            b = angles_b[ib, k]

            if not (np.isfinite(a) and np.isfinite(b)):
                continue

            d = v11.ang_diff(a, b, circular)
            all_values.append(d)
            by_ref[ia].append(d)
            by_test[ib].append(d)

        if all_values:
            path_weighted[k] = float(np.mean(all_values))

        if by_ref:
            per_ref = [np.mean(v) for v in by_ref.values()]
            ref_balanced[k] = float(np.mean(per_ref))

        if by_test:
            per_test = [np.mean(v) for v in by_test.values()]
            test_balanced[k] = float(np.mean(per_test))

        if np.isfinite(ref_balanced[k]) and np.isfinite(test_balanced[k]):
            symmetric_balanced[k] = 0.5 * (
                ref_balanced[k] + test_balanced[k]
            )

    return path_weighted, ref_balanced, test_balanced, symmetric_balanced


def describe(x):
    x = np.asarray(x, dtype=np.float64)
    return {
        "mean": float(np.mean(x)),
        "median": float(np.median(x)),
        "min": float(np.min(x)),
        "max": float(np.max(x)),
        "std": float(np.std(x, ddof=1)) if len(x) > 1 else 0.0,
    }


def main():
    v11 = load_v11()
    os.makedirs(OUT_DIR, exist_ok=True)

    print("=" * 88)
    print("Phase 3.3B - DTW Repeated-Frame Weighting Diagnostic")
    print("=" * 88)
    print("Net resolution :", v11.NET_RESOLUTION)
    print("Frame step     :", v11.FRAME_STEP)
    print("Angle smoothing: 3-frame median")
    print("DTW path       : unchanged")
    print("Joint weights  : unchanged")
    print("Legacy score   : NOT USED")

    sequences = {}
    for i, (key, filename) in enumerate(VIDEO_KEYS.items(), start=1):
        print("\n[%d/%d] %s" % (i, len(VIDEO_KEYS), filename))
        sequences[key] = load_prepare(v11, filename)

    pairs = [(pid, "D", a, b) for pid, a, b in D_PAIRS]
    pairs += [(pid, "E", a, b) for pid, a, b in E_PAIRS]

    pair_rows = []
    joint_rows = []

    for pair_id, group, ref_key, test_key in pairs:
        a = sequences[ref_key]
        b = sequences[test_key]

        cost = v11.build_cost_matrix(
            a["norm"], a["conf"], b["norm"], b["conf"]
        )
        path, dtw_dist, win = v11.dtw_align(
            cost, v11.DTW_WINDOW_RATIO
        )

        p, r, t, s = grouped_joint_errors(
            v11, a["angles"], b["angles"], path
        )

        score_p = v11.weighted_overall_from_joint_errors(p)
        score_r = v11.weighted_overall_from_joint_errors(r)
        score_t = v11.weighted_overall_from_joint_errors(t)
        score_s = v11.weighted_overall_from_joint_errors(s)

        unique_ref = len(set(ia for ia, ib in path))
        unique_test = len(set(ib for ia, ib in path))

        row = {
            "pair_id": pair_id,
            "group": group,
            "reference": a["filename"],
            "test": b["filename"],
            "dtw_pose_distance": float(dtw_dist),
            "path_length": len(path),
            "unique_reference_frames": unique_ref,
            "unique_test_frames": unique_test,
            "path_extra_vs_reference": len(path) - unique_ref,
            "path_extra_vs_test": len(path) - unique_test,
            "path_weighted_angle_deg": float(score_p),
            "ref_balanced_angle_deg": float(score_r),
            "test_balanced_angle_deg": float(score_t),
            "symmetric_balanced_angle_deg": float(score_s),
            "symmetric_minus_path_deg": float(score_s - score_p),
        }
        pair_rows.append(row)

        print("\n%s [%s]" % (pair_id, group))
        print("  Path length / unique ref / unique test : %d / %d / %d" % (
            len(path), unique_ref, unique_test
        ))
        print("  Path-weighted       : %.3f deg" % score_p)
        print("  Reference-balanced  : %.3f deg" % score_r)
        print("  Test-balanced       : %.3f deg" % score_t)
        print("  Symmetric-balanced  : %.3f deg" % score_s)
        print("  Symmetric - current : %+.3f deg" % (score_s - score_p))

        for k, (name, vertex, e1, e2, weight) in enumerate(v11.JOINT_DEFS):
            joint_rows.append({
                "pair_id": pair_id,
                "group": group,
                "joint": name,
                "weight": weight,
                "path_weighted_deg": p[k],
                "ref_balanced_deg": r[k],
                "test_balanced_deg": t[k],
                "symmetric_balanced_deg": s[k],
                "symmetric_minus_path_deg": s[k] - p[k],
            })

    # Pair results
    pair_csv = os.path.join(OUT_DIR, "path_weighting_pair_results.csv")
    with open(pair_csv, "w", newline="", encoding="utf-8-sig") as f:
        wr = csv.DictWriter(f, fieldnames=list(pair_rows[0].keys()))
        wr.writeheader()
        wr.writerows(pair_rows)

    # Joint results
    joint_csv = os.path.join(OUT_DIR, "path_weighting_joint_results.csv")
    with open(joint_csv, "w", newline="", encoding="utf-8-sig") as f:
        wr = csv.DictWriter(f, fieldnames=list(joint_rows[0].keys()))
        wr.writeheader()
        wr.writerows(joint_rows)

    # Group summary
    summary_csv = os.path.join(OUT_DIR, "path_weighting_group_statistics.csv")
    metrics = [
        "path_weighted_angle_deg",
        "ref_balanced_angle_deg",
        "test_balanced_angle_deg",
        "symmetric_balanced_angle_deg",
        "symmetric_minus_path_deg",
    ]

    with open(summary_csv, "w", newline="", encoding="utf-8-sig") as f:
        wr = csv.writer(f)
        wr.writerow([
            "Group", "Metric", "Count",
            "Mean", "Median", "Min", "Max", "Std"
        ])

        for group in ("D", "E"):
            rows = [r for r in pair_rows if r["group"] == group]
            for metric in metrics:
                vals = [r[metric] for r in rows]
                st = describe(vals)
                wr.writerow([
                    group, metric, len(vals),
                    "%.6f" % st["mean"],
                    "%.6f" % st["median"],
                    "%.6f" % st["min"],
                    "%.6f" % st["max"],
                    "%.6f" % st["std"],
                ])

    print("\n" + "=" * 88)
    print("GROUP SUMMARY")
    print("=" * 88)

    for group in ("D", "E"):
        rows = [r for r in pair_rows if r["group"] == group]
        cur = describe([r["path_weighted_angle_deg"] for r in rows])
        sym = describe([r["symmetric_balanced_angle_deg"] for r in rows])

        print("\nGroup %s" % group)
        print("  Current path-weighted : mean %.2f | min %.2f | max %.2f | std %.2f" % (
            cur["mean"], cur["min"], cur["max"], cur["std"]
        ))
        print("  Symmetric-balanced    : mean %.2f | min %.2f | max %.2f | std %.2f" % (
            sym["mean"], sym["min"], sym["max"], sym["std"]
        ))

    d_cur = [r["path_weighted_angle_deg"] for r in pair_rows if r["group"] == "D"]
    e_cur = [r["path_weighted_angle_deg"] for r in pair_rows if r["group"] == "E"]
    d_sym = [r["symmetric_balanced_angle_deg"] for r in pair_rows if r["group"] == "D"]
    e_sym = [r["symmetric_balanced_angle_deg"] for r in pair_rows if r["group"] == "E"]

    print("\nD/E separation")
    print("  Current path-weighted gap : %.3f deg" % (min(e_cur) - max(d_cur)))
    print("  Symmetric-balanced gap    : %.3f deg" % (min(e_sym) - max(d_sym)))
    print("  Positive = complete separation; negative = overlap.")
    print("\nInterpretation:")
    print("  Small symmetric-minus-path changes -> repeated DTW pairs have little scoring effect.")
    print("  Large/systematic changes -> current path-pair averaging materially reweights some frames.")

    print("\nSaved:")
    print(" ", pair_csv)
    print(" ", joint_csv)
    print(" ", summary_csv)


if __name__ == "__main__":
    main()
