# -*- coding: utf-8 -*-
"""
Phase 3.3A - Smoothing A/B Diagnostic

Goal:
Compare RAW joint-angle scoring vs 3-frame median-smoothed joint-angle scoring
on the SAME 18 D/E calibration pairs and the SAME DTW paths.

Important:
- DTW cost/path is NOT changed.
- Joint weights are NOT changed.
- Legacy 0-100 score is NOT used.
- Smoothing is diagnostic only.
- All core functions are reused from Dance_Motion_Similarity_Evaluation_V1.1_final.py.
"""

import os
import csv
import importlib.util
import numpy as np

V11_FILE = "Dance_Motion_Similarity_Evaluation_V1.1_final.py"
VIDEO_DIR = os.path.join("data", "aistpp", "videos")
OUT_DIR = "phase3_smoothing_diagnostic"
SMOOTH_RADIUS = 1  # radius=1 -> 3-frame median

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

    angles_raw = v11.build_joint_angle_sequence(norm, conf)
    angles_smooth = v11.median_smooth_angle_sequence(
        angles_raw, SMOOTH_RADIUS
    )

    return {
        "filename": filename,
        "norm": norm,
        "conf": conf,
        "angles_raw": angles_raw,
        "angles_smooth": angles_smooth,
    }


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


def main():
    v11 = load_v11()
    os.makedirs(OUT_DIR, exist_ok=True)

    print("=" * 86)
    print("Phase 3.3A - Smoothing A/B Diagnostic")
    print("=" * 86)
    print("Net resolution :", v11.NET_RESOLUTION)
    print("Frame step     :", v11.FRAME_STEP)
    print("Smooth radius  :", SMOOTH_RADIUS, "(3-frame median)")
    print("DTW path       : RAW normalized-coordinate Pose Cost; fixed for raw/smoothed scoring")
    print("Legacy score   : NOT USED")

    # Load all 18 unique videos once. Existing JSON cache should be reused.
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

        # DTW is computed exactly as before, from raw normalized coordinates.
        cost = v11.build_cost_matrix(a["norm"], a["conf"], b["norm"], b["conf"])
        path, dtw_dist, win = v11.dtw_align(cost, v11.DTW_WINDOW_RATIO)

        raw_joint = v11.per_joint_path_mean_errors(
            a["angles_raw"], b["angles_raw"], path
        )
        smooth_joint = v11.per_joint_path_mean_errors(
            a["angles_smooth"], b["angles_smooth"], path
        )

        raw_overall = v11.weighted_overall_from_joint_errors(raw_joint)
        smooth_overall = v11.weighted_overall_from_joint_errors(smooth_joint)

        row = {
            "pair_id": pair_id,
            "group": group,
            "reference": a["filename"],
            "test": b["filename"],
            "dtw_pose_distance": float(dtw_dist),
            "raw_angle_error_deg": float(raw_overall),
            "smoothed_angle_error_deg": float(smooth_overall),
            "change_deg": float(smooth_overall - raw_overall),
            "change_percent": float(
                100.0 * (smooth_overall - raw_overall) / raw_overall
            ) if raw_overall > 1e-12 else 0.0,
            "path_length": len(path),
        }
        pair_rows.append(row)

        print("\n%s [%s]" % (pair_id, group))
        print("  Raw      : %.3f deg" % raw_overall)
        print("  Smoothed : %.3f deg" % smooth_overall)
        print("  Change   : %+.3f deg (%+.2f%%)" % (
            row["change_deg"], row["change_percent"]
        ))

        for k, (name, v, e1, e2, weight) in enumerate(v11.JOINT_DEFS):
            joint_rows.append({
                "pair_id": pair_id,
                "group": group,
                "joint": name,
                "weight": weight,
                "raw_error_deg": raw_joint[k],
                "smoothed_error_deg": smooth_joint[k],
                "change_deg": smooth_joint[k] - raw_joint[k],
            })

    # Pair-level CSV
    pair_path = os.path.join(OUT_DIR, "smoothing_pair_results.csv")
    with open(pair_path, "w", newline="", encoding="utf-8-sig") as f:
        wr = csv.DictWriter(f, fieldnames=list(pair_rows[0].keys()))
        wr.writeheader()
        wr.writerows(pair_rows)

    # Joint-level CSV
    joint_path = os.path.join(OUT_DIR, "smoothing_joint_results.csv")
    with open(joint_path, "w", newline="", encoding="utf-8-sig") as f:
        wr = csv.DictWriter(f, fieldnames=list(joint_rows[0].keys()))
        wr.writeheader()
        wr.writerows(joint_rows)

    # Group summary
    group_path = os.path.join(OUT_DIR, "smoothing_group_statistics.csv")
    with open(group_path, "w", newline="", encoding="utf-8-sig") as f:
        wr = csv.writer(f)
        wr.writerow([
            "Group", "Metric", "Count",
            "Mean", "Median", "Min", "Max", "Std"
        ])

        for group in ("D", "E"):
            rows = [r for r in pair_rows if r["group"] == group]
            for metric in (
                "raw_angle_error_deg",
                "smoothed_angle_error_deg",
                "change_deg",
                "change_percent",
            ):
                s = summarize([r[metric] for r in rows])
                wr.writerow([
                    group, metric, s["count"],
                    "%.6f" % s["mean"],
                    "%.6f" % s["median"],
                    "%.6f" % s["min"],
                    "%.6f" % s["max"],
                    "%.6f" % s["std"],
                ])

    # Per-joint group summary
    joint_group_path = os.path.join(
        OUT_DIR, "smoothing_joint_group_statistics.csv"
    )
    with open(joint_group_path, "w", newline="", encoding="utf-8-sig") as f:
        wr = csv.writer(f)
        wr.writerow([
            "Group", "Joint", "Count",
            "Raw Mean", "Smoothed Mean",
            "Mean Change", "Raw Std", "Smoothed Std"
        ])

        for group in ("D", "E"):
            for name, v, e1, e2, weight in v11.JOINT_DEFS:
                rows = [
                    r for r in joint_rows
                    if r["group"] == group and r["joint"] == name
                ]

                raw = np.asarray(
                    [r["raw_error_deg"] for r in rows if np.isfinite(r["raw_error_deg"])],
                    dtype=np.float64,
                )
                smooth = np.asarray(
                    [r["smoothed_error_deg"] for r in rows if np.isfinite(r["smoothed_error_deg"])],
                    dtype=np.float64,
                )

                wr.writerow([
                    group, name, len(raw),
                    "%.6f" % np.mean(raw),
                    "%.6f" % np.mean(smooth),
                    "%.6f" % (np.mean(smooth) - np.mean(raw)),
                    "%.6f" % (np.std(raw, ddof=1) if len(raw) > 1 else 0.0),
                    "%.6f" % (np.std(smooth, ddof=1) if len(smooth) > 1 else 0.0),
                ])

    # Console summary
    print("\n" + "=" * 86)
    print("GROUP SUMMARY")
    print("=" * 86)

    for group in ("D", "E"):
        rows = [r for r in pair_rows if r["group"] == group]
        raw = summarize([r["raw_angle_error_deg"] for r in rows])
        smooth = summarize([r["smoothed_angle_error_deg"] for r in rows])

        print("\nGroup %s" % group)
        print("  Raw      : mean %.2f | median %.2f | min %.2f | max %.2f | std %.2f" % (
            raw["mean"], raw["median"], raw["min"], raw["max"], raw["std"]
        ))
        print("  Smoothed : mean %.2f | median %.2f | min %.2f | max %.2f | std %.2f" % (
            smooth["mean"], smooth["median"], smooth["min"], smooth["max"], smooth["std"]
        ))

    d_raw = [r["raw_angle_error_deg"] for r in pair_rows if r["group"] == "D"]
    e_raw = [r["raw_angle_error_deg"] for r in pair_rows if r["group"] == "E"]
    d_sm = [r["smoothed_angle_error_deg"] for r in pair_rows if r["group"] == "D"]
    e_sm = [r["smoothed_angle_error_deg"] for r in pair_rows if r["group"] == "E"]

    print("\nD/E angle separation")
    print("  Raw gap      min(E)-max(D): %.3f deg" % (min(e_raw) - max(d_raw)))
    print("  Smoothed gap min(E)-max(D): %.3f deg" % (min(e_sm) - max(d_sm)))
    print("  Positive = complete separation; negative = overlap.")
    print("  Do not choose smoothing only because the gap becomes larger.")
    print("  Also inspect whether D variability and known jitter-sensitive joints improve.")

    print("\nSaved:")
    print(" ", pair_path)
    print(" ", joint_path)
    print(" ", group_path)
    print(" ", joint_group_path)


if __name__ == "__main__":
    main()
