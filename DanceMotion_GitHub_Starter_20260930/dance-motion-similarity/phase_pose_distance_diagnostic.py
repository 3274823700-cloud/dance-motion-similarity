# -*- coding: utf-8 -*-
"""
Phase 3.4C - Pose Distance Failure Diagnostic

Why this script exists
----------------------
Unseen ch04-ch06 validation showed that absolute DTW Pose Distance did not
generalize as an overall similarity score. Several ch05 comparisons produced
very large pose distances even though smoothed angle error and timing alignment
were not correspondingly extreme.

This script does NOT change the algorithm. It diagnoses whether the failure is:
1) sparse local-cost outliers,
2) a systematic high pose cost across the whole DTW path,
3) unstable torso-length normalization / normalized-coordinate magnitude.

It reuses the existing 432x240 OpenPose JSON caches.

Outputs
-------
phase3_pose_distance_diagnostic/
    video_normalization_stats.csv
    pair_pose_cost_stats.csv
"""

import os
import csv
import importlib.util
import numpy as np


V11_FILE = "Dance_Motion_Similarity_Evaluation_V1.1_final.py"
VIDEO_DIR = os.path.join("data", "aistpp", "videos")
OUT_DIR = "phase3_pose_distance_diagnostic"


VIDEO_KEYS = {
    # Block A
    "d04_mBR0_ch04": "gBR_sBM_c01_d04_mBR0_ch04.mp4",
    "d04_mBR0_ch05": "gBR_sBM_c01_d04_mBR0_ch05.mp4",
    "d04_mBR0_ch06": "gBR_sBM_c01_d04_mBR0_ch06.mp4",
    "d05_mBR0_ch04": "gBR_sBM_c01_d05_mBR0_ch04.mp4",
    "d05_mBR0_ch05": "gBR_sBM_c01_d05_mBR0_ch05.mp4",
    "d05_mBR0_ch06": "gBR_sBM_c01_d05_mBR0_ch06.mp4",

    # Block B
    "d04_mBR2_ch04": "gBR_sBM_c01_d04_mBR2_ch04.mp4",
    "d04_mBR2_ch05": "gBR_sBM_c01_d04_mBR2_ch05.mp4",
    "d04_mBR2_ch06": "gBR_sBM_c01_d04_mBR2_ch06.mp4",
    "d06_mBR2_ch04": "gBR_sBM_c01_d06_mBR2_ch04.mp4",
    "d06_mBR2_ch05": "gBR_sBM_c01_d06_mBR2_ch05.mp4",
    "d06_mBR2_ch06": "gBR_sBM_c01_d06_mBR2_ch06.mp4",

    # Block C
    "d05_mBR4_ch04": "gBR_sBM_c01_d05_mBR4_ch04.mp4",
    "d05_mBR4_ch05": "gBR_sBM_c01_d05_mBR4_ch05.mp4",
    "d05_mBR4_ch06": "gBR_sBM_c01_d05_mBR4_ch06.mp4",
    "d06_mBR4_ch04": "gBR_sBM_c01_d06_mBR4_ch04.mp4",
    "d06_mBR4_ch05": "gBR_sBM_c01_d06_mBR4_ch05.mp4",
    "d06_mBR4_ch06": "gBR_sBM_c01_d06_mBR4_ch06.mp4",
}

D_PAIRS = [
    ("VD1", "d04_mBR0_ch04", "d05_mBR0_ch04"),
    ("VD2", "d04_mBR0_ch05", "d05_mBR0_ch05"),
    ("VD3", "d04_mBR0_ch06", "d05_mBR0_ch06"),
    ("VD4", "d04_mBR2_ch04", "d06_mBR2_ch04"),
    ("VD5", "d04_mBR2_ch05", "d06_mBR2_ch05"),
    ("VD6", "d04_mBR2_ch06", "d06_mBR2_ch06"),
    ("VD7", "d05_mBR4_ch04", "d06_mBR4_ch04"),
    ("VD8", "d05_mBR4_ch05", "d06_mBR4_ch05"),
    ("VD9", "d05_mBR4_ch06", "d06_mBR4_ch06"),
]

E_PAIRS = [
    ("VE1", "d04_mBR0_ch04", "d04_mBR0_ch05"),
    ("VE2", "d04_mBR0_ch04", "d04_mBR0_ch06"),
    ("VE3", "d04_mBR0_ch05", "d04_mBR0_ch06"),
    ("VE4", "d06_mBR2_ch04", "d06_mBR2_ch05"),
    ("VE5", "d06_mBR2_ch04", "d06_mBR2_ch06"),
    ("VE6", "d06_mBR2_ch05", "d06_mBR2_ch06"),
    ("VE7", "d05_mBR4_ch04", "d05_mBR4_ch05"),
    ("VE8", "d05_mBR4_ch04", "d05_mBR4_ch06"),
    ("VE9", "d05_mBR4_ch05", "d05_mBR4_ch06"),
]


def load_v11():
    path = os.path.abspath(V11_FILE)
    spec = importlib.util.spec_from_file_location("dance_v11", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def safe_stats(x):
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]

    if len(x) == 0:
        return {
            "count": 0,
            "mean": np.nan,
            "median": np.nan,
            "p05": np.nan,
            "p10": np.nan,
            "p90": np.nan,
            "p95": np.nan,
            "min": np.nan,
            "max": np.nan,
        }

    return {
        "count": len(x),
        "mean": float(np.mean(x)),
        "median": float(np.median(x)),
        "p05": float(np.percentile(x, 5)),
        "p10": float(np.percentile(x, 10)),
        "p90": float(np.percentile(x, 90)),
        "p95": float(np.percentile(x, 95)),
        "min": float(np.min(x)),
        "max": float(np.max(x)),
    }


def frame_torso_lengths(v11, kps):
    """Reproduce the torso scale used by normalize_frames, but keep pixel units."""
    conf = kps[:, :, 2]
    xy = kps[:, :, :2]
    values = np.full(len(kps), np.nan, dtype=np.float64)

    for i in range(len(kps)):
        if conf[i, v11.MID_HIP] > v11.CONF_THRESHOLD:
            hip = xy[i, v11.MID_HIP]
        elif (
            conf[i, v11.R_HIP] > v11.CONF_THRESHOLD
            and conf[i, v11.L_HIP] > v11.CONF_THRESHOLD
        ):
            hip = 0.5 * (xy[i, v11.R_HIP] + xy[i, v11.L_HIP])
        else:
            continue

        if conf[i, v11.NECK] > v11.CONF_THRESHOLD:
            neck = xy[i, v11.NECK]
        elif (
            conf[i, v11.R_SHOULDER] > v11.CONF_THRESHOLD
            and conf[i, v11.L_SHOULDER] > v11.CONF_THRESHOLD
        ):
            neck = 0.5 * (
                xy[i, v11.R_SHOULDER] + xy[i, v11.L_SHOULDER]
            )
        else:
            continue

        values[i] = float(np.linalg.norm(neck - hip))

    return values


def normalized_radius_stats(v11, norm_xy, conf):
    """
    Per-frame normalized skeleton radius.
    Uses only currently confident points.
    """
    frame_median_radius = np.full(len(norm_xy), np.nan)
    frame_max_radius = np.full(len(norm_xy), np.nan)

    for i in range(len(norm_xy)):
        valid = conf[i] > v11.CONF_THRESHOLD
        if not np.any(valid):
            continue

        r = np.linalg.norm(norm_xy[i, valid], axis=1)
        frame_median_radius[i] = float(np.median(r))
        frame_max_radius[i] = float(np.max(r))

    return frame_median_radius, frame_max_radius


def load_sequence(v11, filename):
    path = os.path.join(VIDEO_DIR, filename)
    if not os.path.isfile(path):
        raise IOError("Missing video: %s" % path)

    kps, ids, fps, total = v11.get_or_extract(
        path, v11.NET_RESOLUTION, v11.FRAME_STEP
    )
    kps_i = v11.fill_missing(kps)
    norm, conf = v11.normalize_frames(kps_i)

    torso = frame_torso_lengths(v11, kps_i)
    med_radius, max_radius = normalized_radius_stats(v11, norm, conf)

    return {
        "filename": filename,
        "kps": kps_i,
        "norm": norm,
        "conf": conf,
        "torso": torso,
        "median_radius": med_radius,
        "max_radius": max_radius,
    }


def trimmed_mean_top_fraction_removed(x, frac=0.10):
    x = np.sort(np.asarray(x, dtype=np.float64))
    x = x[np.isfinite(x)]

    if len(x) == 0:
        return np.nan

    keep = max(1, int(np.floor(len(x) * (1.0 - frac))))
    return float(np.mean(x[:keep]))


def write_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)


def main():
    v11 = load_v11()
    os.makedirs(OUT_DIR, exist_ok=True)

    print("=" * 90)
    print("Phase 3.4C - Pose Distance Failure Diagnostic")
    print("=" * 90)
    print("Net resolution :", v11.NET_RESOLUTION)
    print("Frame step     :", v11.FRAME_STEP)
    print("Algorithm      : unchanged; diagnostic only")

    # --------------------------------------------------------------
    # Video-level normalization diagnostics
    # --------------------------------------------------------------
    seq = {}
    video_rows = []

    for idx, (key, filename) in enumerate(VIDEO_KEYS.items(), start=1):
        print("\n[%d/%d] %s" % (idx, len(VIDEO_KEYS), filename))
        s = load_sequence(v11, filename)
        seq[key] = s

        torso = safe_stats(s["torso"])
        medr = safe_stats(s["median_radius"])
        maxr = safe_stats(s["max_radius"])

        finite_torso = s["torso"][np.isfinite(s["torso"])]
        torso_med = np.median(finite_torso) if len(finite_torso) else np.nan

        if np.isfinite(torso_med) and torso_med > 0:
            frac_below_50pct = float(np.mean(finite_torso < 0.50 * torso_med))
            frac_below_70pct = float(np.mean(finite_torso < 0.70 * torso_med))
        else:
            frac_below_50pct = np.nan
            frac_below_70pct = np.nan

        row = {
            "key": key,
            "filename": filename,
            "choreography": "ch05" if "_ch05" in filename else (
                "ch04" if "_ch04" in filename else "ch06"
            ),
            "torso_px_mean": torso["mean"],
            "torso_px_median": torso["median"],
            "torso_px_p05": torso["p05"],
            "torso_px_min": torso["min"],
            "torso_fraction_below_50pct_median": frac_below_50pct,
            "torso_fraction_below_70pct_median": frac_below_70pct,
            "norm_median_radius_mean": medr["mean"],
            "norm_median_radius_p95": medr["p95"],
            "norm_max_radius_mean": maxr["mean"],
            "norm_max_radius_p95": maxr["p95"],
            "norm_max_radius_max": maxr["max"],
        }
        video_rows.append(row)

        print(
            "  torso median %.2f px | p05 %.2f | min %.2f | "
            "norm max-radius p95 %.2f | max %.2f"
            % (
                row["torso_px_median"],
                row["torso_px_p05"],
                row["torso_px_min"],
                row["norm_max_radius_p95"],
                row["norm_max_radius_max"],
            )
        )

    # --------------------------------------------------------------
    # Pair-level local-cost diagnostics
    # --------------------------------------------------------------
    pair_rows = []
    pairs = [(pid, "D", a, b) for pid, a, b in D_PAIRS]
    pairs += [(pid, "E", a, b) for pid, a, b in E_PAIRS]

    for pair_id, group, a_key, b_key in pairs:
        a = seq[a_key]
        b = seq[b_key]

        cost = v11.build_cost_matrix(
            a["norm"], a["conf"],
            b["norm"], b["conf"]
        )
        path, dtw_dist, win = v11.dtw_align(
            cost, v11.DTW_WINDOW_RATIO
        )

        local = np.asarray(
            [cost[ia, ib] for ia, ib in path],
            dtype=np.float64
        )
        st = safe_stats(local)

        row = {
            "pair_id": pair_id,
            "group": group,
            "reference": a["filename"],
            "test": b["filename"],
            "contains_ch05": (
                "_ch05" in a["filename"] or "_ch05" in b["filename"]
            ),
            "dtw_pose_distance": float(dtw_dist),
            "path_cost_mean": st["mean"],
            "path_cost_median": st["median"],
            "path_cost_p90": st["p90"],
            "path_cost_p95": st["p95"],
            "path_cost_max": st["max"],
            "trimmed_mean_drop_top10pct":
                trimmed_mean_top_fraction_removed(local, 0.10),
            "fraction_cost_gt_0_5": float(np.mean(local > 0.5)),
            "fraction_cost_gt_1_0": float(np.mean(local > 1.0)),
            "fraction_cost_gt_2_0": float(np.mean(local > 2.0)),
            "fraction_cost_gt_5_0": float(np.mean(local > 5.0)),
            "path_length": len(path),
        }
        pair_rows.append(row)

        print("\n%s [%s]" % (pair_id, group))
        print("  DTW mean : %.6f" % row["dtw_pose_distance"])
        print("  median   : %.6f" % row["path_cost_median"])
        print("  p90/p95  : %.6f / %.6f" % (
            row["path_cost_p90"], row["path_cost_p95"]
        ))
        print("  max      : %.6f" % row["path_cost_max"])
        print("  trim10%%  : %.6f" % row["trimmed_mean_drop_top10pct"])
        print(
            "  >1 / >2 / >5 : %.1f%% / %.1f%% / %.1f%%"
            % (
                100.0 * row["fraction_cost_gt_1_0"],
                100.0 * row["fraction_cost_gt_2_0"],
                100.0 * row["fraction_cost_gt_5_0"],
            )
        )

    # --------------------------------------------------------------
    # Aggregate ch05 vs non-ch05
    # --------------------------------------------------------------
    print("\n" + "=" * 90)
    print("CH05 VS NON-CH05 SUMMARY")
    print("=" * 90)

    for has_ch05 in (False, True):
        subset = [
            r for r in pair_rows
            if r["contains_ch05"] == has_ch05
        ]

        print("\ncontains_ch05 =", has_ch05)
        for metric in (
            "dtw_pose_distance",
            "path_cost_median",
            "path_cost_p95",
            "trimmed_mean_drop_top10pct",
        ):
            x = [r[metric] for r in subset]
            st = safe_stats(x)
            print(
                "  %-31s mean %.4f | median %.4f | min %.4f | max %.4f"
                % (
                    metric,
                    st["mean"], st["median"],
                    st["min"], st["max"],
                )
            )

    # --------------------------------------------------------------
    # Save
    # --------------------------------------------------------------
    video_csv = os.path.join(
        OUT_DIR, "video_normalization_stats.csv"
    )
    pair_csv = os.path.join(
        OUT_DIR, "pair_pose_cost_stats.csv"
    )

    write_csv(video_csv, video_rows)
    write_csv(pair_csv, pair_rows)

    print("\nInterpretation guide:")
    print("  A) Mean huge but median/trim10 normal -> sparse local-cost outliers dominate.")
    print("  B) Mean, median and trim10 all huge -> systematic normalized-coordinate mismatch.")
    print("  C) Tiny torso p05/min + huge normalized radii -> torso-scale normalization instability.")
    print("  D) If normalization looks normal but ch05 costs remain huge, inspect pose geometry/view")
    print("     and keep Pose Distance as an alignment cost rather than an absolute score.")

    print("\nSaved:")
    print(" ", video_csv)
    print(" ", pair_csv)


if __name__ == "__main__":
    main()
