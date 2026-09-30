# -*- coding: utf-8 -*-
"""
Phase 3.4D - Robust Normalization A/B Test

Background
----------
The ch04-ch06 holdout exposed a failure of absolute DTW Pose Distance.
Diagnostic results showed that ch05 often makes the frame-wise Neck-MidHip
torso length collapse, which can explosively magnify normalized coordinates.

This script compares two normalization methods:

A) CURRENT
   Current V1.1 behavior:
       per-frame hip origin
       per-frame Neck-MidHip torso length as scale

B) SEQUENCE_MEDIAN
   Proposed robust behavior:
       per-frame hip origin
       ONE robust scale for the whole sequence:
       median(valid per-frame torso lengths)

Experimental protocol
---------------------
Calibration/development:
    ch01, ch02, ch03

Holdout validation:
    ch04, ch05, ch06

For EACH normalization method:
1) compute D/E pose distances on ch01-ch03;
2) derive the threshold ONLY from ch01-ch03;
3) apply the unchanged threshold to ch04-ch06;
4) report AUC, balanced accuracy and D/E gap.

ch07-ch10 are NOT touched and remain reserved for final validation.

Important
---------
- This script does not modify Dance_Motion_Similarity_Evaluation_V1.1_final.py.
- DTW algorithm and window are unchanged.
- OpenPose extraction settings are unchanged.
- Only the normalization scale is changed in method B.
"""

import os
import csv
import importlib.util
from itertools import combinations
import numpy as np


# ======================================================================
# Configuration
# ======================================================================
V11_FILE = "Dance_Motion_Similarity_Evaluation_V1.1_final.py"
VIDEO_DIR = os.path.join("data", "aistpp", "videos")
OUT_DIR = "phase3_robust_normalization_ab"

SPLITS = {
    "calibration_ch01_ch03": ["ch01", "ch02", "ch03"],
    "holdout_ch04_ch06": ["ch04", "ch05", "ch06"],
}

# Each block preserves same camera/situation/music within every comparison.
BLOCKS = {
    "A": {
        "music": "mBR0",
        "dancer_a": "d04",
        "dancer_b": "d05",
        "e_dancer": "d04",
    },
    "B": {
        "music": "mBR2",
        "dancer_a": "d04",
        "dancer_b": "d06",
        "e_dancer": "d06",
    },
    "C": {
        "music": "mBR4",
        "dancer_a": "d05",
        "dancer_b": "d06",
        "e_dancer": "d05",
    },
}


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


def make_filename(dancer, music, choreo):
    return "gBR_sBM_c01_%s_%s_%s.mp4" % (dancer, music, choreo)


def video_key(dancer, music, choreo):
    return "%s_%s_%s" % (dancer, music, choreo)


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
    """AUC where larger score means more likely E."""
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
    d_ok = []
    e_ok = []

    for y, p in zip(labels, predictions):
        if y == 0:
            d_ok.append(1.0 if p == 0 else 0.0)
        else:
            e_ok.append(1.0 if p == 1 else 0.0)

    return 0.5 * (float(np.mean(d_ok)) + float(np.mean(e_ok)))


def write_csv(path, rows):
    if not rows:
        return

    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)


# ======================================================================
# Reproduce the V1.1 hip/torso references
# ======================================================================
def frame_hip_and_torso(v11, xy, conf):
    """
    Return:
        hip_origin
        torso_length_or_nan

    Uses the same fallbacks as V1.1 normalize_frames().
    """

    # Hip origin
    if conf[v11.MID_HIP] > v11.CONF_THRESHOLD:
        hip = xy[v11.MID_HIP]
        hip_valid = True
    elif (
        conf[v11.R_HIP] > v11.CONF_THRESHOLD
        and conf[v11.L_HIP] > v11.CONF_THRESHOLD
    ):
        hip = 0.5 * (xy[v11.R_HIP] + xy[v11.L_HIP])
        hip_valid = True
    else:
        hip = np.zeros(2, dtype=np.float64)
        hip_valid = False

    # Neck reference
    if conf[v11.NECK] > v11.CONF_THRESHOLD:
        neck = xy[v11.NECK]
        neck_valid = True
    elif (
        conf[v11.R_SHOULDER] > v11.CONF_THRESHOLD
        and conf[v11.L_SHOULDER] > v11.CONF_THRESHOLD
    ):
        neck = 0.5 * (
            xy[v11.R_SHOULDER] + xy[v11.L_SHOULDER]
        )
        neck_valid = True
    else:
        neck = hip
        neck_valid = False

    if hip_valid and neck_valid:
        torso = float(np.linalg.norm(neck - hip))
        if torso >= 1e-6:
            return hip, torso

    return hip, np.nan


def normalize_sequence_median(v11, kps):
    """
    Proposed robust normalization.

    Translation:
        per-frame hip origin, identical to V1.1.

    Scale:
        one sequence-level median valid torso length.

    This removes catastrophic per-frame division by tiny torso lengths.
    """
    S = kps.shape[0]
    conf = kps[:, :, 2].copy()
    norm_xy = np.zeros((S, v11.N_PARTS, 2), dtype=np.float64)

    hips = []
    torso_values = np.full(S, np.nan, dtype=np.float64)

    for i in range(S):
        xy = kps[i, :, :2]
        hip, torso = frame_hip_and_torso(
            v11, xy, conf[i]
        )
        hips.append(hip)
        torso_values[i] = torso

    valid_torso = torso_values[np.isfinite(torso_values)]

    if len(valid_torso) == 0:
        sequence_scale = 1.0
    else:
        sequence_scale = float(np.median(valid_torso))

    if sequence_scale < 1e-6:
        sequence_scale = 1.0

    for i in range(S):
        xy = kps[i, :, :2]
        norm_xy[i] = (xy - hips[i]) / sequence_scale

    return norm_xy, conf, sequence_scale, torso_values


# ======================================================================
# Load all needed skeleton sequences once
# ======================================================================
def collect_required_videos():
    videos = {}

    all_choreos = []
    for vals in SPLITS.values():
        all_choreos.extend(vals)

    for block_name, b in BLOCKS.items():
        for choreo in all_choreos:
            for dancer in {b["dancer_a"], b["dancer_b"], b["e_dancer"]}:
                key = video_key(dancer, b["music"], choreo)
                videos[key] = make_filename(
                    dancer, b["music"], choreo
                )

    return videos


def load_sequences(v11):
    required = collect_required_videos()

    missing = [
        os.path.join(VIDEO_DIR, filename)
        for filename in required.values()
        if not os.path.isfile(
            os.path.join(VIDEO_DIR, filename)
        )
    ]

    if missing:
        print("\n[ERROR] Missing required videos:")
        for path in missing:
            print(" ", path)
        return None, None

    seq = {}
    scale_rows = []

    for idx, (key, filename) in enumerate(
        required.items(), start=1
    ):
        print(
            "\n[%d/%d] %s"
            % (idx, len(required), filename)
        )

        path = os.path.join(VIDEO_DIR, filename)

        kps, ids, fps, total = v11.get_or_extract(
            path,
            v11.NET_RESOLUTION,
            v11.FRAME_STEP,
        )

        kps_i = v11.fill_missing(kps)

        # A) Current V1.1
        current_norm, current_conf = (
            v11.normalize_frames(kps_i)
        )

        # B) Proposed robust version
        median_norm, median_conf, seq_scale, torso_values = (
            normalize_sequence_median(v11, kps_i)
        )

        finite_torso = torso_values[
            np.isfinite(torso_values)
        ]

        torso_median = (
            float(np.median(finite_torso))
            if len(finite_torso)
            else np.nan
        )

        torso_p05 = (
            float(np.percentile(finite_torso, 5))
            if len(finite_torso)
            else np.nan
        )

        torso_min = (
            float(np.min(finite_torso))
            if len(finite_torso)
            else np.nan
        )

        frac_below_half = (
            float(
                np.mean(
                    finite_torso < 0.5 * torso_median
                )
            )
            if len(finite_torso)
            and torso_median > 0
            else np.nan
        )

        seq[key] = {
            "filename": filename,
            "current_norm": current_norm,
            "current_conf": current_conf,
            "median_norm": median_norm,
            "median_conf": median_conf,
        }

        scale_rows.append({
            "key": key,
            "filename": filename,
            "sequence_median_scale_px": seq_scale,
            "frame_torso_p05_px": torso_p05,
            "frame_torso_min_px": torso_min,
            "fraction_frame_torso_below_50pct_sequence_median":
                frac_below_half,
        })

        print(
            "  sequence median scale %.2f px | "
            "frame torso p05 %.2f | min %.2f"
            % (seq_scale, torso_p05, torso_min)
        )

    return seq, scale_rows


# ======================================================================
# Build D/E pair list for one split
# ======================================================================
def build_pairs(split_name, choreos):
    pairs = []
    d_counter = 1
    e_counter = 1

    for block_name, b in BLOCKS.items():
        # D: dancer changes; choreography/music stay fixed.
        for choreo in choreos:
            a = video_key(
                b["dancer_a"], b["music"], choreo
            )
            c = video_key(
                b["dancer_b"], b["music"], choreo
            )

            pairs.append({
                "pair_id":
                    "%s_D%d" % (split_name, d_counter),
                "split": split_name,
                "block": block_name,
                "group": "D",
                "reference_key": a,
                "test_key": c,
            })
            d_counter += 1

        # E: same dancer/music; choreography changes.
        e_dancer = b["e_dancer"]

        for ch_a, ch_b in combinations(choreos, 2):
            a = video_key(
                e_dancer, b["music"], ch_a
            )
            c = video_key(
                e_dancer, b["music"], ch_b
            )

            pairs.append({
                "pair_id":
                    "%s_E%d" % (split_name, e_counter),
                "split": split_name,
                "block": block_name,
                "group": "E",
                "reference_key": a,
                "test_key": c,
            })
            e_counter += 1

    return pairs


# ======================================================================
# Evaluate one normalization method
# ======================================================================
def evaluate_pairs(v11, seq, pairs, method):
    if method == "CURRENT":
        norm_key = "current_norm"
        conf_key = "current_conf"
    elif method == "SEQUENCE_MEDIAN":
        norm_key = "median_norm"
        conf_key = "median_conf"
    else:
        raise ValueError(method)

    rows = []

    for p in pairs:
        a = seq[p["reference_key"]]
        b = seq[p["test_key"]]

        cost = v11.build_cost_matrix(
            a[norm_key],
            a[conf_key],
            b[norm_key],
            b[conf_key],
        )

        path, pose_dist, win = v11.dtw_align(
            cost,
            v11.DTW_WINDOW_RATIO,
        )

        local_cost = np.asarray(
            [cost[ia, ib] for ia, ib in path],
            dtype=np.float64,
        )

        rows.append({
            "normalization": method,
            "pair_id": p["pair_id"],
            "split": p["split"],
            "block": p["block"],
            "group": p["group"],
            "reference":
                a["filename"],
            "test":
                b["filename"],
            "dtw_pose_distance":
                float(pose_dist),
            "path_cost_median":
                float(np.median(local_cost)),
            "path_cost_p95":
                float(np.percentile(local_cost, 95)),
            "path_cost_max":
                float(np.max(local_cost)),
            "path_length":
                len(path),
        })

    return rows


# ======================================================================
# Calibration threshold + held-out validation
# ======================================================================
def calibration_threshold(rows, method):
    cal = [
        r for r in rows
        if r["normalization"] == method
        and r["split"] == "calibration_ch01_ch03"
    ]

    d = [
        r["dtw_pose_distance"]
        for r in cal if r["group"] == "D"
    ]
    e = [
        r["dtw_pose_distance"]
        for r in cal if r["group"] == "E"
    ]

    gap = min(e) - max(d)

    if gap > 0:
        threshold = 0.5 * (max(d) + min(e))
        method_name = "midpoint max(D)-min(E)"
    else:
        threshold = 0.5 * (
            np.median(d) + np.median(e)
        )
        method_name = "midpoint class medians"

    return float(threshold), method_name, float(gap)


def validation_summary(rows, method, threshold):
    result = []

    for split_name in SPLITS.keys():
        subset = [
            r for r in rows
            if r["normalization"] == method
            and r["split"] == split_name
        ]

        labels = [
            0 if r["group"] == "D" else 1
            for r in subset
        ]
        scores = [
            r["dtw_pose_distance"]
            for r in subset
        ]
        preds = [
            1 if s >= threshold else 0
            for s in scores
        ]

        d = [
            r["dtw_pose_distance"]
            for r in subset
            if r["group"] == "D"
        ]
        e = [
            r["dtw_pose_distance"]
            for r in subset
            if r["group"] == "E"
        ]

        ds = describe(d)
        es = describe(e)

        result.append({
            "normalization": method,
            "split": split_name,
            "threshold_from_ch01_ch03":
                threshold,
            "D_mean": ds["mean"],
            "D_median": ds["median"],
            "D_min": ds["min"],
            "D_max": ds["max"],
            "E_mean": es["mean"],
            "E_median": es["median"],
            "E_min": es["min"],
            "E_max": es["max"],
            "gap_minE_minus_maxD":
                min(e) - max(d),
            "auc":
                auc_binary(labels, scores),
            "balanced_accuracy":
                balanced_accuracy(labels, preds),
        })

    return result


# ======================================================================
# Main
# ======================================================================
def main():
    v11 = load_v11()
    os.makedirs(OUT_DIR, exist_ok=True)

    print("=" * 92)
    print("Phase 3.4D - Robust Normalization A/B Test")
    print("=" * 92)
    print("A: CURRENT          = per-frame torso scale")
    print("B: SEQUENCE_MEDIAN  = sequence-median torso scale")
    print("Calibration         = ch01-ch03")
    print("Holdout             = ch04-ch06")
    print("Reserved            = ch07-ch10")
    print("DTW                 = unchanged")
    print("Joint weights       = unchanged")

    seq, scale_rows = load_sequences(v11)

    if seq is None:
        return

    all_pairs = []

    for split_name, choreos in SPLITS.items():
        all_pairs.extend(
            build_pairs(split_name, choreos)
        )

    all_rows = []

    for method in ("CURRENT", "SEQUENCE_MEDIAN"):
        print("\n" + "=" * 92)
        print("Evaluating normalization:", method)
        print("=" * 92)

        all_rows.extend(
            evaluate_pairs(
                v11, seq, all_pairs, method
            )
        )

    summary_rows = []

    for method in ("CURRENT", "SEQUENCE_MEDIAN"):
        threshold, threshold_method, cal_gap = (
            calibration_threshold(
                all_rows, method
            )
        )

        print("\n" + "=" * 92)
        print(method)
        print("=" * 92)
        print(
            "Calibration threshold: %.6f (%s)"
            % (threshold, threshold_method)
        )
        print(
            "Calibration gap min(E)-max(D): %+.6f"
            % cal_gap
        )

        method_summary = validation_summary(
            all_rows,
            method,
            threshold,
        )

        for row in method_summary:
            row["threshold_method"] = (
                threshold_method
            )
            summary_rows.append(row)

            print(
                "%s | D %.4f..%.4f | E %.4f..%.4f | "
                "gap %+.4f | AUC %.3f | BalAcc %.3f"
                % (
                    row["split"],
                    row["D_min"],
                    row["D_max"],
                    row["E_min"],
                    row["E_max"],
                    row["gap_minE_minus_maxD"],
                    row["auc"],
                    row["balanced_accuracy"],
                )
            )

    pair_csv = os.path.join(
        OUT_DIR,
        "normalization_ab_pair_results.csv",
    )
    summary_csv = os.path.join(
        OUT_DIR,
        "normalization_ab_summary.csv",
    )
    scale_csv = os.path.join(
        OUT_DIR,
        "sequence_scale_stats.csv",
    )

    write_csv(pair_csv, all_rows)
    write_csv(summary_csv, summary_rows)
    write_csv(scale_csv, scale_rows)

    print("\nInterpretation:")
    print(
        "  If SEQUENCE_MEDIAN sharply reduces ch05 explosions and improves"
    )
    print(
        "  ch04-ch06 AUC / balanced accuracy using ONLY the ch01-ch03 threshold,"
    )
    print(
        "  the failure is primarily a normalization-scale problem."
    )
    print(
        "  If it does not, Pose Distance is too choreography-sensitive to use"
    )
    print(
        "  as an absolute overall score and should remain only a DTW cost."
    )
    print(
        "  Do not touch ch07-ch10 yet; keep them for the final validation."
    )

    print("\nSaved:")
    print(" ", pair_csv)
    print(" ", summary_csv)
    print(" ", scale_csv)


if __name__ == "__main__":
    main()
