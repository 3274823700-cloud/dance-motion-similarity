# -*- coding: utf-8 -*-
"""
Phase 3.4E - Normalization Downstream Validation

Goal
----
The sequence-median torso scale fixed the catastrophic Pose Distance explosions,
but Pose Distance still showed choreography-dependent absolute scale shift.

Before adopting SEQUENCE_MEDIAN in V1.2, check whether changing normalization
also improves or harms the metrics we actually care about downstream:

1) Smoothed weighted joint-angle error
2) DTW timing deviation
3) Pose distance (for reference only)

Protocol
--------
Calibration/development:
    ch01, ch02, ch03

Holdout:
    ch04, ch05, ch06

Reserved:
    ch07, ch08, ch09, ch10  <-- untouched

For each normalization:
    CURRENT
    SEQUENCE_MEDIAN

we recompute:
    normalization -> Pose Cost -> DTW path

Then, on that SAME path, evaluate:
    - 3-frame-median joint-angle error
    - mean relative-time deviation
    - DTW pose distance

No final 0-100 score is designed here.
"""

import os
import csv
import importlib.util
from itertools import combinations
import numpy as np


V11_FILE = "Dance_Motion_Similarity_Evaluation_V1.1_final.py"
VIDEO_DIR = os.path.join("data", "aistpp", "videos")
OUT_DIR = "phase3_normalization_downstream"

SMOOTH_RADIUS = 1

SPLITS = {
    "calibration_ch01_ch03": ["ch01", "ch02", "ch03"],
    "holdout_ch04_ch06": ["ch04", "ch05", "ch06"],
}

BLOCKS = {
    "A": {"music": "mBR0", "dancer_a": "d04", "dancer_b": "d05", "e_dancer": "d04"},
    "B": {"music": "mBR2", "dancer_a": "d04", "dancer_b": "d06", "e_dancer": "d06"},
    "C": {"music": "mBR4", "dancer_a": "d05", "dancer_b": "d06", "e_dancer": "d05"},
}


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
# Robust normalization
# ======================================================================
def frame_hip_and_torso(v11, xy, conf):
    if conf[v11.MID_HIP] > v11.CONF_THRESHOLD:
        hip = xy[v11.MID_HIP]
        hip_valid = True
    elif conf[v11.R_HIP] > v11.CONF_THRESHOLD and conf[v11.L_HIP] > v11.CONF_THRESHOLD:
        hip = 0.5 * (xy[v11.R_HIP] + xy[v11.L_HIP])
        hip_valid = True
    else:
        hip = np.zeros(2, dtype=np.float64)
        hip_valid = False

    if conf[v11.NECK] > v11.CONF_THRESHOLD:
        neck = xy[v11.NECK]
        neck_valid = True
    elif conf[v11.R_SHOULDER] > v11.CONF_THRESHOLD and conf[v11.L_SHOULDER] > v11.CONF_THRESHOLD:
        neck = 0.5 * (xy[v11.R_SHOULDER] + xy[v11.L_SHOULDER])
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
    S = len(kps)
    conf = kps[:, :, 2].copy()
    hips = []
    torso_values = []

    for i in range(S):
        hip, torso = frame_hip_and_torso(v11, kps[i, :, :2], conf[i])
        hips.append(hip)
        torso_values.append(torso)

    torso_values = np.asarray(torso_values, dtype=np.float64)
    valid = torso_values[np.isfinite(torso_values)]

    scale = float(np.median(valid)) if len(valid) else 1.0
    if scale < 1e-6:
        scale = 1.0

    norm = np.zeros((S, v11.N_PARTS, 2), dtype=np.float64)

    for i in range(S):
        norm[i] = (kps[i, :, :2] - hips[i]) / scale

    return norm, conf, scale


# ======================================================================
# Data construction
# ======================================================================
def collect_required_videos():
    videos = {}
    all_choreos = []

    for choreos in SPLITS.values():
        all_choreos.extend(choreos)

    for b in BLOCKS.values():
        dancers = {b["dancer_a"], b["dancer_b"], b["e_dancer"]}

        for choreo in all_choreos:
            for dancer in dancers:
                key = video_key(dancer, b["music"], choreo)
                videos[key] = make_filename(dancer, b["music"], choreo)

    return videos


def load_sequences(v11):
    required = collect_required_videos()
    seq = {}

    missing = [
        os.path.join(VIDEO_DIR, filename)
        for filename in required.values()
        if not os.path.isfile(os.path.join(VIDEO_DIR, filename))
    ]

    if missing:
        print("\n[ERROR] Missing required videos:")
        for path in missing:
            print(" ", path)
        return None

    for i, (key, filename) in enumerate(required.items(), start=1):
        print("\n[%d/%d] %s" % (i, len(required), filename))

        kps, ids, fps, total = v11.get_or_extract(
            os.path.join(VIDEO_DIR, filename),
            v11.NET_RESOLUTION,
            v11.FRAME_STEP,
        )

        kps_i = v11.fill_missing(kps)

        current_norm, current_conf = v11.normalize_frames(kps_i)
        median_norm, median_conf, median_scale = normalize_sequence_median(v11, kps_i)

        # Joint angles are invariant to translation and uniform scale.
        # Build them once from CURRENT normalized coordinates.
        raw_angles = v11.build_joint_angle_sequence(current_norm, current_conf)
        smooth_angles = v11.median_smooth_angle_sequence(raw_angles, SMOOTH_RADIUS)

        seq[key] = {
            "filename": filename,
            "current_norm": current_norm,
            "current_conf": current_conf,
            "median_norm": median_norm,
            "median_conf": median_conf,
            "smooth_angles": smooth_angles,
            "median_scale_px": median_scale,
        }

    return seq


def build_pairs(split_name, choreos):
    pairs = []
    d_counter = 1
    e_counter = 1

    for block_name, b in BLOCKS.items():
        for choreo in choreos:
            pairs.append({
                "pair_id": "%s_D%d" % (split_name, d_counter),
                "split": split_name,
                "block": block_name,
                "group": "D",
                "reference_key": video_key(b["dancer_a"], b["music"], choreo),
                "test_key": video_key(b["dancer_b"], b["music"], choreo),
            })
            d_counter += 1

        for ch_a, ch_b in combinations(choreos, 2):
            pairs.append({
                "pair_id": "%s_E%d" % (split_name, e_counter),
                "split": split_name,
                "block": block_name,
                "group": "E",
                "reference_key": video_key(b["e_dancer"], b["music"], ch_a),
                "test_key": video_key(b["e_dancer"], b["music"], ch_b),
            })
            e_counter += 1

    return pairs


# ======================================================================
# DTW timing metric
# ======================================================================
def mean_relative_time_deviation(path, n_ref, n_test):
    values = []

    for ia, ib in path:
        ta = ia / float(max(n_ref - 1, 1))
        tb = ib / float(max(n_test - 1, 1))
        values.append(abs(ta - tb))

    return float(np.mean(values))


# ======================================================================
# Evaluate both normalizations
# ======================================================================
def evaluate_method(v11, seq, pairs, method):
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
            a[norm_key], a[conf_key],
            b[norm_key], b[conf_key],
        )

        path, pose_dist, win = v11.dtw_align(
            cost, v11.DTW_WINDOW_RATIO
        )

        joint_err = v11.per_joint_path_mean_errors(
            a["smooth_angles"],
            b["smooth_angles"],
            path,
        )

        angle_error = v11.weighted_overall_from_joint_errors(joint_err)

        time_dev = mean_relative_time_deviation(
            path,
            len(a[norm_key]),
            len(b[norm_key]),
        )

        rows.append({
            "normalization": method,
            "pair_id": p["pair_id"],
            "split": p["split"],
            "block": p["block"],
            "group": p["group"],
            "reference": a["filename"],
            "test": b["filename"],
            "dtw_pose_distance": float(pose_dist),
            "smoothed_angle_error_deg": float(angle_error),
            "mean_relative_time_deviation": float(time_dev),
            "path_length": len(path),
        })

    return rows


# ======================================================================
# Threshold learned only from ch01-ch03
# ======================================================================
def learn_threshold(cal_rows, metric):
    d = [r[metric] for r in cal_rows if r["group"] == "D"]
    e = [r[metric] for r in cal_rows if r["group"] == "E"]

    gap = min(e) - max(d)

    if gap > 0:
        threshold = 0.5 * (max(d) + min(e))
        method = "gap midpoint"
    else:
        threshold = 0.5 * (np.median(d) + np.median(e))
        method = "class-median midpoint"

    return float(threshold), method, float(gap)


def metric_summary(rows, normalization, metric):
    cal = [
        r for r in rows
        if r["normalization"] == normalization
        and r["split"] == "calibration_ch01_ch03"
    ]

    threshold, threshold_method, cal_gap = learn_threshold(cal, metric)

    output = []

    for split in SPLITS.keys():
        subset = [
            r for r in rows
            if r["normalization"] == normalization
            and r["split"] == split
        ]

        labels = [0 if r["group"] == "D" else 1 for r in subset]
        scores = [r[metric] for r in subset]
        preds = [1 if s >= threshold else 0 for s in scores]

        d = [r[metric] for r in subset if r["group"] == "D"]
        e = [r[metric] for r in subset if r["group"] == "E"]

        ds = describe(d)
        es = describe(e)

        output.append({
            "normalization": normalization,
            "metric": metric,
            "split": split,
            "threshold_from_ch01_ch03": threshold,
            "threshold_method": threshold_method,
            "D_mean": ds["mean"],
            "D_min": ds["min"],
            "D_max": ds["max"],
            "E_mean": es["mean"],
            "E_min": es["min"],
            "E_max": es["max"],
            "gap_minE_minus_maxD": min(e) - max(d),
            "auc": auc_binary(labels, scores),
            "balanced_accuracy": balanced_accuracy(labels, preds),
        })

    return output


# ======================================================================
# Main
# ======================================================================
def main():
    v11 = load_v11()
    os.makedirs(OUT_DIR, exist_ok=True)

    print("=" * 92)
    print("Phase 3.4E - Normalization Downstream Validation")
    print("=" * 92)
    print("CURRENT         : per-frame torso scale")
    print("SEQUENCE_MEDIAN : one median torso scale per video")
    print("Calibration     : ch01-ch03")
    print("Holdout         : ch04-ch06")
    print("Reserved        : ch07-ch10")
    print("Angle smoothing : 3-frame median")
    print("Final score     : NOT designed yet")

    seq = load_sequences(v11)

    if seq is None:
        return

    pairs = []

    for split_name, choreos in SPLITS.items():
        pairs.extend(build_pairs(split_name, choreos))

    rows = []

    for method in ("CURRENT", "SEQUENCE_MEDIAN"):
        print("\nEvaluating:", method)
        rows.extend(evaluate_method(v11, seq, pairs, method))

    summary_rows = []

    for method in ("CURRENT", "SEQUENCE_MEDIAN"):
        for metric in (
            "dtw_pose_distance",
            "smoothed_angle_error_deg",
            "mean_relative_time_deviation",
        ):
            summary_rows.extend(
                metric_summary(rows, method, metric)
            )

    print("\n" + "=" * 92)
    print("DOWNSTREAM SUMMARY")
    print("=" * 92)

    for method in ("CURRENT", "SEQUENCE_MEDIAN"):
        print("\n" + method)

        for metric in (
            "dtw_pose_distance",
            "smoothed_angle_error_deg",
            "mean_relative_time_deviation",
        ):
            holdout = [
                r for r in summary_rows
                if r["normalization"] == method
                and r["metric"] == metric
                and r["split"] == "holdout_ch04_ch06"
            ][0]

            print(
                "  %-29s | AUC %.3f | BalAcc %.3f | gap %+.4f | Dmean %.4f | Emean %.4f"
                % (
                    metric,
                    holdout["auc"],
                    holdout["balanced_accuracy"],
                    holdout["gap_minE_minus_maxD"],
                    holdout["D_mean"],
                    holdout["E_mean"],
                )
            )

    pair_csv = os.path.join(
        OUT_DIR, "normalization_downstream_pair_results.csv"
    )
    summary_csv = os.path.join(
        OUT_DIR, "normalization_downstream_summary.csv"
    )

    write_csv(pair_csv, rows)
    write_csv(summary_csv, summary_rows)

    print("\nDecision guide:")
    print("  1) Adopt SEQUENCE_MEDIAN only if it removes Pose explosions without")
    print("     materially degrading smoothed-angle or timing discrimination.")
    print("  2) Even if Pose AUC improves, poor threshold transfer means Pose Distance")
    print("     should not be used as a universal absolute 0-100 score by itself.")
    print("  3) If smoothed angle remains the most stable downstream metric, use it")
    print("     as the main Overall-score candidate and keep Pose Distance for DTW.")
    print("  4) Keep ch07-ch10 untouched for final validation.")

    print("\nSaved:")
    print(" ", pair_csv)
    print(" ", summary_csv)


if __name__ == "__main__":
    main()
