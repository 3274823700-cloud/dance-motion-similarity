# -*- coding: utf-8 -*-
"""
Phase 3.6A - OpenPose Resolution Downstream Validation

Goal
----
Controlled A/B validation of:
    BASELINE  = 432x240
    CANDIDATE = -1x368

Only NET_RESOLUTION changes.

Frozen V1.2 settings:
    FRAME_STEP = 2
    CONF_THRESHOLD = 0.30
    DTW_WINDOW_RATIO = 0.30
    SEQUENCE_MEDIAN normalization
    3-frame median joint-angle smoothing
    frozen joint definitions / weights
    frozen logistic score mapping:
        center = 33.046977 deg
        scale  = 6.292398 deg

Data protocol preserved from Phase 3:
    Development:
        calibration = ch01-ch03
        holdout     = ch04-ch06
    Final validation:
        ch07-ch10

Pair definitions:
    D = different dancer, same music, same choreography
    E = same dancer/music, different choreography

This script does NOT recalibrate score parameters.
It only measures whether -1x368 improves upstream OpenPose measurement and
downstream alignment/evaluation without destroying D/E discrimination.
"""

import os
import csv
import time
import importlib.util
from itertools import combinations

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ======================================================================
# Configuration
# ======================================================================
V12_FILE = "Dance_Motion_Similarity_Evaluation_V1.2.py"
VIDEO_DIR = os.path.join("data", "aistpp", "videos")
OUT_DIR = "phase3_resolution_downstream_validation"

RESOLUTIONS = [
    ("432x240", "BASELINE"),
    ("-1x368", "CANDIDATE"),
]

FRAME_STEP = 2
CONF_THRESHOLD = 0.30
DTW_WINDOW_RATIO = 0.30
SMOOTH_RADIUS = 1
SCORE_CENTER = 33.046977
SCORE_SCALE = 6.292398

SPLITS = {
    "calibration_ch01_ch03": ["ch01", "ch02", "ch03"],
    "holdout_ch04_ch06": ["ch04", "ch05", "ch06"],
    "final_ch07_ch10": ["ch07", "ch08", "ch09", "ch10"],
}

BLOCKS = {
    "A": {"music": "mBR0", "dancer_a": "d04", "dancer_b": "d05", "e_dancer": "d04"},
    "B": {"music": "mBR2", "dancer_a": "d04", "dancer_b": "d06", "e_dancer": "d06"},
    "C": {"music": "mBR4", "dancer_a": "d05", "dancer_b": "d06", "e_dancer": "d05"},
}


# ======================================================================
# Generic helpers
# ======================================================================
def load_v12():
    path = os.path.abspath(V12_FILE)
    if not os.path.isfile(path):
        raise IOError("V1.2 file not found: %s" % path)

    spec = importlib.util.spec_from_file_location("dance_v12", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    # Explicitly freeze every downstream setting used in this A/B test.
    module.FRAME_STEP = FRAME_STEP
    module.CONF_THRESHOLD = CONF_THRESHOLD
    module.DTW_WINDOW_RATIO = DTW_WINDOW_RATIO
    module.SMOOTH_RADIUS = SMOOTH_RADIUS
    module.SCORE_CENTER = SCORE_CENTER
    module.SCORE_SCALE = SCORE_SCALE

    # Resolution-specific JSON cache names already prevent 432/368 mixing.
    # Existing complete caches are reused; missing caches are extracted once.
    module.USE_CACHE = True
    module.FORCE_REEXTRACT = False

    return module


def make_filename(dancer, music, choreo):
    return "gBR_sBM_c01_%s_%s_%s.mp4" % (dancer, music, choreo)


def video_key(dancer, music, choreo):
    return "%s_%s_%s" % (dancer, music, choreo)


def write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)


def describe(values):
    x = np.asarray([v for v in values if np.isfinite(v)], dtype=np.float64)
    if len(x) == 0:
        return {"count": 0, "mean": np.nan, "median": np.nan, "min": np.nan, "max": np.nan, "std": np.nan}
    return {
        "count": int(len(x)),
        "mean": float(np.mean(x)),
        "median": float(np.median(x)),
        "min": float(np.min(x)),
        "max": float(np.max(x)),
        "std": float(np.std(x, ddof=1)) if len(x) > 1 else 0.0,
    }


def auc_binary(labels, scores):
    """AUC with D=0, E=1 and larger metric expected to indicate E."""
    d = [s for y, s in zip(labels, scores) if y == 0 and np.isfinite(s)]
    e = [s for y, s in zip(labels, scores) if y == 1 and np.isfinite(s)]
    wins, total = 0.0, 0
    for se in e:
        for sd in d:
            total += 1
            if se > sd:
                wins += 1.0
            elif se == sd:
                wins += 0.5
    return wins / total if total else np.nan


def balanced_accuracy(labels, predictions):
    d_ok, e_ok = [], []
    for y, p in zip(labels, predictions):
        if y == 0:
            d_ok.append(1.0 if p == 0 else 0.0)
        else:
            e_ok.append(1.0 if p == 1 else 0.0)
    if not d_ok or not e_ok:
        return np.nan
    return 0.5 * (float(np.mean(d_ok)) + float(np.mean(e_ok)))


def error_to_score(error_deg):
    if not np.isfinite(error_deg):
        return np.nan
    z = (float(error_deg) - SCORE_CENTER) / SCORE_SCALE
    if z > 60.0:
        return 0.0
    if z < -60.0:
        return 100.0
    return float(100.0 / (1.0 + np.exp(z)))


# ======================================================================
# Dataset / pair construction
# ======================================================================
def collect_required_videos():
    videos = {}
    for b in BLOCKS.values():
        for choreos in SPLITS.values():
            for ch in choreos:
                for dancer in {b["dancer_a"], b["dancer_b"], b["e_dancer"]}:
                    k = video_key(dancer, b["music"], ch)
                    videos[k] = make_filename(dancer, b["music"], ch)
    return videos


def build_pairs(split_name, choreos):
    pairs = []
    d_counter, e_counter = 1, 1

    for block_name, b in BLOCKS.items():
        # D: different dancer, same music + same choreography
        for ch in choreos:
            pairs.append({
                "pair_id": "%s_D%d" % (split_name, d_counter),
                "split": split_name,
                "block": block_name,
                "group": "D",
                "reference_key": video_key(b["dancer_a"], b["music"], ch),
                "test_key": video_key(b["dancer_b"], b["music"], ch),
            })
            d_counter += 1

        # E: same dancer/music, all pairs of different choreographies
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


def build_all_pairs():
    pairs = []
    for split_name, choreos in SPLITS.items():
        pairs.extend(build_pairs(split_name, choreos))
    return pairs


# ======================================================================
# Sequence loading / measurement diagnostics
# ======================================================================
def raw_body25_coverage(kps):
    """Per BODY_25 point visibility before interpolation."""
    valid = kps[:, :, 2] > CONF_THRESHOLD
    per_point = valid.mean(axis=0)
    return per_point


def load_one_sequence(v12, filename, resolution):
    path = os.path.join(VIDEO_DIR, filename)
    if not os.path.isfile(path):
        raise IOError("Video not found: %s" % path)

    # Determine whether this call starts from a completed JSON cache.
    cache_dir = v12._cache_json_dir(path, resolution)
    complete_flag = os.path.join(cache_dir, "_complete.txt")
    was_cached = os.path.isfile(complete_flag)

    t0 = time.time()
    kps, ids, fps, total = v12.get_or_extract(path, resolution, FRAME_STEP)
    load_seconds = time.time() - t0

    valid = kps[:, :, 2] > CONF_THRESHOLD
    detection_rate = float(valid.any(axis=1).mean())
    per_point_cov = raw_body25_coverage(kps)

    kps_i = v12.fill_missing(kps)
    norm, conf, scale = v12.normalize_frames(kps_i)

    return {
        "filename": filename,
        "resolution": resolution,
        "kps": kps,
        "ids": ids,
        "fps": float(fps),
        "total": int(total),
        "norm": norm,
        "conf": conf,
        "scale_px": float(scale),
        "detection_rate": detection_rate,
        "body25_mean_coverage": float(np.mean(per_point_cov)),
        "body25_min_coverage": float(np.min(per_point_cov)),
        "body25_point_coverage": per_point_cov,
        "cache_status": "cache" if was_cached else "extracted",
        "load_seconds": float(load_seconds),
    }


def load_sequences_for_resolution(v12, resolution, required):
    seq = {}
    measurement_rows = []

    print("\n" + "=" * 96)
    print("Loading resolution: %s" % resolution)
    print("=" * 96)

    for i, (key, filename) in enumerate(required.items(), 1):
        print("\n[%d/%d] %s" % (i, len(required), filename))
        s = load_one_sequence(v12, filename, resolution)
        seq[key] = s

        row = {
            "resolution": resolution,
            "video_key": key,
            "filename": filename,
            "sampled_frames": len(s["ids"]),
            "original_frames": s["total"],
            "fps": s["fps"],
            "person_detection_rate": s["detection_rate"],
            "body25_mean_keypoint_coverage": s["body25_mean_coverage"],
            "body25_min_keypoint_coverage": s["body25_min_coverage"],
            "sequence_torso_scale_px": s["scale_px"],
            "cache_status": s["cache_status"],
            "load_seconds": s["load_seconds"],
        }

        # Preserve every raw BODY_25 point coverage for debugging.
        for j, name in enumerate(v12.BODY_25_PARTS):
            row["kp_%s_coverage" % name] = float(s["body25_point_coverage"][j])

        measurement_rows.append(row)

        print("  Detection %.1f%% | mean BODY_25 coverage %.1f%% | min %.1f%% | scale %.3f px | %s" % (
            100.0 * s["detection_rate"],
            100.0 * s["body25_mean_coverage"],
            100.0 * s["body25_min_coverage"],
            s["scale_px"],
            s["cache_status"],
        ))

    return seq, measurement_rows


# ======================================================================
# Pair evaluation
# ======================================================================
def named_joint_coverage(v12, coverage, name):
    for i, item in enumerate(v12.JOINT_DEFS):
        if item[0] == name:
            return float(coverage[i])
    return np.nan


def evaluate_pairs(v12, seq, pairs, resolution):
    rows = []

    print("\nEvaluating %d pairs at %s ..." % (len(pairs), resolution))

    for i, p in enumerate(pairs, 1):
        a = seq[p["reference_key"]]
        b = seq[p["test_key"]]

        t0 = time.time()
        result = v12.evaluate_pair(a["norm"], a["conf"], b["norm"], b["conf"], a["fps"], b["fps"])
        eval_seconds = time.time() - t0

        cov = np.asarray(result["coverage"], dtype=np.float64)
        timing = result["timing"]

        rk = named_joint_coverage(v12, cov, "Right Knee")
        lk = named_joint_coverage(v12, cov, "Left Knee")
        ra = named_joint_coverage(v12, cov, "Right Ankle")
        la = named_joint_coverage(v12, cov, "Left Ankle")

        row = {
            "resolution": resolution,
            "pair_id": p["pair_id"],
            "split": p["split"],
            "block": p["block"],
            "group": p["group"],
            "reference": a["filename"],
            "test": b["filename"],

            "reference_detection_rate": a["detection_rate"],
            "test_detection_rate": b["detection_rate"],
            "reference_body25_mean_coverage": a["body25_mean_coverage"],
            "test_body25_mean_coverage": b["body25_mean_coverage"],
            "reference_sequence_scale_px": a["scale_px"],
            "test_sequence_scale_px": b["scale_px"],

            "dtw_pose_distance": float(result["dtw_pose_distance"]),
            "path_length": int(len(result["path"])),
            "non_diagonal_ratio": float(timing["non_diagonal_ratio"]),
            "mean_relative_time_deviation": float(timing["mean_relative_time_deviation"]),
            "max_relative_time_deviation": float(timing["max_relative_time_deviation"]),

            "mean_joint_angle_coverage": float(np.mean(cov)),
            "min_joint_angle_coverage": float(np.min(cov)),
            "right_knee_coverage": rk,
            "left_knee_coverage": lk,
            "mean_knee_coverage": float(np.nanmean([rk, lk])),
            "right_ankle_coverage": ra,
            "left_ankle_coverage": la,
            "mean_ankle_coverage": float(np.nanmean([ra, la])),

            "smoothed_angle_error_deg": float(result["weighted_angle_error_deg"]),
            "frozen_old_mapping_score": float(result["overall_similarity_score"]),
            "evaluation_seconds": float(eval_seconds),
        }
        rows.append(row)

        print("[%2d/%2d] %-25s %s/%s | angle %6.2f deg | score %6.2f | pose %.4f | non-diag %5.1f%%" % (
            i, len(pairs), p["pair_id"], p["group"], p["block"],
            row["smoothed_angle_error_deg"],
            row["frozen_old_mapping_score"],
            row["dtw_pose_distance"],
            100.0 * row["non_diagonal_ratio"],
        ))

    return rows


# ======================================================================
# Summary
# ======================================================================
def rows_for_split(rows, resolution, split):
    if split == "development_ch01_ch06":
        return [
            r for r in rows
            if r["resolution"] == resolution
            and r["split"] in ("calibration_ch01_ch03", "holdout_ch04_ch06")
        ]
    return [r for r in rows if r["resolution"] == resolution and r["split"] == split]


def make_metric_summary(rows, baseline_threshold):
    metrics = [
        ("smoothed_angle_error_deg", "overall raw metric"),
        ("dtw_pose_distance", "alignment diagnostic"),
        ("mean_relative_time_deviation", "timing diagnostic"),
        ("non_diagonal_ratio", "DTW path diagnostic"),
    ]

    split_order = [
        "calibration_ch01_ch03",
        "holdout_ch04_ch06",
        "development_ch01_ch06",
        "final_ch07_ch10",
    ]

    out = []

    for resolution, role in RESOLUTIONS:
        for split in split_order:
            sub = rows_for_split(rows, resolution, split)
            labels = [0 if r["group"] == "D" else 1 for r in sub]

            for metric, metric_role in metrics:
                vals = [r[metric] for r in sub]
                d = [r[metric] for r in sub if r["group"] == "D"]
                e = [r[metric] for r in sub if r["group"] == "E"]
                ds, es = describe(d), describe(e)

                bal = ""
                if metric == "smoothed_angle_error_deg":
                    preds = [1 if v >= baseline_threshold else 0 for v in vals]
                    bal = balanced_accuracy(labels, preds)

                out.append({
                    "resolution": resolution,
                    "resolution_role": role,
                    "split": split,
                    "metric": metric,
                    "metric_role": metric_role,
                    "D_count": ds["count"],
                    "D_mean": ds["mean"],
                    "D_median": ds["median"],
                    "D_min": ds["min"],
                    "D_max": ds["max"],
                    "D_std": ds["std"],
                    "E_count": es["count"],
                    "E_mean": es["mean"],
                    "E_median": es["median"],
                    "E_min": es["min"],
                    "E_max": es["max"],
                    "E_std": es["std"],
                    "gap_minE_minus_maxD": es["min"] - ds["max"],
                    "auc": auc_binary(labels, vals),
                    "baseline_432_development_threshold": baseline_threshold if metric == "smoothed_angle_error_deg" else "",
                    "balanced_accuracy_with_baseline_threshold": bal,
                })

    return out


def make_resolution_delta_rows(rows):
    """Pairwise candidate-minus-baseline deltas for exactly matched pair IDs."""
    baseline = {r["pair_id"]: r for r in rows if r["resolution"] == "432x240"}
    candidate = {r["pair_id"]: r for r in rows if r["resolution"] == "-1x368"}

    metrics = [
        "reference_detection_rate",
        "test_detection_rate",
        "mean_joint_angle_coverage",
        "min_joint_angle_coverage",
        "mean_knee_coverage",
        "mean_ankle_coverage",
        "dtw_pose_distance",
        "path_length",
        "non_diagonal_ratio",
        "mean_relative_time_deviation",
        "smoothed_angle_error_deg",
        "frozen_old_mapping_score",
    ]

    out = []
    for pair_id in baseline:
        if pair_id not in candidate:
            continue

        a, b = baseline[pair_id], candidate[pair_id]
        row = {
            "pair_id": pair_id,
            "split": a["split"],
            "block": a["block"],
            "group": a["group"],
            "reference": a["reference"],
            "test": a["test"],
        }

        for metric in metrics:
            row["baseline_%s" % metric] = a[metric]
            row["candidate_%s" % metric] = b[metric]
            row["delta_candidate_minus_baseline_%s" % metric] = b[metric] - a[metric]

        out.append(row)

    return out


# ======================================================================
# Plots
# ======================================================================
def plot_detection(measurement_rows, out_path):
    base = [100.0 * r["person_detection_rate"] for r in measurement_rows if r["resolution"] == "432x240"]
    cand = [100.0 * r["person_detection_rate"] for r in measurement_rows if r["resolution"] == "-1x368"]

    plt.figure(figsize=(7.5, 5.5))
    plt.boxplot([base, cand], labels=["432x240", "-1x368"], showmeans=True)
    plt.ylabel("Person Detection Rate (%)")
    plt.title("OpenPose Person Detection by Resolution")
    plt.ylim(0, 102)
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def plot_angle_distributions(pair_rows, out_path):
    data, labels = [], []

    for split in ("development_ch01_ch06", "final_ch07_ch10"):
        for resolution, _ in RESOLUTIONS:
            sub = rows_for_split(pair_rows, resolution, split)
            for group in ("D", "E"):
                data.append([r["smoothed_angle_error_deg"] for r in sub if r["group"] == group])
                short_split = "DEV" if split == "development_ch01_ch06" else "FINAL"
                short_res = "432" if resolution == "432x240" else "368"
                labels.append("%s-%s-%s" % (short_split, short_res, group))

    plt.figure(figsize=(12, 6))
    plt.boxplot(data, labels=labels, showmeans=True)
    plt.ylabel("Smoothed Weighted Joint-Angle Error (deg)")
    plt.title("D/E Angle-Error Distributions: 432x240 vs -1x368")
    plt.xticks(rotation=30, ha="right")
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def plot_paired_angle_scatter(pair_rows, out_path):
    baseline = {r["pair_id"]: r for r in pair_rows if r["resolution"] == "432x240"}
    candidate = {r["pair_id"]: r for r in pair_rows if r["resolution"] == "-1x368"}

    plt.figure(figsize=(7, 7))

    for group, marker in (("D", "o"), ("E", "x")):
        xs, ys = [], []
        for pair_id, a in baseline.items():
            if pair_id in candidate and a["group"] == group:
                xs.append(a["smoothed_angle_error_deg"])
                ys.append(candidate[pair_id]["smoothed_angle_error_deg"])
        plt.scatter(xs, ys, marker=marker, label=group)

    all_vals = [
        r["smoothed_angle_error_deg"]
        for r in pair_rows
        if np.isfinite(r["smoothed_angle_error_deg"])
    ]
    lo = min(all_vals) if all_vals else 0.0
    hi = max(all_vals) if all_vals else 50.0
    pad = max(1.0, 0.05 * (hi - lo))
    plt.plot([lo - pad, hi + pad], [lo - pad, hi + pad], linestyle="--", linewidth=1.0, label="No change")
    plt.xlim(lo - pad, hi + pad)
    plt.ylim(lo - pad, hi + pad)
    plt.xlabel("432x240 Angle Error (deg)")
    plt.ylabel("-1x368 Angle Error (deg)")
    plt.title("Pairwise Effect of Higher OpenPose Resolution")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


# ======================================================================
# Console comparison
# ======================================================================
def find_summary(summary_rows, resolution, split, metric):
    for r in summary_rows:
        if r["resolution"] == resolution and r["split"] == split and r["metric"] == metric:
            return r
    return None


def print_key_comparison(summary_rows, measurement_rows, baseline_threshold):
    print("\n" + "=" * 104)
    print("PHASE 3.6A KEY COMPARISON")
    print("=" * 104)

    for resolution, role in RESOLUTIONS:
        m = [r for r in measurement_rows if r["resolution"] == resolution]
        det = describe([r["person_detection_rate"] for r in m])
        raw_cov = describe([r["body25_mean_keypoint_coverage"] for r in m])

        print("\n%s [%s]" % (resolution, role))
        print("  Video person detection : mean %.1f%% | min %.1f%%" % (100.0 * det["mean"], 100.0 * det["min"]))
        print("  BODY_25 mean coverage  : mean %.1f%% | min-video %.1f%%" % (100.0 * raw_cov["mean"], 100.0 * raw_cov["min"]))

        for split in ("development_ch01_ch06", "final_ch07_ch10"):
            angle = find_summary(summary_rows, resolution, split, "smoothed_angle_error_deg")
            pose = find_summary(summary_rows, resolution, split, "dtw_pose_distance")
            timing = find_summary(summary_rows, resolution, split, "mean_relative_time_deviation")
            nd = find_summary(summary_rows, resolution, split, "non_diagonal_ratio")

            print("\n  %s" % split)
            print("    Angle D/E mean : %.3f / %.3f deg | AUC %.3f | BalAcc(frozen baseline threshold) %.3f" % (
                angle["D_mean"], angle["E_mean"], angle["auc"], angle["balanced_accuracy_with_baseline_threshold"]))
            print("    Angle gap      : min(E)-max(D) = %+.3f deg" % angle["gap_minE_minus_maxD"])
            print("    Pose D/E mean  : %.4f / %.4f | AUC %.3f" % (pose["D_mean"], pose["E_mean"], pose["auc"]))
            print("    Timing D/E mean: %.5f / %.5f | AUC %.3f" % (timing["D_mean"], timing["E_mean"], timing["auc"]))
            print("    Non-diag D/E   : %.2f%% / %.2f%%" % (100.0 * nd["D_mean"], 100.0 * nd["E_mean"]))

    print("\nFrozen threshold used only for compatibility checking:")
    print("  432x240 development midpoint of D/E medians = %.6f deg" % baseline_threshold)
    print("\nInterpretation rule:")
    print("  - Do NOT refit SCORE_CENTER/SCORE_SCALE here.")
    print("  - Prefer -1x368 only if measurement reliability improves broadly and")
    print("    D/E angle-error discrimination is preserved or improved.")
    print("  - Old 0-100 score compatibility is checked only after this Phase 3.6A result.")


# ======================================================================
# Main
# ======================================================================
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    v12 = load_v12()

    print("=" * 104)
    print("Phase 3.6A - OpenPose Resolution Downstream Validation")
    print("=" * 104)
    print("Baseline resolution : 432x240")
    print("Candidate resolution: -1x368")
    print("Only changed variable: NET_RESOLUTION")
    print("FRAME_STEP          : %d [FROZEN]" % FRAME_STEP)
    print("CONF_THRESHOLD      : %.2f [FROZEN]" % CONF_THRESHOLD)
    print("DTW_WINDOW_RATIO    : %.2f [FROZEN]" % DTW_WINDOW_RATIO)
    print("Normalization       : SEQUENCE_MEDIAN [FROZEN]")
    print("Angle smoothing     : 3-frame median [FROZEN]")
    print("Score mapping       : center %.6f / scale %.6f [FROZEN; NOT REFIT]" % (SCORE_CENTER, SCORE_SCALE))

    required = collect_required_videos()
    missing = [
        os.path.join(VIDEO_DIR, filename)
        for filename in required.values()
        if not os.path.isfile(os.path.join(VIDEO_DIR, filename))
    ]

    if missing:
        print("\n[ERROR] Missing required videos:")
        for p in missing:
            print(" ", p)
        return

    pairs = build_all_pairs()
    print("\nRequired unique videos: %d" % len(required))
    print("Pair matrix:")
    for split_name, choreos in SPLITS.items():
        sub = build_pairs(split_name, choreos)
        print("  %-24s : %d D + %d E = %d pairs" % (
            split_name,
            sum(p["group"] == "D" for p in sub),
            sum(p["group"] == "E" for p in sub),
            len(sub),
        ))
    print("  Total                    : %d pairs per resolution" % len(pairs))

    all_measurement_rows = []
    all_pair_rows = []

    for resolution, role in RESOLUTIONS:
        seq, measurement_rows = load_sequences_for_resolution(v12, resolution, required)
        pair_rows = evaluate_pairs(v12, seq, pairs, resolution)
        all_measurement_rows.extend(measurement_rows)
        all_pair_rows.extend(pair_rows)

    # Baseline frozen threshold: development ch01-ch06 midpoint of D/E medians.
    baseline_dev = rows_for_split(all_pair_rows, "432x240", "development_ch01_ch06")
    d_med = float(np.median([r["smoothed_angle_error_deg"] for r in baseline_dev if r["group"] == "D"]))
    e_med = float(np.median([r["smoothed_angle_error_deg"] for r in baseline_dev if r["group"] == "E"]))
    baseline_threshold = 0.5 * (d_med + e_med)

    summary_rows = make_metric_summary(all_pair_rows, baseline_threshold)
    delta_rows = make_resolution_delta_rows(all_pair_rows)

    measurement_csv = os.path.join(OUT_DIR, "resolution_sequence_measurement.csv")
    pair_csv = os.path.join(OUT_DIR, "resolution_pair_results.csv")
    summary_csv = os.path.join(OUT_DIR, "resolution_summary.csv")
    delta_csv = os.path.join(OUT_DIR, "resolution_pair_deltas.csv")

    write_csv(measurement_csv, all_measurement_rows)
    write_csv(pair_csv, all_pair_rows)
    write_csv(summary_csv, summary_rows)
    write_csv(delta_csv, delta_rows)

    detection_png = os.path.join(OUT_DIR, "resolution_detection_comparison.png")
    angle_png = os.path.join(OUT_DIR, "resolution_angle_distribution.png")
    scatter_png = os.path.join(OUT_DIR, "resolution_pairwise_angle_scatter.png")

    plot_detection(all_measurement_rows, detection_png)
    plot_angle_distributions(all_pair_rows, angle_png)
    plot_paired_angle_scatter(all_pair_rows, scatter_png)

    print_key_comparison(summary_rows, all_measurement_rows, baseline_threshold)

    print("\nSaved:")
    print(" ", measurement_csv)
    print(" ", pair_csv)
    print(" ", summary_csv)
    print(" ", delta_csv)
    print(" ", detection_png)
    print(" ", angle_png)
    print(" ", scatter_png)

    print("\nPhase 3.6A complete.")
    print("Do not change the frozen score mapping yet. Analyze these A/B results first.")


if __name__ == "__main__":
    main()
