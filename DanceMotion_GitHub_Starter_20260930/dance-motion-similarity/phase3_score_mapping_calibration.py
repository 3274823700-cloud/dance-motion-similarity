# -*- coding: utf-8 -*-
"""
Phase 3.5A - Overall 0-100 Score Mapping Calibration

Final architecture candidate
----------------------------
DTW alignment cost:
    SEQUENCE_MEDIAN normalized-coordinate Pose Distance

Overall motion score:
    3-frame-median weighted joint-angle error

Timing:
    kept separate for later rhythm/timing feedback

Development data
----------------
ch01-ch06 are now development data.
ch07-ch10 remain untouched final-validation data.

Score mapping
-------------
We use a simple logistic mapping:

    score(error) = 100 / (1 + exp((error - center) / scale))

The parameters are derived from DEVELOPMENT data only:

    median same-choreography D error -> 80 points
    median different-choreography E error -> 50 points

These are transparent engineering anchors, NOT human-rated ground truth.
No "Excellent/Good/Pass" semantic labels are assigned yet.

The script also checks:
A) same video vs itself
B) temporal subsampling sanity check
C) same dancer + same choreography + different music

Expected role:
- A should be ~100
- B should remain very high
- C should also be high
"""

import os
import csv
import math
import importlib.util
import numpy as np
import matplotlib.pyplot as plt


# ======================================================================
# Configuration
# ======================================================================
V11_FILE = "Dance_Motion_Similarity_Evaluation_V1.1_final.py"
VIDEO_DIR = os.path.join("data", "aistpp", "videos")

DEVELOPMENT_CSV = os.path.join(
    "phase3_normalization_downstream",
    "normalization_downstream_pair_results.csv",
)

OUT_DIR = "phase3_score_mapping"

NORMALIZATION = "SEQUENCE_MEDIAN"
D_ANCHOR_SCORE = 80.0
E_ANCHOR_SCORE = 50.0
SMOOTH_RADIUS = 1

# Sanity-check videos
A_VIDEO = "gBR_sBM_c01_d04_mBR0_ch01.mp4"
C_VIDEO = "gBR_sBM_c01_d04_mBR1_ch01.mp4"


# ======================================================================
# Generic helpers
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
        raise IOError("Development CSV not found: %s" % path)

    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


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


def write_csv(path, rows):
    if not rows:
        return

    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)


# ======================================================================
# Logistic mapping
# ======================================================================
def logit(p):
    return math.log(p / (1.0 - p))


def fit_logistic_from_two_anchors(d_error, d_score, e_error, e_score):
    """
    score/100 = 1 / (1 + exp((error-center)/scale))

    logit(score/100) = (center-error)/scale
    """
    pd = d_score / 100.0
    pe = e_score / 100.0

    ld = logit(pd)
    le = logit(pe)

    if abs(ld - le) < 1e-12:
        raise ValueError("Anchor scores must be different.")

    scale = (e_error - d_error) / (ld - le)
    center = d_error + scale * ld

    if scale <= 0:
        raise ValueError(
            "Invalid mapping: expected E error > D error and D score > E score."
        )

    return float(center), float(scale)


def error_to_score(error_deg, center, scale):
    z = (float(error_deg) - center) / scale

    # Numerically stable enough for our error range.
    if z > 60:
        return 0.0
    if z < -60:
        return 100.0

    return float(100.0 / (1.0 + math.exp(z)))


# ======================================================================
# Sequence-median normalization
# ======================================================================
def frame_hip_and_torso(v11, xy, conf):
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
    S = len(kps)
    conf = kps[:, :, 2].copy()
    hips = []
    torsos = []

    for i in range(S):
        hip, torso = frame_hip_and_torso(
            v11, kps[i, :, :2], conf[i]
        )
        hips.append(hip)
        torsos.append(torso)

    torsos = np.asarray(torsos, dtype=np.float64)
    valid = torsos[np.isfinite(torsos)]

    scale = float(np.median(valid)) if len(valid) else 1.0
    if scale < 1e-6:
        scale = 1.0

    norm = np.zeros((S, v11.N_PARTS, 2), dtype=np.float64)

    for i in range(S):
        norm[i] = (kps[i, :, :2] - hips[i]) / scale

    return norm, conf


# ======================================================================
# Load one video under the candidate final pipeline
# ======================================================================
def load_sequence(v11, filename):
    path = os.path.join(VIDEO_DIR, filename)

    if not os.path.isfile(path):
        raise IOError("Video not found: %s" % path)

    kps, ids, fps, total = v11.get_or_extract(
        path,
        v11.NET_RESOLUTION,
        v11.FRAME_STEP,
    )

    kps_i = v11.fill_missing(kps)
    norm, conf = normalize_sequence_median(v11, kps_i)

    raw_angles = v11.build_joint_angle_sequence(norm, conf)
    smooth_angles = v11.median_smooth_angle_sequence(
        raw_angles, SMOOTH_RADIUS
    )

    return {
        "filename": filename,
        "norm": norm,
        "conf": conf,
        "angles": smooth_angles,
    }


def aligned_angle_error(v11, a, b):
    cost = v11.build_cost_matrix(
        a["norm"], a["conf"],
        b["norm"], b["conf"],
    )

    path, pose_dist, win = v11.dtw_align(
        cost, v11.DTW_WINDOW_RATIO
    )

    joint_err = v11.per_joint_path_mean_errors(
        a["angles"], b["angles"], path
    )

    overall = v11.weighted_overall_from_joint_errors(
        joint_err
    )

    return float(overall), float(pose_dist), path


# ======================================================================
# Development data mapping
# ======================================================================
def load_development_rows():
    rows = read_csv_dict(DEVELOPMENT_CSV)

    selected = [
        r for r in rows
        if r["normalization"] == NORMALIZATION
    ]

    if len(selected) != 36:
        print(
            "[Warning] Expected 36 SEQUENCE_MEDIAN D/E rows across ch01-ch06, got %d."
            % len(selected)
        )

    result = []

    for r in selected:
        result.append({
            "pair_id": r["pair_id"],
            "split": r["split"],
            "block": r["block"],
            "group": r["group"],
            "reference": r["reference"],
            "test": r["test"],
            "angle_error_deg": float(
                r["smoothed_angle_error_deg"]
            ),
        })

    return result


# ======================================================================
# Sanity checks A/B/C
# ======================================================================
def run_sanity_checks(v11, center, scale):
    rows = []

    a = load_sequence(v11, A_VIDEO)

    # --------------------------------------------------------------
    # A: same video vs itself
    # --------------------------------------------------------------
    err_a, pose_a, path_a = aligned_angle_error(v11, a, a)

    rows.append({
        "case": "A_same_video",
        "description": "same video vs itself",
        "angle_error_deg": err_a,
        "overall_score": error_to_score(err_a, center, scale),
        "dtw_pose_distance": pose_a,
        "path_length": len(path_a),
    })

    # --------------------------------------------------------------
    # B: pure temporal subsampling
    # Same already-extracted sequence, keep every second sampled frame.
    # --------------------------------------------------------------
    b = {
        "filename": "temporal_subsample_of_" + A_VIDEO,
        "norm": a["norm"][::2].copy(),
        "conf": a["conf"][::2].copy(),
        "angles": a["angles"][::2].copy(),
    }

    err_b, pose_b, path_b = aligned_angle_error(v11, a, b)

    rows.append({
        "case": "B_temporal_subsample",
        "description": "same skeleton sequence, every second sampled frame",
        "angle_error_deg": err_b,
        "overall_score": error_to_score(err_b, center, scale),
        "dtw_pose_distance": pose_b,
        "path_length": len(path_b),
    })

    # --------------------------------------------------------------
    # C: same dancer + same choreography + different music
    # --------------------------------------------------------------
    c_path = os.path.join(VIDEO_DIR, C_VIDEO)

    if os.path.isfile(c_path):
        c = load_sequence(v11, C_VIDEO)
        err_c, pose_c, path_c = aligned_angle_error(v11, a, c)

        rows.append({
            "case": "C_same_dancer_same_choreo_different_music",
            "description": "same dancer/choreography, different music",
            "angle_error_deg": err_c,
            "overall_score": error_to_score(err_c, center, scale),
            "dtw_pose_distance": pose_c,
            "path_length": len(path_c),
        })
    else:
        print(
            "[Warning] C sanity video missing, skipped: %s"
            % C_VIDEO
        )

    return rows


# ======================================================================
# Plot
# ======================================================================
def plot_mapping(center, scale, d_errors, e_errors, out_path):
    max_err = max(
        50.0,
        max(d_errors + e_errors) + 5.0,
    )

    x = np.linspace(0.0, max_err, 400)
    y = [
        error_to_score(v, center, scale)
        for v in x
    ]

    plt.figure(figsize=(8, 6))
    plt.plot(x, y, label="Overall score mapping")

    plt.scatter(
        d_errors,
        [error_to_score(v, center, scale) for v in d_errors],
        marker="o",
        label="D: same choreography",
    )

    plt.scatter(
        e_errors,
        [error_to_score(v, center, scale) for v in e_errors],
        marker="x",
        label="E: different choreography",
    )

    plt.xlabel("Smoothed Weighted Joint-Angle Error (deg)")
    plt.ylabel("Overall Similarity Score")
    plt.ylim(0, 102)
    plt.title("Phase 3.5A: Development-Calibrated 0-100 Mapping")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


# ======================================================================
# Main
# ======================================================================
def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print("=" * 92)
    print("Phase 3.5A - Overall 0-100 Score Mapping Calibration")
    print("=" * 92)
    print("Development data : ch01-ch06")
    print("Final validation : ch07-ch10 untouched")
    print("Overall metric   : 3-frame-median weighted joint-angle error")
    print("DTW normalization: SEQUENCE_MEDIAN")
    print("D median anchor  : %.1f points" % D_ANCHOR_SCORE)
    print("E median anchor  : %.1f points" % E_ANCHOR_SCORE)
    print("Semantic grades  : NOT assigned")

    dev = load_development_rows()

    d_errors = [
        r["angle_error_deg"]
        for r in dev if r["group"] == "D"
    ]

    e_errors = [
        r["angle_error_deg"]
        for r in dev if r["group"] == "E"
    ]

    d_stats = describe(d_errors)
    e_stats = describe(e_errors)

    center, scale = fit_logistic_from_two_anchors(
        d_stats["median"],
        D_ANCHOR_SCORE,
        e_stats["median"],
        E_ANCHOR_SCORE,
    )

    print("\nDevelopment angle-error distributions")
    print(
        "  D: mean %.3f | median %.3f | min %.3f | max %.3f | std %.3f"
        % (
            d_stats["mean"], d_stats["median"],
            d_stats["min"], d_stats["max"], d_stats["std"],
        )
    )
    print(
        "  E: mean %.3f | median %.3f | min %.3f | max %.3f | std %.3f"
        % (
            e_stats["mean"], e_stats["median"],
            e_stats["min"], e_stats["max"], e_stats["std"],
        )
    )

    print("\nLogistic mapping")
    print("  center = %.6f deg" % center)
    print("  scale  = %.6f deg" % scale)
    print(
        "  score(error) = 100 / (1 + exp((error - %.6f) / %.6f))"
        % (center, scale)
    )

    # Map all development pairs.
    mapped_rows = []

    for r in dev:
        out = dict(r)
        out["overall_score"] = error_to_score(
            r["angle_error_deg"], center, scale
        )
        mapped_rows.append(out)

    print("\nDevelopment score distributions")

    for group in ("D", "E"):
        vals = [
            r["overall_score"]
            for r in mapped_rows
            if r["group"] == group
        ]
        st = describe(vals)
        print(
            "  %s: mean %.2f | median %.2f | min %.2f | max %.2f | std %.2f"
            % (
                group,
                st["mean"], st["median"],
                st["min"], st["max"], st["std"],
            )
        )

    # A/B/C checks
    v11 = load_v11()
    sanity_rows = run_sanity_checks(
        v11, center, scale
    )

    print("\nSanity checks")

    for r in sanity_rows:
        print(
            "  %-43s | angle %7.3f deg | score %6.2f | pose %.6f"
            % (
                r["case"],
                r["angle_error_deg"],
                r["overall_score"],
                r["dtw_pose_distance"],
            )
        )

    # Example mapping table
    table_errors = [
        0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50
    ]

    mapping_rows = [
        {
            "angle_error_deg": e,
            "overall_score": error_to_score(
                e, center, scale
            ),
        }
        for e in table_errors
    ]

    parameter_rows = [{
        "development_normalization": NORMALIZATION,
        "development_choreographies": "ch01-ch06",
        "reserved_final_validation": "ch07-ch10",
        "D_median_error_deg": d_stats["median"],
        "D_anchor_score": D_ANCHOR_SCORE,
        "E_median_error_deg": e_stats["median"],
        "E_anchor_score": E_ANCHOR_SCORE,
        "logistic_center_deg": center,
        "logistic_scale_deg": scale,
    }]

    param_csv = os.path.join(
        OUT_DIR, "score_mapping_parameters.csv"
    )
    dev_csv = os.path.join(
        OUT_DIR, "development_score_results.csv"
    )
    sanity_csv = os.path.join(
        OUT_DIR, "score_mapping_sanity_checks.csv"
    )
    table_csv = os.path.join(
        OUT_DIR, "score_mapping_table.csv"
    )
    plot_png = os.path.join(
        OUT_DIR, "score_mapping_curve.png"
    )

    write_csv(param_csv, parameter_rows)
    write_csv(dev_csv, mapped_rows)
    write_csv(sanity_csv, sanity_rows)
    write_csv(table_csv, mapping_rows)

    plot_mapping(
        center, scale,
        d_errors, e_errors,
        plot_png,
    )

    print("\nImportant:")
    print(
        "  This 0-100 value is an empirically anchored similarity index,"
    )
    print(
        "  not a human-validated dance-quality grade."
    )
    print(
        "  Do NOT tune center/scale again after looking at ch07-ch10."
    )
    print(
        "  If these sanity checks are reasonable, freeze the mapping and"
    )
    print(
        "  use ch07-ch10 exactly once for final validation."
    )

    print("\nSaved:")
    print(" ", param_csv)
    print(" ", dev_csv)
    print(" ", sanity_csv)
    print(" ", table_csv)
    print(" ", plot_png)


if __name__ == "__main__":
    main()
