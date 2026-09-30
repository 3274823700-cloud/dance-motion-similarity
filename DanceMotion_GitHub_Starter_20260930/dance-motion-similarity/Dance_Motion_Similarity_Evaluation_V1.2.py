# -*- coding: utf-8 -*-
"""
================================================================================
Dance Motion Similarity Evaluation V1.2
(OpenPose BODY_25 + SEQUENCE_MEDIAN Normalization + DTW + Smoothed Joint Angles)
--------------------------------------------------------------------------------
V1.2 is the final-system integration of the validated Phase 3 design.

Frozen design used here:
  1. OpenPose backend remains decoupled from Python:
       Video -> OpenPoseDemo.exe -> BODY_25 JSON -> Python
  2. Existing JSON cache is reused whenever possible.
  3. FRAME_STEP = 2, NET_RESOLUTION = "-1x368", CONF_THRESHOLD = 0.30.
  4. Pose normalization uses one SEQUENCE_MEDIAN torso scale per video while
     keeping a per-frame hip origin.
  5. Pose Distance is used only as the DTW alignment cost / diagnostic.
  6. DTW_WINDOW_RATIO remains 0.30.
  7. Joint-angle sequences use a 3-frame median filter (SMOOTH_RADIUS = 1).
  8. Per-joint errors use ordinary DTW path-pair averaging.
  9. Overall raw metric is the frozen weighted smoothed joint-angle error.
 10. Overall Similarity Score uses the frozen logistic mapping:
       100 / (1 + exp((error - 33.046977) / 6.292398))
 11. Timing is reported separately using DTW relative-time deviation.
 12. No Excellent/Good/Pass/Fail semantic grades are assigned.

This file intentionally excludes Phase 3 calibration / A-B-test / holdout code.
================================================================================
"""

import os
import sys
import time
import csv
import json
import re
import shutil
import subprocess
import math

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ======================================================================
# Step 1: Configuration
# ======================================================================
DEMO_VIDEO = r"data\aistpp\videos\gBR_sBM_c01_d04_mBR0_ch10.mp4"
IMIT_VIDEO = r"data\aistpp\videos\gBR_sBM_c01_d05_mBR0_ch10.mp4"

# ---- OpenPose executable backend ----
OPENPOSE_ROOT = r"D:\OpenPose\openpose"
OPENPOSE_DEMO = os.path.join(OPENPOSE_ROOT, "bin", "OpenPoseDemo.exe")
OPENPOSE_MODELS = os.path.join(OPENPOSE_ROOT, "models")

# ---- Frozen extraction / sampling settings ----
NET_RESOLUTION = "-1x368"
FRAME_STEP = 2
CONF_THRESHOLD = 0.30

# ---- Skeleton JSON cache ----
USE_CACHE = True
FORCE_REEXTRACT = False
CACHE_DIR = "pose_cache_json"

# ---- Frozen DTW setting ----
DTW_WINDOW_RATIO = 0.30

# ---- Frozen final scoring settings ----
SMOOTH_RADIUS = 1               # radius=1 -> 3-frame median
SCORE_CENTER = 33.046977        # degrees
SCORE_SCALE = 6.292398          # degrees

# ---- Output ----
OUT_DIR = "similarity_result_v12"
MAKE_CSV = True
MAKE_PLOTS = True
PLOT_DPI = 180
MAKE_COMPARE_VIDEO = True
COMPARE_DISPLAY_HEIGHT = 540
COMPARE_FPS = None  # None -> Reference sampled FPS = reference_fps / FRAME_STEP


# ======================================================================
# Step 2: BODY_25 constants and frozen joint definitions
# ======================================================================
BODY_25_PARTS = [
    "Nose", "Neck", "RShoulder", "RElbow", "RWrist", "LShoulder", "LElbow", "LWrist",
    "MidHip", "RHip", "RKnee", "RAnkle", "LHip", "LKnee", "LAnkle",
    "REye", "LEye", "REar", "LEar", "LBigToe", "LSmallToe", "LHeel",
    "RBigToe", "RSmallToe", "RHeel",
]
N_PARTS = 25

MID_HIP, NECK = 8, 1
R_SHOULDER, R_ELBOW, R_WRIST = 2, 3, 4
L_SHOULDER, L_ELBOW, L_WRIST = 5, 6, 7
R_HIP, R_KNEE, R_ANKLE = 9, 10, 11
L_HIP, L_KNEE, L_ANKLE = 12, 13, 14
R_HEEL, L_HEEL = 24, 21

# BODY_25 bone connections used only for visualization.
BODY_25_PAIRS = [
    (1, 8), (1, 2), (1, 5), (2, 3), (3, 4), (5, 6), (6, 7),
    (8, 9), (9, 10), (10, 11), (8, 12), (12, 13), (13, 14),
    (1, 0), (0, 15), (15, 17), (0, 16), (16, 18),
    (14, 19), (19, 20), (14, 21), (11, 22), (22, 23), (11, 24),
]

# (name, vertex, endpoint1, endpoint2, weight); frozen weights sum to 1.0.
JOINT_DEFS = [
    ("Right Elbow",    R_ELBOW,    R_SHOULDER, R_WRIST,  0.14),
    ("Left Elbow",     L_ELBOW,    L_SHOULDER, L_WRIST,  0.14),
    ("Right Shoulder", R_SHOULDER, NECK,       R_ELBOW,  0.09),
    ("Left Shoulder",  L_SHOULDER, NECK,       L_ELBOW,  0.09),
    ("Right Hip",      R_HIP,      MID_HIP,    R_KNEE,   0.07),
    ("Left Hip",       L_HIP,      MID_HIP,    L_KNEE,   0.07),
    ("Right Knee",     R_KNEE,     R_HIP,      R_ANKLE,  0.14),
    ("Left Knee",      L_KNEE,     L_HIP,      L_ANKLE,  0.14),
    ("Right Ankle",    R_ANKLE,    R_KNEE,     R_HEEL,   0.04),
    ("Left Ankle",     L_ANKLE,    L_KNEE,     L_HEEL,   0.04),
    ("Torso Tilt",     NECK,       MID_HIP,    -1,       0.04),
]

# Body-part reporting only. These groups NEVER replace or modify JOINT_DEFS weights.
BODY_PART_GROUPS = {
    "Elbow": ("Right Elbow", "Left Elbow"),
    "Shoulder": ("Right Shoulder", "Left Shoulder"),
    "Hip": ("Right Hip", "Left Hip"),
    "Knee": ("Right Knee", "Left Knee"),
    "Ankle": ("Right Ankle", "Left Ankle"),
    "Torso": ("Torso Tilt",),
}


# ======================================================================
# Step 3: Extract / load BODY_25 skeleton sequences
# ======================================================================
def read_openpose_json(json_path):
    """Read all BODY_25 people from one OpenPose JSON file. Returns (P,25,3), or None."""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    people = []
    for person in data.get("people", []):
        raw = np.asarray(person.get("pose_keypoints_2d", []), dtype=np.float32)
        if raw.size == N_PARTS * 3:
            people.append(raw.reshape(N_PARTS, 3))
    return np.stack(people) if people else None


def pick_primary(pose_keypoints):
    """Select primary person using valid-keypoint count first, then mean confidence."""
    if pose_keypoints is None or not hasattr(pose_keypoints, "shape") or len(pose_keypoints.shape) != 3 or pose_keypoints.shape[0] == 0:
        return None

    best, best_score = None, -1.0
    for person in pose_keypoints:
        valid_mask = person[:, 2] > CONF_THRESHOLD
        n_valid = int(valid_mask.sum())
        mean_conf = float(person[valid_mask, 2].mean()) if n_valid else 0.0
        score = n_valid * 10.0 + mean_conf
        if score > best_score:
            best_score, best = score, person
    return best


def _frame_id_from_json_name(path, fallback):
    """Parse original OpenPose frame number from ..._000000000123_keypoints.json."""
    match = re.search(r"_(\d+)_keypoints\.json$", os.path.basename(path))
    return int(match.group(1)) if match else int(fallback)


def load_json_sequence(json_dir, frame_step=1):
    """Load cached OpenPose JSON folder and return kps(S,25,3) + original frame IDs."""
    json_files = sorted([os.path.join(json_dir, name) for name in os.listdir(json_dir) if name.endswith("_keypoints.json")])
    if not json_files:
        raise IOError("No OpenPose JSON files found in: %s" % json_dir)

    kps, frame_ids = [], []
    for file_idx, json_path in enumerate(json_files):
        frame_id = _frame_id_from_json_name(json_path, file_idx)
        if frame_id % frame_step != 0:
            continue

        primary = pick_primary(read_openpose_json(json_path))
        if primary is None:
            primary = np.zeros((N_PARTS, 3), dtype=np.float32)

        kps.append(primary.astype(np.float32))
        frame_ids.append(frame_id)

    if not kps:
        raise RuntimeError("No frames remained after FRAME_STEP=%d in: %s" % (frame_step, json_dir))
    return np.stack(kps), np.asarray(frame_ids, dtype=np.int64)


def _video_info(video_path):
    """Return original video fps and frame count."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError("Cannot open video: %s" % video_path)

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    cap.release()
    return fps, total


def _cache_json_dir(video_path, net_resolution):
    """Create stable cache-folder name from video basename and OpenPose resolution."""
    base = os.path.splitext(os.path.basename(video_path))[0]
    resolution_tag = net_resolution.replace("-", "auto").replace(":", "_")
    return os.path.join(CACHE_DIR, "%s_%s" % (base, resolution_tag))


def extract_sequence(video_path, net_resolution, frame_step, json_dir):
    """Run OpenPoseDemo.exe, write BODY_25 JSON, then load sampled sequence."""
    if not os.path.isfile(OPENPOSE_DEMO):
        raise IOError("OpenPoseDemo.exe not found: %s" % OPENPOSE_DEMO)
    if not os.path.isdir(OPENPOSE_MODELS):
        raise IOError("OpenPose model folder not found: %s" % OPENPOSE_MODELS)

    fps, total = _video_info(video_path)

    if os.path.isdir(json_dir):
        shutil.rmtree(json_dir)
    os.makedirs(json_dir, exist_ok=True)

    command = [
        OPENPOSE_DEMO,
        "--video", os.path.abspath(video_path),
        "--model_folder", OPENPOSE_MODELS,
        "--model_pose", "BODY_25",
        "--net_resolution", net_resolution,
        "--write_json", os.path.abspath(json_dir),
        "--display", "0",
        "--render_pose", "0",
    ]

    print("  Running OpenPoseDemo.exe ...")
    print("  JSON cache: %s" % os.path.abspath(json_dir))

    try:
        subprocess.run(command, cwd=OPENPOSE_ROOT, check=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError("OpenPoseDemo.exe failed with exit code %s." % e.returncode)

    kps, frame_ids = load_json_sequence(json_dir, frame_step)
    with open(os.path.join(json_dir, "_complete.txt"), "w", encoding="utf-8") as f:
        f.write("OpenPose JSON extraction completed successfully.\n")

    return kps, frame_ids, fps, total


def get_or_extract(video_path, net_resolution, frame_step):
    """Video -> skeleton sequence. Reuse completed JSON cache unless FORCE_REEXTRACT=True."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    json_dir = _cache_json_dir(video_path, net_resolution)
    complete_flag = os.path.join(json_dir, "_complete.txt")

    if USE_CACHE and not FORCE_REEXTRACT and os.path.isfile(complete_flag):
        fps, total = _video_info(video_path)
        kps, frame_ids = load_json_sequence(json_dir, frame_step)
        print("  Loaded from JSON cache: %s (%d sampled frames)" % (json_dir, len(kps)))
        return kps, frame_ids, fps, total

    print("  Extracting skeletons with OpenPoseDemo.exe (first run may be slow; JSON is cached afterwards)...")
    kps, frame_ids, fps, total = extract_sequence(video_path, net_resolution, frame_step, json_dir)
    print("  Skeleton JSON cached: %s" % json_dir)
    return kps, frame_ids, fps, total


# ======================================================================
# Step 4: Missing-keypoint handling
# ======================================================================
def fill_missing(kps):
    """Interpolate missing x/y through time; confidence remains unchanged."""
    out = kps.copy()
    S = out.shape[0]

    for j in range(N_PARTS):
        valid = kps[:, j, 2] > CONF_THRESHOLD
        idxs = np.where(valid)[0]
        if len(idxs) == 0:
            continue
        for axis in (0, 1):
            out[:, j, axis] = np.interp(np.arange(S), idxs, kps[idxs, j, axis])
    return out


# ======================================================================
# Step 5: Frozen SEQUENCE_MEDIAN normalization
# ======================================================================
def frame_hip_and_torso(xy, conf):
    """Return per-frame hip origin and valid Neck/Hip torso length (or NaN)."""
    if conf[MID_HIP] > CONF_THRESHOLD:
        hip = xy[MID_HIP]
        hip_valid = True
    elif conf[R_HIP] > CONF_THRESHOLD and conf[L_HIP] > CONF_THRESHOLD:
        hip = 0.5 * (xy[R_HIP] + xy[L_HIP])
        hip_valid = True
    else:
        hip = np.zeros(2, dtype=np.float64)
        hip_valid = False

    if conf[NECK] > CONF_THRESHOLD:
        neck = xy[NECK]
        neck_valid = True
    elif conf[R_SHOULDER] > CONF_THRESHOLD and conf[L_SHOULDER] > CONF_THRESHOLD:
        neck = 0.5 * (xy[R_SHOULDER] + xy[L_SHOULDER])
        neck_valid = True
    else:
        neck = hip
        neck_valid = False

    if hip_valid and neck_valid:
        torso = float(np.linalg.norm(neck - hip))
        if torso >= 1e-6:
            return hip, torso
    return hip, np.nan


def normalize_frames(kps):
    """Frozen V1.2 normalization: per-frame hip origin + one median torso scale per sequence."""
    S = len(kps)
    conf = kps[:, :, 2].copy()
    hips = []
    torsos = []

    for i in range(S):
        hip, torso = frame_hip_and_torso(kps[i, :, :2], conf[i])
        hips.append(hip)
        torsos.append(torso)

    torsos = np.asarray(torsos, dtype=np.float64)
    valid_torsos = torsos[np.isfinite(torsos)]
    sequence_scale = float(np.median(valid_torsos)) if len(valid_torsos) else 1.0
    if sequence_scale < 1e-6:
        sequence_scale = 1.0

    norm_xy = np.zeros((S, N_PARTS, 2), dtype=np.float64)
    for i in range(S):
        norm_xy[i] = (kps[i, :, :2] - hips[i]) / sequence_scale

    return norm_xy, conf, sequence_scale


# ======================================================================
# Step 6: Pose Cost Matrix + DTW
# ======================================================================
def _pairwise_cost_block(a_xy, a_c, b_xy, b_c):
    """Confidence-weighted normalized-coordinate pose distance for frame blocks."""
    diff = a_xy[:, None, :, :] - b_xy[None, :, :, :]
    w = np.minimum(a_c[:, None, :], b_c[None, :, :])
    w[w < CONF_THRESHOLD] = 0.0
    sq = (diff ** 2).sum(axis=3)
    wsum = w.sum(axis=2)
    dist = np.sqrt((w * sq).sum(axis=2) / np.maximum(wsum, 1e-9))
    dist[wsum < 1e-9] = 1.0
    return dist


def build_cost_matrix(norm_a, conf_a, norm_b, conf_b, block=64):
    """Build n×m pose-distance matrix in blocks."""
    n, m = norm_a.shape[0], norm_b.shape[0]
    cost = np.zeros((n, m), dtype=np.float64)
    for i0 in range(0, n, block):
        cost[i0:i0 + block] = _pairwise_cost_block(norm_a[i0:i0 + block], conf_a[i0:i0 + block], norm_b, conf_b)
    return cost


def dtw_align(cost, window_ratio):
    """DTW with frozen scaled-relative-time Sakoe-Chiba window."""
    n, m = cost.shape
    win = max(1, int(round(window_ratio * max(m - 1, 1))))
    INF = float("inf")

    D = np.full((n + 1, m + 1), INF, dtype=np.float64)
    D[0, 0] = 0.0

    for i in range(1, n + 1):
        if n <= 1:
            j_center = 1
        else:
            j_center = 1 + int(round((i - 1) * (m - 1) / float(n - 1)))

        j_lo = max(1, j_center - win)
        j_hi = min(m, j_center + win)

        for j in range(j_lo, j_hi + 1):
            diag = D[i - 1, j - 1]
            down = D[i - 1, j]
            right = D[i, j - 1]
            D[i, j] = cost[i - 1, j - 1] + min(diag, down, right)

    if not np.isfinite(D[n, m]):
        raise RuntimeError("DTW failed to reach the end point. DTW_WINDOW_RATIO is frozen; check input sequences.")

    i, j, path = n, m, []
    while i > 0 and j > 0:
        path.append((i - 1, j - 1))
        diag, down, right = D[i - 1, j - 1], D[i - 1, j], D[i, j - 1]
        step = np.argmin([diag, down, right])
        if step == 0:
            i, j = i - 1, j - 1
        elif step == 1:
            i -= 1
        else:
            j -= 1

    path.reverse()
    norm_dist = D[n, m] / len(path)
    return path, norm_dist, win


# ======================================================================
# Step 7: Joint-angle sequence + 3-frame median smoothing
# ======================================================================
def joint_angle(xy, vertex, e1, e2):
    """Angle between two bones at vertex, degrees in [0,180]."""
    v1 = xy[e1] - xy[vertex]
    v2 = xy[e2] - xy[vertex]
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 < 1e-6 or n2 < 1e-6:
        return None
    cos = float(np.dot(v1, v2) / (n1 * n2))
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))


def trunk_angle(xy, conf):
    """Signed MidHip->Neck tilt relative to image vertical upward."""
    if conf[MID_HIP] <= CONF_THRESHOLD or conf[NECK] <= CONF_THRESHOLD:
        return None
    v = xy[NECK] - xy[MID_HIP]
    return float(np.degrees(np.arctan2(v[0], -v[1])))


def ang_diff(a, b, circular=False):
    """Absolute angle difference; circular=True gives shortest signed-angle difference."""
    d = abs(a - b)
    if circular:
        d = d % 360.0
        if d > 180.0:
            d = 360.0 - d
    return d


def frame_joint_angles(xy, conf):
    """Compute all scoring joint angles for one frame; invalid joints are omitted."""
    res = {}
    for k, (name, v, e1, e2, w) in enumerate(JOINT_DEFS):
        if e2 == -1:
            a = trunk_angle(xy, conf)
        else:
            ok = conf[v] > CONF_THRESHOLD and conf[e1] > CONF_THRESHOLD and conf[e2] > CONF_THRESHOLD
            a = joint_angle(xy, v, e1, e2) if ok else None
        if a is not None:
            res[k] = a
    return res


def build_joint_angle_sequence(norm_xy, conf):
    """Build raw angle matrix with shape (S, number_of_joints)."""
    S, J = len(norm_xy), len(JOINT_DEFS)
    angles = np.full((S, J), np.nan, dtype=np.float64)
    for i in range(S):
        for k, value in frame_joint_angles(norm_xy[i], conf[i]).items():
            angles[i, k] = value
    return angles


def median_smooth_angle_sequence(angles, radius=1):
    """Frozen temporal median filter. radius=1 means a 3-frame median window."""
    out = angles.copy()
    S, J = angles.shape

    for k in range(J):
        for i in range(S):
            lo, hi = max(0, i - radius), min(S, i + radius + 1)
            values = angles[lo:hi, k]
            values = values[np.isfinite(values)]
            if len(values) > 0:
                out[i, k] = float(np.median(values))
    return out


# ======================================================================
# Step 8: Frozen DTW-path joint error aggregation
# ======================================================================
def per_joint_path_mean_errors(angles_a, angles_b, path):
    """Frozen path-pair averaging: mean aligned error for every joint."""
    J = len(JOINT_DEFS)
    result = np.full(J, np.nan, dtype=np.float64)

    for k, (name, v, e1, e2, w) in enumerate(JOINT_DEFS):
        circular = (e2 == -1)
        values = []
        for ia, ib in path:
            a, b = angles_a[ia, k], angles_b[ib, k]
            if np.isfinite(a) and np.isfinite(b):
                values.append(ang_diff(a, b, circular))
        if values:
            result[k] = float(np.mean(values))
    return result


def joint_path_coverage(angles_a, angles_b, path):
    """Fraction of DTW pairs that contain valid smoothed angles for each joint."""
    coverage = np.zeros(len(JOINT_DEFS), dtype=np.float64)
    if not path:
        return coverage

    for k in range(len(JOINT_DEFS)):
        valid = sum(np.isfinite(angles_a[ia, k]) and np.isfinite(angles_b[ib, k]) for ia, ib in path)
        coverage[k] = valid / float(len(path))
    return coverage


def weighted_overall_from_joint_errors(joint_errors):
    """Frozen weighted overall angle error using existing JOINT_DEFS weights."""
    weights = np.asarray([j[4] for j in JOINT_DEFS], dtype=np.float64)
    valid = np.isfinite(joint_errors)
    if not valid.any():
        return np.nan
    return float(np.sum(weights[valid] * joint_errors[valid]) / np.sum(weights[valid]))


def aligned_pair_angle_errors(angles_a, angles_b, path):
    """Per-DTW-pair weighted smoothed angle error, kept for later CSV/visualization work."""
    detail = []
    for ia, ib in path:
        total, weight_sum = 0.0, 0.0
        for k, (name, v, e1, e2, w) in enumerate(JOINT_DEFS):
            a, b = angles_a[ia, k], angles_b[ib, k]
            if not (np.isfinite(a) and np.isfinite(b)):
                continue
            total += w * ang_diff(a, b, circular=(e2 == -1))
            weight_sum += w
        pair_error = total / weight_sum if weight_sum > 0 else np.nan
        detail.append((ia, ib, pair_error))
    return np.asarray(detail, dtype=np.float64)


# ======================================================================
# Step 9: Frozen 0-100 similarity mapping
# ======================================================================
def error_to_similarity_score(error_deg):
    """Frozen Phase 3.5A logistic mapping; no calibration occurs in V1.2."""
    if not np.isfinite(error_deg):
        return np.nan

    z = (float(error_deg) - SCORE_CENTER) / SCORE_SCALE
    if z > 60.0:
        return 0.0
    if z < -60.0:
        return 100.0
    return float(100.0 / (1.0 + math.exp(z)))


# ======================================================================
# Step 10: Body-part feedback (reporting only)
# ======================================================================
def aggregate_body_part_errors(joint_errors):
    """Aggregate existing joint errors by body part using the same existing joint weights."""
    name_to_index = {item[0]: i for i, item in enumerate(JOINT_DEFS)}
    result = {}

    for part, names in BODY_PART_GROUPS.items():
        values, weights = [], []
        for name in names:
            k = name_to_index[name]
            if np.isfinite(joint_errors[k]):
                values.append(float(joint_errors[k]))
                weights.append(float(JOINT_DEFS[k][4]))

        if values:
            result[part] = float(np.average(values, weights=weights))
        else:
            result[part] = np.nan
    return result


def rank_body_part_feedback(body_part_errors):
    """Rank valid body parts by measured error only; no subjective quality thresholds."""
    return sorted([(part, err) for part, err in body_part_errors.items() if np.isfinite(err)], key=lambda x: x[1], reverse=True)


# ======================================================================
# Step 11: Independent timing / rhythm diagnostics
# ======================================================================
def timing_diagnostics(path, n_ref, n_test, fps_ref=None, fps_test=None):
    """DTW path timing diagnostics. None of these values enter Overall Similarity Score."""
    if not path:
        return {
            "path_length": 0,
            "diagonal_steps": 0,
            "ref_only_steps": 0,
            "test_only_steps": 0,
            "non_diagonal_ratio": np.nan,
            "mean_relative_time_deviation": np.nan,
            "max_relative_time_deviation": np.nan,
            "duration_ratio_test_over_ref": np.nan,
        }

    diagonal = ref_only = test_only = 0
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

    relative_dev = []
    for ia, ib in path:
        ta = ia / float(max(n_ref - 1, 1))
        tb = ib / float(max(n_test - 1, 1))
        relative_dev.append(abs(ta - tb))

    steps = max(len(path) - 1, 1)

    if fps_ref and fps_test and fps_ref > 0 and fps_test > 0:
        ref_duration = max(n_ref - 1, 0) * FRAME_STEP / float(fps_ref)
        test_duration = max(n_test - 1, 0) * FRAME_STEP / float(fps_test)
        duration_ratio = test_duration / ref_duration if ref_duration > 1e-12 else np.nan
    else:
        duration_ratio = (n_test - 1) / float(max(n_ref - 1, 1))

    return {
        "path_length": len(path),
        "diagonal_steps": diagonal,
        "ref_only_steps": ref_only,
        "test_only_steps": test_only,
        "non_diagonal_ratio": (ref_only + test_only) / float(steps),
        "mean_relative_time_deviation": float(np.mean(relative_dev)),
        "max_relative_time_deviation": float(np.max(relative_dev)),
        "duration_ratio_test_over_ref": float(duration_ratio),
    }


# ======================================================================
# Step 12: Core pair evaluation
# ======================================================================
def evaluate_pair(norm_a, conf_a, norm_b, conf_b, fps_a=None, fps_b=None):
    """Run the frozen V1.2 core evaluation after skeleton preprocessing."""
    cost = build_cost_matrix(norm_a, conf_a, norm_b, conf_b)
    path, pose_dist, win = dtw_align(cost, DTW_WINDOW_RATIO)

    angles_a_raw = build_joint_angle_sequence(norm_a, conf_a)
    angles_b_raw = build_joint_angle_sequence(norm_b, conf_b)
    angles_a = median_smooth_angle_sequence(angles_a_raw, SMOOTH_RADIUS)
    angles_b = median_smooth_angle_sequence(angles_b_raw, SMOOTH_RADIUS)

    joint_errors = per_joint_path_mean_errors(angles_a, angles_b, path)
    coverage = joint_path_coverage(angles_a, angles_b, path)
    overall_error = weighted_overall_from_joint_errors(joint_errors)
    score = error_to_similarity_score(overall_error)

    body_parts = aggregate_body_part_errors(joint_errors)
    body_rank = rank_body_part_feedback(body_parts)
    timing = timing_diagnostics(path, len(norm_a), len(norm_b), fps_a, fps_b)
    detail = aligned_pair_angle_errors(angles_a, angles_b, path)

    return {
        "cost_matrix": cost,
        "path": path,
        "dtw_pose_distance": float(pose_dist),
        "dtw_window": int(win),
        "angles_a_raw": angles_a_raw,
        "angles_b_raw": angles_b_raw,
        "angles_a_smooth": angles_a,
        "angles_b_smooth": angles_b,
        "joint_errors": joint_errors,
        "coverage": coverage,
        "weighted_angle_error_deg": float(overall_error),
        "overall_similarity_score": float(score),
        "body_part_errors": body_parts,
        "body_part_ranking": body_rank,
        "timing": timing,
        "alignment_detail": detail,
    }


# ======================================================================
# Step 13: CSV output (presentation layer only; does not change scoring)
# ======================================================================
def joint_body_part_name(joint_name):
    """Return reporting body-part group for one frozen joint name."""
    for part, names in BODY_PART_GROUPS.items():
        if joint_name in names:
            return part
    return ""


def save_summary_csv(out_path, demo, imit, result, scale_a, scale_b, ids_a, ids_b, fps_a, fps_b, total_a, total_b, det_a, det_b):
    """Save one-row final summary. All values come from the already-computed frozen pipeline."""
    t = result["timing"]
    row = {
        "Reference Video": demo,
        "Learner Video": imit,
        "Overall Similarity Score": "%.6f" % result["overall_similarity_score"],
        "Weighted Angle Error (deg)": "%.6f" % result["weighted_angle_error_deg"],
        "DTW Pose Distance": "%.9f" % result["dtw_pose_distance"],
        "Mean Relative-Time Deviation": "%.9f" % t["mean_relative_time_deviation"],
        "Max Relative-Time Deviation": "%.9f" % t["max_relative_time_deviation"],
        "Non-Diagonal DTW Ratio": "%.9f" % t["non_diagonal_ratio"],
        "Duration Ratio Learner/Reference": "%.9f" % t["duration_ratio_test_over_ref"],
        "Reference Sampled Frames": len(ids_a),
        "Learner Sampled Frames": len(ids_b),
        "Reference Original Frames": total_a,
        "Learner Original Frames": total_b,
        "Reference FPS": "%.6f" % fps_a,
        "Learner FPS": "%.6f" % fps_b,
        "Reference Detection Rate": "%.6f" % (det_a / 100.0),
        "Learner Detection Rate": "%.6f" % (det_b / 100.0),
        "DTW Path Length": len(result["path"]),
        "DTW Window (learner sampled frames)": result["dtw_window"],
        "DTW Diagonal Steps": t["diagonal_steps"],
        "DTW Reference-Only Steps": t["ref_only_steps"],
        "DTW Learner-Only Steps": t["test_only_steps"],
        "Reference Sequence Scale (px)": "%.6f" % scale_a,
        "Learner Sequence Scale (px)": "%.6f" % scale_b,
        "NET_RESOLUTION": NET_RESOLUTION,
        "FRAME_STEP": FRAME_STEP,
        "CONF_THRESHOLD": CONF_THRESHOLD,
        "DTW_WINDOW_RATIO": DTW_WINDOW_RATIO,
        "SMOOTH_RADIUS": SMOOTH_RADIUS,
        "SCORE_CENTER (deg)": SCORE_CENTER,
        "SCORE_SCALE (deg)": SCORE_SCALE,
    }
    with open(out_path, "w", newline="", encoding="utf-8-sig") as f:
        wr = csv.DictWriter(f, fieldnames=list(row.keys()))
        wr.writeheader()
        wr.writerow(row)


def save_joint_report_csv(out_path, joint_errors, coverage):
    """Save the 11 frozen per-joint errors, coverage, weights and reporting body-part group."""
    with open(out_path, "w", newline="", encoding="utf-8-sig") as f:
        wr = csv.writer(f)
        wr.writerow(["Joint", "Mean Angle Error (deg)", "Coverage", "Weight", "Body Part"])
        for k, (name, v, e1, e2, weight) in enumerate(JOINT_DEFS):
            error = "" if not np.isfinite(joint_errors[k]) else "%.6f" % joint_errors[k]
            wr.writerow([name, error, "%.6f" % coverage[k], "%.6f" % weight, joint_body_part_name(name)])


def save_alignment_detail_csv(out_path, result, ids_a, ids_b):
    """Save every DTW path pair for later curves/video without recomputing the frozen pipeline."""
    path = result["path"]
    cost = result["cost_matrix"]
    detail = result["alignment_detail"]
    n_ref, n_test = len(ids_a), len(ids_b)

    with open(out_path, "w", newline="", encoding="utf-8-sig") as f:
        wr = csv.writer(f)
        wr.writerow([
            "Path Index",
            "Reference Sample Index", "Reference Original Frame",
            "Learner Sample Index", "Learner Original Frame",
            "Reference Relative Time", "Learner Relative Time",
            "Relative-Time Deviation", "Pose Cost",
            "Weighted Smoothed Angle Error (deg)",
        ])

        for p, (ia, ib) in enumerate(path):
            ta = ia / float(max(n_ref - 1, 1))
            tb = ib / float(max(n_test - 1, 1))
            pair_error = float(detail[p, 2]) if p < len(detail) and np.isfinite(detail[p, 2]) else np.nan
            wr.writerow([
                p, ia, int(ids_a[ia]), ib, int(ids_b[ib]),
                "%.9f" % ta, "%.9f" % tb, "%.9f" % abs(ta - tb),
                "%.9f" % float(cost[ia, ib]),
                "" if not np.isfinite(pair_error) else "%.6f" % pair_error,
            ])


def save_csv_outputs(out_dir, demo, imit, result, scale_a, scale_b, ids_a, ids_b, fps_a, fps_b, total_a, total_b, det_a, det_b):
    """Write V1.2 CSV reports. This layer consumes result only and never changes the frozen metrics."""
    os.makedirs(out_dir, exist_ok=True)
    summary_path = os.path.join(out_dir, "summary.csv")
    joint_path = os.path.join(out_dir, "joint_report.csv")
    detail_path = os.path.join(out_dir, "alignment_detail.csv")

    save_summary_csv(summary_path, demo, imit, result, scale_a, scale_b, ids_a, ids_b, fps_a, fps_b, total_a, total_b, det_a, det_b)
    save_joint_report_csv(joint_path, result["joint_errors"], result["coverage"])
    save_alignment_detail_csv(detail_path, result, ids_a, ids_b)
    return summary_path, joint_path, detail_path


# ======================================================================
# Step 14: Plot outputs (presentation only)
# ======================================================================
def save_joint_error_plot(result, out_path):
    """Plot the frozen 11-joint mean angle errors; visualization only."""
    names = [item[0] for item in JOINT_DEFS]
    values = np.asarray(result["joint_errors"], dtype=np.float64)
    x = np.arange(len(names))

    plt.figure(figsize=(11, 5.5))
    plt.bar(x, values)
    plt.xticks(x, names, rotation=35, ha="right")
    plt.ylabel("Mean Angle Error (deg)")
    plt.title("Per-Joint Motion Difference")
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_path, dpi=PLOT_DPI)
    plt.close()


def save_alignment_error_curve(result, n_ref, out_path):
    """Plot weighted smoothed angle error along the frozen DTW path."""
    path = result["path"]
    detail = result["alignment_detail"]
    progress = np.asarray([100.0 * ia / float(max(n_ref - 1, 1)) for ia, ib in path], dtype=np.float64)
    errors = np.asarray(detail[:, 2], dtype=np.float64)

    plt.figure(figsize=(10, 5))
    plt.plot(progress, errors, linewidth=1.2, label="Aligned-pair error")
    plt.axhline(result["weighted_angle_error_deg"], linestyle="--", linewidth=1.2, label="Overall weighted error = %.2f deg" % result["weighted_angle_error_deg"])
    plt.xlabel("Reference Motion Progress (%)")
    plt.ylabel("Weighted Smoothed Angle Error (deg)")
    plt.title("Motion Difference Along DTW Alignment")
    plt.xlim(0.0, 100.0)
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=PLOT_DPI)
    plt.close()


def save_timing_alignment_plot(result, n_ref, n_test, out_path):
    """Plot normalized Reference/Learner progress along the frozen DTW path."""
    path = result["path"]
    ref_t = np.asarray([ia / float(max(n_ref - 1, 1)) for ia, ib in path], dtype=np.float64)
    test_t = np.asarray([ib / float(max(n_test - 1, 1)) for ia, ib in path], dtype=np.float64)

    plt.figure(figsize=(6.5, 6.0))
    plt.plot(ref_t, test_t, linewidth=1.5, label="DTW alignment path")
    plt.plot([0.0, 1.0], [0.0, 1.0], linestyle="--", linewidth=1.0, label="Equal relative time")
    plt.xlabel("Reference Relative Time")
    plt.ylabel("Learner Relative Time")
    plt.title("DTW Relative-Time Alignment")
    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 1.0)
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=PLOT_DPI)
    plt.close()


def save_plot_outputs(out_dir, result, ids_a, ids_b):
    """Save V1.2 presentation plots without recomputing any frozen metric."""
    os.makedirs(out_dir, exist_ok=True)
    joint_path = os.path.join(out_dir, "joint_error_profile.png")
    error_path = os.path.join(out_dir, "aligned_angle_error_curve.png")
    timing_path = os.path.join(out_dir, "timing_alignment_curve.png")

    save_joint_error_plot(result, joint_path)
    save_alignment_error_curve(result, len(ids_a), error_path)
    save_timing_alignment_plot(result, len(ids_a), len(ids_b), timing_path)
    return joint_path, error_path, timing_path



# ======================================================================
# Step 15: Side-by-side comparison video (presentation only)
# ======================================================================
def draw_skeleton(frame, keypoints, color):
    """Draw BODY_25 skeleton using original OpenPose coordinates after display scaling."""
    for a, b in BODY_25_PAIRS:
        if keypoints[a, 2] > CONF_THRESHOLD and keypoints[b, 2] > CONF_THRESHOLD:
            pa = tuple(np.round(keypoints[a, :2]).astype(int))
            pb = tuple(np.round(keypoints[b, :2]).astype(int))
            cv2.line(frame, pa, pb, color, 3, cv2.LINE_AA)
    for j in range(N_PARTS):
        if keypoints[j, 2] > CONF_THRESHOLD:
            pt = tuple(np.round(keypoints[j, :2]).astype(int))
            cv2.circle(frame, pt, 4, (255, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(frame, pt, 3, color, -1, cv2.LINE_AA)
    return frame


class FrameIDReader(object):
    """Sequential reader driven by actual original frame IDs stored in the OpenPose JSON sequence."""
    def __init__(self, video_path):
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise IOError("Cannot open video for comparison output: %s" % video_path)
        self.raw_index = -1
        self.last_frame = None

    def get(self, original_frame_id):
        target = int(original_frame_id)
        if target < self.raw_index:
            raise ValueError("FrameIDReader requires non-decreasing frame IDs.")
        while self.raw_index < target:
            ok, frame = self.cap.read()
            self.raw_index += 1
            if not ok:
                return None
            self.last_frame = frame
        return None if self.last_frame is None else self.last_frame.copy()

    def release(self):
        self.cap.release()


def _resize_frame_and_keypoints(frame, keypoints, target_height):
    """Resize one video frame to target height and scale keypoint x/y coordinates identically."""
    h, w = frame.shape[:2]
    scale = target_height / float(max(h, 1))
    target_width = max(1, int(round(w * scale)))
    resized = cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR)
    kp = keypoints.copy()
    kp[:, :2] *= scale
    return resized, kp


def _put_text_shadow(frame, text_value, org, font_scale=0.65, color=(255, 255, 255), thickness=2):
    """Readable OpenCV text with a black shadow; ASCII-only labels for codec/font portability."""
    cv2.putText(frame, text_value, org, cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness + 3, cv2.LINE_AA)
    cv2.putText(frame, text_value, org, cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness, cv2.LINE_AA)


def save_compare_video(demo_path, imit_path, kps_a, kps_b, frame_ids_a, frame_ids_b, result, out_path, fps_ref):
    """Render the already-computed DTW path as a synchronized side-by-side comparison video."""
    path = result["path"]
    detail = result["alignment_detail"]
    if not path:
        raise RuntimeError("Cannot create comparison video: DTW path is empty.")

    reader_a = FrameIDReader(demo_path)
    reader_b = FrameIDReader(imit_path)
    writer = None
    try:
        ia0, ib0 = path[0]
        first_a = reader_a.get(frame_ids_a[ia0])
        first_b = reader_b.get(frame_ids_b[ib0])
        if first_a is None or first_b is None:
            raise RuntimeError("Cannot read first aligned source frames for comparison video.")

        panel_h = int(COMPARE_DISPLAY_HEIGHT)
        first_a_r, _ = _resize_frame_and_keypoints(first_a, kps_a[ia0], panel_h)
        first_b_r, _ = _resize_frame_and_keypoints(first_b, kps_b[ib0], panel_h)
        wa, wb = first_a_r.shape[1], first_b_r.shape[1]
        status_h = 92
        out_size = (wa + wb, panel_h + status_h)
        fps_out = float(COMPARE_FPS) if COMPARE_FPS is not None else float(fps_ref) / float(FRAME_STEP)
        fps_out = max(fps_out, 1.0)

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(out_path, fourcc, fps_out, out_size)
        if not writer.isOpened():
            raise RuntimeError("Cannot open VideoWriter for: %s" % out_path)

        n_ref, n_test = len(frame_ids_a), len(frame_ids_b)
        written = 0
        for p, (ia, ib) in enumerate(path):
            fa = first_a.copy() if p == 0 else reader_a.get(frame_ids_a[ia])
            fb = first_b.copy() if p == 0 else reader_b.get(frame_ids_b[ib])
            if fa is None or fb is None:
                print("  [Warning] Comparison video stopped early at DTW path index %d." % p)
                break

            fa, ka = _resize_frame_and_keypoints(fa, kps_a[ia], panel_h)
            fb, kb = _resize_frame_and_keypoints(fb, kps_b[ib], panel_h)
            if fa.shape[1] != wa: fa = cv2.resize(fa, (wa, panel_h))
            if fb.shape[1] != wb: fb = cv2.resize(fb, (wb, panel_h))

            draw_skeleton(fa, ka, (60, 220, 60))
            draw_skeleton(fb, kb, (0, 165, 255))

            ref_progress = 100.0 * ia / float(max(n_ref - 1, 1))
            learner_progress = 100.0 * ib / float(max(n_test - 1, 1))
            rel_dev = abs(ref_progress - learner_progress) / 100.0
            pair_error = float(detail[p, 2]) if p < len(detail) and np.isfinite(detail[p, 2]) else np.nan

            _put_text_shadow(fa, "REFERENCE  frame %d  progress %.1f%%" % (int(frame_ids_a[ia]), ref_progress), (14, 30), 0.62, (80, 255, 80), 2)
            _put_text_shadow(fb, "LEARNER  frame %d  progress %.1f%%" % (int(frame_ids_b[ib]), learner_progress), (14, 30), 0.62, (0, 190, 255), 2)

            canvas = np.full((panel_h + status_h, wa + wb, 3), 28, dtype=np.uint8)
            canvas[:panel_h, :wa] = fa
            canvas[:panel_h, wa:wa + wb] = fb
            cv2.line(canvas, (wa, 0), (wa, panel_h), (235, 235, 235), 2)

            pair_text = "--" if not np.isfinite(pair_error) else "%.2f deg" % pair_error
            line1 = "Aligned-pair error: %s   |   Overall weighted error: %.2f deg   |   Similarity: %.2f / 100" % (pair_text, result["weighted_angle_error_deg"], result["overall_similarity_score"])
            line2 = "DTW progress: Reference %.1f%% <-> Learner %.1f%%   |   Relative-time deviation: %.4f" % (ref_progress, learner_progress, rel_dev)
            _put_text_shadow(canvas, line1, (14, panel_h + 34), 0.60, (255, 255, 255), 1)
            _put_text_shadow(canvas, line2, (14, panel_h + 70), 0.56, (220, 220, 220), 1)

            writer.write(canvas)
            written += 1

        if written == 0:
            raise RuntimeError("Comparison video contained zero output frames.")
        return out_path, fps_out, written
    finally:
        if writer is not None:
            writer.release()
        reader_a.release()
        reader_b.release()


def save_comparison_video_output(out_dir, demo, imit, kps_a, kps_b, ids_a, ids_b, result, fps_a):
    """Presentation wrapper; consumes the frozen DTW result and raw OpenPose keypoints only."""
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "side_by_side_compare.mp4")
    return save_compare_video(demo, imit, kps_a, kps_b, ids_a, ids_b, result, out_path, fps_a)


# ======================================================================
# Step 16: Console reporting
# ======================================================================
def print_joint_results(joint_errors, coverage):
    print("\nPer-joint Errors")
    print("  %-16s %13s %13s %8s" % ("Joint", "Mean Error", "Coverage", "Weight"))
    print("  " + "-" * 56)

    for k, (name, v, e1, e2, w) in enumerate(JOINT_DEFS):
        err_text = "--" if not np.isfinite(joint_errors[k]) else "%.2f deg" % joint_errors[k]
        print("  %-16s %13s %12.1f%% %8.2f" % (name, err_text, coverage[k] * 100.0, w))


def print_body_part_feedback(body_part_errors, ranking):
    print("\nBody-part Feedback")
    for part in ("Elbow", "Shoulder", "Hip", "Knee", "Ankle", "Torso"):
        value = body_part_errors.get(part, np.nan)
        text = "--" if not np.isfinite(value) else "%.2f deg" % value
        print("  %-10s : %s" % (part, text))

    if ranking:
        print("\n  Largest measured body-part differences")
        for rank, (part, error) in enumerate(ranking, start=1):
            print("    %d. %-10s %.2f deg" % (rank, part, error))


def print_timing_results(timing):
    print("\nTiming / Rhythm Diagnostic  [separate from Overall Score]")
    print("  Mean relative-time deviation : %.6f" % timing["mean_relative_time_deviation"])
    print("  Max relative-time deviation  : %.6f" % timing["max_relative_time_deviation"])
    print("  Non-diagonal DTW ratio       : %.2f%%" % (100.0 * timing["non_diagonal_ratio"]))
    print("  Duration ratio Learner/Ref   : %.4f" % timing["duration_ratio_test_over_ref"])
    print("  DTW steps diag/ref-only/test-only: %d / %d / %d" % (
        timing["diagonal_steps"], timing["ref_only_steps"], timing["test_only_steps"]))


# ======================================================================
# Main pipeline
# ======================================================================
def main():
    demo = sys.argv[1] if len(sys.argv) > 1 else DEMO_VIDEO
    imit = sys.argv[2] if len(sys.argv) > 2 else IMIT_VIDEO

    for p in (demo, imit):
        if not os.path.isfile(p):
            print("[Error] Video not found: %s" % p)
            return

    t_all = time.time()

    print("=" * 84)
    print("Dance Motion Similarity Evaluation V1.2")
    print("Reference video: %s" % demo)
    print("Learner video  : %s" % imit)
    print("=" * 84)
    print("Frozen settings: %s | FRAME_STEP=%d | CONF=%.2f | DTW_WINDOW_RATIO=%.2f" % (
        NET_RESOLUTION, FRAME_STEP, CONF_THRESHOLD, DTW_WINDOW_RATIO))
    print("Scoring        : SEQUENCE_MEDIAN + 3-frame median angles + frozen logistic mapping")

    # 1) OpenPose / cache
    print("\n[1/8] Extract/load BODY_25 skeleton sequences (OpenPoseDemo.exe -> JSON cache)")
    kps_a, ids_a, fps_a, total_a = get_or_extract(demo, NET_RESOLUTION, FRAME_STEP)
    kps_b, ids_b, fps_b, total_b = get_or_extract(imit, NET_RESOLUTION, FRAME_STEP)

    det_a = 100.0 * (kps_a[:, :, 2] > CONF_THRESHOLD).any(axis=1).mean()
    det_b = 100.0 * (kps_b[:, :, 2] > CONF_THRESHOLD).any(axis=1).mean()
    print("  Reference: %d sampled frames (original %d, %.2f fps), person detected %.1f%%" % (len(kps_a), total_a, fps_a, det_a))
    print("  Learner  : %d sampled frames (original %d, %.2f fps), person detected %.1f%%" % (len(kps_b), total_b, fps_b, det_b))

    if det_a < 70.0 or det_b < 70.0:
        print("  [Warning] Low person-detection rate may make the comparison unstable.")

    # 2) Fill missing + frozen normalization
    print("\n[2/8] Missing-keypoint interpolation + SEQUENCE_MEDIAN normalization")
    kps_a_i = fill_missing(kps_a)
    kps_b_i = fill_missing(kps_b)
    norm_a, conf_a, scale_a = normalize_frames(kps_a_i)
    norm_b, conf_b, scale_b = normalize_frames(kps_b_i)
    print("  Reference sequence torso scale: %.3f px" % scale_a)
    print("  Learner sequence torso scale  : %.3f px" % scale_b)

    # 3) DTW alignment
    print("\n[3/8] Pose Cost Matrix + DTW temporal alignment")
    t_dtw = time.time()
    result = evaluate_pair(norm_a, conf_a, norm_b, conf_b, fps_a, fps_b)
    print("  DTW finished in %.2f s" % (time.time() - t_dtw))
    print("  Aligned pairs        : %d" % len(result["path"]))
    print("  DTW window           : ±%d learner sampled frames" % result["dtw_window"])
    print("  Avg Pose Distance    : %.6f torso lengths [alignment diagnostic only]" % result["dtw_pose_distance"])

    # 4) Final motion score and feedback
    print("\n[4/8] Smoothed joint-angle comparison + frozen Overall Similarity Score")
    print_joint_results(result["joint_errors"], result["coverage"])
    print("\n  Weighted Angle Error    : %.3f deg" % result["weighted_angle_error_deg"])
    print("  Overall Similarity Score: %.2f / 100" % result["overall_similarity_score"])
    print_body_part_feedback(result["body_part_errors"], result["body_part_ranking"])

    # 5) Timing diagnostic
    print("\n[5/8] Independent timing diagnostics")
    print_timing_results(result["timing"])

    # 6) CSV output; presentation only, frozen core result is not recomputed
    print("\n[6/8] Save CSV reports")
    if MAKE_CSV:
        summary_csv, joint_csv, detail_csv = save_csv_outputs(
            OUT_DIR, demo, imit, result, scale_a, scale_b, ids_a, ids_b, fps_a, fps_b, total_a, total_b, det_a, det_b)
        print("  Saved: %s" % summary_csv)
        print("  Saved: %s" % joint_csv)
        print("  Saved: %s" % detail_csv)
    else:
        print("  CSV output disabled (MAKE_CSV=False)")

    # 7) Plot output; consumes frozen result only
    print("\n[7/8] Save visualization plots")
    if MAKE_PLOTS:
        joint_png, error_png, timing_png = save_plot_outputs(OUT_DIR, result, ids_a, ids_b)
        print("  Saved: %s" % joint_png)
        print("  Saved: %s" % error_png)
        print("  Saved: %s" % timing_png)
    else:
        print("  Plot output disabled (MAKE_PLOTS=False)")

    # 8) Side-by-side comparison video; consumes existing path/result only
    print("\n[8/8] Save DTW-aligned side-by-side comparison video")
    if MAKE_COMPARE_VIDEO:
        compare_mp4, compare_fps, compare_frames = save_comparison_video_output(
            OUT_DIR, demo, imit, kps_a, kps_b, ids_a, ids_b, result, fps_a)
        print("  Saved: %s" % compare_mp4)
        print("  Video: %d aligned frames at %.2f fps" % (compare_frames, compare_fps))
    else:
        print("  Comparison video disabled (MAKE_COMPARE_VIDEO=False)")

    print("\n" + "#" * 84)
    print("# Overall Similarity Score : %.2f / 100" % result["overall_similarity_score"])
    print("# Weighted Angle Error     : %.3f deg" % result["weighted_angle_error_deg"])
    print("# Pose Distance            : %.6f  [DTW / diagnostic only]" % result["dtw_pose_distance"])
    print("# Timing Deviation         : %.6f  [separate diagnostic]" % result["timing"]["mean_relative_time_deviation"])
    print("# Total runtime            : %.2f s" % (time.time() - t_all))
    print("#" * 84)
    print("\nV1.2 core calculation + CSV + plots + DTW-aligned comparison video complete.")


if __name__ == "__main__":
    main()
