# -*- coding: utf-8 -*-
"""
================================================================================
Dance Motion Similarity Evaluation V1.1
(OpenPose BODY_25 + Normalization + DTW Alignment + Joint Angles)
--------------------------------------------------------------------------------
Based on V1.0. The overall pipeline and scoring/output modules are kept.

Main V1.1 changes:
  1. Replace the Python 3.7 pyopenpose backend with:
       Video -> OpenPoseDemo.exe -> BODY_25 JSON -> Python
     The external program interface is still video-in, result-out.
  2. Keep the downstream skeleton interface unchanged:
       kps.shape = (S, 25, 3), each keypoint = [x, y, confidence].
  3. Improve the DTW window for sequences with different lengths:
       j / (M - 1) ~= i / (N - 1)
     instead of centering the window directly on j ~= i.

DTW sanity-test result used to validate the change:
  Reference = 719 frames, Test = Reference[::2] = 360 frames
  V1.0 window = 359, V1.1 scaled-time window = 108
  Both found the same optimal path:
    path length = 719
    mean alignment index error = 0.4993 reference frames
    maximum alignment index error = 1 reference frame

Note:
  DTW_WINDOW_RATIO = 0.30 remains an experimental parameter.
  No pyopenpose installation is required by this V1.1 file.
================================================================================
"""

import os
import sys
import csv
import time
import json
import re
import shutil
import subprocess

import cv2
import numpy as np


# ======================================================================
# Step 1: Configuration
# ======================================================================
DEMO_VIDEO = r"data\aistpp\videos\gBR_sBM_c01_d04_mBR0_ch01.mp4"
IMIT_VIDEO = r"data\aistpp\videos\gBR_sBM_c01_d04_mBR1_ch01.mp4"

# ---- OpenPose executable backend ----
OPENPOSE_ROOT = r"D:\OpenPose\openpose"
OPENPOSE_DEMO = os.path.join(OPENPOSE_ROOT, "bin", "OpenPoseDemo.exe")
OPENPOSE_MODELS = os.path.join(OPENPOSE_ROOT, "models")

# OpenPoseDemo extracts JSON for all video frames; Python applies FRAME_STEP afterwards.
NET_RESOLUTION = "432x240"
FRAME_STEP = 2
CONF_THRESHOLD = 0.30

# ---- Skeleton JSON cache ----
USE_CACHE = True
FORCE_REEXTRACT = False
CACHE_DIR = "pose_cache_json"

# ---- DTW time alignment ----
DTW_WINDOW_RATIO = 0.30  # Experimental half-window ratio around scaled relative-time mapping

# ---- Scoring parameters inherited from V1.0; not recalibrated in the current DTW phase ----
SCORE_TOL = 8.0
SCORE_TAU = 30.0

# ---- Output controls ----
OUT_DIR = "similarity_result_432x240"
MAKE_DETAIL_CSV = True
MAKE_CURVE_IMG = True
MAKE_DTW_IMG = True
MAKE_COMPARE_VIDEO = True
COMPARE_FPS = 30
DISPLAY_HEIGHT = 540
# ---- Phase 3.1: joint-angle diagnostic outputs ----
MAKE_PHASE3_DIAGNOSTICS = True
PHASE3_TOP_N = 10
PHASE3_CURVE_JOINTS = ("Right Elbow", "Left Elbow", "Right Ankle", "Left Ankle")
# ---- Phase 3.1B: temporal stability and local alignment diagnostics ----
MAKE_PHASE31B_DIAGNOSTICS = True
PHASE31B_SMOOTH_RADIUS = 1     # radius=1 -> 3-frame median
PHASE31B_LOCAL_RADIUS = 2      # inspect imitation frame ib±2
# ---- Phase 3.1D: cross-resolution same-frame consistency ----
MAKE_PHASE31D_DIAGNOSTICS = True
PHASE31D_RES_A = "256x144"
PHASE31D_RES_B = "432x240"

# ======================================================================
# BODY_25 Model Constants
# ======================================================================
BODY_25_PARTS = [
    "Nose", "Neck", "RShoulder", "RElbow", "RWrist", "LShoulder", "LElbow", "LWrist",
    "MidHip", "RHip", "RKnee", "RAnkle", "LHip", "LKnee", "LAnkle",
    "REye", "LEye", "REar", "LEar",
    "LBigToe", "LSmallToe", "LHeel",
    "RBigToe", "RSmallToe", "RHeel",
]
N_PARTS = 25

MID_HIP, NECK = 8, 1
R_SHOULDER, R_ELBOW, R_WRIST = 2, 3, 4
L_SHOULDER, L_ELBOW, L_WRIST = 5, 6, 7
R_HIP, R_KNEE, R_ANKLE = 9, 10, 11
L_HIP, L_KNEE, L_ANKLE = 12, 13, 14
R_HEEL, L_HEEL = 24, 21

# Skeleton bone connections for visualization (main BODY_25 limbs)
BODY_25_PAIRS = [
    (1, 8), (1, 2), (1, 5), (2, 3), (3, 4), (5, 6), (6, 7),
    (8, 9), (9, 10), (10, 11), (8, 12), (12, 13), (13, 14),
    (1, 0), (0, 15), (15, 17), (0, 16), (16, 18),
    (14, 19), (19, 20), (14, 21), (11, 22), (22, 23), (11, 24),
]

# Scoring joint definitions: (name, vertex, endpoint1, endpoint2, weight); sum of weights = 1
# Angle = angle formed by two bones at the vertex (0~180°), naturally invariant to translation and scaling
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
    ("Torso Tilt",     NECK,       MID_HIP,    -1,       0.04),  # Special: torso angle relative to vertical
]

# ======================================================================

# ======================================================================
# Step 2: Extract/load BODY_25 skeleton sequences
# OpenPose backend: OpenPoseDemo.exe -> JSON -> (S,25,3)
# ======================================================================
def read_openpose_json(json_path):
    """Read all BODY_25 people from one OpenPose JSON file. Returns (P,25,3), or None if no valid person exists."""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    people = []
    for person in data.get("people", []):
        raw = np.asarray(person.get("pose_keypoints_2d", []), dtype=np.float32)
        if raw.size == N_PARTS * 3:
            people.append(raw.reshape(N_PARTS, 3))

    return np.stack(people) if people else None


def pick_primary(pose_keypoints):
    """Select the primary person using valid-keypoint count first, then mean confidence."""
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
    """Parse the original OpenPose frame number from ..._000000000123_keypoints.json."""
    match = re.search(r"_(\d+)_keypoints\.json$", os.path.basename(path))
    return int(match.group(1)) if match else int(fallback)


def load_json_sequence(json_dir, frame_step=1):
    """Load a cached OpenPose JSON folder and return sampled kps(S,25,3) and original frame indices."""
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
    """Create a stable cache-folder name from video basename and OpenPose resolution."""
    base = os.path.splitext(os.path.basename(video_path))[0]
    resolution_tag = net_resolution.replace("-", "auto").replace(":", "_")
    return os.path.join(CACHE_DIR, "%s_%s" % (base, resolution_tag))


def extract_sequence(video_path, net_resolution, frame_step, json_dir):
    """Run OpenPoseDemo.exe, write BODY_25 JSON, then load it as kps(S,25,3)."""
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
    """V1.0-compatible interface: video -> kps, frame_ids, fps, total. JSON is an internal cached backend."""
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
# Step 3: Linear interpolation for missing keypoints
# ======================================================================
def fill_missing(kps):
    """Linear interpolation of missing keypoint x/y coordinates along time axis.
    Sequence ends are filled with nearest valid value.
    Confidence column remains unchanged (interpolated points stay low-confidence)."""
    out = kps.copy()
    S = out.shape[0]

    for j in range(N_PARTS):
        valid = kps[:, j, 2] > CONF_THRESHOLD
        idxs = np.where(valid)[0]
        if len(idxs) == 0:
            continue  # Keypoint missing entire sequence, cannot interpolate
        for axis in (0, 1):
            out[:, j, axis] = np.interp(np.arange(S), idxs, kps[idxs, j, axis])
    return out

# ======================================================================
# Step 4: Skeleton normalization (hip center origin + torso length scaling)
# ======================================================================
def normalize_frames(kps):
    """Return norm_xy(S,25,2) (hip center = 0, neck-hip length = 1) and original confidence conf(S,25)."""
    S = kps.shape[0]
    norm_xy = np.zeros((S, N_PARTS, 2), dtype=np.float64)
    conf = kps[:, :, 2].copy()

    for i in range(S):
        xy = kps[i, :, :2]

        # --- Position reference: mid hip point 8; fallback to midpoint of left/right hips ---
        if conf[i, MID_HIP] > CONF_THRESHOLD:
            hip = xy[MID_HIP]
        elif conf[i, R_HIP] > CONF_THRESHOLD and conf[i, L_HIP] > CONF_THRESHOLD:
            hip = (xy[R_HIP] + xy[L_HIP]) / 2.0
        else:
            hip = np.zeros(2)

        # --- Scale reference: torso length from neck point 1 to hip center; fallback to midpoint of shoulders ---
        if conf[i, NECK] > CONF_THRESHOLD:
            neck = xy[NECK]
        elif conf[i, R_SHOULDER] > CONF_THRESHOLD and conf[i, L_SHOULDER] > CONF_THRESHOLD:
            neck = (xy[R_SHOULDER] + xy[L_SHOULDER]) / 2.0
        else:
            neck = hip

        torso = float(np.linalg.norm(neck - hip))
        if torso < 1e-6:
            torso = 1.0  # Fallback to avoid division by zero

        norm_xy[i] = (xy - hip) / torso

    return norm_xy, conf

# ======================================================================
# Step 5: DTW Dynamic Time Warping (flexible time alignment)
# ======================================================================
def _pairwise_cost_block(a_xy, a_c, b_xy, b_c):
    """Compute pose distance matrix for a block of frames × all b frames.
    Distance = confidence-weighted Euclidean distance of normalized coordinates (unit: torso length), occluded points excluded."""
    diff = a_xy[:, None, :, :] - b_xy[None, :, :, :]     # shape: (sa, sb, 25, 2)
    w = np.minimum(a_c[:, None, :], b_c[None, :, :])      # shape: (sa, sb, 25)
    w[w < CONF_THRESHOLD] = 0.0
    sq = (diff ** 2).sum(axis=3)                          # shape: (sa, sb, 25)
    wsum = w.sum(axis=2)
    dist = np.sqrt((w * sq).sum(axis=2) / np.maximum(wsum, 1e-9))
    dist[wsum < 1e-9] = 1.0  # No common visible points: assign neutral large distance
    return dist


def build_cost_matrix(norm_a, conf_a, norm_b, conf_b, block=64):
    """Build n×m cost matrix in blocks to avoid excessive memory usage."""
    n, m = norm_a.shape[0], norm_b.shape[0]
    cost = np.zeros((n, m), dtype=np.float64)
    for i0 in range(0, n, block):
        cost[i0:i0 + block] = _pairwise_cost_block(
            norm_a[i0:i0 + block], conf_a[i0:i0 + block], norm_b, conf_b)
    return cost




def dtw_align(cost, window_ratio):
    """DTW with scaled-time Sakoe-Chiba window.
    Returns (alignment_path, normalized_DTW_distance, window_size).
    The window is centered on equal relative progress: j/(m-1) ~= i/(n-1)."""
    n, m = cost.shape

    # Half-window width measured on the imitation/test time axis.
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
        raise RuntimeError("DTW failed to reach the end point. Increase DTW_WINDOW_RATIO.")

    # Backtrack to find optimal path.
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
# Step 6: Joint angle calculation and scoring
# ======================================================================
def joint_angle(xy, vertex, e1, e2):
    """Calculate angle between two bones at the vertex (degrees, 0~180).
    All three points must be valid."""
    v1 = xy[e1] - xy[vertex]
    v2 = xy[e2] - xy[vertex]
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 < 1e-6 or n2 < 1e-6:
        return None
    cos = float(np.dot(v1, v2) / (n1 * n2))
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))


def trunk_angle(xy, conf):
    """Calculate signed tilt angle of torso (MidHip -> Neck) relative to vertical upward (degrees).
    Special joint metric."""
    if conf[MID_HIP] <= CONF_THRESHOLD or conf[NECK] <= CONF_THRESHOLD:
        return None
    v = xy[NECK] - xy[MID_HIP]
    # Image y-axis points downward; vertical upward direction is (0, -1)
    return float(np.degrees(np.arctan2(v[0], -v[1])))


def ang_diff(a, b, circular=False):
    """Absolute angle difference; use circular=True for shortest difference on signed angles (-180~180)."""
    d = abs(a - b)
    if circular:
        d = d % 360.0
        if d > 180.0:
            d = 360.0 - d
    return d


def frame_joint_angles(xy, conf):
    """Compute all scoring joint angles for one frame.
    Returns {joint_index: angle_value}, invalid joints are omitted."""
    res = {}
    for k, (name, v, e1, e2, w) in enumerate(JOINT_DEFS):
        if e2 == -1:  # Torso tilt special case
            a = trunk_angle(xy, conf)
        else:
            ok = conf[v] > CONF_THRESHOLD and conf[e1] > CONF_THRESHOLD and conf[e2] > CONF_THRESHOLD
            a = joint_angle(xy, v, e1, e2) if ok else None
        if a is not None:
            res[k] = a
    return res


def evaluate_alignment(norm_a, conf_a, norm_b, conf_b, path):
    """Calculate joint angle errors frame by frame along DTW path.
    Returns overall error, per-joint error, coverage rate, and per-pair detail list."""
    n_joints = len(JOINT_DEFS)
    err_sum = np.zeros(n_joints)
    err_cnt = np.zeros(n_joints, dtype=np.int64)
    detail = []  # (demo_idx, imi_idx, weighted_error_for_this_pair)

    for (ia, ib) in path:
        ang_a = frame_joint_angles(norm_a[ia], conf_a[ia])
        ang_b = frame_joint_angles(norm_b[ib], conf_b[ib])
        pair_err, pair_w = 0.0, 0.0

        for k, (name, v, e1, e2, w) in enumerate(JOINT_DEFS):
            if k in ang_a and k in ang_b:
                circular = (e2 == -1)
                d = ang_diff(ang_a[k], ang_b[k], circular)
                err_sum[k] += d
                err_cnt[k] += 1
                pair_err += w * d
                pair_w += w

        pair_score_err = pair_err / pair_w if pair_w > 0 else 0.0
        detail.append((ia, ib, pair_score_err))

    joint_err = np.full(n_joints, np.nan)
    for k in range(n_joints):
        if err_cnt[k] > 0:
            joint_err[k] = err_sum[k] / err_cnt[k]

    # Overall error: weighted average only over joints with valid data
    weights = np.array([j[4] for j in JOINT_DEFS])
    valid = ~np.isnan(joint_err)
    overall = float(np.sum(weights[valid] * joint_err[valid]) / np.sum(weights[valid]))
    coverage = err_cnt / len(path)  # Valid sample ratio per joint

    return overall, joint_err, coverage, np.array(detail)

# ======================================================================
# Phase 3.1: Per-joint diagnostic analysis
# ======================================================================
def joint_min_confidence(conf, vertex, e1, e2):
    """Minimum confidence among keypoints used by one joint angle."""
    if e2 == -1:
        return float(min(conf[MID_HIP], conf[NECK]))
    return float(min(conf[vertex], conf[e1], conf[e2]))


def build_phase3_diagnostics(norm_a, conf_a, norm_b, conf_b, path):
    """Store per-aligned-pair demo angle, imitation angle, error and minimum confidence for every scoring joint."""
    n_pairs, n_joints = len(path), len(JOINT_DEFS)

    demo_angle = np.full((n_pairs, n_joints), np.nan, dtype=np.float64)
    imit_angle = np.full((n_pairs, n_joints), np.nan, dtype=np.float64)
    error = np.full((n_pairs, n_joints), np.nan, dtype=np.float64)
    demo_conf = np.zeros((n_pairs, n_joints), dtype=np.float64)
    imit_conf = np.zeros((n_pairs, n_joints), dtype=np.float64)

    for p, (ia, ib) in enumerate(path):
        ang_a = frame_joint_angles(norm_a[ia], conf_a[ia])
        ang_b = frame_joint_angles(norm_b[ib], conf_b[ib])

        for k, (name, v, e1, e2, w) in enumerate(JOINT_DEFS):
            demo_conf[p, k] = joint_min_confidence(conf_a[ia], v, e1, e2)
            imit_conf[p, k] = joint_min_confidence(conf_b[ib], v, e1, e2)

            if k in ang_a:
                demo_angle[p, k] = ang_a[k]
            if k in ang_b:
                imit_angle[p, k] = ang_b[k]

            if k in ang_a and k in ang_b:
                error[p, k] = ang_diff(ang_a[k], ang_b[k], circular=(e2 == -1))

    return {
        "demo_angle": demo_angle,
        "imit_angle": imit_angle,
        "error": error,
        "demo_conf": demo_conf,
        "imit_conf": imit_conf,
    }


def save_phase3_detail_csv(diag, detail, path, ids_a, ids_b, out_path):
    """Save complete per-frame/per-joint angle diagnostics."""
    header = [
        "Path Index", "Demo Sample Index", "Demo Original Frame",
        "Imitation Sample Index", "Imitation Original Frame",
        "Weighted Angle Error (deg)"
    ]

    for name, v, e1, e2, w in JOINT_DEFS:
        header.extend([
            "%s Demo Angle (deg)" % name,
            "%s Imitation Angle (deg)" % name,
            "%s Error (deg)" % name,
            "%s Demo Min Confidence" % name,
            "%s Imitation Min Confidence" % name,
        ])

    with open(out_path, "w", newline="", encoding="utf-8-sig") as f:
        wr = csv.writer(f)
        wr.writerow(header)

        for p, (ia, ib) in enumerate(path):
            row = [
                p, ia, int(ids_a[ia]), ib, int(ids_b[ib]),
                "%.3f" % float(detail[p, 2])
            ]

            for k in range(len(JOINT_DEFS)):
                values = [
                    diag["demo_angle"][p, k],
                    diag["imit_angle"][p, k],
                    diag["error"][p, k],
                    diag["demo_conf"][p, k],
                    diag["imit_conf"][p, k],
                ]

                for value in values:
                    row.append("" if np.isnan(value) else "%.3f" % value)

            wr.writerow(row)


def save_phase3_top_errors_csv(diag, path, ids_a, ids_b, out_path, top_n=10):
    """Save Top-N largest aligned-frame errors separately for every joint."""
    with open(out_path, "w", newline="", encoding="utf-8-sig") as f:
        wr = csv.writer(f)

        wr.writerow([
            "Joint", "Rank", "Path Index",
            "Demo Sample Index", "Demo Original Frame",
            "Imitation Sample Index", "Imitation Original Frame",
            "Demo Angle (deg)", "Imitation Angle (deg)", "Error (deg)",
            "Demo Min Confidence", "Imitation Min Confidence"
        ])

        for k, (name, v, e1, e2, w) in enumerate(JOINT_DEFS):
            err = diag["error"][:, k]
            valid = np.where(np.isfinite(err))[0]

            if len(valid) == 0:
                continue

            order = valid[np.argsort(err[valid])[::-1]]

            for rank, p in enumerate(order[:top_n], start=1):
                ia, ib = path[p]

                wr.writerow([
                    name, rank, p,
                    ia, int(ids_a[ia]),
                    ib, int(ids_b[ib]),
                    "%.3f" % diag["demo_angle"][p, k],
                    "%.3f" % diag["imit_angle"][p, k],
                    "%.3f" % diag["error"][p, k],
                    "%.3f" % diag["demo_conf"][p, k],
                    "%.3f" % diag["imit_conf"][p, k],
                ])


def save_phase3_joint_curve(errors, joint_name, out_path):
    """Save one joint's angle-error curve along the DTW path."""
    valid = np.isfinite(errors)

    if not valid.any():
        return

    W, H = 1200, 420
    canvas = np.full((H, W, 3), 255, np.uint8)
    margin_l, margin_b, margin_t = 60, 40, 40
    plot_w, plot_h = W - margin_l - 20, H - margin_b - margin_t

    emax = max(60.0, float(np.nanpercentile(errors, 98)))

    cv2.rectangle(canvas, (margin_l, margin_t), (margin_l + plot_w, margin_t + plot_h), (0, 0, 0), 1)

    for gy in range(0, int(emax) + 1, 10):
        y = margin_t + plot_h - int(gy / emax * plot_h)
        cv2.line(canvas, (margin_l, y), (margin_l + plot_w, y), (220, 220, 220), 1)
        cv2.putText(canvas, "%d" % gy, (15, y + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)

    cv2.putText(canvas, "%s error" % joint_name, (margin_l, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 2)

    prev = None

    for p in range(len(errors)):
        if not np.isfinite(errors[p]):
            prev = None
            continue

        x = margin_l + int(p / max(len(errors) - 1, 1) * plot_w)
        y = margin_t + plot_h - int(min(errors[p], emax) / emax * plot_h)

        if prev is not None:
            cv2.line(canvas, prev, (x, y), (0, 128, 255), 1, cv2.LINE_AA)

        prev = (x, y)

    mean = float(np.nanmean(errors))
    ym = margin_t + plot_h - int(min(mean, emax) / emax * plot_h)

    cv2.line(canvas, (margin_l, ym), (margin_l + plot_w, ym), (0, 0, 255), 2)
    cv2.putText(canvas, "mean=%.1f deg" % mean, (margin_l + 10, ym - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 200), 2)

    cv2.imwrite(out_path, canvas)


def print_phase3_summary(diag):
    """Print diagnostic statistics for selected high-error joints."""
    print("\n  Phase 3.1 diagnostic summary")

    name_to_idx = {joint[0]: k for k, joint in enumerate(JOINT_DEFS)}

    print("  %-16s %8s %8s %8s %8s" % ("Joint", "Mean", "Median", "P90", "Max"))

    for name in PHASE3_CURVE_JOINTS:
        k = name_to_idx[name]
        err = diag["error"][:, k]
        err = err[np.isfinite(err)]

        if len(err) == 0:
            continue

        print("  %-16s %7.2f° %7.2f° %7.2f° %7.2f°" % (
            name,
            np.mean(err),
            np.median(err),
            np.percentile(err, 90),
            np.max(err),
        ))

# ======================================================================
# Phase 3.1B: Temporal stability + local alignment diagnostics
# ======================================================================
def build_joint_angle_sequence(norm_xy, conf):
    """Build angle matrix (S, n_joints) for the complete sampled sequence."""
    S, J = len(norm_xy), len(JOINT_DEFS)
    angles = np.full((S, J), np.nan, dtype=np.float64)

    for i in range(S):
        frame_angles = frame_joint_angles(norm_xy[i], conf[i])
        for k, value in frame_angles.items():
            angles[i, k] = value

    return angles


def median_smooth_angle_sequence(angles, radius=1):
    """Temporal median filter used only for Phase 3.1B diagnostics."""
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

def temporal_jump_statistics(angles, frame_ids, sequence_name):
    """Statistics of absolute angle changes between adjacent sampled frames."""
    rows = []

    for k, (name, v, e1, e2, w) in enumerate(JOINT_DEFS):
        circular = (e2 == -1)
        jumps, jump_indices = [], []

        for i in range(1, len(angles)):
            a, b = angles[i - 1, k], angles[i, k]

            if np.isfinite(a) and np.isfinite(b):
                jumps.append(ang_diff(a, b, circular))
                jump_indices.append(i)

        if len(jumps) == 0:
            continue

        jumps = np.asarray(jumps, dtype=np.float64)
        max_pos = int(np.argmax(jumps))
        i = jump_indices[max_pos]

        rows.append({
            "sequence": sequence_name,
            "joint": name,
            "count": len(jumps),
            "mean": float(np.mean(jumps)),
            "median": float(np.median(jumps)),
            "p90": float(np.percentile(jumps, 90)),
            "p95": float(np.percentile(jumps, 95)),
            "max": float(np.max(jumps)),
            "over30": float(np.mean(jumps > 30.0)),
            "over60": float(np.mean(jumps > 60.0)),
            "max_from_sample": i - 1,
            "max_to_sample": i,
            "max_from_frame": int(frame_ids[i - 1]),
            "max_to_frame": int(frame_ids[i]),
        })

    return rows


def save_temporal_jump_report(rows_a, rows_b, out_path):
    """Save adjacent-frame angle jump statistics."""
    rows = rows_a + rows_b

    with open(out_path, "w", newline="", encoding="utf-8-sig") as f:
        wr = csv.writer(f)
        wr.writerow([
            "Sequence", "Joint", "Valid Transitions",
            "Mean Jump (deg)", "Median Jump (deg)",
            "P90 Jump (deg)", "P95 Jump (deg)", "Max Jump (deg)",
            "Jump >30 deg Ratio", "Jump >60 deg Ratio",
            "Max From Sample", "Max To Sample",
            "Max From Original Frame", "Max To Original Frame"
        ])

        for r in rows:
            wr.writerow([
                r["sequence"], r["joint"], r["count"],
                "%.3f" % r["mean"], "%.3f" % r["median"],
                "%.3f" % r["p90"], "%.3f" % r["p95"], "%.3f" % r["max"],
                "%.4f" % r["over30"], "%.4f" % r["over60"],
                r["max_from_sample"], r["max_to_sample"],
                r["max_from_frame"], r["max_to_frame"],
            ])

def per_joint_path_mean_errors(angles_a, angles_b, path):
    """Mean error of every joint along a fixed DTW path."""
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


def weighted_overall_from_joint_errors(joint_errors):
    """Same weighted aggregation idea as the official overall angle error."""
    weights = np.asarray([j[4] for j in JOINT_DEFS], dtype=np.float64)
    valid = np.isfinite(joint_errors)

    if not valid.any():
        return np.nan

    return float(np.sum(weights[valid] * joint_errors[valid]) / np.sum(weights[valid]))


def save_smoothing_report(raw_joint_err, smooth_joint_err, out_path):
    """Compare raw and diagnostic 3-frame-median joint errors."""
    with open(out_path, "w", newline="", encoding="utf-8-sig") as f:
        wr = csv.writer(f)
        wr.writerow(["Joint", "Raw Mean Error (deg)", "Smoothed Mean Error (deg)", "Change (deg)"])

        for k, (name, v, e1, e2, w) in enumerate(JOINT_DEFS):
            raw, smooth = raw_joint_err[k], smooth_joint_err[k]
            change = smooth - raw if np.isfinite(raw) and np.isfinite(smooth) else np.nan
            wr.writerow([
                name,
                "" if not np.isfinite(raw) else "%.3f" % raw,
                "" if not np.isfinite(smooth) else "%.3f" % smooth,
                "" if not np.isfinite(change) else "%.3f" % change,
            ])

def weighted_pair_angle_error(angles_a, angles_b, ia, ib):
    """Weighted joint-angle error for one frame pair."""
    total, weight_sum = 0.0, 0.0

    for k, (name, v, e1, e2, w) in enumerate(JOINT_DEFS):
        a, b = angles_a[ia, k], angles_b[ib, k]

        if not np.isfinite(a) or not np.isfinite(b):
            continue

        total += w * ang_diff(a, b, circular=(e2 == -1))
        weight_sum += w

    return total / weight_sum if weight_sum > 0 else np.nan


def local_alignment_diagnostic(angles_a, angles_b, path, radius=2):
    """For each existing DTW pair, inspect imitation frames ib±radius without changing the DTW path."""
    n = len(path)
    current = np.full(n, np.nan)
    best = np.full(n, np.nan)
    best_ib = np.full(n, -1, dtype=np.int64)

    for p, (ia, ib) in enumerate(path):
        current[p] = weighted_pair_angle_error(angles_a, angles_b, ia, ib)

        lo, hi = max(0, ib - radius), min(len(angles_b) - 1, ib + radius)
        best_error, best_index = np.inf, ib

        for jb in range(lo, hi + 1):
            error = weighted_pair_angle_error(angles_a, angles_b, ia, jb)

            if np.isfinite(error) and error < best_error:
                best_error, best_index = error, jb

        if np.isfinite(best_error):
            best[p] = best_error
            best_ib[p] = best_index

    return current, best, best_ib

def save_local_alignment_report(path, ids_a, ids_b, raw_result, smooth_result, out_path):
    """Save raw and smoothed local ±2-frame alignment diagnostics."""
    raw_current, raw_best, raw_best_ib = raw_result
    smooth_current, smooth_best, smooth_best_ib = smooth_result

    with open(out_path, "w", newline="", encoding="utf-8-sig") as f:
        wr = csv.writer(f)
        wr.writerow([
            "Path Index",
            "Demo Sample Index", "Demo Original Frame",
            "DTW Imitation Sample Index", "DTW Imitation Original Frame",
            "Raw Current Error", "Raw Best Error", "Raw Improvement",
            "Raw Best Sample Index", "Raw Best Original Frame", "Raw Best Offset",
            "Smoothed Current Error", "Smoothed Best Error", "Smoothed Improvement",
            "Smoothed Best Sample Index", "Smoothed Best Original Frame", "Smoothed Best Offset"
        ])

        for p, (ia, ib) in enumerate(path):
            rb = int(raw_best_ib[p])
            sb = int(smooth_best_ib[p])

            wr.writerow([
                p,
                ia, int(ids_a[ia]),
                ib, int(ids_b[ib]),
                "%.3f" % raw_current[p], "%.3f" % raw_best[p], "%.3f" % (raw_current[p] - raw_best[p]),
                rb, int(ids_b[rb]), rb - ib,
                "%.3f" % smooth_current[p], "%.3f" % smooth_best[p], "%.3f" % (smooth_current[p] - smooth_best[p]),
                sb, int(ids_b[sb]), sb - ib,
            ])

def print_phase31b_summary(path, jump_a, jump_b, raw_joint_err, smooth_joint_err, raw_local, smooth_local):
    """Print the main Phase 3.1B diagnostic results."""
    print("\n  Phase 3.1B temporal stability / local alignment summary")

    focus = PHASE3_CURVE_JOINTS

    for sequence_name, rows in (("Demo", jump_a), ("Imitation", jump_b)):
        lookup = {r["joint"]: r for r in rows}
        print("\n  %s adjacent-frame angle jumps" % sequence_name)
        print("  %-16s %8s %8s %8s %8s" % ("Joint", "Median", "P90", "Max", ">30deg"))

        for name in focus:
            r = lookup[name]
            print("  %-16s %7.2f° %7.2f° %7.2f° %7.1f%%" % (
                name, r["median"], r["p90"], r["max"], 100.0 * r["over30"]
            ))

    raw_overall = weighted_overall_from_joint_errors(raw_joint_err)
    smooth_overall = weighted_overall_from_joint_errors(smooth_joint_err)

    print("\n  Fixed DTW path")
    print("  Raw weighted joint error      : %.3f deg" % raw_overall)
    print("  3-frame median joint error   : %.3f deg" % smooth_overall)
    print("  Difference                   : %.3f deg" % (smooth_overall - raw_overall))

    for title, result in (("Raw", raw_local), ("Smoothed", smooth_local)):
        current, best, best_ib = result
        valid = np.isfinite(current) & np.isfinite(best)
        improvement = current[valid] - best[valid]

        print("\n  %s local ±%d-frame diagnostic" % (title, PHASE31B_LOCAL_RADIUS))
        print("  Current mean pair error      : %.3f deg" % np.mean(current[valid]))
        print("  Local-best mean pair error   : %.3f deg" % np.mean(best[valid]))
        print("  Improved > 2 deg             : %.1f%%" % (100.0 * np.mean(improvement > 2.0)))
        print("  Improved > 5 deg             : %.1f%%" % (100.0 * np.mean(improvement > 5.0)))

        original_ib = np.asarray([ib for ia, ib in path], dtype=np.int64)
        same = best_ib[valid] == original_ib[valid]
        print("  DTW frame already local best : %.1f%%" % (100.0 * np.mean(same)))

# ======================================================================
# Phase 3.1D: Cross-resolution same-frame consistency
# ======================================================================
def load_cached_resolution_sequence(video_path, net_resolution, frame_step):
    """Load an existing OpenPose JSON cache for one resolution without running OpenPose again."""
    json_dir = _cache_json_dir(video_path, net_resolution)
    complete_flag = os.path.join(json_dir, "_complete.txt")

    if not os.path.isfile(complete_flag):
        raise IOError("Completed JSON cache not found for %s: %s" % (net_resolution, json_dir))

    kps, frame_ids = load_json_sequence(json_dir, frame_step)
    return kps, frame_ids

def match_frame_ids(ids_a, ids_b):
    """Return matched sequence indices for exactly identical original frame IDs."""
    map_b = {int(frame_id): i for i, frame_id in enumerate(ids_b)}
    matches = []

    for ia, frame_id in enumerate(ids_a):
        frame_id = int(frame_id)
        if frame_id in map_b:
            matches.append((ia, map_b[frame_id], frame_id))

    return matches

def cross_resolution_joint_diagnostics(video_path, res_a, res_b, frame_step):
    """Compare joint angles from two OpenPose resolutions on exactly the same original video frames."""
    kps_a, ids_a = load_cached_resolution_sequence(video_path, res_a, frame_step)
    kps_b, ids_b = load_cached_resolution_sequence(video_path, res_b, frame_step)

    kps_a_i = fill_missing(kps_a)
    kps_b_i = fill_missing(kps_b)

    norm_a, conf_a = normalize_frames(kps_a_i)
    norm_b, conf_b = normalize_frames(kps_b_i)

    angles_a = build_joint_angle_sequence(norm_a, conf_a)
    angles_b = build_joint_angle_sequence(norm_b, conf_b)

    matches = match_frame_ids(ids_a, ids_b)

    rows = []

    for ia, ib, frame_id in matches:
        for k, (name, v, e1, e2, w) in enumerate(JOINT_DEFS):
            angle_a = angles_a[ia, k]
            angle_b = angles_b[ib, k]

            if not np.isfinite(angle_a) or not np.isfinite(angle_b):
                continue

            circular = (e2 == -1)
            difference = ang_diff(angle_a, angle_b, circular)

            conf_min_a = joint_min_confidence(conf_a[ia], v, e1, e2)
            conf_min_b = joint_min_confidence(conf_b[ib], v, e1, e2)

            rows.append({
                "frame_id": frame_id,
                "joint": name,
                "angle_a": float(angle_a),
                "angle_b": float(angle_b),
                "difference": float(difference),
                "conf_a": float(conf_min_a),
                "conf_b": float(conf_min_b),
            })

    return rows, len(matches)

def save_cross_resolution_detail(rows_demo, rows_imit, res_a, res_b, out_path):
    """Save every valid same-frame cross-resolution joint comparison."""
    with open(out_path, "w", newline="", encoding="utf-8-sig") as f:
        wr = csv.writer(f)

        wr.writerow([
            "Sequence", "Original Frame", "Joint",
            "%s Angle (deg)" % res_a,
            "%s Angle (deg)" % res_b,
            "Absolute Difference (deg)",
            "%s Min Confidence" % res_a,
            "%s Min Confidence" % res_b
        ])

        for sequence_name, rows in (("Demo", rows_demo), ("Imitation", rows_imit)):
            for r in rows:
                wr.writerow([
                    sequence_name,
                    r["frame_id"],
                    r["joint"],
                    "%.3f" % r["angle_a"],
                    "%.3f" % r["angle_b"],
                    "%.3f" % r["difference"],
                    "%.3f" % r["conf_a"],
                    "%.3f" % r["conf_b"],
                ])

def summarize_cross_resolution(rows, sequence_name):
    """Summarize cross-resolution angle differences for every joint."""
    summary = []

    for name, v, e1, e2, w in JOINT_DEFS:
        values = np.asarray([r["difference"] for r in rows if r["joint"] == name], dtype=np.float64)

        if len(values) == 0:
            continue

        summary.append({
            "sequence": sequence_name,
            "joint": name,
            "count": len(values),
            "mean": float(np.mean(values)),
            "median": float(np.median(values)),
            "p90": float(np.percentile(values, 90)),
            "p95": float(np.percentile(values, 95)),
            "max": float(np.max(values)),
            "over10": float(np.mean(values > 10.0)),
            "over20": float(np.mean(values > 20.0)),
        })

    return summary

def summarize_cross_resolution_combined(rows_demo, rows_imit):
    """Combined cross-resolution statistics over both videos."""
    return summarize_cross_resolution(rows_demo + rows_imit, "Combined")

def save_cross_resolution_summary(summary_demo, summary_imit, summary_combined, out_path):
    """Save cross-resolution statistics."""
    with open(out_path, "w", newline="", encoding="utf-8-sig") as f:
        wr = csv.writer(f)

        wr.writerow([
            "Sequence", "Joint", "Valid Frames",
            "Mean Difference (deg)", "Median Difference (deg)",
            "P90 Difference (deg)", "P95 Difference (deg)",
            "Max Difference (deg)",
            "Difference >10 deg Ratio",
            "Difference >20 deg Ratio"
        ])

        for r in summary_demo + summary_imit + summary_combined:
            wr.writerow([
                r["sequence"],
                r["joint"],
                r["count"],
                "%.3f" % r["mean"],
                "%.3f" % r["median"],
                "%.3f" % r["p90"],
                "%.3f" % r["p95"],
                "%.3f" % r["max"],
                "%.4f" % r["over10"],
                "%.4f" % r["over20"],
            ])

def print_phase31d_summary(summary_combined, res_a, res_b, demo_matches, imit_matches):
    """Print combined cross-resolution consistency summary."""
    print("\n  Phase 3.1D cross-resolution same-frame consistency")
    print("  Resolution comparison: %s vs %s" % (res_a, res_b))
    print("  Matched frames: Demo=%d, Imitation=%d" % (demo_matches, imit_matches))

    print("\n  %-16s %8s %8s %8s %8s %9s %9s" % (
        "Joint", "Mean", "Median", "P90", "Max", ">10deg", ">20deg"
    ))

    lookup = {r["joint"]: r for r in summary_combined}

    for name, v, e1, e2, w in JOINT_DEFS:
        if name not in lookup:
            continue

        r = lookup[name]

        print("  %-16s %7.2f° %7.2f° %7.2f° %7.2f° %8.1f%% %8.1f%%" % (
            name,
            r["mean"],
            r["median"],
            r["p90"],
            r["max"],
            100.0 * r["over10"],
            100.0 * r["over20"],
        ))

def error_to_score(mean_err_deg):
    """Convert average angle error to 0-100 score: full score within tolerance, exponential smooth decay beyond."""
    return 100.0 * np.exp(-max(0.0, mean_err_deg - SCORE_TOL) / SCORE_TAU)


def grade(score):
    if score >= 90: return "Excellent (highly consistent motion)"
    if score >= 80: return "Good (mostly correct, minor deviations)"
    if score >= 70: return "Fair (main motions match, details need improvement)"
    if score >= 60: return "Pass (motion outline recognizable, noticeable deviations)"
    return "Needs improvement (large rhythm/posture deviations, practice with reference)"

# ======================================================================
# Step 7: Visualization output
# ======================================================================
def draw_skeleton(frame, kp, color):
    """Draw BODY_25 skeleton on frame using original keypoint coordinates."""
    for (a, b) in BODY_25_PAIRS:
        if kp[a, 2] > CONF_THRESHOLD and kp[b, 2] > CONF_THRESHOLD:
            pa = tuple(np.round(kp[a, :2]).astype(int))
            pb = tuple(np.round(kp[b, :2]).astype(int))
            cv2.line(frame, pa, pb, color, 3, cv2.LINE_AA)
    for j in range(N_PARTS):
        if kp[j, 2] > CONF_THRESHOLD:
            p = tuple(np.round(kp[j, :2]).astype(int))
            cv2.circle(frame, p, 4, (255, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(frame, p, 3, color, -1, cv2.LINE_AA)
    return frame


def save_error_curve(detail, out_path):
    """Generate error-over-time curve plot (x = alignment path index, y = joint angle error in degrees)."""
    W, H = 1200, 420
    canvas = np.full((H, W, 3), 255, np.uint8)
    margin_l, margin_b, margin_t = 60, 40, 30
    plot_w, plot_h = W - margin_l - 20, H - margin_b - margin_t

    err = detail[:, 2]
    n = len(err)
    emax = max(60.0, float(np.percentile(err, 98)))

    # Axes
    cv2.rectangle(canvas, (margin_l, margin_t), (margin_l + plot_w, margin_t + plot_h), (0, 0, 0), 1)
    for gy in range(0, int(emax) + 1, 10):
        y = margin_t + plot_h - int(gy / emax * plot_h)
        cv2.line(canvas, (margin_l, y), (margin_l + plot_w, y), (220, 220, 220), 1)
        cv2.putText(canvas, "%d" % gy, (15, y + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)
    cv2.putText(canvas, "Joint angle error (deg) ->", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    # Error line
    pts = []
    for k in range(n):
        x = margin_l + int(k / max(n - 1, 1) * plot_w)
        y = margin_t + plot_h - int(min(err[k], emax) / emax * plot_h)
        pts.append((x, y))
    for k in range(1, n):
        cv2.line(canvas, pts[k - 1], pts[k], (0, 128, 255), 1, cv2.LINE_AA)

    # Mean reference line
    mean = float(err.mean())
    ym = margin_t + plot_h - int(min(mean, emax) / emax * plot_h)
    cv2.line(canvas, (margin_l, ym), (margin_l + plot_w, ym), (0, 0, 255), 2)
    cv2.putText(canvas, "mean=%.1f deg" % mean, (margin_l + 10, ym - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 200), 2)

    cv2.imwrite(out_path, canvas)


def save_dtw_map(cost, path, out_path):
    """Generate DTW cost matrix heatmap with optimal alignment path overlay."""
    img = cost.copy()
    lo, hi = np.percentile(img, 5), np.percentile(img, 95)
    img = np.clip((img - lo) / max(hi - lo, 1e-9), 0, 1)
    img = (img * 255).astype(np.uint8)
    img = cv2.resize(img, (720, 720), interpolation=cv2.INTER_NEAREST)
    img = cv2.applyColorMap(255 - img, cv2.COLORMAP_VIRIDIS)

    n, m = cost.shape
    for k, (ia, ib) in enumerate(path):
        x = int(ib / max(m - 1, 1) * 719)
        y = int(ia / max(n - 1, 1) * 719)
        cv2.circle(img, (x, y), 1, (255, 255, 255), -1)

    cv2.putText(img, "DTW path (x=imitation, y=demonstration)", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.imwrite(out_path, img)


class SeqReader(object):
    """Sequential video reader: DTW path is monotonic, so we just advance frames sequentially to target (faster than random seek).
    Note: input is sampled frame index; convert to original frame index: seq_idx * step."""
    def __init__(self, path, step):
        self.cap = cv2.VideoCapture(path)
        self.step = step
        self.raw_cur = -1   # Current original frame index read
        self.last = None

    def get(self, seq_idx):
        target = seq_idx * self.step   # Sampled index -> original frame index
        while self.raw_cur < target:
            ret, frame = self.cap.read()
            self.raw_cur += 1
            if not ret:
                return None
            self.last = frame
        return self.last

    def release(self):
        self.cap.release()


def save_compare_video(demo_path, imit_path, kps_a, kps_b, frame_ids_a, frame_ids_b,
                       detail, path, score, out_path):
    """Generate side-by-side skeleton comparison video along the alignment path."""
    ra = SeqReader(demo_path, FRAME_STEP)
    rb = SeqReader(imit_path, FRAME_STEP)

    fa = ra.get(0)
    if fa is None:
        ra.release(); rb.release(); return

    h0, w0 = fa.shape[:2]
    scale = DISPLAY_HEIGHT / float(h0)
    dw, dh = int(w0 * scale), DISPLAY_HEIGHT

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(out_path, fourcc, COMPARE_FPS, (dw * 2, dh + 50))

    err_lookup = {(int(a), int(b)): e for a, b, e in detail}

    for (ia, ib) in path:
        fa = ra.get(ia)
        fb = rb.get(ib)
        if fa is None or fb is None:
            break

        fa = cv2.resize(fa, (dw, dh)); fb = cv2.resize(fb, (dw, dh))
        ka = kps_a[ia].copy(); ka[:, :2] *= scale
        kb = kps_b[ib].copy(); kb[:, :2] *= scale

        draw_skeleton(fa, ka, (0, 200, 0))     # Demo = green
        draw_skeleton(fb, kb, (0, 160, 255))   # Imitation = orange

        bar = np.full((50, dw * 2, 3), 30, np.uint8)
        e = err_lookup.get((ia, ib), 0.0)

        cv2.putText(fa, "DEMO frame %d" % frame_ids_a[ia], (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(fb, "IMITATION frame %d" % frame_ids_b[ib], (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 160, 255), 2)
        cv2.putText(bar, "Aligned pair  err=%.1f deg   Overall Similarity=%.1f / 100" % (e, score),
                    (15, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        vw.write(np.vstack([np.hstack([fa, fb]), bar]))

    vw.release(); ra.release(); rb.release()

# ======================================================================
# Main pipeline
# ======================================================================
def main():
    demo = sys.argv[1] if len(sys.argv) > 1 else DEMO_VIDEO
    imit = sys.argv[2] if len(sys.argv) > 2 else IMIT_VIDEO

    for p in (demo, imit):
        if not os.path.isfile(p):
            print("[Error] Video not found: %s" % p); return

    os.makedirs(OUT_DIR, exist_ok=True)
    t_all = time.time()

    print("=" * 76)
    print("Dance Motion Similarity Evaluation V1.1")
    print("Demo video: %s\nImitation video: %s" % (demo, imit))
    print("=" * 76)

    # ① Extract skeletons (with caching)
    print("\n[1/5] Extract/load BODY_25 skeleton sequences (OpenPoseDemo.exe -> JSON)")
    kps_a, ids_a, fps_a, total_a = get_or_extract(demo, NET_RESOLUTION, FRAME_STEP)
    kps_b, ids_b, fps_b, total_b = get_or_extract(imit, NET_RESOLUTION, FRAME_STEP)

    det_a = 100.0 * (kps_a[:, :, 2] > CONF_THRESHOLD).any(axis=1).mean()
    det_b = 100.0 * (kps_b[:, :, 2] > CONF_THRESHOLD).any(axis=1).mean()

    print("  Demo: %d sampled frames (original %d frames, %.1f fps), person detected in %.1f%% frames" % (len(kps_a), total_a, fps_a, det_a))
    print("  Imitation: %d sampled frames (original %d frames, %.1f fps), person detected in %.1f%% frames" % (len(kps_b), total_b, fps_b, det_b))

    if det_a < 70 or det_b < 70:
        print("  [Warning] Low person detection rate in some video(s), results may be unstable. Recommend clearer video with full body visible.")

    # ② Interpolate missing points + normalize
    print("\n[2/5] Interpolate missing keypoints + normalize skeletons (hip center origin, torso length = 1)")
    kps_a_i = fill_missing(kps_a)
    kps_b_i = fill_missing(kps_b)
    norm_a, conf_a = normalize_frames(kps_a_i)
    norm_b, conf_b = normalize_frames(kps_b_i)

    # ③ DTW time alignment
    print("\n[3/5] Building pose cost matrix and running DTW flexible time alignment ...")
    t = time.time()
    cost = build_cost_matrix(norm_a, conf_a, norm_b, conf_b)
    path, dtw_dist, win = dtw_align(cost, DTW_WINDOW_RATIO)

    print("  Alignment finished in %.1f s; flexible window ±%d frames; %d aligned pairs; average pose distance %.4f torso lengths"
          % (time.time() - t, win, len(path), dtw_dist))

    # Rhythm diagnosis from path slope
    slope = (ids_b[-1] - ids_b[path[0][1]]) / max(ids_a[-1] - ids_a[path[0][0]], 1)
    if slope > 1.08:
        print("  Rhythm diagnosis: imitation is overall slower (duration ~%.2fx of demo, DTW has auto-stretched to align)" % slope)
    elif slope < 0.92:
        print("  Rhythm diagnosis: imitation is overall faster (duration ~%.2fx of demo, DTW has auto-compressed to align)" % slope)
    else:
        print("  Rhythm diagnosis: overall rhythm similar (duration ratio %.2f)" % slope)

    # ④ Joint angle error + scoring
    print("\n[4/5] Calculate joint angle errors along alignment path and weighted scoring")
    overall_err, joint_err, coverage, detail = evaluate_alignment(
        norm_a, conf_a, norm_b, conf_b, path)
    score = float(error_to_score(overall_err))
    phase3_diag = build_phase3_diagnostics(norm_a, conf_a, norm_b, conf_b, path) if MAKE_PHASE3_DIAGNOSTICS else None
    if MAKE_PHASE31B_DIAGNOSTICS:
        angles_a_raw = build_joint_angle_sequence(norm_a, conf_a)
        angles_b_raw = build_joint_angle_sequence(norm_b, conf_b)

        angles_a_smooth = median_smooth_angle_sequence(angles_a_raw, PHASE31B_SMOOTH_RADIUS)
        angles_b_smooth = median_smooth_angle_sequence(angles_b_raw, PHASE31B_SMOOTH_RADIUS)

        jump_a = temporal_jump_statistics(angles_a_raw, ids_a, "Demo")
        jump_b = temporal_jump_statistics(angles_b_raw, ids_b, "Imitation")

        raw_joint_err_31b = per_joint_path_mean_errors(angles_a_raw, angles_b_raw, path)
        smooth_joint_err_31b = per_joint_path_mean_errors(angles_a_smooth, angles_b_smooth, path)

        raw_local_31b = local_alignment_diagnostic(angles_a_raw, angles_b_raw, path, PHASE31B_LOCAL_RADIUS)
        smooth_local_31b = local_alignment_diagnostic(angles_a_smooth, angles_b_smooth, path, PHASE31B_LOCAL_RADIUS)

    print("  %-16s %12s %14s %10s" % ("Joint", "Avg Error", "Valid Coverage", "Weight"))
    for k, (name, v, e1, e2, w) in enumerate(JOINT_DEFS):
        je = joint_err[k]
        je_s = "  --  " if np.isnan(je) else "%6.1f°" % je
        print("  %-16s %12s %13.0f%% %10.2f" % (name, je_s, coverage[k] * 100, w))
    print("  " + "-" * 56)
    print("  Weighted average joint angle error: %.2f degrees" % overall_err)

    # ⑤ Generate outputs
    print("\n[5/5] Generating reports and visualizations ...")

    if MAKE_DETAIL_CSV:
        joint_csv = os.path.join(OUT_DIR, "joint_report.csv")
        with open(joint_csv, "w", newline="", encoding="utf-8-sig") as f:
            wr = csv.writer(f)
            wr.writerow(["Joint", "Average Angle Error (deg)", "Valid Coverage", "Weight"])
            for k, (name, v, e1, e2, w) in enumerate(JOINT_DEFS):
                wr.writerow([name, "" if np.isnan(joint_err[k]) else "%.3f" % joint_err[k],
                             "%.3f" % coverage[k], w])
            wr.writerow(["Overall Weighted Avg Error (deg)", "%.3f" % overall_err, "", ""])
            wr.writerow(["Similarity Score (0-100)", "%.2f" % score, "", grade(score)])

        detail_csv = os.path.join(OUT_DIR, "alignment_detail.csv")
        with open(detail_csv, "w", newline="", encoding="utf-8-sig") as f:
            wr = csv.writer(f)
            wr.writerow(["Demo Sample Index", "Demo Original Frame", "Imitation Sample Index", "Imitation Original Frame", "Weighted Angle Error (deg)"])
            for ia, ib, e in detail:
                ia, ib = int(ia), int(ib)
                wr.writerow([ia, int(ids_a[ia]), ib, int(ids_b[ib]), "%.3f" % e])

        print("  Saved: %s , %s" % (joint_csv, detail_csv))

    if MAKE_CURVE_IMG:
        p = os.path.join(OUT_DIR, "error_curve.png"); save_error_curve(detail, p); print("  Saved:", p)
    if MAKE_DTW_IMG:
        p = os.path.join(OUT_DIR, "dtw_alignment.png"); save_dtw_map(cost, path, p); print("  Saved:", p)
    if MAKE_COMPARE_VIDEO:
        p = os.path.join(OUT_DIR, "side_by_side_compare.mp4")
        save_compare_video(demo, imit, kps_a, kps_b, ids_a, ids_b, detail, path, score, p)
        if os.path.isfile(p):
            print("  Saved:", p)
    if MAKE_PHASE3_DIAGNOSTICS:
        phase3_dir = os.path.join(OUT_DIR, "phase3_diagnostics")
        os.makedirs(phase3_dir, exist_ok=True)

        phase3_detail_csv = os.path.join(phase3_dir, "joint_detail.csv")
        save_phase3_detail_csv(phase3_diag, detail, path, ids_a, ids_b, phase3_detail_csv)

        phase3_top_csv = os.path.join(phase3_dir, "top_joint_errors.csv")
        save_phase3_top_errors_csv(phase3_diag, path, ids_a, ids_b, phase3_top_csv, PHASE3_TOP_N)

        print_phase3_summary(phase3_diag)

        name_to_idx = {joint[0]: k for k, joint in enumerate(JOINT_DEFS)}

        for name in PHASE3_CURVE_JOINTS:
            k = name_to_idx[name]
            filename = name.lower().replace(" ", "_") + "_error_curve.png"
            save_phase3_joint_curve(phase3_diag["error"][:, k], name, os.path.join(phase3_dir, filename))

        print("  Phase 3.1 diagnostics saved:", phase3_dir)
    if MAKE_PHASE31B_DIAGNOSTICS:
        phase3_dir = os.path.join(OUT_DIR, "phase3_diagnostics")
        os.makedirs(phase3_dir, exist_ok=True)

        jump_csv = os.path.join(phase3_dir, "temporal_jump_report.csv")
        save_temporal_jump_report(jump_a, jump_b, jump_csv)

        smoothing_csv = os.path.join(phase3_dir, "smoothing_comparison.csv")
        save_smoothing_report(raw_joint_err_31b, smooth_joint_err_31b, smoothing_csv)

        local_csv = os.path.join(phase3_dir, "local_alignment_diagnostic.csv")
        save_local_alignment_report(path, ids_a, ids_b, raw_local_31b, smooth_local_31b, local_csv)

        print_phase31b_summary(path, jump_a, jump_b, raw_joint_err_31b, smooth_joint_err_31b, raw_local_31b, smooth_local_31b)

        print("  Phase 3.1B diagnostics saved:", phase3_dir)
    if MAKE_PHASE31D_DIAGNOSTICS:
        phase3d_dir = os.path.join(OUT_DIR, "phase3_diagnostics")
        os.makedirs(phase3d_dir, exist_ok=True)

        rows_demo_31d, demo_matches_31d = cross_resolution_joint_diagnostics(
            demo, PHASE31D_RES_A, PHASE31D_RES_B, FRAME_STEP)

        rows_imit_31d, imit_matches_31d = cross_resolution_joint_diagnostics(
            imit, PHASE31D_RES_A, PHASE31D_RES_B, FRAME_STEP)

        summary_demo_31d = summarize_cross_resolution(rows_demo_31d, "Demo")
        summary_imit_31d = summarize_cross_resolution(rows_imit_31d, "Imitation")
        summary_combined_31d = summarize_cross_resolution_combined(rows_demo_31d, rows_imit_31d)

        detail_31d_csv = os.path.join(phase3d_dir, "cross_resolution_detail.csv")
        save_cross_resolution_detail(
            rows_demo_31d, rows_imit_31d,
            PHASE31D_RES_A, PHASE31D_RES_B,
            detail_31d_csv)

        summary_31d_csv = os.path.join(phase3d_dir, "cross_resolution_summary.csv")
        save_cross_resolution_summary(
            summary_demo_31d,
            summary_imit_31d,
            summary_combined_31d,
            summary_31d_csv)

        print_phase31d_summary(
            summary_combined_31d,
            PHASE31D_RES_A,
            PHASE31D_RES_B,
            demo_matches_31d,
            imit_matches_31d)

        print("  Phase 3.1D diagnostics saved:", phase3d_dir)

    # Final result
    print("\n" + "#" * 76)
    print("#  Overall motion similarity: %.1f / 100  —— %s" % (score, grade(score)))
    print("#  (Weighted average joint angle error %.2f degrees; lower = more similar)" % overall_err)
    print("#  Total runtime %.1f s" % (time.time() - t_all))
    print("#" * 76)


if __name__ == "__main__":
    main()