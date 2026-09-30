# -*- coding: utf-8 -*-
"""
================================================================================
Dance Motion Similarity Evaluation (OpenPose Skeleton + Normalization + DTW Alignment + Joint Angles)
--------------------------------------------------------------------------------
Pipeline:
  ① Extract BODY_25 skeletons frame by frame from both demo video and imitation video via OpenPose;
  ② Skeleton normalization: set hip center as origin (position normalization), scale "neck-hip" torso length to 1 (scale normalization),
     to eliminate differences in height, body shape and shooting distance; fill missing keypoints with linear interpolation along time axis;
  ③ DTW (Dynamic Time Warping) performs flexible alignment on two sequences with different length/rhythm, automatically finds corresponding frames,
     solves "half-beat fast/slow" mismatch problem;
  ④ Calculate per-frame joint angle errors along the aligned path, weighted by joint importance, to get average angle error;
  ⑤ Map to percentage similarity score, and output itemized report, error curve, DTW alignment plot, side-by-side comparison video.
Runtime: Python 3.7.9 (matches pyopenpose.cp37-win_amd64.pyd)

================================================================================
"""
import os
import sys
import csv
import site
import time
# ----------------------------------------------------------------------
# Step 0: Environment setup (ensures libraries are found regardless of launch method)
# ----------------------------------------------------------------------
try:
    import cv2
    import numpy as np
except ImportError:
    _user_site = site.getusersitepackages()
    if os.path.isdir(_user_site):
        site.addsitedir(_user_site)
    import cv2
    import numpy as np

OPENPOSE_ROOT = r"D:\openpose-1.7.0\openpose"
_openpose_bin = os.path.join(OPENPOSE_ROOT, "bin")
os.environ["PATH"] = _openpose_bin + os.pathsep + os.environ["PATH"]
sys.path.append(os.path.join(_openpose_bin, "python", "openpose", "Release"))

try:
    import pyopenpose as op
except ImportError as e:
    print("[Error] Failed to import pyopenpose. Please run with Python 3.7.x (current %s)." % sys.version.split()[0])
    raise e

# ======================================================================
# Step 1: Configuration
# ======================================================================
DEMO_VIDEO = "dance_1_demo.mp4"            # Demo reference video
IMIT_VIDEO = "dance_1_imitation.mp4"       # User imitation video

# ---- Skeleton extraction (CPU speed related; values measured on 1080p landscape video) ----
NET_RESOLUTION = "256x144"   # Network input resolution (multiple of 16, 16:9 landscape). 256x144 ≈0.95s/frame, good accuracy
                             #   Faster options: "224x128"/"192x112"; More accurate: "-1x368" (very slow, ~5s/frame)
FRAME_STEP = 2               # Frame sampling step: original ~59.94fps, step=2 gives ~30fps (sufficient for dance, halves workload)
CONF_THRESHOLD = 0.30        # Keypoint confidence threshold; values below this are treated as "not detected"

# ---- Skeleton cache: extraction is time-consuming, save raw keypoints as .npz; load directly on subsequent runs ----
USE_CACHE = True
FORCE_REEXTRACT = False      # True = ignore cache and force re-extraction
CACHE_DIR = "pose_cache"

# ---- DTW time alignment ----
DTW_WINDOW_RATIO = 0.30      # Sakoe-Chiba flexible window = sequence length × ratio; allows ±30% speed deviation, prevents pathological warping

# ---- Scoring parameters ----
SCORE_TOL = 8.0              # Full-score tolerance: average angle error ≤ this value = 100 points (degrees)
SCORE_TAU = 30.0             # Decay coefficient beyond tolerance (degrees); smaller value = harsher penalty

# ---- Output controls ----
OUT_DIR = "similarity_result"
MAKE_DETAIL_CSV = True       # Export per-aligned-frame error detail CSV
MAKE_CURVE_IMG = True        # Generate error-over-time curve image
MAKE_DTW_IMG = True          # Generate DTW alignment path heatmap
MAKE_COMPARE_VIDEO = True    # Generate side-by-side skeleton comparison video
COMPARE_FPS = 30             # Framerate of comparison video
DISPLAY_HEIGHT = 540         # Height of each video panel in comparison output

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
# Step 2: Extract skeleton sequences frame by frame
# ======================================================================
def pick_primary(pose_keypoints):
    """Select the primary person with most valid keypoints and highest average confidence when multiple people are detected.
    Returns 25x3 array, or None if no person is detected."""
    if pose_keypoints is None or not hasattr(pose_keypoints, "shape") \
            or len(pose_keypoints.shape) != 3 or pose_keypoints.shape[0] == 0:
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


def extract_sequence(video_path, net_resolution, frame_step):
    """Extract primary skeleton frame by frame (sampled by frame_step).
    Returns kps(S,25,3), corresponding original frame indices, fps, total frame count."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError("Cannot open video: %s" % video_path)

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

    params = {
        "model_folder": os.path.join(OPENPOSE_ROOT, "models"),
        "model_pose": "BODY_25",
        "net_resolution": net_resolution,
        "render_pose": 0,            # Output keypoints only, no skeleton rendering (faster extraction)
        "logging_level": 255,
    }
    wrapper = op.WrapperPython()
    wrapper.configure(params)
    wrapper.start()

    kps, frame_ids = [], []
    idx = -1
    t0 = time.time()
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            idx += 1
            if idx % frame_step != 0:
                continue

            datum = op.Datum()
            datum.cvInputData = frame
            wrapper.emplaceAndPop(op.VectorDatum([datum]))
            primary = pick_primary(datum.poseKeypoints)

            if primary is None:
                primary = np.zeros((N_PARTS, 3), dtype=np.float32)  # No person: placeholder, interpolated later
            kps.append(primary.astype(np.float32))
            frame_ids.append(idx)

            done = len(kps)
            if done % 20 == 0:
                plan = (total + frame_step - 1) // frame_step
                el = time.time() - t0
                speed = done / el if el > 0 else 0
                eta = (plan - done) / speed if speed > 0 else 0
                sys.stdout.write("\r  Extracting %5d/%d  %5.1f%%  %4.1f fps  ETA %0.0f s   "
                                 % (done, plan, 100.0 * done / max(plan, 1), speed, eta))
                sys.stdout.flush()
    finally:
        cap.release()
        wrapper.stop()
    print()
    return np.stack(kps), np.array(frame_ids, dtype=np.int64), fps, total


def get_or_extract(video_path, net_resolution, frame_step):
    """Cached skeleton extraction: cache key includes resolution and frame step; auto re-extract when parameters change."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    base = os.path.splitext(os.path.basename(video_path))[0]
    cache = os.path.join(CACHE_DIR, "%s_%s_step%d.npz" % (base, net_resolution.replace("-1", "auto"), frame_step))

    if USE_CACHE and not FORCE_REEXTRACT and os.path.isfile(cache):
        d = np.load(cache)
        print("  Loaded from cache: %s (%d frames)" % (cache, len(d["kps"])))
        return d["kps"], d["frame_ids"], float(d["fps"]), int(d["total"])

    print("  Extracting skeletons with OpenPose (slow on first run, cached afterwards)...")
    kps, frame_ids, fps, total = extract_sequence(video_path, net_resolution, frame_step)
    np.savez_compressed(cache, kps=kps, frame_ids=frame_ids, fps=fps, total=total)
    print("  Skeleton cached: %s" % cache)
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
    """DTW with Sakoe-Chiba window constraint.
    Returns (alignment_path, normalized_DTW_distance, window_size).
    path = [(demo_frame_idx, imitation_frame_idx), ...], monotonic in time, minimizes total pose cost."""
    n, m = cost.shape
    win = max(abs(n - m), int(round(window_ratio * max(n, m))))
    INF = float("inf")

    D = np.full((n + 1, m + 1), INF, dtype=np.float64)
    D[0, 0] = 0.0

    for i in range(1, n + 1):
        j_lo = max(1, i - win)
        j_hi = min(m, i + win)
        for j in range(j_lo, j_hi + 1):
            diag = D[i - 1, j - 1]
            down = D[i - 1, j]
            right = D[i, j - 1]
            D[i, j] = cost[i - 1, j - 1] + min(diag, down, right)

    # Backtrack to find optimal path
    i, j, path = n, m, []
    while i > 0 and j > 0:
        path.append((i - 1, j - 1))
        diag, down, right = D[i - 1, j - 1], D[i - 1, j], D[i, j - 1]
        step = np.argmin([diag, down, right])  # 0=diagonal, 1=down, 2=right
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
    print("Dance Motion Similarity Evaluation V1.0")
    print("Demo video: %s\nImitation video: %s" % (demo, imit))
    print("=" * 76)

    # ① Extract skeletons (with caching)
    print("\n[1/5] Extract skeleton sequences")
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

    # Final result
    print("\n" + "#" * 76)
    print("#  Overall motion similarity: %.1f / 100  —— %s" % (score, grade(score)))
    print("#  (Weighted average joint angle error %.2f degrees; lower = more similar)" % overall_err)
    print("#  Total runtime %.1f s" % (time.time() - t_all))
    print("#" * 76)


if __name__ == "__main__":
    main()
