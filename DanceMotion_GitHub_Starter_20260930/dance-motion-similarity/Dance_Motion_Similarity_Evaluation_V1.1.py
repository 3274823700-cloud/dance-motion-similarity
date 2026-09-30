from pathlib import Path
import numpy as np

from keypoint_utils import BODY_25_INDEX, load_openpose_sequence


CONF_THRESHOLD = 0.30
DTW_WINDOW_RATIO = 0.30

MID_HIP = BODY_25_INDEX["MidHip"]
NECK = BODY_25_INDEX["Neck"]

R_HIP = BODY_25_INDEX["RHip"]
L_HIP = BODY_25_INDEX["LHip"]

R_SHOULDER = BODY_25_INDEX["RShoulder"]
L_SHOULDER = BODY_25_INDEX["LShoulder"]


def normalize_frames(kps):
    """
    对每一帧人体骨架进行：
    1. MidHip 中心化
    2. Neck-MidHip 躯干长度归一化

    返回：
    norm_xy     (S, 25, 2)
    confidence  (S, 25)
    frame_valid (S,)
    """

    n_frames = kps.shape[0]

    norm_xy = np.zeros((n_frames, 25, 2), dtype=np.float64)
    confidence = kps[:, :, 2].copy()
    frame_valid = np.zeros(n_frames, dtype=bool)

    for i in range(n_frames):
        xy = kps[i, :, :2]
        conf = confidence[i]

        # 1. 确定身体中心
        if conf[MID_HIP] > CONF_THRESHOLD:
            hip_center = xy[MID_HIP]
        elif conf[R_HIP] > CONF_THRESHOLD and conf[L_HIP] > CONF_THRESHOLD:
            hip_center = (xy[R_HIP] + xy[L_HIP]) / 2.0
        else:
            continue

        # 2. 确定躯干上端
        if conf[NECK] > CONF_THRESHOLD:
            neck_center = xy[NECK]
        elif conf[R_SHOULDER] > CONF_THRESHOLD and conf[L_SHOULDER] > CONF_THRESHOLD:
            neck_center = (xy[R_SHOULDER] + xy[L_SHOULDER]) / 2.0
        else:
            continue

        # 3. 计算躯干长度
        torso_length = np.linalg.norm(neck_center - hip_center)

        if torso_length < 1e-6:
            continue

        # 4. 平移 + 尺度归一化
        norm_xy[i] = (xy - hip_center) / torso_length
        frame_valid[i] = True

    return norm_xy, confidence, frame_valid


def _pairwise_cost_block(a_xy, a_conf, b_xy, b_conf):
    """
    计算一组 Reference frames 与全部 Test frames 之间的姿态距离。

    距离采用：
    confidence-weighted normalized keypoint distance
    """

    # shape: (Reference帧数, Test帧数, 25, 2)
    diff = a_xy[:, None, :, :] - b_xy[None, :, :, :]

    # 两边 confidence 取较小值作为关键点权重
    # shape: (Reference帧数, Test帧数, 25)
    weights = np.minimum(a_conf[:, None, :], b_conf[None, :, :])

    # confidence 太低的关键点不参与比较
    weights[weights < CONF_THRESHOLD] = 0.0

    # 每个关键点二维欧氏距离平方
    squared_distance = (diff ** 2).sum(axis=3)

    # 每一对姿态有效关键点权重总和
    weight_sum = weights.sum(axis=2)

    # confidence-weighted pose distance
    distance = np.sqrt((weights * squared_distance).sum(axis=2) / np.maximum(weight_sum, 1e-9))

    # 如果两个姿态之间没有共同有效关键点，赋默认较大距离
    distance[weight_sum < 1e-9] = 1.0

    return distance


def build_cost_matrix(norm_a, conf_a, norm_b, conf_b, block=64):
    """
    构建两段动作之间完整的 Pose Cost Matrix。

    返回：
    cost.shape = (Reference帧数, Test帧数)
    """

    n = norm_a.shape[0]
    m = norm_b.shape[0]

    cost = np.zeros((n, m), dtype=np.float64)

    # 分块计算，避免一次生成过大的中间数组
    for start in range(0, n, block):
        end = start + block
        cost[start:end] = _pairwise_cost_block(norm_a[start:end], conf_a[start:end], norm_b, conf_b)

    return cost


def dtw_align(cost, window_ratio):
    """
    使用 V1.0 的 DTW 对两段动作进行时间对齐。

    输入：
    cost         Pose Cost Matrix，shape = (N, M)
    window_ratio Sakoe-Chiba window 比例

    返回：
    path         最优时间对齐路径 [(ref_idx, test_idx), ...]
    norm_dist    最优路径上的平均 pose cost
    win          实际使用的窗口大小
    """

    n, m = cost.shape

    # V1.0 原始窗口设计：窗口围绕 j ≈ i
    win = max(abs(n - m), int(round(window_ratio * max(n, m))))

    INF = float("inf")

    # 累积代价矩阵，多一行一列用于初始化
    D = np.full((n + 1, m + 1), INF, dtype=np.float64)
    D[0, 0] = 0.0

    # Dynamic Programming
    for i in range(1, n + 1):
        j_lo = max(1, i - win)
        j_hi = min(m, i + win)

        for j in range(j_lo, j_hi + 1):
            diag = D[i - 1, j - 1]
            vertical = D[i - 1, j]
            horizontal = D[i, j - 1]
            D[i, j] = cost[i - 1, j - 1] + min(diag, vertical, horizontal)

    # 从右下角 backtracking 得到最优路径
    i, j = n, m
    path = []

    while i > 0 and j > 0:
        path.append((i - 1, j - 1))

        diag = D[i - 1, j - 1]
        vertical = D[i - 1, j]
        horizontal = D[i, j - 1]

        step = np.argmin([diag, vertical, horizontal])

        if step == 0:
            i -= 1
            j -= 1
        elif step == 1:
            i -= 1
        else:
            j -= 1

    path.reverse()

    norm_dist = D[n, m] / len(path)

    return path, norm_dist, win

def dtw_align_scaled_window(cost, window_ratio):
    """
    使用按序列长度比例缩放中心线的 DTW window。

    中心关系：
    j / (M - 1) ≈ i / (N - 1)

    即：
    j_center ≈ i * (M - 1) / (N - 1)
    """

    n, m = cost.shape

    # window 的半宽使用 Test 时间轴长度计算
    win = max(1, int(round(window_ratio * (m - 1))))

    INF = float("inf")
    D = np.full((n + 1, m + 1), INF, dtype=np.float64)
    D[0, 0] = 0.0

    for i in range(1, n + 1):
        if n == 1:
            j_center = 1
        else:
            j_center = 1 + int(round((i - 1) * (m - 1) / (n - 1)))

        j_lo = max(1, j_center - win)
        j_hi = min(m, j_center + win)

        for j in range(j_lo, j_hi + 1):
            diag = D[i - 1, j - 1]
            vertical = D[i - 1, j]
            horizontal = D[i, j - 1]

            D[i, j] = cost[i - 1, j - 1] + min(diag, vertical, horizontal)

    i, j = n, m
    path = []

    while i > 0 and j > 0:
        path.append((i - 1, j - 1))

        diag = D[i - 1, j - 1]
        vertical = D[i - 1, j]
        horizontal = D[i, j - 1]

        step = np.argmin([diag, vertical, horizontal])

        if step == 0:
            i -= 1
            j -= 1
        elif step == 1:
            i -= 1
        else:
            j -= 1

    path.reverse()

    norm_dist = D[n, m] / len(path)

    return path, norm_dist, win

VIDEO_NAME = "gBR_sBM_c01_d04_mBR0_ch01"

JSON_DIR = Path("data/aistpp/openpose_json") / VIDEO_NAME

keypoints, frame_ids = load_openpose_sequence(JSON_DIR)

norm_xy, confidence, frame_valid = normalize_frames(keypoints)


# ============================================================
# Phase 2.1: Temporal Resampling Cost Matrix Sanity Test
# ============================================================

# Reference 使用完整动作序列
norm_ref = norm_xy
conf_ref = confidence

# Test 每隔一帧取一帧：
# Test[0] = Reference[0]
# Test[1] = Reference[2]
# Test[2] = Reference[4]
# ...
norm_test = norm_xy[::2]
conf_test = confidence[::2]

print()
print("=" * 60)
print("DTW Phase 2.1 - Temporal Resampling Sanity Test")
print("=" * 60)

print(f"Reference frames: {len(norm_ref)}")
print(f"Test frames:      {len(norm_test)}")

# Reference × Test Pose Cost Matrix
cost_resampled = build_cost_matrix(norm_ref, conf_ref, norm_test, conf_test)

print()
print(f"Cost matrix shape: {cost_resampled.shape}")

print()
print("Known mapping pose costs:")

print(f"Ref 0   vs Test 0   : {cost_resampled[0, 0]:.8f}")
print(f"Ref 2   vs Test 1   : {cost_resampled[2, 1]:.8f}")
print(f"Ref 100 vs Test 50  : {cost_resampled[100, 50]:.8f}")
print(f"Ref 200 vs Test 100 : {cost_resampled[200, 100]:.8f}")
print(f"Ref 718 vs Test 359 : {cost_resampled[718, 359]:.8f}")

# Test[j] = Reference[2*j]
test_indices = np.arange(len(norm_test))
ref_indices = 2 * test_indices

known_mapping_costs = cost_resampled[ref_indices, test_indices]

print()
print("All known mappings:")

print(f"Mean known-mapping cost: {known_mapping_costs.mean():.10f}")
print(f"Maximum known-mapping cost: {known_mapping_costs.max():.10f}")
print(f"Minimum known-mapping cost: {known_mapping_costs.min():.10f}")


# ============================================================
# Phase 2.2: V1.0 DTW Baseline
# ============================================================

print()
print("=" * 60)
print("DTW Phase 2.2 - V1.0 Baseline")
print("=" * 60)

path, dtw_distance, window_size = dtw_align(cost_resampled, DTW_WINDOW_RATIO)

print(f"DTW window ratio: {DTW_WINDOW_RATIO}")
print(f"DTW window size: {window_size}")
print(f"DTW path length: {len(path)}")
print(f"Normalized DTW distance: {dtw_distance:.8f}")

print()
print(f"Path start: {path[0]}")
print(f"Path end:   {path[-1]}")
print()

print("=" * 60)
print("DTW Phase 2.3 - Scaled-Time Window")
print("=" * 60)

scaled_path, scaled_distance, scaled_window = dtw_align_scaled_window(cost_resampled, DTW_WINDOW_RATIO)

print(f"DTW window ratio: {DTW_WINDOW_RATIO}")
print(f"Scaled window size: {scaled_window}")
print(f"DTW path length: {len(scaled_path)}")
print(f"Normalized DTW distance: {scaled_distance:.8f}")

print()
print(f"Path start: {scaled_path[0]}")
print(f"Path end:   {scaled_path[-1]}")

# ------------------------------------------------------------
# 统计 diagonal / vertical / horizontal 三种路径
# ------------------------------------------------------------

diagonal_steps = 0
vertical_steps = 0
horizontal_steps = 0
path_valid = True

for k in range(1, len(path)):
    prev_i, prev_j = path[k - 1]
    curr_i, curr_j = path[k]

    di = curr_i - prev_i
    dj = curr_j - prev_j

    if di == 1 and dj == 1:
        diagonal_steps += 1
    elif di == 1 and dj == 0:
        vertical_steps += 1
    elif di == 0 and dj == 1:
        horizontal_steps += 1
    else:
        path_valid = False

print()
print("DTW step counts:")

print(f"Diagonal steps:   {diagonal_steps}")
print(f"Vertical steps:   {vertical_steps}")
print(f"Horizontal steps: {horizontal_steps}")

print()
print(f"DTW path valid: {path_valid}")


# ------------------------------------------------------------
# 与真实时间映射比较
#
# Test[j] = Reference[2*j]
# 所以理论对应 Reference index = 2 * Test index
# ------------------------------------------------------------

alignment_errors = []

for ref_idx, test_idx in path:
    expected_ref_idx = 2 * test_idx
    error = abs(ref_idx - expected_ref_idx)
    alignment_errors.append(error)

alignment_errors = np.array(alignment_errors, dtype=np.float64)

print()
print("Alignment index error:")

print(f"Mean error:    {alignment_errors.mean():.4f} reference frames")
print(f"Maximum error: {alignment_errors.max():.0f} reference frames")


# ------------------------------------------------------------
# 打印若干代表性 aligned pairs
# ------------------------------------------------------------

print()
print("Example aligned pairs:")

sample_positions = np.linspace(0, len(path) - 1, 10, dtype=int)

for pos in sample_positions:
    ref_idx, test_idx = path[pos]
    expected_ref_idx = 2 * test_idx
    error = abs(ref_idx - expected_ref_idx)

    print(f"Ref {ref_idx:3d}  <->  Test {test_idx:3d}   Expected Ref {expected_ref_idx:3d}   Error {error}")

print("=" * 60)


# ============================================================
# Existing normalization information
# ============================================================

print()
print("=" * 60)
print("Dance Motion Similarity Evaluation V1.1")
print("=" * 60)

print(f"Input frames: {len(keypoints)}")
print(f"Original keypoints shape: {keypoints.shape}")
print(f"Normalized coordinates shape: {norm_xy.shape}")

valid_count = frame_valid.sum()

print(f"Valid normalized frames: {valid_count}/{len(frame_valid)} ({valid_count / len(frame_valid) * 100:.2f}%)")

print()
print("First normalized frame:")

print("MidHip:", norm_xy[0, MID_HIP])
print("Neck:", norm_xy[0, NECK])

torso_after = np.linalg.norm(norm_xy[0, NECK] - norm_xy[0, MID_HIP])

print(f"Normalized torso length: {torso_after:.6f}")