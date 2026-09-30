import json
from pathlib import Path
import numpy as np

BODY_25_NAMES = [
    "Nose", "Neck",
    "RShoulder", "RElbow", "RWrist",
    "LShoulder", "LElbow", "LWrist",
    "MidHip",
    "RHip", "RKnee", "RAnkle",
    "LHip", "LKnee", "LAnkle",
    "REye", "LEye", "REar", "LEar",
    "LBigToe", "LSmallToe", "LHeel",
    "RBigToe", "RSmallToe", "RHeel"
]

BODY_25_INDEX = {name: i for i, name in enumerate(BODY_25_NAMES)}

CONF_THRESHOLD = 0.30


def read_openpose_json(json_file):
    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    people_keypoints = []

    for person in data["people"]:
        keypoints = np.array(person["pose_keypoints_2d"], dtype=float)
        keypoints = keypoints.reshape(25, 3)
        people_keypoints.append(keypoints)

    return people_keypoints


def pick_primary(people_keypoints):
    """从一帧中选择主要人物。"""

    if len(people_keypoints) == 0:
        return None

    best_person = None
    best_score = -1

    for person in people_keypoints:
        valid_mask = person[:, 2] > CONF_THRESHOLD
        valid_count = np.sum(valid_mask)

        if valid_count == 0:
            continue

        mean_conf = np.mean(person[valid_mask, 2])

        # 优先选择有效关键点更多的人；
        # 有效点数量相同时，再考虑平均 confidence
        score = valid_count * 10 + mean_conf

        if score > best_score:
            best_score = score
            best_person = person

    return best_person


def load_openpose_sequence(json_dir):
    """读取一个视频对应的全部 OpenPose JSON，生成单人关键点序列。"""

    json_dir = Path(json_dir)
    json_files = sorted(json_dir.glob("*_keypoints.json"))

    if not json_files:
        raise FileNotFoundError(f"没有找到 OpenPose JSON：{json_dir}")

    sequence = []
    frame_ids = []

    for frame_index, json_file in enumerate(json_files):
        people = read_openpose_json(json_file)
        primary = pick_primary(people)

        if primary is None:
            # 当前帧没有可靠人物，先用全零占位
            primary = np.zeros((25, 3), dtype=np.float32)

        sequence.append(primary)

        # OpenPose 文件名倒数第二部分通常就是原始视频帧编号
        try:
            frame_id = int(json_file.stem.split("_")[-2])
        except ValueError:
            frame_id = frame_index

        frame_ids.append(frame_id)

    sequence = np.stack(sequence).astype(np.float32)
    frame_ids = np.array(frame_ids, dtype=np.int64)

    return sequence, frame_ids