# -*- coding: utf-8 -*-
"""
Final Regression Check for Dance Motion Similarity Evaluation V1.2

Purpose
-------
After permanently freezing V1.2 to NET_RESOLUTION="-1x368", verify that the
main computational pipeline reproduces the validated Phase 3.6A results.

Checks:
1) ch01 normal-case D pair
2) ch10 difficult floor-motion D pair

This script does NOT recalibrate or modify any parameter.
"""

import os
import importlib.util
import numpy as np

V12_FILE = "Dance_Motion_Similarity_Evaluation_V1.2.py"
VIDEO_DIR = os.path.join("data", "aistpp", "videos")

EXPECTED_SETTINGS = {
    "NET_RESOLUTION": "-1x368",
    "FRAME_STEP": 2,
    "CONF_THRESHOLD": 0.30,
    "DTW_WINDOW_RATIO": 0.30,
    "SMOOTH_RADIUS": 1,
    "SCORE_CENTER": 33.046977,
    "SCORE_SCALE": 6.292398,
}

CASES = [
    {
        "name": "ch01 normal-case regression",
        "reference": "gBR_sBM_c01_d04_mBR0_ch01.mp4",
        "learner": "gBR_sBM_c01_d05_mBR0_ch01.mp4",
        "expected": {
            "angle_error": 17.058927,
            "score": 92.695651,
            "pose_distance": 0.192032,
            "non_diagonal_ratio": 0.059459,
            "timing_deviation": 0.006727,
            "path_length": 371,
        },
    },
    {
        "name": "ch10 floor-motion regression",
        "reference": "gBR_sBM_c01_d04_mBR0_ch10.mp4",
        "learner": "gBR_sBM_c01_d05_mBR0_ch10.mp4",
        "expected": {
            "angle_error": 19.153210,
            "score": 90.096777,
            "pose_distance": 0.663112,
            "non_diagonal_ratio": 0.182278,
            "timing_deviation": 0.014772,
            "path_length": 396,
        },
    },
]

# Small numerical tolerance. With the same cached JSON and frozen code,
# results should normally match much more closely than these limits.
TOL = {
    "angle_error": 0.02,
    "score": 0.05,
    "pose_distance": 0.002,
    "non_diagonal_ratio": 0.002,
    "timing_deviation": 0.0002,
    "path_length": 0,
}


def load_v12():
    path = os.path.abspath(V12_FILE)
    if not os.path.isfile(path):
        raise IOError("V1.2 file not found: %s" % path)
    spec = importlib.util.spec_from_file_location("dance_v12_final", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def check_settings(v12):
    print("=" * 88)
    print("FINAL V1.2 FROZEN-SETTING CHECK")
    print("=" * 88)
    ok = True
    for name, expected in EXPECTED_SETTINGS.items():
        actual = getattr(v12, name, None)
        if isinstance(expected, float):
            passed = actual is not None and abs(float(actual) - expected) < 1e-12
        else:
            passed = actual == expected
        print("  %-20s actual=%-12s expected=%-12s %s" %
              (name, str(actual), str(expected), "PASS" if passed else "FAIL"))
        ok = ok and passed
    return ok


def load_sequence(v12, filename):
    path = os.path.join(VIDEO_DIR, filename)
    if not os.path.isfile(path):
        raise IOError("Video not found: %s" % path)

    kps, ids, fps, total = v12.get_or_extract(path, v12.NET_RESOLUTION, v12.FRAME_STEP)
    detection = float((kps[:, :, 2] > v12.CONF_THRESHOLD).any(axis=1).mean())

    kps_i = v12.fill_missing(kps)
    norm, conf, scale = v12.normalize_frames(kps_i)

    return {
        "filename": filename,
        "kps": kps,
        "ids": ids,
        "fps": float(fps),
        "total": int(total),
        "detection": detection,
        "norm": norm,
        "conf": conf,
        "scale": float(scale),
    }


def compare_value(name, actual, expected):
    tol = TOL[name]
    if name == "path_length":
        passed = int(actual) == int(expected)
        delta = int(actual) - int(expected)
    else:
        passed = abs(float(actual) - float(expected)) <= tol
        delta = float(actual) - float(expected)

    print("    %-20s actual=%-12.6f expected=%-12.6f delta=%+10.6f  %s" %
          (name, float(actual), float(expected), float(delta), "PASS" if passed else "FAIL"))
    return passed


def run_case(v12, case):
    print("\n" + "-" * 88)
    print(case["name"])
    print("-" * 88)

    a = load_sequence(v12, case["reference"])
    b = load_sequence(v12, case["learner"])

    print("  Reference detection: %.1f%% | scale %.3f px" % (100.0 * a["detection"], a["scale"]))
    print("  Learner   detection: %.1f%% | scale %.3f px" % (100.0 * b["detection"], b["scale"]))

    result = v12.evaluate_pair(a["norm"], a["conf"], b["norm"], b["conf"], a["fps"], b["fps"])
    timing = result["timing"]

    actual = {
        "angle_error": result["weighted_angle_error_deg"],
        "score": result["overall_similarity_score"],
        "pose_distance": result["dtw_pose_distance"],
        "non_diagonal_ratio": timing["non_diagonal_ratio"],
        "timing_deviation": timing["mean_relative_time_deviation"],
        "path_length": len(result["path"]),
    }

    print("\n  Regression metrics")
    passed = True
    for key in ("angle_error", "score", "pose_distance", "non_diagonal_ratio", "timing_deviation", "path_length"):
        passed = compare_value(key, actual[key], case["expected"][key]) and passed

    # Measurement sanity: both validated -1x368 cases reached 100% person detection.
    detection_pass = a["detection"] >= 0.999 and b["detection"] >= 0.999
    print("    %-20s ref=%6.2f%% test=%6.2f%%               %s" %
          ("person_detection", 100.0 * a["detection"], 100.0 * b["detection"],
           "PASS" if detection_pass else "FAIL"))

    return passed and detection_pass


def main():
    v12 = load_v12()

    settings_ok = check_settings(v12)
    if not settings_ok:
        print("\n[FAIL] V1.2 frozen settings do not match the validated final configuration.")
        print('Set NET_RESOLUTION = "-1x368" and keep all other frozen settings unchanged.')
        return

    results = [run_case(v12, case) for case in CASES]

    print("\n" + "=" * 88)
    if all(results):
        print("FINAL REGRESSION RESULT: PASS")
        print("V1.2 reproduces the validated -1x368 Phase 3.6A pipeline.")
        print("The implementation can now be treated as FROZEN.")
    else:
        print("FINAL REGRESSION RESULT: FAIL")
        print("At least one validated result was not reproduced.")
        print("Do not recalibrate. First check cache, settings, or unintended code changes.")
    print("=" * 88)


if __name__ == "__main__":
    main()
