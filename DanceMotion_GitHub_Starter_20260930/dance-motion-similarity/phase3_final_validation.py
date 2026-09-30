# -*- coding: utf-8 -*-
"""
Phase 3.5B - Untouched Final Validation on ch07-ch10

Frozen before this test:
- SEQUENCE_MEDIAN normalization
- DTW settings
- 3-frame median angle smoothing
- joint weights
- logistic 0-100 score mapping from ch01-ch06

IMPORTANT: ch07-ch10 are final test data. Do not tune parameters after seeing them.
"""

import os
import csv
import importlib.util
from itertools import combinations
import numpy as np

HELPER_FILE = "phase3_score_mapping_calibration.py"
VIDEO_DIR = os.path.join("data", "aistpp", "videos")
PARAMETER_CSV = os.path.join("phase3_score_mapping", "score_mapping_parameters.csv")
DEVELOPMENT_SCORE_CSV = os.path.join("phase3_score_mapping", "development_score_results.csv")
OUT_DIR = "phase3_final_validation"
FINAL_CHOREOS = ["ch07", "ch08", "ch09", "ch10"]

BLOCKS = {
    "A": {"music": "mBR0", "dancer_a": "d04", "dancer_b": "d05", "e_dancer": "d04"},
    "B": {"music": "mBR2", "dancer_a": "d04", "dancer_b": "d06", "e_dancer": "d06"},
    "C": {"music": "mBR4", "dancer_a": "d05", "dancer_b": "d06", "e_dancer": "d05"},
}


def import_helper():
    path = os.path.abspath(HELPER_FILE)
    if not os.path.isfile(path):
        raise IOError("Helper file not found: %s" % path)
    spec = importlib.util.spec_from_file_location("phase35a", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def read_csv(path):
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)


def make_filename(dancer, music, choreo):
    return "gBR_sBM_c01_%s_%s_%s.mp4" % (dancer, music, choreo)


def key(dancer, music, choreo):
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
    wins, total = 0.0, 0
    for se in e:
        for sd in d:
            total += 1
            if se > sd:
                wins += 1.0
            elif se == sd:
                wins += 0.5
    return wins / total if total else float("nan")


def balanced_accuracy(labels, preds):
    d = [1.0 if p == 0 else 0.0 for y, p in zip(labels, preds) if y == 0]
    e = [1.0 if p == 1 else 0.0 for y, p in zip(labels, preds) if y == 1]
    return 0.5 * (float(np.mean(d)) + float(np.mean(e)))


def timing_deviation(path, n_ref, n_test):
    vals = []
    for ia, ib in path:
        ta = ia / float(max(n_ref - 1, 1))
        tb = ib / float(max(n_test - 1, 1))
        vals.append(abs(ta - tb))
    return float(np.mean(vals))


def load_frozen_parameters():
    rows = read_csv(PARAMETER_CSV)
    if len(rows) != 1:
        raise ValueError("Expected exactly one row in %s" % PARAMETER_CSV)
    center = float(rows[0]["logistic_center_deg"])
    scale = float(rows[0]["logistic_scale_deg"])

    dev = read_csv(DEVELOPMENT_SCORE_CSV)
    d = [float(r["angle_error_deg"]) for r in dev if r["group"] == "D"]
    e = [float(r["angle_error_deg"]) for r in dev if r["group"] == "E"]
    d_med = float(np.median(d))
    e_med = float(np.median(e))
    threshold = 0.5 * (d_med + e_med)
    return center, scale, threshold, d_med, e_med


def collect_videos_and_pairs():
    videos = {}
    pairs = []
    d_id, e_id = 1, 1

    for block, b in BLOCKS.items():
        for ch in FINAL_CHOREOS:
            for dancer in {b["dancer_a"], b["dancer_b"], b["e_dancer"]}:
                videos[key(dancer, b["music"], ch)] = make_filename(dancer, b["music"], ch)

        # D: different dancer, same music + same choreography
        for ch in FINAL_CHOREOS:
            pairs.append(("FD%d" % d_id, block, "D",
                          key(b["dancer_a"], b["music"], ch),
                          key(b["dancer_b"], b["music"], ch)))
            d_id += 1

        # E: same dancer/music, every pair of different choreographies
        for ch1, ch2 in combinations(FINAL_CHOREOS, 2):
            pairs.append(("FE%d" % e_id, block, "E",
                          key(b["e_dancer"], b["music"], ch1),
                          key(b["e_dancer"], b["music"], ch2)))
            e_id += 1

    return videos, pairs


def main():
    helper = import_helper()
    v11 = helper.load_v11()
    center, scale, dev_threshold, dev_d_med, dev_e_med = load_frozen_parameters()
    os.makedirs(OUT_DIR, exist_ok=True)

    print("=" * 94)
    print("Phase 3.5B - Untouched Final Validation on ch07-ch10")
    print("=" * 94)
    print("Development used earlier : ch01-ch06")
    print("Final validation now      : ch07-ch10")
    print("Normalization             : SEQUENCE_MEDIAN [FROZEN]")
    print("Angle smoothing           : 3-frame median [FROZEN]")
    print("Score center              : %.6f deg [FROZEN]" % center)
    print("Score scale               : %.6f deg [FROZEN]" % scale)
    print("Development D/E medians   : %.3f / %.3f deg" % (dev_d_med, dev_e_med))
    print("Diagnostic error threshold: %.3f deg [FROZEN]" % dev_threshold)
    print("No parameter is fitted from ch07-ch10.")

    videos, pairs = collect_videos_and_pairs()
    missing = [os.path.join(VIDEO_DIR, f) for f in videos.values()
               if not os.path.isfile(os.path.join(VIDEO_DIR, f))]
    if missing:
        print("\n[ERROR] Missing final-validation videos:")
        for p in missing:
            print(" ", p)
        return

    print("Local file validation: all %d final-validation videos are present." % len(videos))
    print("Final pair matrix: %d D + %d E = %d pairs" %
          (sum(g == "D" for _, _, g, _, _ in pairs),
           sum(g == "E" for _, _, g, _, _ in pairs), len(pairs)))

    seq = {}
    for i, (k, filename) in enumerate(videos.items(), 1):
        print("\n[%d/%d] %s" % (i, len(videos), filename))
        seq[k] = helper.load_sequence(v11, filename)

    rows = []
    for pair_id, block, group, a_key, b_key in pairs:
        a, b = seq[a_key], seq[b_key]
        angle_err, pose_dist, path = helper.aligned_angle_error(v11, a, b)
        score = helper.error_to_score(angle_err, center, scale)
        timing = timing_deviation(path, len(a["norm"]), len(b["norm"]))

        row = {
            "pair_id": pair_id,
            "block": block,
            "group": group,
            "reference": a["filename"],
            "test": b["filename"],
            "smoothed_angle_error_deg": angle_err,
            "overall_similarity_score": score,
            "dtw_pose_distance": pose_dist,
            "mean_relative_time_deviation": timing,
            "path_length": len(path),
        }
        rows.append(row)

        print("\n%s [%s / Block %s]" % (pair_id, group, block))
        print("  Angle error : %.3f deg" % angle_err)
        print("  Score       : %.2f" % score)
        print("  Pose dist.  : %.6f" % pose_dist)
        print("  Timing dev. : %.6f" % timing)

    # Overall raw metric validation
    labels = [0 if r["group"] == "D" else 1 for r in rows]
    angle_vals = [r["smoothed_angle_error_deg"] for r in rows]
    preds = [1 if e >= dev_threshold else 0 for e in angle_vals]
    angle_auc = auc_binary(labels, angle_vals)
    angle_bal = balanced_accuracy(labels, preds)

    d_angle = describe([r["smoothed_angle_error_deg"] for r in rows if r["group"] == "D"])
    e_angle = describe([r["smoothed_angle_error_deg"] for r in rows if r["group"] == "E"])
    d_score = describe([r["overall_similarity_score"] for r in rows if r["group"] == "D"])
    e_score = describe([r["overall_similarity_score"] for r in rows if r["group"] == "E"])
    d_pose = describe([r["dtw_pose_distance"] for r in rows if r["group"] == "D"])
    e_pose = describe([r["dtw_pose_distance"] for r in rows if r["group"] == "E"])
    d_time = describe([r["mean_relative_time_deviation"] for r in rows if r["group"] == "D"])
    e_time = describe([r["mean_relative_time_deviation"] for r in rows if r["group"] == "E"])

    pose_auc = auc_binary(labels, [r["dtw_pose_distance"] for r in rows])
    time_auc = auc_binary(labels, [r["mean_relative_time_deviation"] for r in rows])

    print("\n" + "=" * 94)
    print("FINAL VALIDATION SUMMARY")
    print("=" * 94)
    print("\nOVERALL RAW METRIC: SMOOTHED ANGLE ERROR")
    print("  D: mean %.3f | median %.3f | min %.3f | max %.3f" %
          (d_angle["mean"], d_angle["median"], d_angle["min"], d_angle["max"]))
    print("  E: mean %.3f | median %.3f | min %.3f | max %.3f" %
          (e_angle["mean"], e_angle["median"], e_angle["min"], e_angle["max"]))
    print("  Gap min(E)-max(D): %+.3f deg" % (e_angle["min"] - d_angle["max"]))
    print("  AUC: %.3f" % angle_auc)
    print("  Balanced accuracy with frozen development threshold %.3f deg: %.3f" %
          (dev_threshold, angle_bal))

    print("\nFROZEN 0-100 OVERALL SIMILARITY SCORE")
    print("  D: mean %.2f | median %.2f | min %.2f | max %.2f" %
          (d_score["mean"], d_score["median"], d_score["min"], d_score["max"]))
    print("  E: mean %.2f | median %.2f | min %.2f | max %.2f" %
          (e_score["mean"], e_score["median"], e_score["min"], e_score["max"]))

    print("\nPOSE DISTANCE -- alignment diagnostic only")
    print("  D mean %.4f | E mean %.4f | AUC %.3f | gap %+.4f" %
          (d_pose["mean"], e_pose["mean"], pose_auc, e_pose["min"] - d_pose["max"]))

    print("\nTIMING -- separate rhythm diagnostic")
    print("  D mean %.5f | E mean %.5f | AUC %.3f | gap %+.5f" %
          (d_time["mean"], e_time["mean"], time_auc, e_time["min"] - d_time["max"]))

    pair_csv = os.path.join(OUT_DIR, "final_validation_pair_results.csv")
    summary_csv = os.path.join(OUT_DIR, "final_validation_summary.csv")
    block_csv = os.path.join(OUT_DIR, "final_validation_block_statistics.csv")

    write_csv(pair_csv, rows)

    summary_rows = [
        {"metric": "smoothed_angle_error_deg", "role": "overall raw metric",
         "D_mean": d_angle["mean"], "D_median": d_angle["median"], "D_min": d_angle["min"], "D_max": d_angle["max"],
         "E_mean": e_angle["mean"], "E_median": e_angle["median"], "E_min": e_angle["min"], "E_max": e_angle["max"],
         "auc": angle_auc, "balanced_accuracy": angle_bal},
        {"metric": "overall_similarity_score", "role": "frozen 0-100 similarity index",
         "D_mean": d_score["mean"], "D_median": d_score["median"], "D_min": d_score["min"], "D_max": d_score["max"],
         "E_mean": e_score["mean"], "E_median": e_score["median"], "E_min": e_score["min"], "E_max": e_score["max"],
         "auc": "", "balanced_accuracy": ""},
        {"metric": "dtw_pose_distance", "role": "alignment diagnostic",
         "D_mean": d_pose["mean"], "D_median": d_pose["median"], "D_min": d_pose["min"], "D_max": d_pose["max"],
         "E_mean": e_pose["mean"], "E_median": e_pose["median"], "E_min": e_pose["min"], "E_max": e_pose["max"],
         "auc": pose_auc, "balanced_accuracy": ""},
        {"metric": "mean_relative_time_deviation", "role": "timing diagnostic",
         "D_mean": d_time["mean"], "D_median": d_time["median"], "D_min": d_time["min"], "D_max": d_time["max"],
         "E_mean": e_time["mean"], "E_median": e_time["median"], "E_min": e_time["min"], "E_max": e_time["max"],
         "auc": time_auc, "balanced_accuracy": ""},
    ]
    write_csv(summary_csv, summary_rows)

    block_rows = []
    for block in ("A", "B", "C"):
        for group in ("D", "E"):
            sub = [r for r in rows if r["block"] == block and r["group"] == group]
            for metric in ("smoothed_angle_error_deg", "overall_similarity_score", "dtw_pose_distance", "mean_relative_time_deviation"):
                st = describe([r[metric] for r in sub])
                block_rows.append({"block": block, "group": group, "metric": metric,
                                   "count": st["count"], "mean": st["mean"], "median": st["median"],
                                   "min": st["min"], "max": st["max"], "std": st["std"]})
    write_csv(block_csv, block_rows)

    print("\nFINAL-TEST RULE:")
    print("  Do NOT tune center/scale, normalization, DTW, smoothing or joint weights")
    print("  after seeing these ch07-ch10 results. Any weakness is reported as a limitation.")

    print("\nSaved:")
    print(" ", pair_csv)
    print(" ", summary_csv)
    print(" ", block_csv)


if __name__ == "__main__":
    main()
