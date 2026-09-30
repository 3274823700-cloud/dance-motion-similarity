# Dance Motion Similarity Evaluation

Video-based dance comparison using OpenPose skeletons, dynamic time warping (DTW), and joint-angle analysis.

**Status: research prototype / work in progress.** This repository contains the V1.0–V1.2 implementations and the team's diagnostic and calibration scripts. It does not contain pretrained model weights, datasets, or a verified full set of experimental outputs.

## Project overview

The system compares a reference dance video with another performance. It extracts 2D body keypoints, aligns the two motion sequences, and measures differences in joint angles. Outputs include a motion similarity score, joint-level differences, separate timing diagnostics, and an aligned comparison video.

The aim is to support dance imitation feedback. The score measures similarity under the current algorithm, not an expert judgment of dance quality. Professional performances can differ without either performance being incorrect.

## Current method

1. Extract OpenPose BODY_25 keypoints through `OpenPoseDemo.exe` and load the JSON output.
2. Sample every second frame and screen keypoints using a confidence threshold of 0.30.
3. Center each pose on the hips and use one median torso length per sequence for scale normalization.
4. Align sequences with DTW using confidence-weighted normalized-coordinate distances.
5. Apply a three-frame median filter to 11 joint-angle metrics and compute weighted errors along the alignment path.
6. Map the overall angle error to a 0–100 similarity score. Report timing diagnostics separately.

Each extracted frame has shape `25 × 3`: x coordinate, y coordinate, and detection confidence. The 11 scoring metrics cover the left/right elbows, shoulders, hips, knees, ankles, and torso tilt.

### V1.2 configuration

| Setting | Value |
|---|---|
| Network input resolution | `-1x368` |
| Sampling step | `2` |
| Confidence threshold | `0.30` |
| Normalization | Per-frame hip origin, sequence-median torso scale |
| DTW window ratio | `0.30`, centered on relative sequence progress |
| Angle smoothing | Three sampled frames |
| Score mapping | `100 / (1 + exp((error - 33.046977) / 6.292398))` |

The mapping calibration script uses engineering anchors: the median same-choreography comparison maps to 80, and the median different-choreography comparison maps to 50. These anchors are **not human-rated ground truth**.

## Getting started

The integrated V1.2 program uses a separate Windows OpenPose executable. Installing Python packages alone does not install OpenPose or its BODY_25 model.

1. Set up [OpenPose](https://github.com/CMU-Perceptual-Computing-Lab/openpose) and confirm that BODY_25 inference works on your machine. Follow its own installation and licensing instructions.
2. Create a Python environment and install the Python dependencies:

   ```powershell
   python -m venv .venv
   .\.venv\Scripts\python.exe -m pip install -r requirements.txt
   ```

3. In `Dance_Motion_Similarity_Evaluation_V1.2.py`, change `OPENPOSE_ROOT` to your OpenPose directory. The supplied value, `D:\OpenPose\openpose`, is a local configuration example, not a bundled installation.
4. Open a terminal in this repository's root folder and run:

   ```powershell
   .\.venv\Scripts\python.exe Dance_Motion_Similarity_Evaluation_V1.2.py "C:\path\reference.mp4" "C:\path\learner.mp4"
   ```

Use comparable camera views and full-body footage. Without arguments, the script expects the `ch10` AIST video pair specified at the top of the file under `data/aistpp/videos/`.

Dependencies are listed without version pins because an exact team environment lockfile was not supplied. Full OpenPose execution has not been rerun as part of this repository packaging. The historical V1.0 requires its own compatible `pyopenpose` build; its original setup targets Python 3.7.9. It is not the recommended entry point.

### Outputs

The main program writes these files under `similarity_result_v12/`:

- `summary.csv`, `joint_report.csv`, and `alignment_detail.csv`
- `joint_error_profile.png`, `aligned_angle_error_curve.png`, and `timing_alignment_curve.png`
- `side_by_side_compare.mp4`

Raw JSON is cached in `pose_cache_json/`. Repeated runs reuse complete caches when `FORCE_REEXTRACT = False`. Cache names do not fingerprint the input video or model: replacing a video under the same filename can reuse stale results. Forced extraction recreates the selected cache subdirectory. Keep original data outside cache folders. Repeated evaluations also overwrite the fixed output filenames, so archive important results before rerunning.

## Repository guide

| File or folder | Purpose |
|---|---|
| `Dance_Motion_Similarity_Evaluation_V1.2.py` | Current integrated entry point |
| `Dance_Motion_Similarity_Evaluation_V1.1_final.py` | Historical executable/JSON pipeline and helpers used by experiments |
| `Dance_Motion_Similarity_Evaluation_V1.1.py`, `keypoint_utils.py` | Early JSON loading and DTW sanity-check code |
| `Dance_Motion_Similarity_Evaluation_V1.0.py` | Original pyopenpose baseline |
| `phase*.py` | Diagnostic, calibration, validation, and regression scripts |
| [docs/PROGRESS.md](docs/PROGRESS.md) | Current progress and evidence boundaries |
| [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md) | Experiment map and prerequisites |
| [docs/DATA.md](docs/DATA.md) | Dataset scope and representation differences |
| [materials/README.md](materials/README.md) | Location for future poster, slides, and demo links |

The original 17 Python files are preserved byte-for-byte. They remain together at the root because experiment scripts dynamically load neighboring filenames and use paths relative to the working directory.

## Limitations and next steps

- This is an offline prototype, not a demonstrated real-time system.
- Current calibration/validation scripts focus on selected Breaking (`gBR`) examples, not a completed benchmark over all ten dance genres.
- Person detection means at least one keypoint exceeds the confidence threshold. It does not imply a complete or correct skeleton.
- Angle coverage is measured after smoothing, which can fill isolated missing angles. Raw detection coverage should be reported separately.
- The current feedback groups left/right measurements by joint type. It does not yet generate validated corrective instructions.
- DTW relative-time deviation is not a music-beat accuracy metric. No audio beat analysis is implemented.
- The system still depends on camera view, detection quality, and correct anatomical keypoint assignments.
- Future work includes limb orientation, angular motion features, learner recordings with human feedback, and learned skeleton representations such as ST-GCN. These are plans, not implemented results.

## Data and acknowledgments

The project uses AIST/AIST++ dance material as a research resource. Dataset files and OpenPose binaries/models are deliberately excluded from this repository.

- [AIST++ project and data interface](https://github.com/google/aistplusplus)
- [AIST Dance Video Database](https://aistdancedb.ongaaccel.jp/)
- [OpenPose](https://github.com/CMU-Perceptual-Computing-Lab/openpose)

This is a team project. Confirm contributor names, publication permission, and a code license with the team before a public release. No software license has been selected in this starter package. Third-party resources retain their own terms.
