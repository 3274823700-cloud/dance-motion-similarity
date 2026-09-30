# Data preparation and scope

The broader project has used a local subset of 1,200 basic solo dance skeleton sequences covering ten genres, 30 dancers, and 60 music IDs (six per genre), restricted to camera c01. These are project inventory counts, not a claim that all sequences were evaluated by the supplied Phase 3 scripts. Music IDs in metadata do not imply that separate audio files are included.

## Two skeleton representations

| Source | Representation | Current use |
|---|---|---|
| Previously downloaded AIST++ 2D annotations | `frames × 17 × 3`, COCO-style keypoints | Broader project dataset, not directly loaded by V1.2 |
| V1.2 OpenPose extraction from video | `frames × 25 × 3`, BODY_25 | Input to the current evaluation pipeline |

Both use x, y, confidence. The third value is not a depth coordinate. Directly substituting 17-point arrays for BODY_25 is invalid. A mapping and revised angle definitions would be required, particularly for neck/hip-center proxies and the missing heel points.

## Obtain data separately

- [AIST++](https://github.com/google/aistplusplus)
- [AIST Dance Video Database](https://aistdancedb.ongaaccel.jp/)
- [AIST dataset terms](https://aistdancedb.ongaaccel.jp/terms_of_use/)

Follow the providers' instructions and applicable terms. This package does not redistribute videos, pose annotations, audio, model weights, or dataset metadata tables.

For the default current example, prepare:

```text
data/aistpp/videos/gBR_sBM_c01_d04_mBR0_ch10.mp4
data/aistpp/videos/gBR_sBM_c01_d05_mBR0_ch10.mp4
```

The final regression script additionally needs the corresponding `ch01` pair. Other scripts list their own video selections.

Use professional-to-professional comparisons as reference comparisons, not teacher/student ground-truth labels. Standardize camera view and mirror convention for recordings. Missing detections are missing evidence, not correct poses.
