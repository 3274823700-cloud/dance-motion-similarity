# Progress snapshot: 30 September 2026

## What is implemented

- A complete two-video comparison pipeline with OpenPose extraction, normalization, DTW, angle scoring, CSV reports, plots, and a comparison video.
- V1.1's executable/JSON interface removes the need for Python bindings in the integrated application.
- V1.2 uses sequence-median scale normalization, three-frame median angle smoothing, fixed logistic score mapping, body-part error reporting, and separate timing diagnostics.
- The latest desktop V1.2 adopts `-1x368`, sets `FORCE_REEXTRACT = False`, and uses a `ch10` default example. Relative to the separately supplied older Downloads V1.2, these are configuration changes, not a new scoring algorithm.
- Twelve diagnostic/calibration/validation scripts accompany the four versioned application files and one keypoint helper.

## What the supplied files establish

This release is based on 17 Python source files. The folder did not include the experiment CSV outputs, complete run logs, video inputs, model hashes, or skeleton caches. Script descriptions record the team's experimental history, but a script's existence does not establish that its test passed.

The regression script contains these expected values:

| Case | Expected weighted angle error | Expected similarity |
|---|---:|---:|
| `ch01`, d04 vs d05, mBR0 | 17.058927 degrees | 92.695651 |
| `ch10`, d04 vs d05, mBR0 | 19.153210 degrees | 90.096777 |

Its comments report 100% person detection for those reference runs, and its check requires at least 99.9%. These are **source-recorded regression targets**, not independently reproduced benchmark results. They should be accompanied by the original logs and input/model identifiers before being presented as verified results.

## Evaluation design

The experiments distinguish:

- D: different dancer, same music and choreography.
- E: same dancer/music, different choreography.

This evaluates discrimination between selected choreography pairings. It does not establish sensitivity to novice mistakes, and D is not a perfect-performance label.

The development history uses `ch01–ch03` for initial calibration, `ch04–ch06` for development validation, and later combines `ch01–ch06` for score mapping. `ch07–ch10` were originally reserved for final validation. The subsequent resolution study also examines `ch07–ch10`. If those results influenced resolution selection, they are no longer an untouched final test for the resulting configuration. A fresh held-out set is needed for a strong final generalization claim.

The selected blocks use Breaking, three dancers (`d04`, `d05`, `d06`), and music IDs `mBR0`, `mBR2`, and `mBR4`. This is narrower than the project's broader ten-genre data collection.

## What to add next

1. Original result CSVs and logs, with exact configuration, video hashes, model/build information, and cache provenance.
2. Raw joint visibility and longest missing intervals, in addition to person detection and smoothed-angle coverage.
3. Fresh held-out examples across genres, including real learner performances and human annotations.
4. An approved team attribution list, a code-license decision, and reviewed presentation materials.

No neural-network similarity model, music-beat analysis, or real-time performance benchmark is included yet.
