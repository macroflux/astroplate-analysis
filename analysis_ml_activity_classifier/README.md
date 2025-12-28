# ML Activity Classifier (Stage-2 Test Harness)

This folder trains a lightweight logistic regression to predict "activity frames"
using the pipeline outputs.

## Labels
We use **pseudo-labels** from `data/activity_windows.json`:
- y=1 if frame index is inside any detected activity window
- y=0 otherwise

This is a *first pass* to validate end-to-end ML flow without hand labeling.

## Features
Loaded from `data/metrics.csv`, then we build:
- robust z-scores of: brightness, contrast, focus, streak_count, interest_score
- robust z-scores of absolute deltas: brightness/contrast/focus

## Run

1) Run analysis first:
```bash
python analyze.py /path/to/night_dir --overlay --visualize
