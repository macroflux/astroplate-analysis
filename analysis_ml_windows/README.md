# ML Activity Window Detection

This module detects activity windows from ML per-frame activity probabilities, providing a second layer of analysis that complements the basic interest-score windowing in `analysis_simple`.

## Overview

After running the ML activity classifier (from `analysis_ml_activity_classifier`), you'll have per-frame activity probabilities in `predictions.csv`. This script:

1. **Smooths** the noisy per-frame probabilities using EMA or moving average
2. **Detects** contiguous windows where smoothed probability exceeds a threshold
3. **Merges** nearby windows and ranks by peak probability
4. **Outputs** `ml_windows.json` with window metadata (start/end/peak indices and probabilities)
5. **Optionally generates** per-window artifacts (timelapses, keograms, startrails)

## Why ML Windows?

The basic `analysis_simple` uses rule-based streak detection and interest scoring. This ML approach:

- ✅ Can catch subtle activity that streak detection misses
- ✅ Learns patterns from training data (satellites, meteors, aircraft)
- ✅ Provides probability-based confidence scores
- ✅ Can be refined/validated against manually labeled data
- ✅ Complements interest-score windows with a data-driven alternative

## Prerequisites

1. **Run analysis_simple first** to generate frames, masks, and basic metrics:
   ```bash
   cd analysis_simple
   python analyze.py ../data/night_2025-12-27 --all-tools
   ```

2. **Run ML classifier** to generate per-frame predictions:
   ```bash
   cd analysis_ml_activity_classifier
   python train.py ../data/night_2025-12-27
   ```

This creates `data/night_2025-12-27/ml/predictions.csv` with per-frame activity probabilities.

## Usage

### Basic Window Detection

```bash
python analysis_ml_windows/infer_windows.py data/night_2025-12-27
```

**Output:**
- `data/night_2025-12-27/data/ml_windows.json` - Detected activity windows with metadata

### With Per-Window Artifacts

```bash
python analysis_ml_windows/infer_windows.py data/night_2025-12-27 --artifacts
```

**Additional Output:**
- `data/night_2025-12-27/activity_ml/window_XX_YYYY_ZZZZ/`
  - `timelapse_window.mp4` - Subset video of window frames (raw)
  - `timelapse_annotated_window.mp4` - Annotated version (if available)
  - `keogram.png` - Vertical time-slice visualization
  - `startrails.png` - Max-projected star trails

### Write Smoothed Predictions

```bash
python analysis_ml_windows/infer_windows.py data/night_2025-12-27 --write-smoothed
```

**Additional Output:**
- `data/night_2025-12-27/ml/predictions_smoothed.csv` - Per-frame smoothed probabilities

## Configuration

Edit `analysis_ml_windows/config.yaml` to tune detection parameters:

```yaml
ml_windows:
  # Probability threshold for detection
  prob_threshold: 0.45       # Lower = more sensitive (more windows)
  
  # Window parameters
  min_len: 2                 # Minimum frames to form a window
  pad: 10                    # Padding frames around detected activity
  merge_gap: 8               # Merge windows closer than this many frames
  max_windows: 10            # Maximum windows to report
  
  # Temporal smoothing (reduces frame-to-frame noise)
  smoothing: "ema"           # "ema" or "moving_average"
  ema_alpha: 0.35            # EMA weight (0-1, higher = less smoothing)
  ma_window: 7               # Moving average window size (odd number)
  
  # Artifact generation settings
  fps: 30                    # Timelapse frame rate
  keogram_column_width: 3    # Keogram sampling width
  startrails_gamma: 1.2      # Gamma correction for startrails
```

### Parameter Tuning Guide

**`prob_threshold`**: Balance between sensitivity and false positives
- **Lower (0.3-0.5)**: Catch more activity, more false alarms
- **Higher (0.6-0.8)**: Only high-confidence detections

**`smoothing`**: Reduce frame-to-frame jitter
- **`ema` (recommended)**: Exponential moving average, responsive to changes
- **`moving_average`**: Simple moving average, more uniform smoothing

**`ema_alpha`**: Controls EMA responsiveness
- **Lower (0.1-0.25)**: Heavy smoothing, slow response to changes
- **Higher (0.3-0.5)**: Light smoothing, faster response

**`min_len`**: Minimum frames to form a valid window
- **2-4**: Catch brief events (meteors, quick aircraft passes)
- **6-10**: Only sustained activity (satellites, longer transits)

## Output Format

### ml_windows.json

```json
[
  {
    "start": 157,
    "end": 178,
    "peak_index": 165,
    "peak_value": 0.87,
    "length": 22
  },
  {
    "start": 477,
    "end": 500,
    "peak_index": 488,
    "peak_value": 0.76,
    "length": 24
  }
]
```

**Fields:**
- `start`: First frame index in window (0-based)
- `end`: Last frame index in window (inclusive)
- `peak_index`: Frame with highest activity probability
- `peak_value`: Maximum smoothed probability in window
- `length`: Total frames in window (end - start + 1)

## Workflow Integration

### Complete Analysis Pipeline

```bash
# 1. Basic analysis (streak detection + interest scoring)
cd analysis_simple
python analyze.py ../data/night_2025-12-27 --all-tools

# 2. Train ML classifier (generates predictions.csv)
cd ../analysis_ml_activity_classifier
python train.py ../data/night_2025-12-27

# 3. Detect ML-based activity windows
cd ../analysis_ml_windows
python infer_windows.py ../data/night_2025-12-27 --artifacts

# Compare outputs:
# - data/night_2025-12-27/data/activity_windows.json (interest-score based)
# - data/night_2025-12-27/data/ml_windows.json (ML probability based)
```

### Comparing Detection Methods

You now have two complementary approaches:

1. **Interest-Score Windows** (`activity_windows.json`)
   - Rule-based: streak count + brightness/contrast/focus changes
   - Good for: obvious activity, multiple detection types
   - Fast: no training required

2. **ML Windows** (`ml_windows.json`)
   - Data-driven: learned from labeled/pseudo-labeled examples
   - Good for: subtle patterns, specific event types
   - Requires: training data and classifier

Use **both** for comprehensive coverage! ML may catch subtle activity that rule-based methods miss, while rule-based methods are more interpretable and don't require training.

## Inputs

**Required:**
- `<night>/ml/predictions.csv` - Per-frame ML predictions (from classifier)
- `<night>/frames/*.jpg` - Raw input frames

**Optional (for artifacts):**
- `<night>/masks/combined_mask.png` - Analysis mask for keogram/startrails
- `<night>/annotated/*.jpg` - Annotated frames for annotated timelapse

## Outputs

### Always Generated
- `<night>/data/ml_windows.json` - Window detection results

### With `--write-smoothed`
- `<night>/ml/predictions_smoothed.csv` - Smoothed per-frame probabilities

### With `--artifacts`
- `<night>/activity_ml/window_XX_YYYY_ZZZZ/`
  - `timelapse_window.mp4`
  - `timelapse_annotated_window.mp4` (if annotated frames exist)
  - `keogram.png`
  - `startrails.png`

## Troubleshooting

### No predictions.csv found
```
Error: predictions.csv not found at data/night_2025-12-27/ml/predictions.csv
Run classifier first: python analysis_ml_activity_classifier/train.py <night_dir>
```
**Solution:** Run the ML activity classifier first to generate predictions.

### No probability column found
```
Error: couldn't find probability column in predictions.csv
```
**Solution:** Ensure predictions.csv contains a column like `prob_activity`, `probability`, or `p_activity`.

### No windows detected
- Try lowering `prob_threshold` in config.yaml (e.g., 0.3 instead of 0.45)
- Check smoothed probabilities with `--write-smoothed` to see if any frames are above threshold
- Verify classifier is predicting reasonable probabilities (not all 0 or 1)

### Too many windows / noisy detection
- Increase `prob_threshold` (e.g., 0.6 instead of 0.45)
- Increase `min_len` to filter out brief spikes (e.g., 6 instead of 2)
- Increase smoothing: lower `ema_alpha` or increase `ma_window`

## See Also

- **[analysis_simple](../analysis_simple/README.md)** - Basic streak detection and interest scoring
- **[analysis_ml_activity_classifier](../analysis_ml_activity_classifier/README.md)** - Train the activity classifier
- **[Project README](../README.md)** - Overall project documentation
