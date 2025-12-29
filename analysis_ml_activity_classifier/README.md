# ML Activity Classifier

This module trains a lightweight logistic regression classifier to predict "activity frames" 
using features from the analysis pipeline, and applies peak-seeded windowing to detect 
contiguous activity windows from the per-frame ML probabilities.

## Two-Stage Process

1. **Train Classifier**: Learn frame-level activity patterns from pseudo-labeled data
2. **Detect Windows**: Apply peak-seeded windowing to find contiguous activity regions

## Labels

We use **pseudo-labels** from `data/activity_windows.json`:
- y=1 if frame index is inside any detected activity window
- y=0 otherwise

This is a *first pass* to validate end-to-end ML flow without hand labeling.
As you manually label more data, replace activity_windows.json with ground truth.

## Features

Loaded from `data/metrics.csv`, then we build:
- Robust z-scores of: brightness, contrast, focus, streak_count, interest_score
- Robust z-scores of absolute deltas: brightness/contrast/focus

These 8 features capture both absolute signal and temporal changes.

## Peak-Seeded Windowing

After training, the classifier produces per-frame activity probabilities. We apply 
**peak-seeded windowing** to detect activity windows:

1. **Smooth** probabilities using exponential moving average (EMA) to reduce noise
2. **Find peaks** that exceed minimum thresholds for height and prominence
3. **Expand** each peak left/right while probability stays above tail threshold
4. **Merge** overlapping or nearby windows
5. **Rank** windows by total area under the probability curve
6. **Output** top N windows with start/end/peak indices

This approach is more robust than simple thresholding because it:
- ✅ Handles short events (meteors) that might only span 3-5 frames
- ✅ Survives pulsing/intermittent activity (satellites moving through clouds)
- ✅ Adapts window boundaries to probability curve shape
- ✅ Ranks by evidence strength (area) not just peak height

### Configuration Parameters

**Smoothing:**
- `--ema-alpha` (default: 0.20): EMA smoothing weight (0.15-0.30)
  - Lower = heavier smoothing, slower response
  - Higher = lighter smoothing, faster response

**Peak Detection:**
- `--min-peak` (default: 0.35): Minimum probability to be considered a peak
  - Lower = more sensitive, catches weak events
  - Higher = only strong confident detections
  
- `--min-prominence` (default: 0.04): Peak must rise above local baseline by this amount
  - Prevents noise spikes from being detected as peaks

**Window Expansion:**
- `--tail-threshold` (default: 0.22): Expand window while prob > this threshold
  - Lower = wider windows, captures full event context
  - Higher = tighter windows, only high-confidence regions

**Post-Processing:**
- `--min-len` (default: 5): Minimum frames for valid window
  - Allows short meteor-like events to survive
  
- `--pad` (default: 8): Add this many frames before/after each window
  - Provides context frames for visual inspection
  
- `--merge-gap` (default: 10): Merge windows closer than this many frames
  - Prevents fragmenting a single event into multiple windows
  
- `--max-windows` (default: 10): Maximum windows to output
  - Top N by area score

## Run

1) **Run analysis first** to generate metrics and initial windows:
```bash
cd analysis_simple
python analyze.py ../data/night_2025-12-27 --all-tools
```

2) **Train classifier and detect ML windows**:
```bash
cd ../analysis_ml_activity_classifier
python train.py ../data/night_2025-12-27
```

### Basic Usage

Train with default peak-seeded windowing parameters:
```bash
python train.py ../data/night_2025-12-27
```

### Tuning for Short Events (Meteors)

Lower thresholds to catch brief transients:
```bash
python train.py ../data/night_2025-12-27 \
  --min-peak 0.30 \
  --tail-threshold 0.18 \
  --min-len 3
```

### Tuning for High Confidence Only

Higher thresholds for fewer false positives:
```bash
python train.py ../data/night_2025-12-27 \
  --min-peak 0.50 \
  --tail-threshold 0.30 \
  --min-len 8
```

### Custom Smoothing

Adjust EMA for your noise level:
```bash
python train.py ../data/night_2025-12-27 \
  --ema-alpha 0.15  # Heavier smoothing
```

## Outputs

### ml/ Directory
- `predictions.csv` - Per-frame predictions with columns:
  - `index`: Frame number (0-based)
  - `file`: Frame filename
  - `y_window`: Pseudo-label (1=in activity window, 0=not)
  - `p_activity_raw`: Raw classifier probability
  - `p_activity_smooth`: EMA-smoothed probability
- `report.json` - Training metrics, feature weights
- `topk_frames.json` - Top K frames by predicted probability

### data/ Directory
- `ml_windows.json` - Detected ML activity windows:
  ```json
  [
    {
      "start": 157,
      "end": 178,
      "peak_index": 165,
      "peak_value": 0.87,
      "area": 15.3,
      "mean": 0.76,
      "length": 22
    }
  ]
  ```

**Fields:**
- `start/end`: Frame indices (inclusive)
- `peak_index`: Frame with highest smoothed probability
- `peak_value`: Maximum probability in window
- `area`: Sum of probabilities (evidence strength)
- `mean`: Average probability in window
- `length`: Number of frames (end - start + 1)

## Comparing Detection Methods

You now have two complementary approaches:

1. **Interest-Score Windows** (`data/activity_windows.json`)
   - From `analysis_simple`
   - Rule-based: streak count + brightness/contrast/focus changes
   - Fast, interpretable, no training required

2. **ML Windows** (`data/ml_windows.json`)
   - From this module
   - Data-driven: learned patterns from labeled examples
   - Can catch subtle activity rule-based methods miss

**Best practice**: Use both methods and compare results. ML windows validate 
interest-score windows and may find additional events.

## Next Steps

After generating `ml_windows.json`, use it with `analysis_ml_windows/infer_windows.py` 
to generate per-window artifacts (timelapses, keograms, startrails) for detailed inspection.

## Troubleshooting

### Classifier predicts all 0 or all 1
- Check that `data/activity_windows.json` has reasonable windows
- Verify `data/metrics.csv` has varied feature values
- Try adjusting learning rate (`--lr`) or regularization (`--l2`)

### No ML windows detected
- Lower `--min-peak` threshold (try 0.25-0.30)
- Check `predictions.csv` - are smoothed probabilities ever above threshold?
- Reduce `--min-len` to allow shorter windows

### Too many windows / noisy detection
- Increase `--min-peak` (try 0.45-0.55)
- Increase `--min-prominence` to filter noise
- Increase `--min-len` to require longer sustained activity
- Lower `--ema-alpha` for heavier smoothing

### Windows too narrow / missing event tails
- Lower `--tail-threshold` (try 0.15-0.20)
- Increase `--pad` for more context frames

## See Also

- **[analysis_simple](../analysis_simple/README.md)** - Generate metrics and interest-score windows
- **[analysis_ml_windows](../analysis_ml_windows/README.md)** - Apply ML windows to generate artifacts
- **[Project README](../README.md)** - Overall documentation
