# Data Directory

This directory contains astronomical time-lapse data used by analysis tools in this repository. Each subdirectory represents data from a single observation night.

## Structure

Each night directory follows this structure:

```
data/
├── night_2025-12-24/
│   ├── frames/              # INPUT: Place your .jpg frames here
│   │   ├── frame_001.jpg
│   │   ├── frame_002.jpg
│   │   └── ...
│   ├── masks/               # OUTPUT: Analysis masks
│   │   ├── sky_mask.png
│   │   ├── persistent_edges.png
│   │   └── combined_mask.png
│   ├── data/                # OUTPUT: Structured data files
│   │   ├── metrics.csv      # Per-frame statistics with focus_score and interest_score
│   │   ├── events.json      # Detected transient events
│   │   ├── activity_windows.json  # Interest-based activity periods
│   │   └── ml_windows.json  # ML-based activity periods (optional)
│   ├── activity/            # OUTPUT: Interest-based per-window artifacts
│   │   ├── window_00_0045_0089/
│   │   │   ├── timelapse_window.mp4
│   │   │   ├── keogram.png
│   │   │   └── startrails.png
│   │   └── window_01_0234_0267/
│   │       ├── timelapse_window.mp4
│   │       ├── keogram.png
│   │       └── startrails.png
│   ├── activity_ml/         # OUTPUT: ML-based per-window artifacts (optional)
│   │   └── window_00_0157_0178/
│   │       ├── timelapse_window.mp4
│   │       ├── timelapse_annotated_window.mp4
│   │       ├── keogram.png
│   │       └── startrails.png
│   ├── ml/                  # OUTPUT: ML classifier predictions (optional)
│   │   ├── predictions.csv  # Per-frame probabilities (raw + smoothed)
│   │   ├── predictions_smoothed.csv
│   │   ├── report.json      # Training metrics
│   │   └── topk_frames.json # High-confidence frames
│   ├── plots/               # OUTPUT: Visualization plots
│   │   ├── brightness_over_time.png
│   │   ├── contrast_over_time.png
│   │   └── streak_counts.png
│   ├── annotated/           # OUTPUT: Frames with detected streaks overlaid
│   │   ├── frame_002.jpg
│   │   └── ...
│   └── timelapse/           # OUTPUT: Full-night timelapse videos
│       ├── timelapse.mp4
│       └── timelapse_annotated.mp4
└── night_YYYY-MM-DD/        # Additional observation nights
    └── ...
```

### New in v2.0

**Organized Folder Structure:**
- **masks/** - Organized directory for analysis masks (sky_mask.png, persistent_edges.png, combined_mask.png)
- **data/** - Structured data outputs optimized for ML workflows
  - `metrics.csv` - Now includes `focus_score`, `interest_score`, and `z_streak` columns
  - `activity_windows.json` - Interest-based automatically detected high-interest time periods
  - `ml_windows.json` - ML-based activity windows (optional, from ML classifier)
- **activity/** - Interest-based per-window artifacts for each detected activity period
  - Timelapse videos of just the activity window
  - Keograms showing motion over time
  - Startrail composites
- **activity_ml/** - ML-based per-window artifacts (optional, from ML window detector)
  - Same structure as activity/ but for ML-detected windows
  - Includes both raw and annotated timelapses when available
- **ml/** - ML classifier outputs (optional)
  - Per-frame activity probabilities (raw and smoothed)
  - Training reports and high-confidence frame lists
- **timelapse/** - Organized location for full-night timelapse videos

**Dual Window Detection:**
The pipeline now supports two complementary methods:
1. **Interest-based** (analysis_simple): Rule-based, fast, no training required
2. **ML-based** (analysis_ml_activity_classifier + analysis_ml_windows): Data-driven, learns patterns, provides confidence scores

## Usage

### Option 1: Download Images Automatically

Use the data_fetch tool to automatically download and organize images:

```bash
cd tools/data_fetch
python fetch.py YYYYMMDD
```

This creates `data/night_YYYY-MM-DD/frames/` and downloads all images.

### Option 2: Manual Setup

1. Create a directory named `night_YYYY-MM-DD` where YYYY-MM-DD is the observation date
2. Create a `frames/` subdirectory within it
3. Place your time-lapse `.jpg` images in the `frames/` directory

### Running Analysis

Once you have images in place, run the analysis:

**Basic analysis (interest-based windows):**
```bash
cd analysis_simple/
python analyze.py ../data/night_YYYY-MM-DD/ --all-tools
```

**Optional: ML-based windows:**
```bash
# Train classifier and detect ML windows
cd analysis_ml_activity_classifier/
python train.py ../data/night_YYYY-MM-DD/

# Generate ML window artifacts
cd ../analysis_ml_windows/
python infer_windows.py ../data/night_YYYY-MM-DD/ --artifacts
```

### Image Requirements

- **Format**: JPEG with `.jpg` extension
- **Naming**: Any naming scheme works (files are sorted alphabetically)
- **Content**: Night sky images from a fixed camera position
- **Sequence**: Should be a time-series (sequential frames)

## Example Data

See `night_2025-12-24/` for an example night directory with sample outputs.

## Common Location

This `data/` directory serves as a common location where all analysis tools can:
- Read input frames (downloaded by `tools/data_fetch/` or placed manually)
- Generate output files (masks, metrics, events)
- Store visualization plots
- Share data between different analysis tools

Each analysis tool in the repository can reference data directories here using relative paths like `../data/night_YYYY-MM-DD/`.

## Git Ignore

The `night_*/` directories are git-ignored to avoid committing large image datasets. Only the directory structure and documentation are tracked in version control.
