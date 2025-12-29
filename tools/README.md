# Tools Directory

This directory contains utility tools for the astroplate-analysis repository. Each tool is self-contained with its own dependencies, configuration, and documentation.

## Available Tools

### `data_fetch/`

Robust data fetching tool for downloading astronomical time-lapse images from allsky.local server.

**Purpose:** Download astroplate images with retry logic, progress tracking, and comprehensive error handling.

**Key Features:**
- Configurable via YAML
- Exponential backoff and retry logic
- Progress bars with tqdm
- Date range support
- Dry-run mode
- Checksum verification
- Summary statistics

**Quick Start:**
```bash
cd data_fetch
pip install -r requirements.txt
python fetch.py 20251224
```

See [`data_fetch/README.md`](data_fetch/README.md) for detailed documentation.

### Analysis Pipeline Tools (`analysis_simple/tools/`)

Helper utilities for the analysis_simple pipeline (located in `../analysis_simple/tools/`):

#### `overlay_streaks.py`
Overlay detected streaks on event frames with visual annotations.

**Purpose:** Create annotated frames showing detected transient events.

**Usage:**
```bash
cd ../analysis_simple/tools
python overlay_streaks.py ../../data/night_2025-12-24/
```

**Reads:** `data/events.json`, `frames/*.jpg`  
**Writes:** `annotated/*.jpg`

#### `visualize.py`
Generate time-series plots of per-frame metrics.

**Purpose:** Visualize brightness, contrast, and streak count trends over time.

**Usage:**
```bash
python visualize.py ../../data/night_2025-12-24/
```

**Reads:** `data/metrics.csv`  
**Writes:** `plots/*.png`

#### `validate_data.py`
Validate data structure and image quality before analysis.

**Purpose:** Check for corrupt images, missing files, and folder structure issues.

**Usage:**
```bash
python validate_data.py ../../data/night_2025-12-24/
```

**Checks:** `frames/` directory, image dimensions, file integrity

#### `timelapse.py`
Generate MP4 timelapse videos from frame sequences.

**Purpose:** Create full-night timelapse videos from raw or annotated frames.

**Usage:**
```bash
python timelapse.py ../../data/night_2025-12-24/
```

**Reads:** `frames/*.jpg` or `annotated/*.jpg`  
**Writes:** `timelapse/*.mp4`

**Note:** These tools are typically run automatically via `analyze.py --all-tools` but can be executed individually for custom workflows.

## Tool Structure

Each tool follows a consistent structure pattern:

```
tools/
└── tool_name/
    ├── README.md          # Comprehensive documentation
    ├── requirements.txt   # Python dependencies
    ├── config.yaml        # Configuration settings
    ├── .gitignore         # Tool-specific ignores
    └── tool_script.py     # Main executable
```

## Development Guidelines

When creating a new tool:

1. **Self-Contained**: Each tool should be independently runnable
2. **Documentation**: Include a comprehensive README.md
3. **Configuration**: Use YAML for configuration with sensible defaults
4. **Dependencies**: Pin versions in requirements.txt
5. **Module Support**: Tools should work as both scripts and importable modules
6. **Error Handling**: Include robust error handling and logging
7. **Progress Tracking**: Use progress bars for long-running operations
8. **Testing**: Include validation and dry-run modes where applicable

## Integration with Analysis Pipeline

Complete workflow from data fetch to ML-based window detection:

```bash
# 1. Fetch data
cd tools/data_fetch
python fetch.py 20251224

# 2. Run basic analysis with interest-based windows
cd ../../analysis_simple
python analyze.py ../data/night_2025-12-24/ --all-tools

# 3. Optional: Train ML classifier and detect ML windows
cd ../analysis_ml_activity_classifier
python train.py ../data/night_2025-12-24/

# 4. Optional: Generate ML window artifacts
cd ../analysis_ml_windows
python infer_windows.py ../data/night_2025-12-24/ --artifacts
```

This creates a complete analysis with:
- Interest-based activity windows (rule-based, fast)
- ML-based activity windows (data-driven, confidence scores)
- Per-window artifacts for both methods
- Full-night timelapses and visualizations

## Common Data Directory

All tools read from and write to the common `data/` directory at the repository root:

```
data/
└── night_YYYY-MM-DD/
    ├── frames/              # Input images (from data_fetch)
    ├── masks/               # Generated masks (from analysis_simple)
    ├── data/                # Structured outputs (metrics, events, windows)
    ├── activity/            # Interest-based window artifacts
    ├── activity_ml/         # ML-based window artifacts (optional)
    ├── ml/                  # ML predictions and reports (optional)
    ├── plots/               # Visualizations
    ├── annotated/           # Annotated frames
    └── timelapse/           # Full-night videos
```

## Contributing

New tools should follow the existing patterns. See the main repository README for general contribution guidelines.
