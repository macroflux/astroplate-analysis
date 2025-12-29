# Architecture Documentation

This document describes the preferred code structure and architectural patterns used in the astroplate-analysis pipeline.

## Window Source System

### Overview

The pipeline provides a unified interface for activity window detection through the `--windows-source` flag. This abstraction allows different detection strategies to be used interchangeably while maintaining consistent output structure.

### Core Concept

**Window**: A contiguous sequence of frames representing a period of potential astronomical activity (satellites, meteors, etc.). Windows are defined by:
- `start`: Starting frame index (inclusive)
- `end`: Ending frame index (inclusive)
- `peak_index`: Frame with highest activity score
- `peak_value`: Activity score at peak
- `length`: Number of frames in window
- `source`: Detection method used ('interest', 'ml', or 'hybrid')

### Window Sources

#### 1. Interest-Based Windows (Default)

**File**: `analysis_simple/analyze.py` - `detect_activity_windows()`

**Strategy**: Rule-based scoring using weighted combination of per-frame metrics:
```
interest_score = 0.60×(streak_count) + 0.15×(brightness_change) + 
                 0.15×(contrast_change) + 0.10×(focus_change)
```

**Advantages**:
- Fast, no training required
- Transparent, tunable weights
- Works with first analysis run
- Good for general-purpose detection

**Configuration**: `config.yaml` → `windows` section

**Output**: `data/activity_windows.json`

#### 2. ML-Based Windows

**Files**: 
- `analysis_ml_activity_classifier/train.py` - Trains classifier
- `analysis_ml_windows/infer_windows.py` - Generates windows from predictions

**Strategy**: Peak-seeded windowing on classifier probability output:
1. Train logistic regression on pseudo-labeled frames
2. Smooth probabilities (EMA or moving average)
3. Detect local peaks above threshold
4. Expand peaks into windows using adaptive boundaries
5. Merge overlapping windows

**Advantages**:
- Data-driven, learns from observations
- Provides confidence scores
- Can detect subtle patterns
- Improves with more training data

**Requirements**: Requires initial training on interest-based windows

**Output**: `data/ml_windows.json`

#### 3. Hybrid Windows

**File**: `analysis_simple/analyze.py` - `select_windows()`

**Strategy**: Combines ML and interest-based detection:
1. Start with ML windows (higher confidence typically)
2. For each interest window, calculate IoU with all ML windows
3. If `IoU < threshold` (default 0.3), add as novel detection
4. Merge overlapping windows using `merge_windows_simple()`
5. Sort chronologically

**Advantages**:
- Best of both worlds
- ML's learned patterns + interest's novelty detection
- Maximum coverage of potential events
- Resilient to ML false negatives

**Configuration**: `hybrid_iou_threshold` in `config.yaml` → `windows` section

**Output**: `data/windows_hybrid.json`

### Intersection over Union (IoU)

IoU measures window overlap for hybrid deduplication:

```python
intersection = min(end1, end2) - max(start1, start2) + 1  # Inclusive indices
union = length1 + length2 - intersection
iou = intersection / union
```

**Threshold Tuning**:
- `IoU < 0.3`: Windows considered distinct (default)
- Lower threshold: More strict, more interest windows added
- Higher threshold: More lenient, fewer interest windows added

### Unified Interface

All three sources use the same interface:

```bash
# Interest-based (default)
python analyze.py data/night_2025-12-27/ --windows-source interest

# ML-based
python analyze.py data/night_2025-12-27/ --windows-source ml

# Hybrid
python analyze.py data/night_2025-12-27/ --windows-source hybrid
```

**Key Benefit**: All sources output artifacts to the same location (`activity/`), enabling:
- Consistent downstream processing
- A/B testing between detection methods
- Future dashboard integration with single interface

### File Structure

```
night_YYYY-MM-DD/
├── data/
│   ├── activity_windows.json    # Interest-based (always generated)
│   ├── ml_windows.json          # ML-based (if ML classifier run)
│   └── windows_hybrid.json      # Hybrid (if hybrid mode used)
├── activity/                    # Unified artifact location
│   └── window_XX_YYYY_ZZZZ/     # Generated from selected source
│       ├── timelapse_window.mp4
│       ├── keogram.png
│       └── startrails.png
└── ml/                          # ML classifier outputs
    ├── predictions.csv
    └── report.json
```

## Design Principles

### 1. Single Responsibility

Each function has one clear purpose:
- `load_interest_windows()`: Load windows from JSON
- `load_ml_windows()`: Load windows from JSON
- `calculate_iou()`: Compute overlap metric
- `select_windows()`: Choose windows based on strategy

### 2. Type Safety

Type hints throughout for clarity:
```python
def select_windows(
    source: str,
    interest_windows: List[dict],
    ml_windows: List[dict],
    iou_threshold: float = 0.3,
    merge_gap: int = 8
) -> Tuple[List[dict], Dict[str, int]]:
```

### 3. Fail-Fast Validation

Check prerequisites early with helpful error messages:
```python
if windows_source == 'ml' and not ml_windows_path.exists():
    print(f"Error: ML windows file not found: {ml_windows_path}")
    print(f"Please run ML classifier first:")
    print(f"  cd analysis_ml_activity_classifier && python train.py {night_dir}")
    sys.exit(1)
```

### 4. Configuration Over Convention

All tunable parameters in `config.yaml`:
- Window detection thresholds
- Merge gaps
- IoU thresholds
- Artifact generation settings

### 5. Progressive Enhancement

System works incrementally:
1. Basic interest detection (no setup)
2. Add ML detection (requires training)
3. Use hybrid mode (combines both)

Each level adds capability without breaking previous functionality.

## Future Extensions

### Window Manifest

Future enhancement: `data/windows_manifest.json` to track:
- Which sources exist
- Which source was used for artifacts
- Timestamp of generation
- Configuration used

### Dashboard Integration

The `--windows-source` flag provides a clean API for future dashboards:
```python
# Dashboard can programmatically select source
result = run_analysis(night_dir, windows_source='hybrid')
windows = result['windows']
artifacts = result['artifact_paths']
```

### Custom Window Sources

Architecture allows adding new sources:
1. Implement window detection function
2. Save to `data/{source}_windows.json`
3. Add source option to `select_windows()`
4. Document in `--windows-source` choices

Example: `--windows-source manual` for hand-labeled windows.

## Coding Standards

### Import Organization

```python
# Standard library
import json
import sys
from pathlib import Path

# Third-party
import cv2
import numpy as np
import yaml

# Type hints
from typing import Optional, List, Dict, Tuple
```

### Path Handling

Always use `pathlib.Path` for cross-platform compatibility:
```python
data_dir = night / "data"  # Good
data_dir = night + "/data"  # Avoid
```

### Error Handling

Provide context in error messages:
```python
if not path.exists():
    print(f"Error: File not found: {path}", file=sys.stderr)
    print(f"Expected at: {path.absolute()}", file=sys.stderr)
    sys.exit(1)
```

### Function Documentation

Use clear docstrings with Args/Returns:
```python
def calculate_iou(window1: dict, window2: dict) -> float:
    """
    Calculate Intersection over Union (IoU) for two windows.
    
    Windows use inclusive frame indices.
    
    Args:
        window1: First window dict with 'start' and 'end' keys
        window2: Second window dict with 'start' and 'end' keys
    
    Returns:
        IoU value between 0.0 and 1.0
    """
```

## Commit Style

Follow Conventional Commits:
```
feat(ml): add windows-source flag with hybrid support
fix(ml): handle negative overlap in hybrid merge
docs: update README with unified window concept
test: verify IoU calculation for edge cases
```

Types: `feat`, `fix`, `docs`, `test`, `refactor`, `perf`, `chore`
Scope: Module or area affected (e.g., `ml`, `docs`, `analysis`)

---

**Last Updated**: 2025-12-29
**Version**: 1.0
**Status**: Living document (will be updated as architecture evolves)
