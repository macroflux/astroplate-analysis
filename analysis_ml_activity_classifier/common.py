from __future__ import annotations

import json
from pathlib import Path
import numpy as np
from typing import List, Dict, Optional, Tuple


FEATURE_COLS = [
    "mean_brightness",
    "star_contrast",
    "focus_score",
    "streak_count",
    "interest_score",
]


def load_metrics_csv(csv_path: Path) -> list[dict]:
    import csv
    rows = []
    with csv_path.open("r", newline="") as fp:
        r = csv.DictReader(fp)
        for row in r:
            rows.append(row)
    return rows


def to_float(x: str) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


def robust_zscore(x: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    x = x.astype(np.float64)
    med = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - med))
    scale = 1.4826 * mad + eps
    return (x - med) / scale


def diff1(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float64)
    if x.size == 0:
        return x
    return np.diff(x, prepend=x[0])


def load_windows_json(windows_path: Path) -> list[dict]:
    if not windows_path.exists():
        return []
    return json.loads(windows_path.read_text())


def windows_to_labels(n: int, windows: list[dict]) -> np.ndarray:
    y = np.zeros(n, dtype=np.int64)
    for w in windows:
        s = int(w["start"])
        e = int(w["end"])
        s = max(0, s)
        e = min(n - 1, e)
        y[s : e + 1] = 1
    return y


def build_feature_matrix(rows: list[dict]) -> tuple[np.ndarray, list[str]]:
    """
    Build X with:
      - robust z of base features
      - robust z of abs diffs of a few continuous signals
    """
    n = len(rows)
    if n == 0:
        return np.zeros((0, 0), dtype=np.float64), []

    # base feature vectors
    base = {c: np.array([to_float(r.get(c, "nan")) for r in rows], dtype=np.float64) for c in FEATURE_COLS}

    # fill missing interest_score if older CSV
    if np.all(np.isnan(base["interest_score"])):
        base["interest_score"] = np.zeros(n, dtype=np.float64)

    # change features (deltas)
    d_mean = np.abs(diff1(base["mean_brightness"]))
    d_contr = np.abs(diff1(base["star_contrast"]))
    d_focus = np.abs(diff1(base["focus_score"]))

    feats = []
    names = []

    # robust z of base
    for c in FEATURE_COLS:
        feats.append(robust_zscore(np.nan_to_num(base[c], nan=np.nanmedian(base[c]))))
        names.append(f"z_{c}")

    # robust z of deltas
    feats.append(robust_zscore(np.nan_to_num(d_mean, nan=np.nanmedian(d_mean))))
    names.append("z_abs_d_mean_brightness")

    feats.append(robust_zscore(np.nan_to_num(d_contr, nan=np.nanmedian(d_contr))))
    names.append("z_abs_d_star_contrast")

    feats.append(robust_zscore(np.nan_to_num(d_focus, nan=np.nanmedian(d_focus))))
    names.append("z_abs_d_focus_score")

    X = np.stack(feats, axis=1).astype(np.float64)
    return X, names


def train_val_split(n: int, val_frac: float = 0.2, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    split = int(np.floor((1.0 - val_frac) * n))
    return idx[:split], idx[split:]


def sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.clip(z, -50, 50)
    return 1.0 / (1.0 + np.exp(-z))

def smooth_ema(x: np.ndarray, alpha: float = 0.25) -> np.ndarray:
    """
    Exponential moving average smoothing.
    alpha in (0,1], higher = less smoothing.
    """
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return x
    y = np.empty_like(x)
    y[0] = x[0]
    for i in range(1, len(x)):
        y[i] = alpha * x[i] + (1.0 - alpha) * y[i - 1]
    return y

def find_local_peaks(
    x: np.ndarray,
    min_peak: float = 0.4,
    min_prominence: float = 0.05,
    min_distance: int = 5
) -> np.ndarray:
    """
    Find local maxima indices in 1D array x using simple neighborhood logic.
    - min_peak: absolute peak floor
    - min_prominence: peak must rise above local baseline by at least this amount
    - min_distance: suppress peaks too close together (keep higher one)
    """
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    if n < 3:
        return np.array([], dtype=int)

    # local maxima candidates (strict)
    cand = np.where((x[1:-1] > x[:-2]) & (x[1:-1] >= x[2:]))[0] + 1
    if cand.size == 0:
        return np.array([], dtype=int)

    # apply absolute floor
    cand = cand[x[cand] >= min_peak]
    if cand.size == 0:
        return np.array([], dtype=int)

    # crude prominence: compare to min of a small neighborhood around peak
    # (keeps dependency-free, avoids scipy)
    def local_baseline(i: int, radius: int = 20) -> float:
        a = max(0, i - radius)
        b = min(n, i + radius + 1)
        return float(np.min(x[a:b]))

    keep = []
    for i in cand:
        base = local_baseline(int(i))
        if (x[i] - base) >= min_prominence:
            keep.append(int(i))

    if not keep:
        return np.array([], dtype=int)

    keep = np.array(keep, dtype=int)

    # enforce min_distance: greedy keep-highest, suppress neighbors
    keep = keep[np.argsort(x[keep])[::-1]]  # sort by height desc
    selected = []
    blocked = np.zeros(n, dtype=bool)

    for i in keep:
        if blocked[i]:
            continue
        selected.append(i)
        lo = max(0, i - min_distance)
        hi = min(n, i + min_distance + 1)
        blocked[lo:hi] = True

    selected = np.array(sorted(selected), dtype=int)
    return selected

def expand_peak_to_window(
    x: np.ndarray,
    peak_idx: int,
    tail_threshold: float = 0.25,
    max_expand: Optional[int] = None
) -> Tuple[int, int]:
    """
    Expand left/right from peak while x >= tail_threshold.
    max_expand optionally limits expansion distance.
    Returns (start, end) inclusive indices.
    """
    n = len(x)
    s = e = int(peak_idx)

    limit_left = 0
    limit_right = n - 1
    if max_expand is not None:
        limit_left = max(0, peak_idx - int(max_expand))
        limit_right = min(n - 1, peak_idx + int(max_expand))

    # expand left
    i = peak_idx
    while i - 1 >= limit_left and x[i - 1] >= tail_threshold:
        i -= 1
    s = i

    # expand right
    i = peak_idx
    while i + 1 <= limit_right and x[i + 1] >= tail_threshold:
        i += 1
    e = i

    return s, e

def merge_windows(windows: List[Tuple[int, int]], merge_gap: int = 5) -> List[Tuple[int, int]]:
    """
    Merge windows if overlapping or within merge_gap.
    Input/output windows are (start,end) inclusive.
    """
    if not windows:
        return []

    windows = sorted(windows, key=lambda t: t[0])
    merged = [windows[0]]

    for s, e in windows[1:]:
        ms, me = merged[-1]
        if s <= me + merge_gap:
            merged[-1] = (ms, max(me, e))
        else:
            merged.append((s, e))

    return merged

def windows_from_probability(
    p: np.ndarray,
    *,
    # smoothing
    smooth: str = "ema",           # "ema" or "none"
    ema_alpha: float = 0.25,
    # peak finding
    min_peak: float = 0.40,
    min_prominence: float = 0.05,
    min_peak_distance: int = 6,
    # expansion
    tail_threshold: float = 0.25,
    max_expand: Optional[int] = None,
    # filtering/limits
    min_len: int = 6,
    pad: int = 0,
    merge_gap: int = 8,
    max_windows: int = 10,
    # scoring
    score_mode: str = "area"       # "area" or "peak"
) -> Dict[str, object]:
    """
    Peak-seeded windowing on probability signal p.

    Returns dict with:
      - p_raw, p_smooth
      - peaks (indices)
      - windows (list of dicts: start/end/peak_index/peak_value/length/area/mean)
    """
    p = np.asarray(p, dtype=np.float64)
    n = len(p)
    if n == 0:
        return {"p_raw": p, "p_smooth": p, "peaks": np.array([], dtype=int), "windows": []}

    # clip just in case
    p_raw = np.clip(p, 0.0, 1.0)

    # smooth
    if smooth == "ema":
        p_smooth = smooth_ema(p_raw, alpha=ema_alpha)
    elif smooth == "none":
        p_smooth = p_raw
    else:
        raise ValueError(f"Unknown smooth mode: {smooth}")

    # peaks
    peaks = find_local_peaks(
        p_smooth,
        min_peak=min_peak,
        min_prominence=min_prominence,
        min_distance=min_peak_distance
    )

    if peaks.size == 0:
        return {"p_raw": p_raw, "p_smooth": p_smooth, "peaks": peaks, "windows": []}

    # expand each peak into a window
    raw_windows = []
    peak_to_window = []
    for pk in peaks:
        s, e = expand_peak_to_window(
            p_smooth,
            int(pk),
            tail_threshold=tail_threshold,
            max_expand=max_expand
        )
        if pad:
            s = max(0, s - pad)
            e = min(n - 1, e + pad)
        raw_windows.append((s, e))
        peak_to_window.append((int(pk), s, e))

    # merge overlaps
    merged = merge_windows(raw_windows, merge_gap=merge_gap)

    # compute window stats + choose peak inside merged window
    out = []
    for s, e in merged:
        seg = p_smooth[s:e+1]
        if seg.size < min_len:
            continue

        peak_rel = int(np.argmax(seg))
        peak_idx = s + peak_rel
        peak_val = float(p_smooth[peak_idx])
        area = float(np.sum(seg))
        mean = float(np.mean(seg))

        out.append({
            "start": int(s),
            "end": int(e),
            "peak_index": int(peak_idx),
            "peak_value": peak_val,
            "length": int(e - s + 1),
            "area": area,
            "mean": mean,
        })

    # sort by score + cap
    if score_mode == "area":
        out.sort(key=lambda w: w["area"], reverse=True)
    elif score_mode == "peak":
        out.sort(key=lambda w: w["peak_value"], reverse=True)
    else:
        raise ValueError(f"Unknown score_mode: {score_mode}")

    out = out[:max_windows]
    out.sort(key=lambda w: w["start"])  # chronological

    return {"p_raw": p_raw, "p_smooth": p_smooth, "peaks": peaks, "windows": out}
