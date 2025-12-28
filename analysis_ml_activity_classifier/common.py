from __future__ import annotations

import json
from pathlib import Path
import numpy as np


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
