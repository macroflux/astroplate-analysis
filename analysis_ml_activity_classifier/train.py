from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from common import (
    load_metrics_csv,
    load_windows_json,
    windows_to_labels,
    build_feature_matrix,
    train_val_split,
    sigmoid,
)


def logistic_train(
    X: np.ndarray,
    y: np.ndarray,
    lr: float = 0.05,
    epochs: int = 1500,
    l2: float = 1e-3,
) -> tuple[np.ndarray, float]:
    """
    Simple logistic regression (batch GD), returns (w, b).
    """
    n, d = X.shape
    w = np.zeros(d, dtype=np.float64)
    b = 0.0

    for _ in range(epochs):
        z = X @ w + b
        p = sigmoid(z)

        # gradients
        dw = (X.T @ (p - y)) / n + l2 * w
        db = np.mean(p - y)

        w -= lr * dw
        b -= lr * db

    return w, b


def metrics_at_threshold(y_true: np.ndarray, p: np.ndarray, thr: float = 0.5) -> dict:
    y_hat = (p >= thr).astype(np.int64)
    tp = int(np.sum((y_true == 1) & (y_hat == 1)))
    tn = int(np.sum((y_true == 0) & (y_hat == 0)))
    fp = int(np.sum((y_true == 0) & (y_hat == 1)))
    fn = int(np.sum((y_true == 1) & (y_hat == 0)))

    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    acc = (tp + tn) / max(1, (tp + tn + fp + fn))

    return {
        "threshold": thr,
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "precision": prec,
        "recall": rec,
        "accuracy": acc,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("night_dir", help="Path to night folder (contains data/metrics.csv)")
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--epochs", type=int, default=1500)
    ap.add_argument("--l2", type=float, default=1e-3)
    ap.add_argument("--out", default=None, help="Output directory (default: <night>/ml)")
    ap.add_argument("--topk", type=int, default=50, help="Top K frames to list by predicted probability")
    args = ap.parse_args()

    night = Path(args.night_dir)
    data_dir = night / "data"
    metrics_csv = data_dir / "metrics.csv"
    windows_json = data_dir / "activity_windows.json"

    if not metrics_csv.exists():
        raise SystemExit(f"Missing {metrics_csv}. Run analyze.py first.")

    rows = load_metrics_csv(metrics_csv)
    X, feat_names = build_feature_matrix(rows)
    n = len(rows)

    windows = load_windows_json(windows_json)
    y = windows_to_labels(n, windows)  # pseudo-labels

    # split
    tr_idx, va_idx = train_val_split(n, val_frac=args.val_frac, seed=args.seed)
    Xtr, ytr = X[tr_idx], y[tr_idx]
    Xva, yva = X[va_idx], y[va_idx]

    # train
    w, b = logistic_train(Xtr, ytr, lr=args.lr, epochs=args.epochs, l2=args.l2)

    # predict
    ptr = sigmoid(Xtr @ w + b)
    pva = sigmoid(Xva @ w + b)

    # evaluate
    report = {
        "train": metrics_at_threshold(ytr, ptr, 0.5),
        "val": metrics_at_threshold(yva, pva, 0.5),
        "meta": {
            "n_frames": n,
            "n_pos": int(np.sum(y)),
            "val_frac": args.val_frac,
            "seed": args.seed,
            "lr": args.lr,
            "epochs": args.epochs,
            "l2": args.l2,
            "features": feat_names,
        },
        "weights": {feat_names[i]: float(w[i]) for i in range(len(feat_names))},
        "bias": float(b),
    }

    out_dir = Path(args.out) if args.out else (night / "ml")
    out_dir.mkdir(parents=True, exist_ok=True)

    # score all frames
    p_all = sigmoid(X @ w + b)

    # emit predictions CSV (simple)
    import csv
    pred_csv = out_dir / "predictions.csv"
    with pred_csv.open("w", newline="") as fp:
        writer = csv.writer(fp)
        writer.writerow(["index", "file", "y_window", "p_activity"])
        for i, r in enumerate(rows):
            writer.writerow([i, r.get("file", ""), int(y[i]), float(p_all[i])])

    # ranked list (topk)
    order = np.argsort(-p_all)
    topk = []
    for j in order[: args.topk]:
        topk.append({
            "index": int(j),
            "file": rows[j].get("file", ""),
            "p_activity": float(p_all[j]),
            "y_window": int(y[j]),
        })

    (out_dir / "report.json").write_text(json.dumps(report, indent=2))
    (out_dir / "topk_frames.json").write_text(json.dumps(topk, indent=2))

    print("\n" + "=" * 60)
    print("ML Activity Classifier (logistic regression, pseudo-labels from windows)")
    print("=" * 60)
    print(f"Output: {out_dir}")
    print(f"Predictions: {pred_csv.name}")
    print(f"Report: report.json")
    print(f"TopK: topk_frames.json\n")

    print("Validation metrics @ 0.50 threshold:")
    for k, v in report["val"].items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")

    print("\nMost influential features (abs weight):")
    items = sorted(report["weights"].items(), key=lambda kv: abs(kv[1]), reverse=True)[:8]
    for name, wt in items:
        print(f"  {name:28s}  {wt:+.4f}")


if __name__ == "__main__":
    main()
