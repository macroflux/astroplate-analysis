# analyze1.py
import cv2
import json
import csv
from pathlib import Path
import numpy as np

def build_sky_mask(sample_img_path: Path) -> np.ndarray:
    """
    One-time mask builder idea:
    - exclude very dark pixels (housing)
    - optionally exclude top-left overlay region
    Save and reuse this mask each run.
    """
    img = cv2.imread(str(sample_img_path))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # crude sky vs housing mask
    mask = (gray > 10).astype(np.uint8) * 255

    # remove overlay text region (tune these coords to your layout)
    mask[0:140, 0:450] = 0

    return mask

def star_contrast_score(gray: np.ndarray, mask: np.ndarray) -> float:
    # high-pass: stars increase local contrast
    blur = cv2.GaussianBlur(gray, (0,0), sigmaX=5)
    hp = cv2.subtract(gray, blur)
    vals = hp[mask > 0]
    return float(np.std(vals))

def detect_streaks(gray: np.ndarray, mask: np.ndarray):
    # super basic streak detection (starter):
    # edges -> Hough lines, return list of segments
    g = cv2.bitwise_and(gray, gray, mask=mask)
    edges = cv2.Canny(g, 40, 120)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=60,
                            minLineLength=40, maxLineGap=10)
    if lines is None:
        return []
    return [l[0].tolist() for l in lines]  # [x1,y1,x2,y2]

def main(night_dir: str):
    night = Path(night_dir)
    frames = sorted((night / "frames").glob("*.jpg"))
    if not frames:
        raise SystemExit("No frames found.")

    mask_path = night / "sky_mask.png"
    if mask_path.exists():
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    else:
        mask = build_sky_mask(frames[0])
        cv2.imwrite(str(mask_path), mask)

    metrics_rows = []
    events = []

    for f in frames:
        img = cv2.imread(str(f))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        mean = float(np.mean(gray[mask > 0]))
        contrast = star_contrast_score(gray, mask)
        streaks = detect_streaks(gray, mask)

        metrics_rows.append({
            "file": f.name,
            "mean_brightness": mean,
            "star_contrast": contrast,
            "streak_count": len(streaks),
        })

        # naive event flag:
        if len(streaks) >= 2:
            events.append({
                "file": f.name,
                "reason": "many_streaks",
                "streaks": streaks[:10],
            })

    # write outputs
    out_csv = night / "metrics.csv"
    with out_csv.open("w", newline="") as fp:
        w = csv.DictWriter(fp, fieldnames=metrics_rows[0].keys())
        w.writeheader()
        w.writerows(metrics_rows)

    out_events = night / "events.json"
    out_events.write_text(json.dumps(events, indent=2))

    print(f"Wrote {out_csv} and {out_events}")

if __name__ == "__main__":
    import sys
    main(sys.argv[1])
