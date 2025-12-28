#!/usr/bin/env python3
"""
Timelapse Video Generator

Generates MP4 timelapse videos from image sequences using OpenCV.
Can be used standalone or integrated into analysis pipeline.

Usage:
    python timelapse.py <frames_dir> [options]
    python timelapse.py <frames_dir> --output video.mp4 --fps 30 --quality 8
"""
from pathlib import Path
import sys
import argparse

try:
    import cv2
except ImportError:
    print("Error: opencv-python not installed", file=sys.stderr)
    print("Install with: pip install opencv-python", file=sys.stderr)
    sys.exit(1)


def build_timelapse(
    frames_dir: Path,
    output_path: Path,
    fps: int = 30,
    pattern: str = "*.jpg",
    quality: int = 8,
    verbose: bool = True,
    codec: str = "mp4v",
) -> bool:
    """
    Build an MP4 timelapse from image frames using OpenCV.

    Notes on quality:
      - OpenCV VideoWriter does NOT reliably expose CRF/bitrate control like ffmpeg.
      - We try to hint quality via VIDEOWRITER_PROP_QUALITY when available.
      - Some platforms/codecs ignore this; we report whether it was accepted.
    """
    if not frames_dir.exists():
        print(f"Error: Frames directory not found: {frames_dir}", file=sys.stderr)
        return False

    # Allow a single pattern OR a "comma list" of patterns like "*.jpg,*.png"
    patterns = [p.strip() for p in pattern.split(",")] if "," in pattern else [pattern]

    frames = []
    for pat in patterns:
        frames.extend(frames_dir.glob(pat))
    frames = sorted(set(frames))  # de-dupe + stable sort

    if not frames:
        print(f"Error: No frames found in {frames_dir} matching {patterns}", file=sys.stderr)
        return False

    if verbose:
        print(f"Building MP4 timelapse")
        print(f"  Input: {frames_dir}")
        print(f"  Patterns: {patterns}")
        print(f"  Frames: {len(frames)}")
        print(f"  FPS: {fps}")
        print(f"  Codec: {codec}")
        print(f"  Output: {output_path}")

    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)

        first_frame = cv2.imread(str(frames[0]))
        if first_frame is None:
            print(f"Error: Could not read first frame: {frames[0]}", file=sys.stderr)
            return False

        height, width = first_frame.shape[:2]

        # Create VideoWriter
        fourcc = cv2.VideoWriter_fourcc(*codec)
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

        if not writer.isOpened():
            print(f"Error: Could not open video writer for {output_path}", file=sys.stderr)
            print("Try a different codec: mp4v, avc1 (if supported), X264 (rare), etc.", file=sys.stderr)
            return False

        # ---- Attempt to apply quality hint (may be ignored by backend) ----
        # Your CLI defines: 0-10 where lower = higher quality.
        # OpenCV uses a 0..1 float for some backends. We'll map:
        # quality=0 => 1.0 (best), quality=10 => 0.0 (worst)
        quality_hint = max(0.0, min(1.0, 1.0 - (quality / 10.0)))
        quality_applied = False
        try:
            if hasattr(cv2, "VIDEOWRITER_PROP_QUALITY"):
                quality_applied = bool(writer.set(cv2.VIDEOWRITER_PROP_QUALITY, float(quality_hint)))
        except Exception:
            quality_applied = False

        if verbose:
            if hasattr(cv2, "VIDEOWRITER_PROP_QUALITY"):
                print(f"  Quality requested: {quality} (hint={quality_hint:.2f}) | applied={quality_applied}")
            else:
                print("  Quality requested: (no VIDEOWRITER_PROP_QUALITY in this OpenCV build)")

        # Write frames
        written = 0
        for i, frame_path in enumerate(frames, 1):
            frame = cv2.imread(str(frame_path))
            if frame is None:
                if verbose:
                    print(f"  Warning: Could not read frame {frame_path.name}, skipping", file=sys.stderr)
                continue

            if frame.shape[:2] != (height, width):
                if verbose:
                    print(f"  Warning: Frame {frame_path.name} has different dimensions, skipping", file=sys.stderr)
                continue

            writer.write(frame)
            written += 1

            if verbose and (i % 50 == 0 or i == len(frames)):
                print(f"  Progress: {i}/{len(frames)} frames ({100*i//len(frames)}%)")

        writer.release()

        if verbose:
            print(f"  Successfully wrote {written} frames")

        return written > 0

    except Exception as e:
        print(f"Error creating timelapse: {e}", file=sys.stderr)
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Generate MP4 timelapse videos from image sequences",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create timelapse from frames directory (default: timelapse.mp4)
  python timelapse.py path/to/frames

  # Specify output file and settings
  python timelapse.py path/to/frames --output night.mp4 --fps 25

  # Create from annotated images
  python timelapse.py data/night_2025-12-24/annotated --output annotated.mp4
        """
    )
    
    parser.add_argument(
        'frames_dir',
        type=Path,
        help='Directory containing image frames'
    )
    
    parser.add_argument(
        '-o', '--output',
        type=Path,
        default=None,
        help='Output MP4 path (default: frames_dir/timelapse.mp4)'
    )
    
    parser.add_argument(
        '--fps',
        type=int,
        default=30,
        help='Frames per second (default: 30)'
    )
    
    parser.add_argument(
        '--pattern',
        type=str,
        default='*.jpg',
        help='Glob pattern for frame files (default: *.jpg)'
    )
    
    parser.add_argument(
        '--quality',
        type=int,
        default=8,
        choices=range(0, 11),
        help='Video quality 0-10, lower=higher (default: 8)'
    )
    
    parser.add_argument(
        '-q', '--quiet',
        action='store_true',
        help='Suppress progress messages'
    )
    
    args = parser.parse_args()
    
    # Set default output path if not specified
    if args.output is None:
        args.output = args.frames_dir / "timelapse.mp4"
    
    # Build the timelapse
    success = build_timelapse(
        frames_dir=args.frames_dir,
        output_path=args.output,
        fps=args.fps,
        pattern=args.pattern,
        quality=args.quality,
        verbose=not args.quiet
    )
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()