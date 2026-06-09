#!/usr/bin/env python3
"""Threshold calibration helper.

Reports per-pixel intensity statistics on negative-control images and suggests
threshold values for ac_detection.threshold in config.yaml. Background
subtraction matches the main pipeline so threshold values are directly
transferable.

Usage:

    python calibrate_threshold.py \\
        --neg-control /path/to/no_AC_well/*.TIF \\
        --channel-extraction max_rgb \\
        --background-subtraction \\
        --rolling-ball-radius-um 20 \\
        --um-per-pixel 0.43 \\
        --output threshold_report.txt
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import List

import numpy as np

PIPELINE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PIPELINE_ROOT))

from src import detect_AC, io as pio


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--neg-control', type=Path, nargs='+', required=True,
                        help='One or more negative-control TIFF files (no AC).')
    parser.add_argument('--channel-extraction', default='max_rgb',
                        choices=['max_rgb', 'sum_rgb', 'r', 'g', 'b'],
                        help='How to collapse RGB to grayscale.')
    parser.add_argument('--background-subtraction', action='store_true',
                        help='Apply rolling-ball top-hat (matches pipeline default).')
    parser.add_argument('--rolling-ball-radius-um', type=float, default=20.0,
                        help='Rolling-ball radius in um (only if --background-subtraction).')
    parser.add_argument('--um-per-pixel', type=float, default=None,
                        help='Pixel size in um. Auto-extracted from each file '
                             'if omitted; required if metadata is missing.')
    parser.add_argument('--output', type=Path, default=None,
                        help='Optional path to save the report; otherwise printed to stdout.')
    return parser.parse_args()


def aggregate_pixel_stats(images: List[np.ndarray]) -> dict:
    """Concatenate all pixels and report distribution statistics."""
    all_pixels = np.concatenate([im.ravel() for im in images]).astype(np.float64)
    pcts = [50, 90, 95, 99, 99.5, 99.9, 99.99]
    pct_vals = np.percentile(all_pixels, pcts)
    return {
        'n_images': len(images),
        'n_pixels': int(all_pixels.size),
        'mean': float(all_pixels.mean()),
        'sd': float(all_pixels.std(ddof=0)),
        'min': float(all_pixels.min()),
        'max': float(all_pixels.max()),
        'percentiles': {str(p): float(v) for p, v in zip(pcts, pct_vals)},
    }


def suggested_thresholds(stats: dict) -> dict:
    """Three threshold suggestions from negative-control statistics."""
    return {
        'mean+5SD (conservative)': stats['mean'] + 5 * stats['sd'],
        'p99.9 (moderate)':        stats['percentiles']['99.9'],
        'p99.99 (permissive)':     stats['percentiles']['99.99'],
    }


def format_report(stats: dict, thresholds: dict, files: List[Path],
                  bg_subtracted: bool, channel_extraction: str) -> str:
    lines = []
    lines.append("Negative-control threshold calibration report")
    lines.append("=" * 60)
    lines.append(f"Channel extraction       : {channel_extraction}")
    lines.append(f"Background subtraction   : {'on' if bg_subtracted else 'off'}")
    lines.append(f"Number of images         : {stats['n_images']}")
    lines.append(f"Total pixels analyzed    : {stats['n_pixels']:,}")
    lines.append("")
    lines.append("Files:")
    for f in files:
        lines.append(f"  {f}")
    lines.append("")
    lines.append("Pixel intensity distribution (after preprocessing):")
    lines.append(f"  mean   : {stats['mean']:.3f}")
    lines.append(f"  SD     : {stats['sd']:.3f}")
    lines.append(f"  min    : {stats['min']:.3f}")
    lines.append(f"  max    : {stats['max']:.3f}")
    lines.append("  percentiles:")
    for p, v in stats['percentiles'].items():
        lines.append(f"    p{p:>5} : {v:.3f}")
    lines.append("")
    lines.append("Suggested thresholds (write one to config.yaml ac_detection.threshold):")
    for name, value in thresholds.items():
        lines.append(f"  {name:<28} {value:.2f}")
    lines.append("")
    lines.append("Note: lower threshold is more sensitive (more false positives in")
    lines.append("the negative control); higher threshold is more specific. Inspect")
    lines.append("the no-AC overlay after picking a value to confirm phagocytic_strict")
    lines.append("rate is near zero in the control well.")
    return "\n".join(lines)


def main() -> int:
    configure_logging()
    args = parse_args()

    images: List[np.ndarray] = []
    for path in args.neg_control:
        if not path.exists():
            logging.error(f"File not found: {path}")
            return 2
        img = pio.load_image_grayscale(path, args.channel_extraction)
        if args.background_subtraction:
            um_px = args.um_per_pixel or pio.extract_pixel_size_um(path)
            if um_px is None:
                logging.error(f"Cannot resolve pixel size for {path}; "
                              f"pass --um-per-pixel.")
                return 2
            radius_px = max(int(round(args.rolling_ball_radius_um / um_px)), 1)
            img = detect_AC.subtract_background(img, radius_px)
        images.append(img)
        logging.info(f"Loaded {path} (shape={img.shape}, "
                     f"max={img.max():.1f}, mean={img.mean():.2f})")

    stats = aggregate_pixel_stats(images)
    thresholds = suggested_thresholds(stats)
    report = format_report(stats, thresholds, args.neg_control,
                            args.background_subtraction, args.channel_extraction)

    if args.output:
        args.output.write_text(report)
        logging.info(f"Wrote report to {args.output}")
    print(report)
    return 0


if __name__ == '__main__':
    sys.exit(main())
