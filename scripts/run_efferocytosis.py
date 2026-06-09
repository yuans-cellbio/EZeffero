#!/usr/bin/env python3
"""Efferocytosis quantification pipeline entry point.

Usage examples:

    # Single field
    python run_efferocytosis.py single \\
        --ac path/to/AC.tif \\
        --brightfield path/to/BF.tif \\
        --output-prefix sample01 \\
        --config config.yaml

    # Batch
    python run_efferocytosis.py batch \\
        --input plate1_batch.csv \\
        --config config.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

PIPELINE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PIPELINE_ROOT))

from src import pipeline


def configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='Efferocytosis quantification pipeline.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--config', type=Path, required=True,
                        help='Path to YAML config file.')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Enable debug logging.')

    sub = parser.add_subparsers(dest='mode', required=True)

    single = sub.add_parser('single', help='Process a single field.')
    single.add_argument('--ac', type=Path, required=True,
                        help='Path to AC channel TIFF.')
    single.add_argument('--brightfield', type=Path, required=True,
                        help='Path to brightfield TIFF.')
    single.add_argument('--seg-file', type=Path, default=None,
                        help='Path to existing _seg.npy (skips Cellpose).')
    single.add_argument('--output-prefix', type=str, required=True,
                        help='Used as sample_id and output filename prefix.')
    single.add_argument('--output-dir', type=Path, default=None,
                        help='Override output folder (default: BF image folder).')

    batch = sub.add_parser('batch', help='Process a batch from a CSV.')
    batch.add_argument('--input', type=Path, required=True,
                       help='Path to batch input CSV.')
    batch.add_argument('--summary-out', type=Path, default=None,
                       help='Optional path to write batch summary JSON.')

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    configure_logging(args.verbose)
    logger = logging.getLogger('run_efferocytosis')

    if not args.config.exists():
        logger.error(f"Config not found: {args.config}")
        return 2

    config = pipeline.load_config(args.config)
    logger.info(f"Loaded config from {args.config}")

    if args.mode == 'single':
        result = pipeline.run_single(args, config)
        if result is None:
            logger.error("Single-field run failed; see logs above.")
            return 1
        logger.info(f"Done. Wrote {len(result)} cells.")
        return 0

    summary = pipeline.run_batch(args.input, config)
    if args.summary_out:
        summary['run_date_utc'] = datetime.now(timezone.utc).isoformat()
        with open(args.summary_out, 'w') as f:
            json.dump(summary, f, indent=2)
        logger.info(f"Wrote batch summary to {args.summary_out}")
    return 0 if not summary['failed'] else 1


if __name__ == '__main__':
    sys.exit(main())
