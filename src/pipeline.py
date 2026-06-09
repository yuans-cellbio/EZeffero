"""Pipeline orchestration: per-field processing, batch loop, run_log generation."""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yaml

from . import detect_AC, io as pio, overlay, quantify, segment

logger = logging.getLogger(__name__)


def load_config(config_path: Path) -> dict:
    """Load and lightly validate the YAML config."""
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    for section in ('microscope', 'segmentation', 'ac_detection', 'output'):
        if section not in cfg:
            raise ValueError(f"Config missing required section: {section}")
    return cfg


def resolve_pixel_size_um(bf_path: Path, config: dict) -> float:
    """Resolve pixel size: config override -> metadata -> error."""
    cfg_value = config['microscope'].get('um_per_pixel')
    if cfg_value is not None:
        return float(cfg_value)

    metadata_value = pio.extract_pixel_size_um(bf_path)
    if metadata_value is not None:
        logger.info(f"Pixel size from metadata: {metadata_value:.4f} um/px")
        return metadata_value

    raise ValueError(
        f"Could not extract pixel size from {bf_path}. Set "
        f"microscope.um_per_pixel in config.yaml. The script is designed for "
        f"EVOS OME-TIFF output; non-EVOS images may not carry the expected "
        f"metadata tags."
    )


def process_field(row: pd.Series, config: dict) -> Optional[pd.DataFrame]:
    """Process a single field. Returns per-cell DataFrame or None on failure.

    Failures are logged and converted to None so batch runs continue.
    """
    sample_id = str(row['sample_id'])
    logger.info(f"=== Processing sample_id={sample_id} ===")

    pio.validate_row_paths(row)

    bf_path = Path(row['brightfield_path'])
    ac_path = Path(row['ac_path'])
    seg_path = Path(row['seg_path']) if row.get('seg_path') else None
    output_dir = Path(row['output_dir']) if row.get('output_dir') else bf_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    um_per_pixel = resolve_pixel_size_um(bf_path, config)

    fluorescence_extraction = config['ac_detection'].get('channel_extraction', 'max_rgb')
    brightfield_extraction = config['segmentation'].get('brightfield_channel_extraction',
                                                        'luminance')
    bf_image = pio.load_image_grayscale(bf_path, brightfield_extraction)
    ac_image_raw = pio.load_image_grayscale(ac_path, fluorescence_extraction)

    seg_cfg = config['segmentation']
    if seg_path is not None:
        if not seg_path.exists():
            raise FileNotFoundError(
                f"seg_path was provided but the file does not exist: "
                f"{seg_path}. To run Cellpose instead, leave seg_path empty."
            )
        logger.info(f"Loading existing segmentation from {seg_path}")
        cell_labels = segment.load_seg_file(seg_path)
        seg_file_used = seg_path
    else:
        cell_diameter_um = seg_cfg.get('cell_diameter_um')
        if cell_diameter_um is None or cell_diameter_um <= 0:
            diameter_px = None
        else:
            diameter_px = cell_diameter_um / um_per_pixel
        model_spec = seg_cfg['cellpose_model']
        use_model_diameter = seg_cfg.get('use_model_diameter', False)
        if segment.is_cellpose_model_path(model_spec) and 'use_model_diameter' not in seg_cfg:
            use_model_diameter = True

        cell_labels, flows, model_resolved = segment.run_cellpose(
            brightfield=bf_image,
            model_spec=model_spec,
            cell_diameter_px=diameter_px,
            use_gpu=seg_cfg.get('use_gpu', True),
            use_model_diameter=use_model_diameter,
            flow_threshold=seg_cfg.get('flow_threshold', 0.4),
            cellprob_threshold=seg_cfg.get('cellprob_threshold', 0.0),
        )

        out_cfg = config['output']
        save_gray_bf = out_cfg.get('save_grayscale_brightfield', True)
        if save_gray_bf:
            gray_bf_path = output_dir / f"{bf_path.stem}_gray.tif"
            pio.save_grayscale_tif(bf_image, gray_bf_path)
            seg_file_used = output_dir / f"{bf_path.stem}_gray_seg.npy"
            seg_source_path = gray_bf_path
        else:
            seg_file_used = output_dir / f"{bf_path.stem}_seg.npy"
            seg_source_path = bf_path

        if out_cfg.get('save_seg_file', True):
            segment.save_seg_file(
                seg_out_path=seg_file_used,
                masks=cell_labels,
                flows=flows,
                image=bf_image,
                diameter_px=diameter_px,
                source_image_path=seg_source_path,
            )

    min_cell_area_px = seg_cfg.get('min_cell_area_um2', 80) / (um_per_pixel ** 2)
    cell_labels = segment.filter_labels(
        cell_labels,
        min_cell_area_px=min_cell_area_px,
        exclude_border=seg_cfg.get('exclude_border_cells', True),
    )

    if seg_cfg.get('dilation', {}).get('enabled', False):
        cell_labels = quantify.expand_cell_labels(
            cell_labels,
            distance_um=seg_cfg['dilation']['distance_um'],
            um_per_pixel=um_per_pixel,
        )

    n_cells = int(cell_labels.max())
    if n_cells == 0:
        logger.warning(f"No cells in {sample_id} after filtering; writing empty CSV")

    ac_cfg = config['ac_detection']
    if ac_cfg.get('background_subtraction', {}).get('enabled', True):
        radius_px = int(round(ac_cfg['background_subtraction']
                              .get('rolling_ball_radius_um', 20) / um_per_pixel))
        ac_image_corrected = detect_AC.subtract_background(ac_image_raw, radius_px)
    else:
        ac_image_corrected = ac_image_raw

    threshold = float(ac_cfg['threshold'])
    min_object_area_px = ac_cfg.get('min_object_area_um2', 0.3) / (um_per_pixel ** 2)
    ac_labels, ac_props = detect_AC.detect_ac_objects(
        ac_image_corrected, threshold=threshold, min_object_area_px=min_object_area_px,
    )
    ac_props = detect_AC.classify_ac_by_size(
        ac_props,
        um_per_pixel=um_per_pixel,
        puncta_max_um2=ac_cfg['size_classes_um2']['puncta_max'],
        small_ac_max_um2=ac_cfg['size_classes_um2']['small_ac_max'],
    )
    ac_props = quantify.assign_objects_to_cells(ac_props, cell_labels)

    per_cell = quantify.compute_per_cell_metrics(
        cell_labels=cell_labels,
        ac_props=ac_props,
        ac_image=ac_image_corrected,
        um_per_pixel=um_per_pixel,
    )

    metadata = pio.get_pass_through_metadata(row)
    for k, v in metadata.items():
        per_cell[k] = v
    per_cell['sample_id'] = sample_id
    per_cell['threshold_used'] = threshold
    per_cell['um_per_pixel'] = um_per_pixel
    per_cell['cellpose_model'] = seg_cfg['cellpose_model']
    per_cell['seg_file_sha256'] = (pio.compute_file_sha256(seg_file_used)
                                   if seg_file_used.exists() else '')
    per_cell['run_date_utc'] = datetime.now(timezone.utc).isoformat()

    leading = ['sample_id'] + list(metadata.keys())
    rest = [c for c in per_cell.columns if c not in leading]
    per_cell = per_cell[leading + rest]

    csv_path = output_dir / f"{sample_id}_per_cell.csv"
    per_cell.to_csv(csv_path, index=False)
    logger.info(f"Wrote {csv_path} ({len(per_cell)} cells)")

    if config['output'].get('save_qc_overlays', True):
        overlay_path = output_dir / f"{sample_id}_overlay.png"
        overlay.make_qc_overlay(
            brightfield=bf_image,
            ac_image=ac_image_raw,
            cell_labels=cell_labels,
            ac_props=ac_props,
            output_path=overlay_path,
            title=f"{sample_id}  (n_cells={n_cells}, threshold={threshold:.0f})",
            dpi=config['output'].get('qc_overlay_dpi', 120),
        )

    write_run_log(
        log_path=output_dir / f"{sample_id}_run_log.json",
        config=config,
        row=row,
        bf_path=bf_path,
        ac_path=ac_path,
        seg_file_used=seg_file_used,
        threshold=threshold,
        um_per_pixel=um_per_pixel,
        n_cells=n_cells,
        n_ac_objects=len(ac_props),
    )

    return per_cell


def write_run_log(log_path: Path,
                  config: dict,
                  row: pd.Series,
                  bf_path: Path,
                  ac_path: Path,
                  seg_file_used: Path,
                  threshold: float,
                  um_per_pixel: float,
                  n_cells: int,
                  n_ac_objects: int) -> None:
    """Write a per-field provenance log."""
    versions = _collect_versions()

    inputs = {
        'brightfield': {'path': str(bf_path), 'sha256': pio.compute_file_sha256(bf_path)},
        'ac':          {'path': str(ac_path), 'sha256': pio.compute_file_sha256(ac_path)},
    }
    if seg_file_used.exists():
        inputs['seg'] = {'path': str(seg_file_used),
                         'sha256': pio.compute_file_sha256(seg_file_used)}

    log_data = {
        'sample_id': str(row['sample_id']),
        'run_date_utc': datetime.now(timezone.utc).isoformat(),
        'config_snapshot': config,
        'row_metadata': pio.get_pass_through_metadata(row),
        'inputs': inputs,
        'resolved_parameters': {
            'um_per_pixel': um_per_pixel,
            'threshold': threshold,
            'n_cells_after_filter': n_cells,
            'n_ac_objects': n_ac_objects,
        },
        'versions': versions,
        'random_seed': 123,
    }
    with open(log_path, 'w') as f:
        json.dump(log_data, f, indent=2, default=str)


def _collect_versions() -> dict:
    """Collect versions of pipeline-relevant packages for provenance.

    Uses importlib.metadata (which queries pip-installed package metadata
    directly), since several relevant packages do not expose __version__
    as a module attribute. Maps the import name to the PyPI distribution
    name where they differ.
    """
    from importlib.metadata import version, PackageNotFoundError
    out = {'python': sys.version.split()[0]}
    distribution_names = {
        'numpy': 'numpy',
        'pandas': 'pandas',
        'scikit-image': 'scikit-image',
        'Pillow': 'Pillow',
        'PyYAML': 'PyYAML',
        'matplotlib': 'matplotlib',
        'cellpose': 'cellpose',
        'torch': 'torch',
    }
    for label, dist in distribution_names.items():
        try:
            out[label] = version(dist)
        except PackageNotFoundError:
            out[label] = 'not_installed'
    return out


def run_batch(csv_path: Path, config: dict) -> dict:
    """Execute batch mode. Returns summary dict."""
    df = pio.parse_batch_csv(csv_path)
    logger.info(f"Batch CSV loaded: {len(df)} rows")

    np.random.seed(123)
    try:
        import torch
        torch.manual_seed(123)
    except ImportError:
        pass

    succeeded, failed = [], []
    for _, row in df.iterrows():
        try:
            result = process_field(row, config)
            if result is not None:
                succeeded.append(row['sample_id'])
        except Exception as exc:
            logger.error(f"Failed on sample_id={row['sample_id']}: {exc}",
                         exc_info=True)
            failed.append({'sample_id': row['sample_id'], 'error': str(exc)})

    summary = {
        'total': len(df),
        'succeeded': succeeded,
        'failed': failed,
    }
    logger.info(f"Batch complete: {len(succeeded)}/{len(df)} succeeded")
    return summary


def run_single(args, config: dict) -> Optional[pd.DataFrame]:
    """Execute single mode by constructing a one-row DataFrame and processing it."""
    row = pd.Series({
        'sample_id': args.output_prefix,
        'ac_path': str(Path(args.ac).resolve()),
        'brightfield_path': str(Path(args.brightfield).resolve()),
        'seg_path': str(Path(args.seg_file).resolve()) if args.seg_file else '',
        'output_dir': str(Path(args.output_dir).resolve()) if args.output_dir else '',
    })
    np.random.seed(123)
    try:
        import torch
        torch.manual_seed(123)
    except ImportError:
        pass
    return process_field(row, config)
