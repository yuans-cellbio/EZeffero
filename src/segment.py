"""Cell segmentation: Cellpose wrapper, seg file I/O, postprocessing.

Cellpose is imported lazily inside run_cellpose() so the rest of the pipeline
remains usable when only loading existing seg files.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from skimage import segmentation

logger = logging.getLogger(__name__)


def is_cellpose_model_path(model_spec: str) -> bool:
    """Return True if model_spec resolves to an existing file (custom model)."""
    return Path(model_spec).is_file()


def _get_cellpose_version(cellpose_module) -> str:
    """Resolve the installed Cellpose version string robustly.

    Cellpose v4 stopped exposing `cellpose.__version__` as a top-level
    attribute. importlib.metadata is the standard fallback and works for any
    pip-installed package, so try that first. If both fail, return '0.0.0'
    and let the caller decide what to do.
    """
    try:
        from importlib.metadata import version
        return version('cellpose')
    except Exception:
        pass

    candidate = getattr(cellpose_module, '__version__', None)
    if isinstance(candidate, str):
        return candidate

    version_attr = getattr(cellpose_module, 'version', None)
    if isinstance(version_attr, str):
        return version_attr
    if version_attr is not None:
        nested = getattr(version_attr, '__version__', None)
        if isinstance(nested, str):
            return nested

    return '0.0.0'


def run_cellpose(brightfield: np.ndarray,
                 model_spec: str,
                 cell_diameter_px: Optional[float],
                 use_gpu: bool,
                 use_model_diameter: bool,
                 flow_threshold: float = 0.4,
                 cellprob_threshold: float = 0.0
                 ) -> Tuple[np.ndarray, list, str]:
    """Run Cellpose on the brightfield image.

    cell_diameter_px:
        Float -> passed to model.eval as `diameter`. Cellpose rescales the
        image so that an object of this size becomes the model's training
        mean diameter (30 pixels for CPSAM, similar for cyto3). For CPSAM,
        which is largely size-invariant in 7.5-120 px, the rescaling can
        cause information loss when the cells are already in that range.
        None -> diameter is not passed; no rescaling. Recommended for CPSAM
        when input cell sizes are within the trained range.

    Cellpose v4 (Cellpose-SAM):
        Single model (CPSAM), channel-agnostic. The `model_type` and `channels`
        arguments were removed.

    Cellpose v3 (cyto3 et al.):
        Multi-model with explicit channel semantics. The pipeline runs
        single-channel on brightfield (`channels=[0, 0]`).

    Returns (masks, flows, resolved_model_descriptor).
    """
    try:
        import cellpose
        from cellpose import models
    except ImportError as exc:
        raise RuntimeError(
            "cellpose is not installed. Install with `pip install cellpose` "
            "or supply a pre-existing seg_path to skip segmentation."
        ) from exc

    cp_version = _get_cellpose_version(cellpose)
    try:
        cp_major = int(cp_version.split('.')[0])
    except (ValueError, IndexError):
        logger.warning(f"Could not parse Cellpose version '{cp_version}'; "
                       f"assuming v4+ API.")
        cp_major = 4
    is_v4 = cp_major >= 4

    if is_cellpose_model_path(model_spec):
        model_path = str(Path(model_spec).resolve())
        logger.info(f"Loading custom Cellpose model from {model_path}")
        model = models.CellposeModel(gpu=use_gpu, pretrained_model=model_path)
        resolved = model_path
    elif is_v4:
        if model_spec not in ('cpsam', 'auto', '', None):
            logger.warning(
                f"Cellpose v{cp_version} only ships Cellpose-SAM (CPSAM); "
                f"requested model_spec='{model_spec}' is ignored. To silence "
                f"this warning, set segmentation.cellpose_model: cpsam in config."
            )
        logger.info(f"Loading Cellpose-SAM (cellpose v{cp_version})")
        model = models.CellposeModel(gpu=use_gpu)
        resolved = f'cpsam-cellpose-v{cp_version}'
    else:
        logger.info(f"Loading built-in Cellpose model: {model_spec} (v{cp_version})")
        try:
            model = models.CellposeModel(gpu=use_gpu, model_type=model_spec)
        except TypeError:
            model = models.CellposeModel(gpu=use_gpu, pretrained_model=model_spec)
        resolved = model_spec

    if use_model_diameter or cell_diameter_px is None:
        diameter = None
    else:
        diameter = float(cell_diameter_px)

    if diameter is None:
        logger.info("Cellpose eval: diameter=None (no image rescaling)")
    else:
        cpsam_train_mean = 30.0
        rescale_factor = cpsam_train_mean / diameter
        logger.info(
            f"Cellpose eval: diameter={diameter:.2f} px -> image will be "
            f"rescaled by {rescale_factor:.2f}x before segmentation "
            f"(target = {cpsam_train_mean:.0f} px training mean)"
        )

    if is_v4:
        eval_kwargs = dict(
            diameter=diameter,
            flow_threshold=flow_threshold,
            cellprob_threshold=cellprob_threshold,
        )
    else:
        eval_kwargs = dict(
            diameter=diameter,
            channels=[0, 0],
            flow_threshold=flow_threshold,
            cellprob_threshold=cellprob_threshold,
        )

    result = model.eval(brightfield, **eval_kwargs)
    masks = result[0]
    flows = result[1] if len(result) > 1 else []

    n_cells = int(masks.max())
    logger.info(f"Cellpose detected {n_cells} cells")

    return masks.astype(np.int32), flows, resolved


def save_seg_file(seg_out_path: Path,
                  masks: np.ndarray,
                  flows: list,
                  image: np.ndarray,
                  diameter_px: Optional[float],
                  source_image_path: Path) -> None:
    """Write a Cellpose-compatible _seg.npy file.

    Format mirrors what cellpose.io.masks_flows_to_seg writes, so the GUI can
    auto-detect masks when the user opens the source image. We write a manual
    dict to avoid forcing cellpose import for non-segmentation runs.
    """
    outlines = _masks_to_outlines(masks)
    n = int(masks.max())

    seg_dict = {
        'masks': masks.astype(np.uint16 if n < 65535 else np.uint32),
        'outlines': outlines.astype(np.uint16 if n < 65535 else np.uint32),
        'flows': flows if flows else [],
        'chan_choose': [0, 0],
        'ismanual': np.zeros(n, dtype=bool),
        'filename': str(source_image_path),
        'diameter': float(diameter_px) if diameter_px else 0.0,
        'img': image,
    }
    np.save(seg_out_path, seg_dict, allow_pickle=True)
    logger.info(f"Saved seg file: {seg_out_path}")


def load_seg_file(seg_path: Path) -> np.ndarray:
    """Load masks from a Cellpose _seg.npy file. Returns a labeled int array."""
    data = np.load(seg_path, allow_pickle=True)
    if hasattr(data, 'item'):
        data = data.item()
    if not isinstance(data, dict) or 'masks' not in data:
        raise ValueError(f"{seg_path} is not a recognized Cellpose seg file")
    masks = np.asarray(data['masks'])
    n_cells = int(masks.max())
    logger.info(f"Loaded seg file with {n_cells} cells from {seg_path}")
    return masks.astype(np.int32)


def _masks_to_outlines(masks: np.ndarray) -> np.ndarray:
    """Convert label image to outline image (each cell's boundary as its label)."""
    boundaries = segmentation.find_boundaries(masks, mode='inner')
    outlines = np.where(boundaries, masks, 0)
    return outlines


def filter_labels(masks: np.ndarray,
                  min_cell_area_px: float,
                  exclude_border: bool) -> np.ndarray:
    """Drop cells smaller than min_cell_area_px and (optionally) border-touching cells.

    Returns a relabeled image (1..N contiguous) for clean downstream indexing.
    Border-touching status is also returned via flag_border_cells in quantify.py
    so pipelines that want to keep border cells but flag them can do that
    instead of dropping. Here we drop entirely if exclude_border is True.
    """
    out = masks.copy()

    if min_cell_area_px > 0:
        sizes = np.bincount(out.ravel())
        sizes[0] = 0
        keep = sizes >= min_cell_area_px
        out = np.where(keep[out], out, 0)

    if exclude_border:
        out = segmentation.clear_border(out)

    out, _, _ = segmentation.relabel_sequential(out)
    return out.astype(np.int32)
