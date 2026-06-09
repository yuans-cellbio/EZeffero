"""Per-cell quantification: dilation, AC-to-cell assignment, metric computation."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from skimage import measure, segmentation

logger = logging.getLogger(__name__)


def expand_cell_labels(labels: np.ndarray,
                       distance_um: float,
                       um_per_pixel: float) -> np.ndarray:
    """Expand each label outward by distance_um using a multi-source watershed.

    skimage.segmentation.expand_labels grows each label simultaneously and
    stops at the midline between neighbors, so dilated regions never overlap
    and no ambiguous zones are created. This is the correct behavior for AC
    ownership assignment in dense fields. Distance is the maximum expansion
    in pixels; in regions with neighbors closer than 2*distance, expansion
    halts at the midline before reaching the full distance.
    """
    distance_px = max(int(round(distance_um / um_per_pixel)), 1)
    logger.info(f"Expanding labels by up to {distance_px} px ({distance_um} um)")
    return segmentation.expand_labels(labels, distance=distance_px)


def assign_objects_to_cells(ac_props: pd.DataFrame,
                            cell_labels: np.ndarray) -> pd.DataFrame:
    """Assign each AC object to a cell by centroid lookup.

    AC objects whose centroid falls outside any cell label receive cell_id=0
    and are excluded from per-cell aggregation. Centroid-based assignment is
    robust for compact AC objects; for elongated objects it could be replaced
    with a majority-overlap rule but that is not the regime here.
    """
    out = ac_props.copy()
    if len(out) == 0:
        out['cell_id'] = pd.Series(dtype=int)
        return out

    cy = out['centroid_y'].astype(int).clip(0, cell_labels.shape[0] - 1)
    cx = out['centroid_x'].astype(int).clip(0, cell_labels.shape[1] - 1)
    out['cell_id'] = cell_labels[cy.values, cx.values]

    n_assigned = int((out['cell_id'] > 0).sum())
    logger.info(f"AC objects assigned to a cell: {n_assigned}/{len(out)}")
    return out


def flag_border_cells(cell_labels: np.ndarray) -> set:
    """Return the set of cell ids that touch any image border."""
    border_ids = set()
    border_ids.update(np.unique(cell_labels[0, :]).tolist())
    border_ids.update(np.unique(cell_labels[-1, :]).tolist())
    border_ids.update(np.unique(cell_labels[:, 0]).tolist())
    border_ids.update(np.unique(cell_labels[:, -1]).tolist())
    border_ids.discard(0)
    return border_ids


def compute_per_cell_metrics(cell_labels: np.ndarray,
                             ac_props: pd.DataFrame,
                             ac_image: np.ndarray,
                             um_per_pixel: float
                             ) -> pd.DataFrame:
    """Aggregate AC counts and intensities per BMDM cell.

    Returns a DataFrame with one row per cell. AC-channel intensity is
    integrated over the entire cell mask, not just AC objects, so cells with
    sub-threshold but elevated AC signal are still distinguishable from truly
    empty cells.
    """
    cell_props = pd.DataFrame(measure.regionprops_table(
        cell_labels, properties=('label', 'area', 'centroid', 'solidity'),
    ))
    cell_props = cell_props.rename(columns={
        'label': 'cell_id',
        'area': 'cell_area_px',
        'centroid-0': 'cell_centroid_y',
        'centroid-1': 'cell_centroid_x',
        'solidity': 'cell_solidity',
    })
    cell_props['cell_area_um2'] = cell_props['cell_area_px'] * (um_per_pixel ** 2)

    border_ids = flag_border_cells(cell_labels)
    cell_props['is_border_cell'] = cell_props['cell_id'].isin(border_ids)

    if len(ac_props) > 0:
        valid = ac_props[ac_props['cell_id'] > 0]
        class_counts = (valid.groupby(['cell_id', 'size_class'])
                              .size().unstack(fill_value=0))
    else:
        class_counts = pd.DataFrame()

    for cls in ('large_AC', 'small_AC', 'puncta'):
        col = f'n_{cls}'
        if cls in class_counts.columns:
            cell_props[col] = (cell_props['cell_id']
                               .map(class_counts[cls]).fillna(0).astype(int))
        else:
            cell_props[col] = 0
    cell_props['n_AC_total'] = (cell_props['n_large_AC']
                                + cell_props['n_small_AC'])

    intensity_per_cell = _integrate_intensity_per_cell(cell_labels, ac_image)
    cell_props['ac_integrated_intensity'] = (cell_props['cell_id']
                                             .map(intensity_per_cell)
                                             .fillna(0.0))
    cell_props['ac_mean_intensity'] = (cell_props['ac_integrated_intensity']
                                       / cell_props['cell_area_px'].replace(0, np.nan))
    cell_props['ac_mean_intensity'] = cell_props['ac_mean_intensity'].fillna(0.0)

    cell_props['phagocytic_strict'] = cell_props['n_AC_total'] >= 1

    column_order = [
        'cell_id', 'cell_area_px', 'cell_area_um2',
        'cell_centroid_x', 'cell_centroid_y', 'cell_solidity',
        'is_border_cell',
        'n_large_AC', 'n_small_AC', 'n_puncta', 'n_AC_total',
        'ac_integrated_intensity', 'ac_mean_intensity',
        'phagocytic_strict',
    ]
    return cell_props[column_order].copy()


def _integrate_intensity_per_cell(cell_labels: np.ndarray,
                                  intensity_image: np.ndarray) -> dict:
    """Sum intensity within each cell label. Returns dict cell_id -> sum."""
    cell_ids = np.unique(cell_labels)
    cell_ids = cell_ids[cell_ids > 0]
    if len(cell_ids) == 0:
        return {}
    sums = {}
    for cid in cell_ids:
        sums[int(cid)] = float(intensity_image[cell_labels == cid].sum())
    return sums
