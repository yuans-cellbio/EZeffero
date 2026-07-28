"""Apoptotic cell (AC) detection in the AC fluorescence channel.

Pipeline: background subtraction (top-hat with disk SE) -> single or
hysteresis thresholding -> connected components -> size-based classification
(puncta / small AC / large AC).
"""

from __future__ import annotations

import logging
from typing import Tuple

import cv2
import numpy as np
import pandas as pd
from skimage import measure

logger = logging.getLogger(__name__)


def subtract_background(image: np.ndarray, radius_px: int) -> np.ndarray:
    """Top-hat background subtraction with a disk structuring element.

    Conceptually equivalent to ImageJ's "Subtract Background" rolling-ball
    operation: a grayscale opening with a disk SE, then subtraction. Uses
    OpenCV's morphologyEx with MORPH_ELLIPSE so the inner loop is the
    SIMD-vectorized C++ implementation; produces output that matches
    skimage's white_tophat with a disk SE to within ~0.01 grayscale units
    on typical EVOS BMDM fields, but is roughly 100x faster.

    OpenCV's morphology runs natively on uint8/uint16/float32. We pass float32
    through unchanged so the returned dtype matches the rest of the pipeline.
    The output is a non-negative image with low-frequency illumination removed.
    """
    if radius_px <= 0:
        return image
    ksize = 2 * int(radius_px) + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    img32 = image.astype(np.float32, copy=False)
    return cv2.morphologyEx(img32, cv2.MORPH_TOPHAT, kernel)


PROP_COLUMNS = [
    'label', 'area', 'centroid_y', 'centroid_x',
    'mean_intensity', 'max_intensity', 'integrated_intensity',
]


def _empty_props() -> pd.DataFrame:
    return pd.DataFrame(columns=PROP_COLUMNS)


def _measure_object_mask(image: np.ndarray,
                         mask: np.ndarray,
                         min_object_area_px: float
                         ) -> Tuple[np.ndarray, pd.DataFrame]:
    """Label, area-filter, and measure a binary AC object mask."""
    labels = measure.label(mask, connectivity=2)

    if labels.max() == 0:
        return labels.astype(np.int32), _empty_props()

    if min_object_area_px > 1:
        sizes = np.bincount(labels.ravel())
        sizes[0] = 0
        keep = sizes >= min_object_area_px
        labels = np.where(keep[labels], labels, 0)
        if labels.max() == 0:
            return labels.astype(np.int32), _empty_props()

    props_table = measure.regionprops_table(
        labels, intensity_image=image,
        properties=('label', 'area', 'centroid', 'mean_intensity', 'max_intensity'),
    )
    props = pd.DataFrame(props_table)
    props = props.rename(columns={
        'centroid-0': 'centroid_y',
        'centroid-1': 'centroid_x',
    })
    props['integrated_intensity'] = props['mean_intensity'] * props['area']
    return labels.astype(np.int32), props


def detect_ac_objects(image: np.ndarray,
                      threshold: float,
                      min_object_area_px: float
                      ) -> Tuple[np.ndarray, pd.DataFrame]:
    """Apply one global threshold and label connected AC objects.

    Returns:
        labels : int label image, 0 = background.
        props  : DataFrame with one row per object (label, area, centroids,
                 mean_intensity, max_intensity, integrated_intensity).
    """
    labels, props = _measure_object_mask(
        image,
        mask=image > threshold,
        min_object_area_px=min_object_area_px,
    )
    logger.info(f"Detected {len(props)} AC objects above threshold {threshold:.2f}")
    return labels, props


def detect_ac_objects_hysteresis(image: np.ndarray,
                                 low_threshold: float,
                                 high_threshold: float,
                                 min_object_area_px: float
                                 ) -> Tuple[np.ndarray, pd.DataFrame]:
    """Keep low-threshold components only when they contain a high-threshold seed.

    The high threshold requires a bright core, while the low threshold restores
    the connected dimmer extent of that object. Eight-connected components are
    used, matching the single-threshold detector. Low-only components without a
    high seed are discarded.
    """
    low_threshold = float(low_threshold)
    high_threshold = float(high_threshold)
    if not np.isfinite(low_threshold) or not np.isfinite(high_threshold):
        raise ValueError("Hysteresis thresholds must be finite")
    if low_threshold > high_threshold:
        raise ValueError(
            "Hysteresis low_threshold must be less than or equal to high_threshold"
        )

    low_labels = measure.label(image > low_threshold, connectivity=2)
    if low_labels.max() == 0:
        return low_labels.astype(np.int32), _empty_props()

    seeded_labels = np.unique(low_labels[image > high_threshold])
    seeded_labels = seeded_labels[seeded_labels > 0]
    if len(seeded_labels) == 0:
        return np.zeros_like(low_labels, dtype=np.int32), _empty_props()

    keep_labels = np.zeros(int(low_labels.max()) + 1, dtype=bool)
    keep_labels[seeded_labels] = True
    hysteresis_mask = keep_labels[low_labels]
    labels, props = _measure_object_mask(
        image,
        mask=hysteresis_mask,
        min_object_area_px=min_object_area_px,
    )
    logger.info(
        "Detected %d AC objects with hysteresis thresholds low=%.2f, high=%.2f",
        len(props),
        low_threshold,
        high_threshold,
    )
    return labels, props


def classify_ac_by_size(props: pd.DataFrame,
                        um_per_pixel: float,
                        puncta_max_um2: float,
                        small_ac_max_um2: float) -> pd.DataFrame:
    """Add a 'size_class' column to the AC properties DataFrame.

    Classes:
        puncta    : area_um2 <= puncta_max_um2
        small_AC  : puncta_max_um2 < area_um2 <= small_ac_max_um2
        large_AC  : area_um2 > small_ac_max_um2

    The puncta class is reported separately because its biological identity is
    mixed (degraded internalized membrane plus possible dye transfer); it
    should not by default count toward phagocytic index.
    """
    if len(props) == 0:
        out = props.copy()
        out['area_um2'] = []
        out['size_class'] = []
        return out

    out = props.copy()
    px2_per_um2 = 1.0 / (um_per_pixel ** 2)
    out['area_um2'] = out['area'] / px2_per_um2

    conditions = [
        out['area_um2'] <= puncta_max_um2,
        out['area_um2'] <= small_ac_max_um2,
    ]
    choices = ['puncta', 'small_AC']
    out['size_class'] = np.select(conditions, choices, default='large_AC')

    counts = out['size_class'].value_counts().to_dict()
    logger.info(f"AC size class counts: {counts}")
    return out
