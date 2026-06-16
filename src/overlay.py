"""QC overlay rendering: BF, BF+AC merge, AC+masks, full annotation.

Four-panel layout:
    [1, 1] Brightfield (grayscale)
    [1, 2] BF + AC merge (max-blend) + red mask outlines
    [2, 1] AC channel on black + red mask outlines
    [2, 2] BF + AC merge + red mask outlines + AC centroids by size class

Designed for visual sanity-checking; not a publication figure.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import binary_dilation
from skimage import segmentation

logger = logging.getLogger(__name__)


def _stretch(image: np.ndarray, lo: float = 1.0, hi: float = 99.5) -> np.ndarray:
    """Percentile-clip an image to [0, 1] for display."""
    a, b = np.percentile(image, [lo, hi])
    return np.clip((image - a) / max(b - a, 1e-9), 0, 1)


def _scale_fixed(image: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
    """Scale an image to [0, 1] using absolute display limits."""
    return np.clip((image - vmin) / max(vmax - vmin, 1e-9), 0, 1)


def _scatter_centroids(ax, ac_props: pd.DataFrame) -> None:
    """Draw size-class-colored centroids on ax. Sizes tuned to leave
    underlying AC signal visible at typical EVOS field magnifications.
    """
    if 'size_class' not in ac_props.columns or len(ac_props) == 0:
        return
    styles = {
        'large_AC': dict(color='red',    marker='+', s=20, label='large AC'),
        'small_AC': dict(color='orange', marker='x', s=12, label='small AC'),
        'puncta':   dict(color='gold',   marker='.', s=3,  label='puncta'),
    }
    for cls, style in styles.items():
        sub = ac_props[ac_props['size_class'] == cls]
        if len(sub):
            ax.scatter(sub['centroid_x'], sub['centroid_y'], **style)
    ax.legend(loc='upper right', fontsize=8, framealpha=0.7)


def make_qc_overlay(brightfield: np.ndarray,
                    ac_image: np.ndarray,
                    cell_labels: np.ndarray,
                    ac_props: pd.DataFrame,
                    output_path: Path,
                    title: str,
                    dpi: int = 120,
                    ac_display_limits: Optional[tuple[float, float]] = None) -> None:
    """Render the QC overlay PNG.

    Mask boundaries are 1-pixel dilated for readable line width at PNG
    rasterization; the merged display uses per-channel max blending which
    matches ImageJ's "Merge Channels" behavior for fluorescence + transmitted
    light composites.
    """
    h, w = brightfield.shape
    bf_n = _stretch(brightfield, 1, 99).astype(np.float32)
    if ac_display_limits is None:
        ac_n = _stretch(ac_image, 5, 99.95).astype(np.float32)
    else:
        ac_n = _scale_fixed(ac_image, *ac_display_limits).astype(np.float32)

    bf_rgb = np.stack([bf_n, bf_n, bf_n], axis=-1)

    ac_only_rgb = np.zeros((h, w, 3), dtype=np.float32)
    ac_only_rgb[..., 1] = ac_n

    merged = bf_rgb.copy()
    merged[..., 1] = np.maximum(merged[..., 1], ac_n)

    boundaries = segmentation.find_boundaries(cell_labels, mode='outer')
    boundaries_thick = binary_dilation(boundaries, iterations=1)
    boundary_red = np.zeros((h, w, 4), dtype=np.float32)
    boundary_red[boundaries_thick, 0] = 1.0
    boundary_red[boundaries_thick, 3] = 0.85

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    axes[0, 0].imshow(bf_n, cmap='gray')
    axes[0, 0].set_title('Brightfield')
    axes[0, 0].axis('off')

    axes[0, 1].imshow(merged)
    axes[0, 1].imshow(boundary_red)
    axes[0, 1].set_title(f'BF + AC merge + cell masks: n={int(cell_labels.max())}')
    axes[0, 1].axis('off')

    axes[1, 0].imshow(ac_only_rgb)
    axes[1, 0].imshow(boundary_red)
    axes[1, 0].set_title('AC channel (background-subtracted) + cell masks')
    axes[1, 0].axis('off')

    axes[1, 1].imshow(merged)
    axes[1, 1].imshow(boundary_red)
    _scatter_centroids(axes[1, 1], ac_props)
    axes[1, 1].set_title('BF + AC merge + cell masks + centroids')
    axes[1, 1].axis('off')

    fig.suptitle(title, y=0.995)
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"Saved QC overlay: {output_path}")
