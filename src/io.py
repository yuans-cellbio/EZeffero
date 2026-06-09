"""File I/O utilities.

Handles EVOS TIFF metadata parsing, RGB to grayscale collapse for fluorescence
channels, batch CSV validation, and file hashing for provenance.
"""

from __future__ import annotations

import hashlib
import logging
import re
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from PIL import Image

logger = logging.getLogger(__name__)

REQUIRED_BATCH_COLUMNS = {'sample_id', 'ac_path', 'brightfield_path'}
OPTIONAL_BATCH_COLUMNS = {'seg_path', 'output_dir'}
RESERVED_COLUMNS = REQUIRED_BATCH_COLUMNS | OPTIONAL_BATCH_COLUMNS


def extract_pixel_size_um(tif_path: Path) -> Optional[float]:
    """Extract pixel size in micrometers from a TIFF file.

    Resolution order: OME-XML PhysicalSizeX, then TIFF XResolution tag.
    Returns None if neither source is available.
    """
    try:
        with Image.open(tif_path) as im:
            tags = im.tag_v2 if hasattr(im, 'tag_v2') else im.tag

            image_description = tags.get(270)
            if image_description:
                if isinstance(image_description, tuple):
                    image_description = image_description[0]
                um = _parse_ome_pixel_size(str(image_description))
                if um is not None:
                    return um

            x_resolution = tags.get(282)
            resolution_unit = tags.get(296, 2)
            if x_resolution:
                um = _resolution_tag_to_um(x_resolution, resolution_unit)
                if um is not None:
                    return um
    except Exception as exc:
        logger.warning(f"Failed to extract pixel size from {tif_path}: {exc}")

    return None


def _parse_ome_pixel_size(xml_text: str) -> Optional[float]:
    """Pull PhysicalSizeX from an OME-XML string."""
    match = re.search(r'PhysicalSizeX="([0-9.eE+-]+)"', xml_text)
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            return None
    return None


def _resolution_tag_to_um(x_resolution, resolution_unit: int) -> Optional[float]:
    """Convert TIFF XResolution + ResolutionUnit to um/pixel.

    ResolutionUnit codes: 2 = inches, 3 = centimeters.
    """
    try:
        if isinstance(x_resolution, tuple) and len(x_resolution) == 1:
            x_resolution = x_resolution[0]
        if hasattr(x_resolution, 'numerator'):
            value = float(x_resolution.numerator) / float(x_resolution.denominator)
        elif isinstance(x_resolution, tuple) and len(x_resolution) == 2:
            value = float(x_resolution[0]) / float(x_resolution[1])
        else:
            value = float(x_resolution)
    except Exception:
        return None

    if value <= 0:
        return None

    if resolution_unit == 2:
        return 25400.0 / value
    if resolution_unit == 3:
        return 10000.0 / value
    return None


def load_image_grayscale(tif_path: Path,
                         channel_extraction: str = 'max_rgb') -> np.ndarray:
    """Load a TIFF and return a 2D grayscale array.

    For RGB pseudo-color fluorescence images (EVOS default), the per-pixel
    maximum across channels is the right collapse: only one of R/G/B carries
    fluorescence signal, so max preserves it without dilution. Brightfield is
    different: EVOS BF often has a strongly dominant red channel (the LED
    color) that may be saturated, so 'luminance' or 'mean_rgb' preserve the
    cell-edge contrast that 'max_rgb' destroys.

    Options:
        max_rgb   : per-pixel max across R, G, B (use for fluorescence).
        mean_rgb  : per-pixel mean across R, G, B.
        sum_rgb   : per-pixel sum across R, G, B.
        luminance : ITU-R BT.601 weighted sum 0.299R + 0.587G + 0.114B,
                    via PIL convert('L'). Recommended for brightfield.
        r/g/b     : pick a single channel.

    Already-grayscale TIFFs pass through unchanged.
    """
    if channel_extraction == 'luminance':
        with Image.open(tif_path) as im:
            return np.array(im.convert('L')).astype(np.float32)

    arr = np.array(Image.open(tif_path))

    if arr.ndim == 2:
        return arr.astype(np.float32)

    if arr.ndim == 3 and arr.shape[-1] >= 3:
        if channel_extraction == 'max_rgb':
            return arr[..., :3].max(axis=-1).astype(np.float32)
        if channel_extraction == 'mean_rgb':
            return arr[..., :3].astype(np.float32).mean(axis=-1)
        if channel_extraction == 'sum_rgb':
            return arr[..., :3].astype(np.float32).sum(axis=-1)
        if channel_extraction in {'r', 'g', 'b'}:
            idx = {'r': 0, 'g': 1, 'b': 2}[channel_extraction]
            return arr[..., idx].astype(np.float32)
        raise ValueError(f"Unknown channel_extraction: {channel_extraction}")

    raise ValueError(f"Unsupported image shape {arr.shape} for {tif_path}")


def compute_file_sha256(path: Path, chunk_size: int = 1 << 20) -> str:
    """Compute sha256 of a file. Used for provenance tracking."""
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(chunk_size), b''):
            h.update(chunk)
    return h.hexdigest()


def save_grayscale_tif(image: np.ndarray, output_path: Path) -> None:
    """Save a 2D float array as an 8-bit grayscale TIFF.

    Used to write a Cellpose-GUI-friendly companion image alongside the
    seg.npy file. Values are clipped to [0, 255] and cast to uint8 to match
    what the GUI expects after ImageJ-style 8-bit conversion. The matching
    seg file (named `<this_stem>_seg.npy`) is auto-detected by the GUI when
    the user opens this TIFF.
    """
    if image.ndim != 2:
        raise ValueError(f"Expected 2D image, got shape {image.shape}")
    arr = np.clip(image, 0, 255).astype(np.uint8)
    Image.fromarray(arr, mode='L').save(output_path)
    logger.info(f"Saved grayscale brightfield: {output_path}")


def parse_batch_csv(csv_path: Path) -> pd.DataFrame:
    """Load and validate a batch input CSV.

    Validates required columns, unique sample_ids, file existence (paths only,
    not seg_path which is allowed to be missing if user wants Cellpose to run),
    and same-folder constraint per row. Optional metadata columns are preserved.
    """
    df = pd.read_csv(csv_path)

    missing = REQUIRED_BATCH_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Batch CSV missing required columns: {sorted(missing)}")

    for col in OPTIONAL_BATCH_COLUMNS:
        if col not in df.columns:
            df[col] = ''
        df[col] = df[col].fillna('').astype(str)

    df['sample_id'] = df['sample_id'].astype(str)
    duplicates = df['sample_id'][df['sample_id'].duplicated()].tolist()
    if duplicates:
        raise ValueError(f"Duplicate sample_id values: {duplicates}")

    for col in ('ac_path', 'brightfield_path'):
        df[col] = df[col].astype(str)

    return df


def validate_row_paths(row: pd.Series) -> None:
    """Validate that AC and BF inputs share a parent directory.

    The same-folder rule governs the default output location. Rows that violate
    it must set output_dir explicitly. Raises ValueError if violated and
    output_dir is empty.
    """
    bf_path = Path(row['brightfield_path'])
    ac_path = Path(row['ac_path'])
    output_dir_raw = row.get('output_dir', '') or ''

    for label, p in (('brightfield_path', bf_path), ('ac_path', ac_path)):
        if not p.exists():
            raise FileNotFoundError(f"{label} does not exist: {p}")

    if output_dir_raw:
        return

    parents = {bf_path.parent, ac_path.parent}
    if len(parents) > 1:
        raise ValueError(
            f"Input files for sample_id={row['sample_id']} live in different "
            f"folders ({sorted(str(p) for p in parents)}); set output_dir "
            f"explicitly to allow this."
        )


def get_pass_through_metadata(row: pd.Series) -> dict:
    """Extract metadata columns from a batch row for pass-through to per-cell CSV."""
    return {k: row[k] for k in row.index if k not in RESERVED_COLUMNS}
