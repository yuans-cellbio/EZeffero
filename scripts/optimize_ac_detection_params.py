#!/usr/bin/env python3
"""Optimize EZeffero AC threshold and large-AC size cutoff.

This script reuses existing Cellpose segmentations from a completed EZeffero
run, then sweeps AC detection parameters without rerunning Cellpose.
"""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import detect_AC, io as pio, quantify, segment  # noqa: E402
from src.pipeline import resolve_pixel_size_um  # noqa: E402


@dataclass
class FieldData:
    sample_id: str
    condition: str
    replicate: str
    field: str
    cell_labels: np.ndarray
    ac_image: np.ndarray
    um_per_pixel: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Sweep ac_detection.threshold and "
            "ac_detection.size_classes_um2.small_ac_max to keep the negative "
            "optimizer class low while separating the positive class."
        )
    )
    parser.add_argument("--batch", type=Path, default=Path("ezeffero_full_batch.csv"))
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/ac_param_optimization"))
    parser.add_argument("--perturbation", default="vehicle")
    parser.add_argument("--positive-condition", default="positive",
                        help="Value identifying positive rows in optimizer_condition.")
    parser.add_argument("--negative-condition", default="negative",
                        help="Value identifying negative rows in optimizer_condition.")
    parser.add_argument("--thresholds", default="20:80:10",
                        help="Grid as start:stop:step, or comma-separated values.")
    parser.add_argument("--small-ac-max", default="20:100:10",
                        help="Grid as start:stop:step, or comma-separated values.")
    parser.add_argument("--negative-penalty", type=float, default=1.0,
                        help="Extra penalty on negative-class positivity in the score column.")
    parser.add_argument("--max-negative-pct", default="10.0",
                        help="Hard filter for mean negative percent. Use 'none' to disable.")
    parser.add_argument("--rank-by", choices=("separation", "score"), default="separation",
                        help="Rank passing candidates by separation or penalized score.")
    parser.add_argument("--top-n", type=int, default=20)
    parser.add_argument("--config-out", type=Path, default=None,
                        help="Optional path to write config.yaml with best parameters.")
    return parser.parse_args()


def parse_grid(spec: str) -> list[float]:
    spec = spec.strip()
    if ":" in spec:
        parts = [float(x) for x in spec.split(":")]
        if len(parts) != 3:
            raise ValueError(f"Grid must be start:stop:step, got {spec!r}")
        start, stop, step = parts
        if step <= 0:
            raise ValueError("Grid step must be positive")
        values = []
        x = start
        while x <= stop + (step / 10.0):
            values.append(round(x, 6))
            x += step
        return values
    return [float(x.strip()) for x in spec.split(",") if x.strip()]


def parse_optional_float(value: object) -> float | None:
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in {"", "none", "null", "nan"}:
        return None
    return float(text)


def clean_id(value: object) -> str:
    text = "" if value is None else str(value).strip()
    if text.endswith(".0") and text[:-2].isdigit():
        return text[:-2]
    return text


def load_config(path: Path) -> dict:
    with path.open() as handle:
        return yaml.safe_load(handle)


def expected_seg_path(row: pd.Series) -> Path:
    output_dir = Path(row["output_dir"])
    bf_stem = Path(row["brightfield_path"]).stem
    return output_dir / f"{bf_stem}_gray_seg.npy"


def load_fields(args: argparse.Namespace, config: dict) -> list[FieldData]:
    batch = pd.read_csv(args.batch).fillna("")
    wanted_conditions = {args.positive_condition, args.negative_condition}

    if "optimizer_condition" not in batch.columns:
        raise ValueError(
            "Batch CSV must contain optimizer_condition with explicit positive "
            "and negative class labels."
        )

    subset = batch[
        (batch.get("perturbation", "") == args.perturbation)
        & (batch["optimizer_condition"].isin(wanted_conditions))
    ].copy()
    if subset.empty:
        raise ValueError("No matching rows found in batch CSV")

    fields: list[FieldData] = []
    ac_extraction = config["ac_detection"].get("channel_extraction", "max_rgb")
    bg_cfg = config["ac_detection"].get("background_subtraction", {})
    seg_cfg = config["segmentation"]

    for _, row in subset.iterrows():
        seg_path = Path(row.get("seg_path") or "") if row.get("seg_path") else expected_seg_path(row)
        if not seg_path.exists():
            raise FileNotFoundError(
                f"Missing segmentation for {row['sample_id']}: {seg_path}. "
                "Run EZeffero once before optimizing."
            )

        ac_path = Path(row["ac_path"])
        bf_path = Path(row["brightfield_path"])
        um_per_pixel = resolve_pixel_size_um(bf_path, config)

        labels = segment.load_seg_file(seg_path)
        min_cell_area_px = seg_cfg.get("min_cell_area_um2", 80) / (um_per_pixel ** 2)
        labels = segment.filter_labels(
            labels,
            min_cell_area_px=min_cell_area_px,
            exclude_border=seg_cfg.get("exclude_border_cells", True),
        )
        if seg_cfg.get("dilation", {}).get("enabled", False):
            labels = quantify.expand_cell_labels(
                labels,
                distance_um=seg_cfg["dilation"]["distance_um"],
                um_per_pixel=um_per_pixel,
            )

        ac_image = pio.load_image_grayscale(ac_path, ac_extraction)
        if bg_cfg.get("enabled", True):
            radius_px = int(round(bg_cfg.get("rolling_ball_radius_um", 20) / um_per_pixel))
            ac_image = detect_AC.subtract_background(ac_image, radius_px)

        fields.append(
            FieldData(
                sample_id=str(row["sample_id"]),
                condition=clean_id(row["optimizer_condition"]),
                replicate=clean_id(row["replicate"]),
                field=clean_id(row["field"]),
                cell_labels=labels,
                ac_image=ac_image,
                um_per_pixel=um_per_pixel,
            )
        )

    return fields


def threshold_props(
    field: FieldData,
    threshold: float,
    min_object_area_um2: float,
) -> pd.DataFrame:
    """Detect objects for one field/threshold and assign them to BMDMs."""
    min_object_area_px = min_object_area_um2 / (field.um_per_pixel ** 2)
    _, props = detect_AC.detect_ac_objects(
        field.ac_image,
        threshold=threshold,
        min_object_area_px=min_object_area_px,
    )
    if len(props) == 0:
        props = props.copy()
        props["area_um2"] = pd.Series(dtype=float)
        props["cell_id"] = pd.Series(dtype=int)
        return props

    props = props.copy()
    props["area_um2"] = props["area"] * (field.um_per_pixel ** 2)
    return quantify.assign_objects_to_cells(props, field.cell_labels)


def counts_from_threshold_props(
    field: FieldData,
    props: pd.DataFrame,
    small_ac_max_um2: float,
) -> dict[str, object]:
    """Count BMDMs with at least one large AC for one size cutoff."""
    n_bmdm = int(field.cell_labels.max())
    if n_bmdm == 0 or len(props) == 0:
        n_positive = 0
    else:
        large = props[(props["cell_id"] > 0) & (props["area_um2"] > small_ac_max_um2)]
        n_positive = int(large["cell_id"].nunique())

    return {
        "sample_id": field.sample_id,
        "condition": field.condition,
        "replicate": field.replicate,
        "field": field.field,
        "n_bmdm": n_bmdm,
        "n_bmdm_with_large_ac": n_positive,
    }


def summarize_counts(counts: list[dict[str, object]]) -> pd.DataFrame:
    df = pd.DataFrame(counts)
    if df.empty:
        return df
    grouped = (
        df.groupby(["condition", "replicate"], dropna=False)
        .agg(
            n_bmdm=("n_bmdm", "sum"),
            n_bmdm_with_large_ac=("n_bmdm_with_large_ac", "sum"),
            n_fields=("sample_id", "nunique"),
        )
        .reset_index()
    )
    grouped["pct_bmdm_with_large_ac"] = np.where(
        grouped["n_bmdm"] > 0,
        100.0 * grouped["n_bmdm_with_large_ac"] / grouped["n_bmdm"],
        0.0,
    )
    return grouped


def score_summary(
    summary: pd.DataFrame,
    negative_condition: str,
    positive_condition: str,
    negative_penalty: float,
) -> dict[str, float]:
    negative = summary[summary["condition"] == negative_condition]["pct_bmdm_with_large_ac"]
    positive = summary[summary["condition"] == positive_condition]["pct_bmdm_with_large_ac"]
    metrics = {
        "negative_mean_pct": float(negative.mean()) if len(negative) else np.nan,
        "negative_max_pct": float(negative.max()) if len(negative) else np.nan,
        "positive_mean_pct": float(positive.mean()) if len(positive) else np.nan,
    }

    metrics["separation_pct"] = metrics["positive_mean_pct"] - metrics["negative_mean_pct"]
    metrics["score"] = (
        metrics["separation_pct"] - negative_penalty * metrics["negative_mean_pct"]
    )
    return metrics


def write_optimized_config(config: dict, output_path: Path, threshold: float, small_ac_max: float) -> None:
    out = dict(config)
    out["ac_detection"] = dict(config["ac_detection"])
    out["ac_detection"]["size_classes_um2"] = dict(config["ac_detection"]["size_classes_um2"])
    out["ac_detection"]["threshold"] = float(threshold)
    out["ac_detection"]["size_classes_um2"]["small_ac_max"] = float(small_ac_max)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as handle:
        yaml.safe_dump(out, handle, sort_keys=False)


def main() -> None:
    args = parse_args()
    max_negative_pct = parse_optional_float(args.max_negative_pct)
    config = load_config(args.config)
    thresholds = parse_grid(args.thresholds)
    small_ac_grid = parse_grid(args.small_ac_max)
    fields = load_fields(args, config)
    min_object_area_um2 = float(config["ac_detection"].get("min_object_area_um2", 0.3))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    candidate_rows: list[dict[str, object]] = []
    best_summary = None
    best_counts = None

    for threshold in thresholds:
        print(f"Detecting objects at threshold={threshold} ...", flush=True)
        detected_by_field = [
            threshold_props(
                field,
                threshold=threshold,
                min_object_area_um2=min_object_area_um2,
            )
            for field in fields
        ]
        for small_ac_max in small_ac_grid:
            counts = [
                counts_from_threshold_props(field, props, small_ac_max_um2=small_ac_max)
                for field, props in zip(fields, detected_by_field)
            ]
            summary = summarize_counts(counts)
            metrics = score_summary(
                summary,
                negative_condition=args.negative_condition,
                positive_condition=args.positive_condition,
                negative_penalty=args.negative_penalty,
            )

            row = {
                "threshold": threshold,
                "small_ac_max_um2": small_ac_max,
                **metrics,
                "passes_max_negative": (
                    True if max_negative_pct is None
                    else metrics["negative_mean_pct"] <= max_negative_pct
                ),
            }
            candidate_rows.append(row)

    candidates = pd.DataFrame(candidate_rows)
    rankable = candidates[candidates["passes_max_negative"]].copy()
    if rankable.empty:
        rankable = candidates.copy()
    primary_rank = "separation_pct" if args.rank_by == "separation" else "score"
    secondary_rank = "score" if args.rank_by == "separation" else "separation_pct"
    rankable = rankable.sort_values(
        [primary_rank, secondary_rank, "positive_mean_pct"],
        ascending=[False, False, False],
    )
    best = rankable.iloc[0]

    best_props = [
        threshold_props(field, threshold=float(best["threshold"]), min_object_area_um2=min_object_area_um2)
        for field in fields
    ]
    final_counts = [
        counts_from_threshold_props(field, props, small_ac_max_um2=float(best["small_ac_max_um2"]))
        for field, props in zip(fields, best_props)
    ]
    best_counts = pd.DataFrame(final_counts)
    best_summary = summarize_counts(final_counts)

    candidates = candidates.sort_values(
        ["passes_max_negative", primary_rank, secondary_rank, "positive_mean_pct"],
        ascending=[False, False, False, False],
    )
    candidates.to_csv(args.output_dir / "candidate_scores.csv", index=False)
    rankable.head(args.top_n).to_csv(args.output_dir / "top_candidates.csv", index=False)
    best_summary.to_csv(args.output_dir / "best_condition_by_mouse_summary.csv", index=False)
    best_counts.to_csv(args.output_dir / "best_field_counts.csv", index=False)

    if args.config_out is not None:
        write_optimized_config(
            config,
            args.config_out,
            threshold=float(best["threshold"]),
            small_ac_max=float(best["small_ac_max_um2"]),
        )

    print(f"Loaded fields: {len(fields)}")
    print(f"Wrote: {args.output_dir / 'candidate_scores.csv'}")
    print(f"Wrote: {args.output_dir / 'best_condition_by_mouse_summary.csv'}")
    print(
        "Best parameters: "
        f"threshold={best['threshold']}, "
        f"small_ac_max={best['small_ac_max_um2']} um2, "
        f"score={best['score']:.2f}, "
        f"negative_mean={best['negative_mean_pct']:.2f}%, "
        f"positive_mean={best['positive_mean_pct']:.2f}%"
    )
    if args.config_out is not None:
        print(f"Wrote optimized config: {args.config_out}")


if __name__ == "__main__":
    main()
