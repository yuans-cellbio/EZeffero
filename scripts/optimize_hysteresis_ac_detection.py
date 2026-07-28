#!/usr/bin/env python3
"""Optimize hysteresis thresholds and the large-AC area boundary.

The optimizer reuses saved Cellpose masks, requires a bright high-threshold
seed for each object, and restores that object's connected extent at the low
threshold. Candidates are screened by Healthy-control positivity and object
area plausibility before ranking AC-versus-Healthy separation.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import detect_AC, quantify  # noqa: E402

from optimize_ac_detection_params import (  # noqa: E402
    FieldData,
    load_config,
    load_fields,
    parse_grid,
    parse_optional_float,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Sweep hysteresis low/high thresholds and the large-AC area "
            "boundary while reusing saved Cellpose masks."
        )
    )
    parser.add_argument("--batch", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--perturbation", default="green_ac_plate")
    parser.add_argument("--positive-condition", default="positive")
    parser.add_argument("--negative-condition", default="negative")
    parser.add_argument("--low-thresholds", default="125:225:25")
    parser.add_argument("--high-thresholds", default="200:350:25")
    parser.add_argument("--small-ac-max", default="60")
    parser.add_argument("--max-negative-large-pct", default="2.5")
    parser.add_argument("--max-negative-any-pct", default="3.5")
    parser.add_argument("--min-median-large-area", default="80")
    parser.add_argument("--max-median-large-area", default="250")
    parser.add_argument("--max-large-p90-area", default="500")
    parser.add_argument("--giant-object-area", type=float, default=500)
    parser.add_argument("--max-giant-object-pct", default="5")
    parser.add_argument("--top-n", type=int, default=30)
    parser.add_argument("--config-out", type=Path, default=None)
    return parser.parse_args()


def hysteresis_props(
    field: FieldData,
    low_threshold: float,
    high_threshold: float,
    min_object_area_um2: float,
) -> pd.DataFrame:
    min_object_area_px = min_object_area_um2 / (field.um_per_pixel**2)
    _, props = detect_AC.detect_ac_objects_hysteresis(
        field.ac_image,
        low_threshold=low_threshold,
        high_threshold=high_threshold,
        min_object_area_px=min_object_area_px,
    )
    if len(props) == 0:
        out = props.copy()
        out["area_um2"] = pd.Series(dtype=float)
        out["cell_id"] = pd.Series(dtype=int)
        return out
    props = props.copy()
    props["area_um2"] = props["area"] * (field.um_per_pixel**2)
    return quantify.assign_objects_to_cells(props, field.cell_labels)


def field_counts(
    field: FieldData,
    props: pd.DataFrame,
    puncta_max_um2: float,
    small_ac_max_um2: float,
) -> dict[str, object]:
    assigned = props[props["cell_id"] > 0]
    large = assigned[assigned["area_um2"] > small_ac_max_um2]
    any_ac = assigned[assigned["area_um2"] > puncta_max_um2]
    return {
        "sample_id": field.sample_id,
        "condition": field.condition,
        "replicate": field.replicate,
        "field": field.field,
        "n_bmdm": int(field.cell_labels.max()),
        "n_large_positive": int(large["cell_id"].nunique()),
        "n_any_positive": int(any_ac["cell_id"].nunique()),
        "n_large_objects": len(large),
        "n_any_objects": len(any_ac),
    }


def summarize_mouse_counts(counts: list[dict[str, object]]) -> pd.DataFrame:
    summary = (
        pd.DataFrame(counts)
        .groupby(["condition", "replicate"], dropna=False)
        .agg(
            n_fields=("sample_id", "nunique"),
            n_bmdm=("n_bmdm", "sum"),
            n_large_positive=("n_large_positive", "sum"),
            n_any_positive=("n_any_positive", "sum"),
            n_large_objects=("n_large_objects", "sum"),
            n_any_objects=("n_any_objects", "sum"),
        )
        .reset_index()
    )
    summary["pct_large_positive"] = np.where(
        summary["n_bmdm"] > 0,
        100 * summary["n_large_positive"] / summary["n_bmdm"],
        0,
    )
    summary["pct_any_positive"] = np.where(
        summary["n_bmdm"] > 0,
        100 * summary["n_any_positive"] / summary["n_bmdm"],
        0,
    )
    return summary


def condition_mean(
    summary: pd.DataFrame,
    condition: str,
    column: str,
) -> float:
    values = summary.loc[summary["condition"] == condition, column]
    return float(values.mean()) if len(values) else np.nan


def score_candidate(
    fields: list[FieldData],
    detected_by_field: list[pd.DataFrame],
    puncta_max_um2: float,
    small_ac_max_um2: float,
    args: argparse.Namespace,
) -> tuple[dict[str, object], pd.DataFrame, pd.DataFrame]:
    counts = [
        field_counts(
            field,
            props,
            puncta_max_um2=puncta_max_um2,
            small_ac_max_um2=small_ac_max_um2,
        )
        for field, props in zip(fields, detected_by_field)
    ]
    mouse_summary = summarize_mouse_counts(counts)

    negative_large = condition_mean(
        mouse_summary, args.negative_condition, "pct_large_positive"
    )
    negative_any = condition_mean(
        mouse_summary, args.negative_condition, "pct_any_positive"
    )
    positive_large = condition_mean(
        mouse_summary, args.positive_condition, "pct_large_positive"
    )
    positive_any = condition_mean(
        mouse_summary, args.positive_condition, "pct_any_positive"
    )

    positive_large_areas = np.concatenate(
        [
            props.loc[
                (props["cell_id"] > 0)
                & (props["area_um2"] > small_ac_max_um2),
                "area_um2",
            ].to_numpy()
            for field, props in zip(fields, detected_by_field)
            if field.condition == args.positive_condition and len(props) > 0
        ]
        or [np.array([], dtype=float)]
    )
    median_large_area = (
        float(np.median(positive_large_areas))
        if len(positive_large_areas)
        else np.nan
    )
    p90_large_area = (
        float(np.quantile(positive_large_areas, 0.90))
        if len(positive_large_areas)
        else np.nan
    )
    giant_object_pct = (
        100
        * float(np.mean(positive_large_areas > args.giant_object_area))
        if len(positive_large_areas)
        else np.nan
    )

    max_negative_large = parse_optional_float(args.max_negative_large_pct)
    max_negative_any = parse_optional_float(args.max_negative_any_pct)
    min_median_large = parse_optional_float(args.min_median_large_area)
    max_median_large = parse_optional_float(args.max_median_large_area)
    max_large_p90 = parse_optional_float(args.max_large_p90_area)
    max_giant_pct = parse_optional_float(args.max_giant_object_pct)

    checks = {
        "passes_negative_large": (
            True
            if max_negative_large is None
            else negative_large <= max_negative_large
        ),
        "passes_negative_any": (
            True if max_negative_any is None else negative_any <= max_negative_any
        ),
        "passes_median_large_min": (
            True
            if min_median_large is None
            else median_large_area >= min_median_large
        ),
        "passes_median_large_max": (
            True
            if max_median_large is None
            else median_large_area <= max_median_large
        ),
        "passes_large_p90": (
            True if max_large_p90 is None else p90_large_area <= max_large_p90
        ),
        "passes_giant_object_pct": (
            True if max_giant_pct is None else giant_object_pct <= max_giant_pct
        ),
    }
    metrics = {
        "negative_large_pct": negative_large,
        "positive_large_pct": positive_large,
        "large_separation_pct": positive_large - negative_large,
        "negative_any_pct": negative_any,
        "positive_any_pct": positive_any,
        "any_separation_pct": positive_any - negative_any,
        "positive_large_object_count": len(positive_large_areas),
        "positive_median_large_area_um2": median_large_area,
        "positive_p90_large_area_um2": p90_large_area,
        "positive_giant_object_pct": giant_object_pct,
        **checks,
    }
    metrics["passes_all_constraints"] = all(checks.values())
    return metrics, pd.DataFrame(counts), mouse_summary


def assigned_object_table(
    fields: list[FieldData],
    detected_by_field: list[pd.DataFrame],
) -> pd.DataFrame:
    tables = []
    for field, props in zip(fields, detected_by_field):
        if len(props) == 0:
            continue
        table = props[props["cell_id"] > 0].copy()
        table.insert(0, "field", field.field)
        table.insert(0, "replicate", field.replicate)
        table.insert(0, "condition", field.condition)
        table.insert(0, "sample_id", field.sample_id)
        tables.append(table)
    return pd.concat(tables, ignore_index=True) if tables else pd.DataFrame()


def write_optimized_config(
    config: dict,
    output_path: Path,
    low_threshold: float,
    high_threshold: float,
    small_ac_max_um2: float,
) -> None:
    output = dict(config)
    output["ac_detection"] = dict(config["ac_detection"])
    output["ac_detection"]["size_classes_um2"] = dict(
        config["ac_detection"]["size_classes_um2"]
    )
    output["ac_detection"]["threshold_strategy"] = "hysteresis"
    output["ac_detection"]["threshold"] = float(high_threshold)
    output["ac_detection"]["hysteresis"] = {
        "low_threshold": float(low_threshold),
        "high_threshold": float(high_threshold),
    }
    output["ac_detection"]["size_classes_um2"]["small_ac_max"] = float(
        small_ac_max_um2
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as handle:
        yaml.safe_dump(output, handle, sort_keys=False)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    fields = load_fields(args, config)
    low_thresholds = parse_grid(args.low_thresholds)
    high_thresholds = parse_grid(args.high_thresholds)
    small_ac_grid = parse_grid(args.small_ac_max)
    puncta_max_um2 = float(
        config["ac_detection"]["size_classes_um2"]["puncta_max"]
    )
    min_object_area_um2 = float(
        config["ac_detection"].get("min_object_area_um2", 0.3)
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    candidates: list[dict[str, object]] = []
    cache: dict[tuple[float, float], list[pd.DataFrame]] = {}

    for low_threshold in low_thresholds:
        for high_threshold in high_thresholds:
            if low_threshold > high_threshold:
                continue
            print(
                f"Detecting hysteresis objects low={low_threshold:g}, "
                f"high={high_threshold:g}",
                flush=True,
            )
            detected_by_field = [
                hysteresis_props(
                    field,
                    low_threshold=low_threshold,
                    high_threshold=high_threshold,
                    min_object_area_um2=min_object_area_um2,
                )
                for field in fields
            ]
            cache[(low_threshold, high_threshold)] = detected_by_field
            for small_ac_max_um2 in small_ac_grid:
                metrics, _, _ = score_candidate(
                    fields,
                    detected_by_field,
                    puncta_max_um2=puncta_max_um2,
                    small_ac_max_um2=small_ac_max_um2,
                    args=args,
                )
                candidates.append(
                    {
                        "low_threshold": low_threshold,
                        "high_threshold": high_threshold,
                        "small_ac_max_um2": small_ac_max_um2,
                        **metrics,
                    }
                )

    candidate_table = pd.DataFrame(candidates)
    passing = candidate_table[candidate_table["passes_all_constraints"]].copy()
    if passing.empty:
        passing = candidate_table.copy()
        print(
            "WARNING: no candidate passed every constraint; ranking all candidates",
            flush=True,
        )
    passing = passing.sort_values(
        [
            "large_separation_pct",
            "any_separation_pct",
            "positive_large_pct",
            "negative_large_pct",
        ],
        ascending=[False, False, False, True],
    )
    best = passing.iloc[0]
    best_key = (float(best["low_threshold"]), float(best["high_threshold"]))
    best_props = cache[best_key]
    _, best_field_counts, best_mouse_summary = score_candidate(
        fields,
        best_props,
        puncta_max_um2=puncta_max_um2,
        small_ac_max_um2=float(best["small_ac_max_um2"]),
        args=args,
    )

    candidate_table.sort_values(
        ["passes_all_constraints", "large_separation_pct", "any_separation_pct"],
        ascending=[False, False, False],
    ).to_csv(args.output_dir / "candidate_scores.csv", index=False)
    passing.head(args.top_n).to_csv(
        args.output_dir / "top_candidates.csv", index=False
    )
    best_field_counts.to_csv(
        args.output_dir / "best_field_counts.csv", index=False
    )
    best_mouse_summary.to_csv(
        args.output_dir / "best_condition_by_mouse_summary.csv", index=False
    )
    assigned_object_table(fields, best_props).to_csv(
        args.output_dir / "best_assigned_objects.csv", index=False
    )

    if args.config_out is not None:
        write_optimized_config(
            config,
            output_path=args.config_out,
            low_threshold=float(best["low_threshold"]),
            high_threshold=float(best["high_threshold"]),
            small_ac_max_um2=float(best["small_ac_max_um2"]),
        )

    print(f"Loaded fields: {len(fields)}", flush=True)
    print(
        "Best parameters: "
        f"low={best['low_threshold']}, high={best['high_threshold']}, "
        f"small_ac_max={best['small_ac_max_um2']} um2; "
        f"Healthy large={best['negative_large_pct']:.2f}%, "
        f"AC large={best['positive_large_pct']:.2f}%, "
        f"median large area={best['positive_median_large_area_um2']:.1f} um2",
        flush=True,
    )
    if args.config_out is not None:
        print(f"Wrote optimized config: {args.config_out}", flush=True)


if __name__ == "__main__":
    main()
