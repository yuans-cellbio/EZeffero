#!/usr/bin/env python3
"""Summarize BMDMs with large ACs from EZeffero per-cell CSV outputs."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path


DEFAULT_GROUP_COLS = ("condition", "replicate")
REQUIRED_COLUMNS = {"sample_id", "field", "n_large_AC"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Summarize the percentage of BMDMs with at least one large AC "
            "from EZeffero *_per_cell.csv files."
        )
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("outputs/ezeffero_full_run"),
        help="Directory to search recursively for *_per_cell.csv files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/large_ac_by_mouse_summary.csv"),
        help="Path for the summary CSV.",
    )
    parser.add_argument(
        "--group-cols",
        default=",".join(DEFAULT_GROUP_COLS),
        help=(
            "Comma-separated columns to group by. Default: condition,replicate. "
            "Use replicate to group only by mouse ID."
        ),
    )
    return parser.parse_args()


def clean_value(value: str) -> str:
    """Normalize blank values and pandas-style integer floats like 4122.0."""
    value = (value or "").strip()
    if value.endswith(".0"):
        left = value[:-2]
        if left.isdigit():
            return left
    return value


def numeric_value(value: str, *, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def read_per_cell_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        columns = reader.fieldnames or []
        missing = REQUIRED_COLUMNS - set(columns)
        if missing:
            raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
        rows = list(reader)
    return columns, rows


def summarize(input_dir: Path, group_cols: list[str]) -> list[dict[str, str]]:
    per_cell_paths = sorted(input_dir.rglob("*_per_cell.csv"))
    if not per_cell_paths:
        raise FileNotFoundError(f"No *_per_cell.csv files found under {input_dir}")

    groups: dict[tuple[str, ...], dict[str, object]] = {}
    missing_group_cols: set[str] = set()

    for path in per_cell_paths:
        columns, rows = read_per_cell_csv(path)
        missing_group_cols.update(col for col in group_cols if col not in columns)

        for row in rows:
            key = tuple(clean_value(row.get(col, "")) for col in group_cols)
            if key not in groups:
                groups[key] = {
                    "n_bmdm": 0,
                    "n_bmdm_with_large_ac": 0,
                    "fields": set(),
                    "sample_ids": set(),
                    "source_files": set(),
                }

            group = groups[key]
            group["n_bmdm"] += 1
            if numeric_value(row.get("n_large_AC", "0")) >= 1:
                group["n_bmdm_with_large_ac"] += 1
            group["fields"].add(clean_value(row.get("field", "")))
            group["sample_ids"].add(clean_value(row.get("sample_id", "")))
            group["source_files"].add(str(path))

    if missing_group_cols:
        raise ValueError(
            "Requested group columns were not found in one or more files: "
            f"{sorted(missing_group_cols)}"
        )

    summary_rows: list[dict[str, str]] = []
    for key, group in sorted(groups.items()):
        n_bmdm = int(group["n_bmdm"])
        n_large = int(group["n_bmdm_with_large_ac"])
        pct = (100.0 * n_large / n_bmdm) if n_bmdm else 0.0

        out_row = {col: key[i] for i, col in enumerate(group_cols)}
        out_row.update(
            {
                "n_bmdm": str(n_bmdm),
                "n_bmdm_with_large_ac": str(n_large),
                "pct_bmdm_with_large_ac": f"{pct:.2f}",
                "n_fields": str(len(group["sample_ids"])),
                "sample_ids": ";".join(sorted(group["sample_ids"])),
                "n_source_files": str(len(group["source_files"])),
            }
        )
        summary_rows.append(out_row)

    return summary_rows


def main() -> None:
    args = parse_args()
    group_cols = [col.strip() for col in args.group_cols.split(",") if col.strip()]
    if not group_cols:
        raise ValueError("--group-cols must contain at least one column")

    rows = summarize(args.input_dir, group_cols)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        *group_cols,
        "n_bmdm",
        "n_bmdm_with_large_ac",
        "pct_bmdm_with_large_ac",
        "n_fields",
        "sample_ids",
        "n_source_files",
    ]
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {args.output} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
