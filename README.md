# Efferocytosis quantification pipeline

Quantify apoptotic cell (AC) association with bone marrow-derived macrophages
(BMDMs) from EVOS fluorescence images. Produces one row per BMDM with AC
counts split by size class, AC-channel intensity, and full provenance.

## What this pipeline measures

The output measures **AC association** (surface-bound + internalized
combined). Distinguishing surface-bound from engulfed requires additional
assay components (pHrodo, outside-only counterstain, or confocal z-stacks)
that are not part of this pipeline. Treat phagocytic_strict accordingly.

## Installation

Requires **Python 3.10 or 3.11**. The binding constraint is Cellpose (3.x
does not install on Python above 3.11; 3.10 is the maintainers' recommended
version, with 3.9 and 3.11 also working). The pipeline code itself only
needs 3.7+ but Cellpose pins the floor higher.

```bash
conda create -n efferocytosis python=3.10
conda activate efferocytosis
pip install -r requirements.txt
```

GPU is auto-used if CUDA is available; otherwise the pipeline runs on CPU
(slower but functional). Cellpose model weights are downloaded on first use.

## Quickstart

1. Copy `config.yaml` next to your data and edit `microscope.um_per_pixel`
   only if your images are not standard EVOS OME-TIFFs.
2. Run the threshold calibration helper on your no-AC control images and
   write the chosen threshold into `ac_detection.threshold`:
   ```bash
   python scripts/calibrate_threshold.py \
       --neg-control /data/plate1/no_AC_well/*.TIF \
       --background-subtraction \
       --output threshold_report.txt
   ```
3. Run the pipeline.

### Single-field

```bash
python scripts/run_efferocytosis.py --config config.yaml single \
    --ac /data/plate1/CM_B05f00d0.TIF \
    --brightfield /data/plate1/CM_B05f00d4.TIF \
    --output-prefix B05f00
```

### Batch

```bash
python scripts/run_efferocytosis.py --config config.yaml batch \
    --input plate1_batch.csv \
    --summary-out plate1_summary.json
```

See `batch_template.csv` for the input CSV format.

### AC parameter optimization

`scripts/optimize_ac_detection_params.py` jointly sweeps
`ac_detection.threshold` and
`ac_detection.size_classes_um2.small_ac_max`. It scores each candidate by
the percentage of BMDMs containing at least one large AC, retaining
parameter sets below a configurable healthy-control ceiling.

Run the normal batch pipeline first. The optimizer reuses the resulting
Cellpose masks and does not rerun segmentation:

```bash
python scripts/run_efferocytosis.py --config config.yaml batch \
    --input optimization_batch.csv
```

The optimization batch CSV uses the normal batch columns plus:

| Column | Purpose |
|---|---|
| `perturbation` | optional second stratum selected by `--perturbation` |
| `optimizer_condition` | explicit optimizer class; defaults are `negative` and `positive` |
| `replicate` | biological replicate used when aggregating fields |

Each row must either provide `seg_path` or have a mask at
`<output_dir>/<brightfield_stem>_gray_seg.npy`. Use distinct output
directories when EVOS acquisitions reuse image basenames; otherwise a mask
from one acquisition can be incorrectly reused for another.

Run independent sweeps for strata with different fluorescence
distributions. For example:

```bash
python scripts/optimize_ac_detection_params.py \
    --batch optimization_batch.csv \
    --config config.yaml \
    --output-dir outputs/ac_param_optimization/cm \
    --perturbation all \
    --negative-condition negative \
    --positive-condition positive \
    --thresholds 1:30:1 \
    --small-ac-max 1:50:1 \
    --max-negative-pct 10 \
    --rank-by separation \
    --config-out config_cm_optimized.yaml
```

Select experimental strata by supplying a batch CSV containing only the
samples that should be compared. Columns such as `medium`, genotype, or
treatment remain ordinary pass-through metadata and are not interpreted by
the optimizer.

The grid syntax is `start:stop:step` with an inclusive stop, or a
comma-separated list. By default, rows with `optimizer_condition=positive`
are compared with rows having `optimizer_condition=negative`; override the
keywords with `--positive-condition` and `--negative-condition`.
`--rank-by separation` maximizes positive minus negative percent among
candidates passing `--max-negative-pct`. `--rank-by score` additionally
penalizes negative-class positivity by `--negative-penalty`. If the selected
value is at a grid boundary, extend the grid and rerun before accepting it.

The optimizer writes:

| File | Purpose |
|---|---|
| `candidate_scores.csv` | every parameter combination and its metrics |
| `top_candidates.csv` | highest-ranked passing candidates |
| `best_condition_by_mouse_summary.csv` | condition-level result for the selected parameters |
| `best_field_counts.csv` | selected-parameter counts for every field |
| `--config-out` path | input config copied with the selected threshold and size cutoff |

## Inputs and channels

Two image channels are required per field: the AC fluorescence channel (any
fluorophore color) and brightfield. The pipeline does not consume a nuclear
channel; segmentation is brightfield-only. Cellpose-SAM (v4) is
channel-agnostic and does not benefit from a separate nuclear input, and in
2D epifluorescence an AC nucleus stacked on a BMDM nucleus cannot be
distinguished from a single nucleus, so a nuclear channel adds no
discriminating information for bound-vs-engulfed counts.

The AC channel is collapsed to grayscale by per-pixel `max(R, G, B)`,
which works for any single-fluorophore EVOS pseudo-color export. The
brightfield channel uses luminance instead (see "Brightfield channel
collapse" below).

## Outputs

Per field, in the brightfield image's folder by default:

| File | Purpose |
|---|---|
| `<sample_id>_per_cell.csv` | one row per BMDM, all metrics + provenance |
| `<sample_id>_overlay.png` | QC overlay (BF, BF+AC merge with masks, AC+masks, full annotation) |
| `<sample_id>_run_log.json` | config snapshot, file hashes, software versions |
| `<bf_basename>_gray.tif` | 8-bit luminance grayscale BF, ready to load in Cellpose GUI |
| `<bf_basename>_gray_seg.npy` | Cellpose-native segmentation, paired with the gray TIFF |

To inspect or correct the segmentation, open `<bf_basename>_gray.tif` in the
Cellpose GUI; the matching `_gray_seg.npy` auto-loads. Edit, save, then
re-run the pipeline pointing at the edited seg file.

If you prefer to keep only the original RGB BF and the original seg naming
(`<bf_basename>_seg.npy` referencing the RGB BF), set
`output.save_grayscale_brightfield: false` in the config. The grayscale
TIFF is preferred because the original EVOS RGB BF often has a saturated
red channel that degrades manual GUI editing as well as Cellpose
segmentation; see "Brightfield channel collapse" below.

## Per-cell CSV columns

| Column | Description |
|---|---|
| sample_id | from input CSV (or --output-prefix in single mode) |
| (pass-through metadata) | any extra columns from the batch CSV |
| cell_id | label index within the field (1..N) |
| cell_area_px, cell_area_um2 | cell footprint area |
| cell_centroid_x, cell_centroid_y | cell centroid (pixel coords) |
| cell_solidity | convex hull area / cell area |
| is_border_cell | cell touches image border |
| n_large_AC | AC objects above small_ac_max threshold |
| n_small_AC | AC objects between puncta_max and small_ac_max |
| n_puncta | AC objects below puncta_max threshold |
| n_AC_total | n_large_AC + n_small_AC |
| ac_integrated_intensity | sum of AC channel within cell mask |
| ac_mean_intensity | per-pixel AC intensity within cell mask |
| phagocytic_strict | n_AC_total >= 1 |
| threshold_used, um_per_pixel | resolved values for this field |
| cellpose_model, seg_file_sha256 | segmentation provenance |
| run_date_utc | run timestamp |

Other definitions of "phagocytic" (intensity-based, large-AC-only, etc.)
should be derived in R from the raw count and intensity columns. They are
intentionally not baked into the CSV.

## Configuration

See `config.yaml` for all parameters. The most commonly tuned ones:

| Parameter | Typical range | Notes |
|---|---|---|
| `segmentation.cellpose_model` | `cyto3`, `cpsam`, or path | custom-trained models go here |
| `segmentation.cell_diameter_um` | 25-40 | typical BMDM body |
| `segmentation.dilation.enabled` | false / true | enables peri-cell expansion |
| `ac_detection.threshold` | ~p99.9 of neg control | from calibrate_threshold.py |
| `ac_detection.size_classes_um2.small_ac_max` | 15-25 | upper bound for "single AC body" |

## Cellpose v3 vs v4

The pipeline auto-detects the installed Cellpose major version and adapts
its API accordingly. Segmentation is brightfield-only under both versions.
The differences that affect this pipeline:

| | Cellpose v3 (cyto3 et al.) | Cellpose v4 (Cellpose-SAM / CPSAM) |
|---|---|---|
| Built-in models | cyto, cyto2, cyto3, nuclei, ... | CPSAM only |
| `model_type` argument | required | removed (silently ignored) |
| `channels` argument | required (`[0,0]` for BF-only) | removed (silently ignored) |
| Diameter handling | mandatory or auto-estimated | optional; CPSAM is robust to scale |

If you set `cellpose_model: cyto3` in the config but have v4 installed,
you will see a warning and CPSAM will be used. To silence the warning,
set `cellpose_model: cpsam`.

The CP4 GUI ships only CPSAM and does not expose a separate nuclear
channel option, because CPSAM is generalist by design. Custom CPSAM
models trained in the GUI inherit the same single-image input
convention.

## Brightfield channel collapse

EVOS exports BF as RGB with strongly unequal channels because the
transmitted-light LED has a dominant color (typically red, which can
saturate the R channel). Collapsing with `max(R, G, B)` then picks the
saturated channel and discards the cell-edge contrast that lives in the
G and B channels. This degrades segmentation quality substantially.

The default for brightfield is now `luminance` (PIL `convert('L')`,
ITU-R BT.601 weighted sum `0.299R + 0.587G + 0.114B`), which preserves
contrast across all three channels. This matches what ImageJ's
`Image > Type > 8-bit` does on RGB images. Override via
`segmentation.brightfield_channel_extraction` in the config if needed.

For the AC fluorescence channel, `max_rgb` remains the right default
because EVOS pseudo-color fluorescence images concentrate signal in only
one of R/G/B.

## Custom Cellpose models

Set `segmentation.cellpose_model` to the absolute path of a model file
output by the Cellpose GUI's "Train new model" feature. The pipeline
detects the file and loads it as a `pretrained_model`. When using a
custom model, `use_model_diameter` defaults to `true` so Cellpose uses the
diameter learned from your training data rather than the config value.

Custom models trained in CP4 are channel-agnostic (CPSAM-derived). Custom
models trained in CP3 follow the v3 channels convention. The pipeline
uses whichever convention matches the installed Cellpose version, so a v3
custom model will not load correctly under v4 and vice versa.

## Reusing existing segmentation

To re-run quantification against a corrected segmentation without invoking
Cellpose:

1. Open `<bf_basename>_gray.tif` in the Cellpose GUI; masks auto-load
   from the matching `<bf_basename>_gray_seg.npy`.
2. Edit boundaries manually and save in the GUI.
3. Re-run the pipeline with `--seg-file path/to/edited_gray_seg.npy`
   (single mode) or fill the `seg_path` column in the batch CSV.

The edited seg file is never overwritten by a re-run. Its sha256 is
recorded on every per-cell row, so two runs against the same edited seg
file produce identical output.

## Empty cells in the batch CSV

Both `seg_path` and `output_dir` are optional. An empty cell triggers the
default behavior:

| Column | Empty | Set |
|---|---|---|
| `seg_path` | run Cellpose, save new `_gray_seg.npy` | load masks from this file, skip Cellpose, do not write a new seg file |
| `output_dir` | write outputs into the brightfield image's parent folder | write outputs into this folder |

There is no special syntax (no `null`, no `NA`); a literally empty cell is
the right input.

## Keeping border cells

Set `segmentation.exclude_border_cells: false` in the config. Cells that
touch any image border are then retained in the per-cell CSV with
`is_border_cell=True`, and you can filter them downstream as you like
(or keep them in some analyses and exclude them in others). The
`is_border_cell` flag is computed and written regardless of the config
value, so dropping border cells later in R is always possible.

## Multi-round workflow with shared segmentation

Use case: image two AC labels in different fluorescence channels (e.g.,
CellBrite Green for AC1, pHrodo Red for AC2), run them through the
pipeline separately, and analyze them as paired observations on the same
BMDMs without re-running Cellpose for the second round.

The pipeline supports this without any new arguments. The seg file is
keyed by brightfield basename, while per-cell CSV / overlay / run_log are
keyed by `sample_id`. Use distinct `sample_id` values per round and the
same `output_dir`. Round 1 segments and writes everything; Round 2 reuses
the seg file and writes only AC-derived outputs.

**Round 1 batch CSV** (`plate1_AC1.csv`):
```csv
sample_id,well,field,condition,ac_label,ac_path,brightfield_path,seg_path,output_dir
B05f00_AC1,B05,00,WT,AC1,/data/CM_B05f00d0.TIF,/data/CM_B05f00d4.TIF,,/data/output
B05f01_AC1,B05,01,WT,AC1,/data/CM_B05f01d0.TIF,/data/CM_B05f01d4.TIF,,/data/output
```

After running, `/data/output/` contains per field:
```
CM_B05f00d4_gray.tif
CM_B05f00d4_gray_seg.npy
B05f00_AC1_per_cell.csv
B05f00_AC1_overlay.png
B05f00_AC1_run_log.json
```

Inspect overlays. If any segmentation needs correction, edit the
matching `_gray_seg.npy` in the Cellpose GUI and save.

**Round 2 batch CSV** (`plate1_AC2.csv`):
```csv
sample_id,well,field,condition,ac_label,ac_path,brightfield_path,seg_path,output_dir
B05f00_AC2,B05,00,WT,AC2,/data/CM_B05f00d2.TIF,/data/CM_B05f00d4.TIF,/data/output/CM_B05f00d4_gray_seg.npy,/data/output
B05f01_AC2,B05,01,WT,AC2,/data/CM_B05f01d2.TIF,/data/CM_B05f01d4.TIF,/data/output/CM_B05f01d4_gray_seg.npy,/data/output
```

Round 2 reuses the seg files, does not invoke Cellpose, does not rewrite
the gray TIFFs, and writes `*_AC2_*` files alongside the existing
`*_AC1_*` files. Nothing is overwritten.

Provenance: the `seg_file_sha256` column on every per-cell row will be
identical between AC1 and AC2 for the same field. Use this in downstream
analysis to verify that paired observations really did use identical
masks. The pass-through metadata column `ac_label` (or whatever you name
it) distinguishes the rounds in a concatenated long-format table.

## Adjacent-cell handling

When dilation is enabled, the pipeline uses
`skimage.segmentation.expand_labels`, a multi-source watershed that grows
each label simultaneously and stops at the midline between neighbors.
Dilated regions never overlap. This is materially different from binary
dilation followed by overlap removal, which would create exclusion zones.

## Reproducibility

Each run writes `<sample_id>_run_log.json` containing the full config
snapshot, sha256 of every input file, software versions, and the random
seed (123). Two runs on the same inputs and seg file produce identical
output.

## Statistical analysis

The pipeline output is per-cell. For across-condition comparisons, the
statistical unit should be the well, with fields as technical replicates.
Aggregate field-level results within each well before applying inferential
tests. This is standard for plate-based imaging assays and avoids
pseudoreplication.

## Known limitations

- Pipeline cannot distinguish surface-bound from internalized ACs.
- Lipophilic dye transfer between membranes (documented for CellBrite-class
  probes) contributes to small-puncta signal independently of true engulfment.
- Custom Cellpose models are only as good as their training set; performance
  can drop on morphologically different conditions.

## Project structure

```
efferocytosis_pipeline/
├── config.yaml
├── README.md
├── requirements.txt
├── batch_template.csv
├── src/
│   ├── __init__.py
│   ├── io.py              # file discovery, channel extraction, metadata
│   ├── segment.py         # Cellpose wrapper, seg file I/O
│   ├── detect_AC.py       # background subtraction, thresholding, classification
│   ├── quantify.py        # per-cell metric computation
│   ├── overlay.py         # QC overlay rendering
│   └── pipeline.py        # orchestration, batch loop
└── scripts/
    ├── run_efferocytosis.py
    └── calibrate_threshold.py
```
