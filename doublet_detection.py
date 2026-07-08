#!/usr/bin/env python3
"""
Reusable doublet detection for Cell Ranger / Parse Biosciences / generic scRNA outputs.

Method:
  - Loads raw UMI count matrices from common scRNA output formats.
  - Runs Scanpy/Scrublet doublet detection globally or per sample/batch.
  - Writes doublet calls, a summary table, an annotated h5ad, and QC plots.

Install once, for example:
  mamba create -n scrna-doublets -c conda-forge -c bioconda \
      python=3.11 scanpy scrublet anndata pandas numpy scipy matplotlib h5py
  mamba activate scrna-doublets

Two ways to use:
  1) Edit the USER SETTINGS block below and run:
       python detect_doublets_configurable.py

  2) Override from the command line:
       python detect_doublets_configurable.py \
         -i sample1/outs sample2/outs \
         --sample-id sample1 sample2 \
         -o doublet_detection/all_samples

Supported inputs:
  - Cell Ranger outs/ directory
  - Cell Ranger filtered_feature_bc_matrix.h5 / raw_feature_bc_matrix.h5
  - 10x-style filtered_feature_bc_matrix/ folder with matrix.mtx + features.tsv + barcodes.tsv
  - ParseBio/generic mtx folders such as DGE.mtx + all_genes.csv + cell_metadata.csv
  - h5ad, optionally using a raw-count layer such as --counts-layer counts
"""

from __future__ import annotations

import argparse
import copy
import gc
import gzip
import inspect
import logging
import sys
from pathlib import Path
from typing import Iterable, Optional

import anndata as ad
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.io
import scipy.sparse as sp


# =============================================================================
# USER SETTINGS: edit these variables for your project
# =============================================================================

# -----------------------------------------------------------------------------
# BATCH/JOB MODE
# -----------------------------------------------------------------------------
# Set RUN_JOBS=True to run several analyses in one go. Each job can have its own
# inputs, sample IDs, output folder, expected_doublet_rate, threshold, etc.
#
# Set RUN_JOBS=False if you only want to use the old single-run INPUTS/OUTDIR
# section below or pass paths from the command line.
RUN_JOBS: bool = True

# Optional: run only selected job names. Example: ["plex1_no_force_auto"]
# Leave empty to run all jobs in JOBS.
ONLY_JOBS: list[str] = []

# Optional combined summary table across all jobs. If None, the script writes it
# next to the first job's output directory as doublet_detection_jobs_summary.tsv.
JOBS_SUMMARY_TSV: Optional[str] = None

# PLEX1_NO_FORCE_OUTS: str = (
#     "/mnt/storage3b/projects/research/22029I_0978_TreatION/analysis_cellranger_ocm/"
#     "22029I_0978_TreatION_Plex1/outs"
# )
#
# PLEX1_FORCE18K_OUTS: str = (
#     "/mnt/storage3b/projects/research/22029I_0978_TreatION/analysis_cellranger_ocm/"
#     "22029I_0978_TreatION_Plex1_18k_4_5adjusted/outs"
# )
#
# PLEX1_SAMPLE_IDS: list[str] = [
#     "M2_5k",
#     "M4_5k",
#     "M5_5k",
#     "M9_5k",
# ]

PLEX2_NO_FORCE_OUTS: str = (
    "/mnt/storage3b/projects/research/22029I_0978_TreatION/analysis_cellranger_ocm/"
    "22029I_0978_TreatION_Plex2/outs"
)

PLEX2_FORCE18K_OUTS: str = (
    "/mnt/storage3b/projects/research/22029I_0978_TreatION/analysis_cellranger_ocm/"
    "22029I_0978_TreatION_Plex2_18k_4_5adjusted/outs"
)

PLEX2_SAMPLE_IDS: list[str] = [
    "M2_10k",
    "M4_10k",
    "M5_10k",
    "M9_10k",
]

def cellranger_ocm_h5_paths(outs_dir: str, sample_ids: list[str]) -> list[str]:
    """Build per-sample Cell Ranger OCM h5 paths from an outs directory."""
    return [
        str(Path(outs_dir) / "per_sample_outs" / sample_id / "count" / "sample_filtered_feature_bc_matrix.h5")
        for sample_id in sample_ids
    ]


# JOBS: list[dict] = [
#     {
#         "name": "plex1_no_force_auto",
#         "inputs": cellranger_ocm_h5_paths(PLEX1_NO_FORCE_OUTS, PLEX1_SAMPLE_IDS),
#         "sample_ids": PLEX1_SAMPLE_IDS,
#         "outdir": str(Path(PLEX1_NO_FORCE_OUTS) / "doublet_detection_out_auto"),
#         "expected_doublet_rate": 0.076,
#         "threshold": None,
#         "call_top_fraction": None,
#     },
#     {
#         "name": "plex1_no_force_manual025",
#         "inputs": cellranger_ocm_h5_paths(PLEX1_NO_FORCE_OUTS, PLEX1_SAMPLE_IDS),
#         "sample_ids": PLEX1_SAMPLE_IDS,
#         "outdir": str(Path(PLEX1_NO_FORCE_OUTS) / "doublet_detection_out_manual025"),
#         "expected_doublet_rate": 0.076,
#         "threshold": 0.25,
#         "call_top_fraction": None,
#     },
#     {
#         "name": "plex1_force18k_auto",
#         "inputs": cellranger_ocm_h5_paths(PLEX1_FORCE18K_OUTS, PLEX1_SAMPLE_IDS),
#         "sample_ids": PLEX1_SAMPLE_IDS,
#         "outdir": str(Path(PLEX1_FORCE18K_OUTS) / "doublet_detection_out_auto"),
#         "expected_doublet_rate": 0.076,
#         "threshold": None,
#         "call_top_fraction": None,
#     },
#     {
#         "name": "plex1_force18k_manual025",
#         "inputs": cellranger_ocm_h5_paths(PLEX1_FORCE18K_OUTS, PLEX1_SAMPLE_IDS),
#         "sample_ids": PLEX1_SAMPLE_IDS,
#         "outdir": str(Path(PLEX1_FORCE18K_OUTS) / "doublet_detection_out_manual025"),
#         "expected_doublet_rate": 0.076,
#         "threshold": 0.25,
#         "call_top_fraction": None,
#     },
# ]

JOBS: list[dict] = [
    {
        "name": "plex2_no_force_auto",
        "inputs": cellranger_ocm_h5_paths(PLEX2_NO_FORCE_OUTS, PLEX2_SAMPLE_IDS),
        "sample_ids": PLEX2_SAMPLE_IDS,
        "outdir": str(Path(PLEX2_NO_FORCE_OUTS) / "doublet_detection_out_auto"),
        "expected_doublet_rate": 0.076,
        "threshold": None,
        "call_top_fraction": None,
    },
    {
        "name": "plex2_no_force_manual025",
        "inputs": cellranger_ocm_h5_paths(PLEX2_NO_FORCE_OUTS, PLEX2_SAMPLE_IDS),
        "sample_ids": PLEX2_SAMPLE_IDS,
        "outdir": str(Path(PLEX2_NO_FORCE_OUTS) / "doublet_detection_out_manual025"),
        "expected_doublet_rate": 0.076,
        "threshold": 0.25,
        "call_top_fraction": None,
    },
    {
        "name": "plex2_force18k_auto",
        "inputs": cellranger_ocm_h5_paths(PLEX2_FORCE18K_OUTS, PLEX2_SAMPLE_IDS),
        "sample_ids": PLEX2_SAMPLE_IDS,
        "outdir": str(Path(PLEX2_FORCE18K_OUTS) / "doublet_detection_out_auto"),
        "expected_doublet_rate": 0.076,
        "threshold": None,
        "call_top_fraction": None,
    },
    {
        "name": "plex2_force18k_manual025",
        "inputs": cellranger_ocm_h5_paths(PLEX2_FORCE18K_OUTS, PLEX2_SAMPLE_IDS),
        "sample_ids": PLEX2_SAMPLE_IDS,
        "outdir": str(Path(PLEX2_FORCE18K_OUTS) / "doublet_detection_out_manual025"),
        "expected_doublet_rate": 0.076,
        "threshold": 0.25,
        "call_top_fraction": None,
    },
]


# -----------------------------------------------------------------------------
# SINGLE-RUN MODE
# -----------------------------------------------------------------------------
# Used only when RUN_JOBS=False, or when you override paths with -i/-o from the
# command line.
#
# Input(s). Examples:
#   ["/path/to/cellranger_sample/outs"]
#   ["/path/to/filtered_feature_bc_matrix.h5"]
#   ["/path/to/parsebio_output_folder"]
#   ["sample1/outs", "sample2/outs"]
INPUTS: list[str] = []

# Output directory.
OUTDIR: str = "doublet_detection_out"

# Optional sample names. Use None to derive names from input folder/file names.
# Must have one value per INPUTS entry if used, for example: ["ctrl", "treated"]
SAMPLE_IDS: Optional[list[str]] = None

# h5ad input only: use this layer as raw counts instead of .X, for example "counts".
COUNTS_LAYER: Optional[str] = None

# Optional metadata CSV/TSV to join onto cells. Useful for donor/sample annotations.
METADATA: Optional[str] = None
METADATA_BARCODE_COL: Optional[str] = None
SAMPLE_COL: Optional[str] = None

# How to run Scrublet batches:
#   "auto" = run per sample if multiple samples are present, otherwise global
#   "none" = force one global model
#   "sample" / "donor" / other obs column = run separately per group
BATCH_KEY: str = "auto"

# Explicit generic matrix triplet. Usually leave these as None and let the script auto-detect.
# Useful for ParseBio-like folders if auto-detection fails.
MATRIX: Optional[str] = None
FEATURES: Optional[str] = None
BARCODES: Optional[str] = None

# Feature names for 10x/generic inputs: "gene_symbols" or "gene_ids".
VAR_NAMES: str = "gene_symbols"

# 10x feature matrices: False keeps only Gene Expression features when Scanpy supports it.
# Set True only if you intentionally want all feature types retained.
NO_GEX_ONLY: bool = False

# Scrublet parameters. For this OCM comparison we use 0.076 as the common starting point.
EXPECTED_DOUBLET_RATE: float = 0.076
SIM_DOUBLET_RATIO: float = 2.0
N_NEIGHBORS: Optional[int] = None
N_PRIN_COMPS: int = 30
THRESHOLD: Optional[float] = None       # None = automatic threshold; or set e.g. 0.25
# Optional rescue mode if automatic Scrublet thresholds are clearly bad.
# If set, calls the top X fraction of cells by doublet_score per batch/sample.
# Example: 0.06 means force approximately 6% calls per batch. Usually leave as None.
CALL_TOP_FRACTION: Optional[float] = None
RANDOM_STATE: int = 0
MIN_COUNTS: int = 2
MIN_CELLS: int = 3
MIN_GENE_VARIABILITY_PCTL: int = 85

# Output options.
SKIP_H5AD: bool = False
WRITE_SINGLETS: bool = False
PREFIX_BARCODES: bool = False          # Useful when combining samples with identical barcodes
VERBOSE: bool = False

# =============================================================================
# End of USER SETTINGS. You usually do not need to edit below this line.
# =============================================================================


LOGGER = logging.getLogger("detect_doublets")

CELLRANGER_H5_NAMES = (
    "filtered_feature_bc_matrix.h5",
    "raw_feature_bc_matrix.h5",
    "filtered_gene_bc_matrices_h5.h5",
    "raw_gene_bc_matrices_h5.h5",
)

CELLRANGER_MTX_DIR_NAMES = (
    "filtered_feature_bc_matrix",
    "raw_feature_bc_matrix",
    "filtered_gene_bc_matrices",
    "raw_gene_bc_matrices",
)

MATRIX_NAMES = (
    "matrix.mtx",
    "matrix.mtx.gz",
    "DGE.mtx",
    "DGE.mtx.gz",
    "dge.mtx",
    "dge.mtx.gz",
    "gene_expression.mtx",
    "gene_expression.mtx.gz",
    "counts.mtx",
    "counts.mtx.gz",
)

FEATURE_NAMES = (
    "features.tsv",
    "features.tsv.gz",
    "genes.tsv",
    "genes.tsv.gz",
    "all_genes.csv",
    "genes.csv",
    "gene_metadata.csv",
    "features.csv",
)

BARCODE_NAMES = (
    "barcodes.tsv",
    "barcodes.tsv.gz",
    "cells.tsv",
    "cells.tsv.gz",
    "cell_barcodes.tsv",
    "cell_barcodes.tsv.gz",
    "cell_metadata.csv",
    "cells.csv",
    "barcodes.csv",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Scrublet doublet detection on Cell Ranger, ParseBio, or generic scRNA outputs."
    )
    parser.add_argument(
        "-i",
        "--input",
        nargs="+",
        default=INPUTS,
        help="Input path(s). Overrides INPUTS at top of script.",
    )
    parser.add_argument(
        "-o",
        "--outdir",
        default=OUTDIR,
        help="Output directory. Overrides OUTDIR at top of script.",
    )
    parser.add_argument(
        "--sample-id",
        nargs="*",
        default=SAMPLE_IDS,
        help="Optional sample IDs, one per input. Overrides SAMPLE_IDS.",
    )
    parser.add_argument("--counts-layer", default=COUNTS_LAYER)
    parser.add_argument("--metadata", default=METADATA)
    parser.add_argument("--metadata-barcode-col", default=METADATA_BARCODE_COL)
    parser.add_argument("--sample-col", default=SAMPLE_COL)
    parser.add_argument("--batch-key", default=BATCH_KEY)
    parser.add_argument("--matrix", default=MATRIX)
    parser.add_argument("--features", default=FEATURES)
    parser.add_argument("--barcodes", default=BARCODES)
    parser.add_argument(
        "--var-names",
        choices=("gene_symbols", "gene_ids"),
        default=VAR_NAMES,
    )

    parser.add_argument("--no-gex-only", dest="no_gex_only", action="store_true", default=NO_GEX_ONLY)
    parser.add_argument("--gex-only", dest="no_gex_only", action="store_false")

    parser.add_argument("--expected-doublet-rate", type=float, default=EXPECTED_DOUBLET_RATE)
    parser.add_argument("--sim-doublet-ratio", type=float, default=SIM_DOUBLET_RATIO)
    parser.add_argument("--n-neighbors", type=int, default=N_NEIGHBORS)
    parser.add_argument("--n-prin-comps", type=int, default=N_PRIN_COMPS)
    parser.add_argument("--threshold", type=float, default=THRESHOLD)
    parser.add_argument("--call-top-fraction", type=float, default=CALL_TOP_FRACTION)
    parser.add_argument("--random-state", type=int, default=RANDOM_STATE)
    parser.add_argument("--min-counts", type=int, default=MIN_COUNTS)
    parser.add_argument("--min-cells", type=int, default=MIN_CELLS)
    parser.add_argument("--min-gene-variability-pctl", type=int, default=MIN_GENE_VARIABILITY_PCTL)

    parser.add_argument("--skip-h5ad", dest="skip_h5ad", action="store_true", default=SKIP_H5AD)
    parser.add_argument("--write-h5ad", dest="skip_h5ad", action="store_false")
    parser.add_argument("--write-singlets", dest="write_singlets", action="store_true", default=WRITE_SINGLETS)
    parser.add_argument("--no-write-singlets", dest="write_singlets", action="store_false")
    parser.add_argument("--prefix-barcodes", dest="prefix_barcodes", action="store_true", default=PREFIX_BARCODES)
    parser.add_argument("--no-prefix-barcodes", dest="prefix_barcodes", action="store_false")
    parser.add_argument("--verbose", dest="verbose", action="store_true", default=VERBOSE)
    parser.add_argument("--quiet", dest="verbose", action="store_false")

    parser.add_argument("--run-jobs", dest="run_jobs", action="store_true", default=RUN_JOBS)
    parser.add_argument("--single-run", dest="run_jobs", action="store_false")
    parser.add_argument(
        "--only-job",
        nargs="*",
        default=ONLY_JOBS,
        help="When using RUN_JOBS, run only these job names. Default: run all jobs.",
    )
    parser.add_argument(
        "--jobs-summary-tsv",
        default=JOBS_SUMMARY_TSV,
        help="Optional combined summary TSV path for RUN_JOBS mode.",
    )

    args = parser.parse_args()
    if not args.run_jobs and not args.input:
        parser.error(
            "No input provided. Either edit INPUTS at the top of the script, "
            "run with -i /path/to/input, or use RUN_JOBS=True."
        )
    return args


def setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(format="[%(asctime)s] %(levelname)s: %(message)s", level=level)


def infer_sample_id(path: Path) -> str:
    path = path.resolve()
    if path.is_file():
        name = path.name
        for suffix in (".h5ad", ".h5", ".hdf5"):
            if name.endswith(suffix):
                return name[: -len(suffix)]
        return path.stem
    if path.name == "outs":
        return path.parent.name
    if path.name in CELLRANGER_MTX_DIR_NAMES and path.parent.name == "outs":
        return path.parent.parent.name
    return path.name


def read_table_auto(path: Path, header="infer") -> pd.DataFrame:
    suffixes = "".join(path.suffixes).lower()
    if ".csv" in suffixes:
        return pd.read_csv(path, header=header)
    return pd.read_csv(path, sep="\t", header=header)


def first_existing(directory: Path, names: Iterable[str]) -> Optional[Path]:
    for name in names:
        candidate = directory / name
        if candidate.exists():
            return candidate
    return None


def looks_like_10x_mtx_dir(directory: Path) -> bool:
    has_matrix = first_existing(directory, ("matrix.mtx", "matrix.mtx.gz")) is not None
    has_barcodes = first_existing(directory, ("barcodes.tsv", "barcodes.tsv.gz")) is not None
    has_features = first_existing(
        directory, ("features.tsv", "features.tsv.gz", "genes.tsv", "genes.tsv.gz")
    ) is not None
    return has_matrix and has_barcodes and has_features


def find_cellranger_candidate(path: Path) -> Optional[Path]:
    roots = [path]
    if (path / "outs").exists():
        roots.insert(0, path / "outs")

    for root in roots:
        for name in CELLRANGER_H5_NAMES:
            candidate = root / name
            if candidate.exists():
                return candidate
        for name in CELLRANGER_MTX_DIR_NAMES:
            candidate = root / name
            if candidate.exists():
                if candidate.name.endswith("gene_bc_matrices"):
                    subdirs = [x for x in candidate.iterdir() if x.is_dir()]
                    if len(subdirs) == 1:
                        return subdirs[0]
                return candidate
        if looks_like_10x_mtx_dir(root):
            return root
    return None


def limited_rglob(root: Path, names: Iterable[str], max_depth: int = 4) -> list[Path]:
    hits: list[Path] = []
    root = root.resolve()
    for name in names:
        for hit in root.rglob(name):
            try:
                depth = len(hit.relative_to(root).parts)
            except ValueError:
                continue
            if depth <= max_depth:
                hits.append(hit)
    return hits


def find_generic_triplet(directory: Path) -> tuple[Path, Path, Path]:
    direct_matrix = first_existing(directory, MATRIX_NAMES)
    if direct_matrix is not None:
        feature = first_existing(directory, FEATURE_NAMES)
        barcode = first_existing(directory, BARCODE_NAMES)
        if feature is not None and barcode is not None:
            return direct_matrix, feature, barcode

    for matrix in limited_rglob(directory, MATRIX_NAMES):
        feature = first_existing(matrix.parent, FEATURE_NAMES)
        barcode = first_existing(matrix.parent, BARCODE_NAMES)
        if feature is not None and barcode is not None:
            return matrix, feature, barcode

    raise FileNotFoundError(
        f"Could not auto-detect matrix/features/barcodes under {directory}. "
        "Set MATRIX, FEATURES, and BARCODES at the top, or pass --matrix --features --barcodes."
    )


def pick_column(df: pd.DataFrame, requested: Optional[str], candidates: Iterable[str], what: str) -> str:
    if requested is not None:
        if requested not in df.columns:
            raise ValueError(f"Requested {what} column '{requested}' was not found. Columns: {list(df.columns)}")
        return requested
    lower_to_col = {str(c).lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in lower_to_col:
            return lower_to_col[c.lower()]
    return df.columns[0]


def read_features(path: Path, var_names: str) -> pd.DataFrame:
    suffixes = "".join(path.suffixes).lower()
    if ".csv" in suffixes:
        df = pd.read_csv(path)
        gene_id_col = pick_column(
            df,
            None,
            ("gene_id", "gene_ids", "id", "ensembl_id", "feature_id"),
            "gene ID",
        )
        symbol_col = pick_column(
            df,
            None,
            ("gene_name", "gene_symbol", "symbol", "gene", "name", "feature_name"),
            "gene symbol",
        )
        out = pd.DataFrame(
            {
                "gene_id": df[gene_id_col].astype(str).values,
                "gene_symbol": df[symbol_col].astype(str).values,
            }
        )
    else:
        df = pd.read_csv(path, sep="\t", header=None)
        if df.shape[1] >= 3:
            out = pd.DataFrame(
                {
                    "gene_id": df.iloc[:, 0].astype(str).values,
                    "gene_symbol": df.iloc[:, 1].astype(str).values,
                    "feature_type": df.iloc[:, 2].astype(str).values,
                }
            )
        elif df.shape[1] == 2:
            out = pd.DataFrame(
                {
                    "gene_id": df.iloc[:, 0].astype(str).values,
                    "gene_symbol": df.iloc[:, 1].astype(str).values,
                }
            )
        else:
            out = pd.DataFrame(
                {
                    "gene_id": df.iloc[:, 0].astype(str).values,
                    "gene_symbol": df.iloc[:, 0].astype(str).values,
                }
            )

    index_col = "gene_symbol" if var_names == "gene_symbols" else "gene_id"
    out.index = out[index_col].astype(str)
    return out


def read_barcodes(path: Path, barcode_col: Optional[str] = None) -> pd.Index:
    suffixes = "".join(path.suffixes).lower()
    if ".csv" in suffixes:
        df = pd.read_csv(path)
        col = pick_column(
            df,
            barcode_col,
            ("barcode", "barcodes", "cell_barcode", "cell", "cell_id", "cellid", "bc"),
            "barcode",
        )
        return pd.Index(df[col].astype(str).values)
    df = pd.read_csv(path, sep="\t", header=None)
    return pd.Index(df.iloc[:, 0].astype(str).values)


def mmread_auto(matrix_path: Path):
    suffixes = "".join(matrix_path.suffixes).lower()
    if suffixes.endswith(".mtx.gz"):
        with gzip.open(matrix_path, "rb") as handle:
            return scipy.io.mmread(handle)
    return scipy.io.mmread(str(matrix_path))


def read_generic_mtx(matrix_path: Path, features_path: Path, barcodes_path: Path, var_names: str) -> ad.AnnData:
    LOGGER.info("Reading generic mtx triplet: matrix=%s features=%s barcodes=%s", matrix_path, features_path, barcodes_path)
    matrix = mmread_auto(matrix_path).tocsr()
    var = read_features(features_path, var_names=var_names)
    barcodes = read_barcodes(barcodes_path)

    n_genes = var.shape[0]
    n_cells = len(barcodes)
    if matrix.shape == (n_genes, n_cells):
        x = matrix.T.tocsr()
    elif matrix.shape == (n_cells, n_genes):
        x = matrix.tocsr()
    else:
        raise ValueError(
            f"Matrix shape {matrix.shape} does not match features={n_genes} and barcodes={n_cells}."
        )

    obs = pd.DataFrame(index=barcodes.astype(str))
    obs["orig_barcode"] = obs.index.astype(str)
    adata = ad.AnnData(X=x, obs=obs, var=var)
    adata.var_names_make_unique()
    adata.obs_names_make_unique()
    return adata


def read_10x_mtx_robust(directory: Path, var_names: str, gex_only: bool) -> ad.AnnData:
    compressed = (directory / "matrix.mtx.gz").exists()
    try:
        return sc.read_10x_mtx(
            directory,
            var_names=var_names,
            make_unique=True,
            gex_only=gex_only,
            compressed=compressed,
        )
    except TypeError:
        return sc.read_10x_mtx(directory, var_names=var_names, make_unique=True)


def load_one_input(path: Path, args: argparse.Namespace) -> ad.AnnData:
    gex_only = not args.no_gex_only
    if args.matrix is not None:
        if args.features is None or args.barcodes is None:
            raise ValueError("When using MATRIX/--matrix, also provide FEATURES/--features and BARCODES/--barcodes.")
        return read_generic_mtx(Path(args.matrix), Path(args.features), Path(args.barcodes), args.var_names)

    path = path.resolve()
    LOGGER.info("Loading input: %s", path)

    if path.is_file():
        name = path.name.lower()
        if name.endswith(".h5ad"):
            adata = sc.read_h5ad(path)
            if args.counts_layer is not None:
                if args.counts_layer not in adata.layers:
                    raise KeyError(f"Layer '{args.counts_layer}' not found in {path}.")
                adata.X = adata.layers[args.counts_layer].copy()
            return adata
        if name.endswith(".h5") or name.endswith(".hdf5"):
            try:
                return sc.read_10x_h5(path, gex_only=gex_only)
            except TypeError:
                return sc.read_10x_h5(path)
        raise ValueError(f"Unsupported file type: {path}")

    if not path.is_dir():
        raise FileNotFoundError(path)

    candidate = find_cellranger_candidate(path)
    if candidate is not None:
        LOGGER.info("Detected Cell Ranger/10x-style input: %s", candidate)
        if candidate.is_file():
            try:
                return sc.read_10x_h5(candidate, gex_only=gex_only)
            except TypeError:
                return sc.read_10x_h5(candidate)
        if looks_like_10x_mtx_dir(candidate):
            return read_10x_mtx_robust(candidate, args.var_names, gex_only=gex_only)

    matrix, features, barcodes = find_generic_triplet(path)
    return read_generic_mtx(matrix, features, barcodes, args.var_names)


def maybe_warn_not_raw_counts(adata: ad.AnnData) -> None:
    x = adata.X
    if sp.issparse(x):
        data = x.data
        if data.size == 0:
            return
        sample = data[: min(10000, data.size)]
    else:
        flat = np.asarray(x).ravel()
        sample = flat[: min(10000, flat.size)]
    sample = sample[np.isfinite(sample)]
    if sample.size == 0:
        return
    if np.nanmax(sample) < 50 and not np.allclose(sample, np.round(sample), atol=1e-6):
        LOGGER.warning(
            "The matrix does not look like raw integer UMI counts. "
            "Scrublet should be run on raw, unnormalized counts. "
            "For h5ad, set COUNTS_LAYER='counts' or pass --counts-layer counts if available."
        )


def load_all_inputs(args: argparse.Namespace) -> ad.AnnData:
    input_paths = [Path(p) for p in args.input]
    if args.sample_id is not None and len(args.sample_id) not in (0, len(input_paths)):
        raise ValueError("Provide either no sample IDs or exactly one sample ID per input.")
    sample_ids = args.sample_id if args.sample_id else [infer_sample_id(p) for p in input_paths]

    adatas: list[ad.AnnData] = []
    for path, sample_id in zip(input_paths, sample_ids):
        one = load_one_input(path, args)
        one.var_names_make_unique()
        one.obs_names_make_unique()
        if "orig_barcode" not in one.obs:
            one.obs["orig_barcode"] = one.obs_names.astype(str)
        one.obs["sample"] = str(sample_id)
        if len(input_paths) > 1 or args.prefix_barcodes:
            one.obs_names = [f"{sample_id}:{bc}" for bc in one.obs["orig_barcode"].astype(str)]
        maybe_warn_not_raw_counts(one)
        adatas.append(one)

    if len(adatas) == 1:
        adata = adatas[0]
    else:
        LOGGER.info("Concatenating %d inputs", len(adatas))
        adata = ad.concat(
            adatas,
            join="outer",
            merge="same",
            label="input_sample",
            keys=list(sample_ids),
            index_unique=None,
        )
        if sp.issparse(adata.X):
            adata.X = adata.X.tocsr()
    return adata


def add_metadata(adata: ad.AnnData, args: argparse.Namespace) -> None:
    if args.metadata is None:
        return

    meta_path = Path(args.metadata)
    LOGGER.info("Joining metadata: %s", meta_path)
    meta = read_table_auto(meta_path)
    barcode_col = pick_column(
        meta,
        args.metadata_barcode_col,
        ("barcode", "barcodes", "cell_barcode", "cell", "cell_id", "cellid", "bc"),
        "metadata barcode",
    )
    meta[barcode_col] = meta[barcode_col].astype(str)
    meta = meta.drop_duplicates(subset=barcode_col).set_index(barcode_col)

    joined = meta.reindex(adata.obs_names.astype(str))
    n_direct = int(joined.notna().any(axis=1).sum())

    if n_direct == 0 and "orig_barcode" in adata.obs:
        joined = meta.reindex(adata.obs["orig_barcode"].astype(str).values)
        joined.index = adata.obs_names
        n_orig = int(joined.notna().any(axis=1).sum())
        LOGGER.info("Metadata matched %d cells using orig_barcode.", n_orig)
    else:
        joined.index = adata.obs_names
        LOGGER.info("Metadata matched %d cells using obs_names.", n_direct)

    for col in joined.columns:
        if col in adata.obs.columns:
            adata.obs[f"metadata_{col}"] = joined[col].values
        else:
            adata.obs[col] = joined[col].values


def add_qc_metrics(adata: ad.AnnData) -> None:
    gene_names = pd.Index(adata.var_names.astype(str))
    adata.var["mt"] = gene_names.str.upper().str.startswith("MT-")
    try:
        sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], percent_top=None, log1p=False, inplace=True)
    except Exception as exc:
        LOGGER.warning("Could not calculate QC metrics: %s", exc)


def choose_batch_key(adata: ad.AnnData, args: argparse.Namespace) -> Optional[str]:
    value = str(args.batch_key).strip()
    if value.lower() in ("none", "false", "no", "0"):
        return None
    if value.lower() != "auto":
        if value not in adata.obs.columns:
            raise KeyError(f"Batch key '{value}' not found in adata.obs. Available columns: {list(adata.obs.columns)}")
        if adata.obs[value].nunique(dropna=True) <= 1:
            LOGGER.info("Batch key '%s' has one value only; running one global Scrublet model.", value)
            return None
        return value

    if args.sample_col is not None and args.sample_col in adata.obs.columns:
        if adata.obs[args.sample_col].nunique(dropna=True) > 1:
            return args.sample_col
    if "sample" in adata.obs.columns and adata.obs["sample"].nunique(dropna=True) > 1:
        return "sample"
    return None


def get_scrublet_function():
    if hasattr(sc.pp, "scrublet"):
        return sc.pp.scrublet
    try:
        import scanpy.external as sce

        return sce.pp.scrublet
    except Exception as exc:
        raise RuntimeError("Could not find Scanpy scrublet wrapper. Install scanpy and scrublet.") from exc


def scrublet_kwargs_for_function(func, args: argparse.Namespace, batch_key: Optional[str]) -> dict:
    candidate_kwargs = {
        "expected_doublet_rate": args.expected_doublet_rate,
        "sim_doublet_ratio": args.sim_doublet_ratio,
        "n_neighbors": args.n_neighbors,
        "n_prin_comps": args.n_prin_comps,
        "threshold": args.threshold,
        "batch_key": batch_key,
        "random_state": args.random_state,
        "min_counts": args.min_counts,
        "min_cells": args.min_cells,
        "min_gene_variability_pctl": args.min_gene_variability_pctl,
        "copy": False,
    }
    signature = inspect.signature(func)
    accepts_var_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in signature.parameters.values())
    kwargs = {
        k: v
        for k, v in candidate_kwargs.items()
        if v is not None and (accepts_var_kwargs or k in signature.parameters)
    }
    if batch_key is None:
        kwargs.pop("batch_key", None)
    return kwargs


def normalize_scrublet_columns(adata: ad.AnnData) -> None:
    if "predicted_doublet" not in adata.obs and "predicted_doublets" in adata.obs:
        adata.obs["predicted_doublet"] = adata.obs["predicted_doublets"]
    if "doublet_score" not in adata.obs and "doublet_scores" in adata.obs:
        adata.obs["doublet_score"] = adata.obs["doublet_scores"]

    required = {"doublet_score", "predicted_doublet"}
    missing = required.difference(adata.obs.columns)
    if missing:
        raise RuntimeError(f"Scrublet finished, but these expected obs columns are missing: {missing}")
    adata.obs["predicted_doublet"] = adata.obs["predicted_doublet"].astype(bool)


def run_scrublet_call(adata: ad.AnnData, args: argparse.Namespace, batch_key: Optional[str]) -> ad.AnnData:
    func = get_scrublet_function()
    kwargs = scrublet_kwargs_for_function(func, args, batch_key=batch_key)
    LOGGER.info("Scrublet parameters: %s", {k: v for k, v in kwargs.items() if k != "copy"})
    result = func(adata, **kwargs)
    if result is not None:
        adata = result
    normalize_scrublet_columns(adata)
    return adata


def run_scrublet(adata: ad.AnnData, args: argparse.Namespace, batch_key: Optional[str]) -> ad.AnnData:
    func = get_scrublet_function()
    signature = inspect.signature(func)
    supports_batch_key = "batch_key" in signature.parameters or any(
        p.kind == inspect.Parameter.VAR_KEYWORD for p in signature.parameters.values()
    )

    if batch_key is None or supports_batch_key:
        LOGGER.info("Running Scrublet%s", f" per batch: {batch_key}" if batch_key else " globally")
        return run_scrublet_call(adata, args, batch_key=batch_key)

    # Fallback for older Scanpy wrappers without batch_key support.
    LOGGER.info("Running Scrublet manually per batch: %s", batch_key)
    adata.obs["doublet_score"] = np.nan
    adata.obs["predicted_doublet"] = False
    adata.uns["scrublet_per_batch"] = {}

    for value, idx in adata.obs.groupby(batch_key, dropna=False).indices.items():
        cell_names = adata.obs_names[list(idx)]
        LOGGER.info("Batch %s: %d cells", value, len(cell_names))
        sub = adata[cell_names].copy()
        sub = run_scrublet_call(sub, args, batch_key=None)
        adata.obs.loc[sub.obs_names, "doublet_score"] = sub.obs["doublet_score"].values
        adata.obs.loc[sub.obs_names, "predicted_doublet"] = sub.obs["predicted_doublet"].values.astype(bool)
        if "scrublet" in sub.uns:
            adata.uns["scrublet_per_batch"][str(value)] = sub.uns["scrublet"]

    adata.obs["predicted_doublet"] = adata.obs["predicted_doublet"].astype(bool)
    return adata


def extract_thresholds(adata: ad.AnnData, manual_threshold: Optional[float]) -> dict[str, float]:
    if manual_threshold is not None:
        return {"manual": float(manual_threshold)}

    thresholds: dict[str, float] = {}

    def walk(obj, prefix=""):
        if isinstance(obj, dict):
            for k, v in obj.items():
                key = f"{prefix}.{k}" if prefix else str(k)
                if str(k).lower() == "threshold":
                    try:
                        thresholds[key] = float(v)
                    except Exception:
                        pass
                walk(v, key)

    if "scrublet" in adata.uns:
        walk(adata.uns["scrublet"], "scrublet")
    if "scrublet_per_batch" in adata.uns:
        walk(adata.uns["scrublet_per_batch"], "scrublet_per_batch")
    return thresholds


def apply_top_fraction_calls(adata: ad.AnnData, batch_key: Optional[str], fraction: Optional[float]) -> None:
    """Optionally override predicted_doublet by calling the top score fraction per batch."""
    if fraction is None:
        return
    if not (0 < fraction < 1):
        raise ValueError("CALL_TOP_FRACTION / --call-top-fraction must be between 0 and 1, e.g. 0.06")

    scores = pd.to_numeric(adata.obs["doublet_score"], errors="coerce")
    adata.obs["predicted_doublet_scrublet_auto"] = adata.obs["predicted_doublet"].astype(bool).values
    adata.obs["doublet_call_method"] = f"top_fraction_{fraction:g}"
    adata.obs["predicted_doublet"] = False

    if batch_key is not None:
        groups = adata.obs.groupby(batch_key, dropna=False).indices.items()
    else:
        groups = [("global", np.arange(adata.n_obs))]

    thresholds = {}
    for label, idx in groups:
        idx = np.asarray(list(idx), dtype=int)
        vals = scores.iloc[idx]
        vals_non_na = vals.dropna()
        if vals_non_na.empty:
            continue
        n_call = max(1, int(round(len(vals_non_na) * fraction)))
        cutoff = float(vals_non_na.sort_values(ascending=False).iloc[n_call - 1])
        selected = vals.index[vals >= cutoff]
        adata.obs.loc[selected, "predicted_doublet"] = True
        thresholds[str(label)] = cutoff
    adata.uns["top_fraction_doublet_call"] = {"fraction": fraction, "score_cutoffs": thresholds}
    adata.obs["predicted_doublet"] = adata.obs["predicted_doublet"].astype(bool)


def unique_keep_order(items: Iterable[Optional[str]]) -> list[str]:
    seen = set()
    out = []
    for item in items:
        if item is None or item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def threshold_for_group(label, thresholds: dict[str, float], manual_threshold: Optional[float]) -> str:
    if manual_threshold is not None:
        return f"manual={manual_threshold:g}"
    label = str(label)
    matches = [f"{k}={v:.6g}" for k, v in thresholds.items() if label in k]
    if matches:
        return ";".join(matches)
    if len(thresholds) == 1:
        k, v = next(iter(thresholds.items()))
        return f"{k}={v:.6g}"
    return ";".join(f"{k}={v:.6g}" for k, v in thresholds.items())


def write_tables(adata: ad.AnnData, outdir: Path, batch_key: Optional[str], args: argparse.Namespace) -> None:
    obs = adata.obs.copy()
    obs.insert(0, "cell_id", adata.obs_names.astype(str))

    preferred = [
        "cell_id",
        "orig_barcode",
        "sample",
        batch_key if batch_key is not None else None,
        "doublet_score",
        "predicted_doublet",
        "total_counts",
        "n_genes_by_counts",
        "pct_counts_mt",
    ]
    preferred = unique_keep_order(c for c in preferred if c is not None and c in obs.columns)
    remaining = [c for c in obs.columns if c not in preferred]
    obs = obs[preferred + remaining]
    obs.to_csv(outdir / "doublet_calls.tsv", sep="\t", index=False)

    group_col = batch_key if batch_key is not None else ("sample" if "sample" in adata.obs else None)
    if group_col is not None:
        summary = (
            adata.obs.groupby(group_col, dropna=False)
            .agg(
                n_cells=("predicted_doublet", "size"),
                n_doublets=("predicted_doublet", "sum"),
                mean_doublet_score=("doublet_score", "mean"),
                median_doublet_score=("doublet_score", "median"),
            )
            .reset_index()
        )
    else:
        summary = pd.DataFrame(
            {
                "n_cells": [adata.n_obs],
                "n_doublets": [int(adata.obs["predicted_doublet"].sum())],
                "mean_doublet_score": [float(adata.obs["doublet_score"].mean())],
                "median_doublet_score": [float(adata.obs["doublet_score"].median())],
            }
        )
    summary["pct_doublets"] = 100 * summary["n_doublets"] / summary["n_cells"]
    thresholds = extract_thresholds(adata, args.threshold)
    if thresholds:
        if group_col is not None and group_col in summary.columns:
            summary["threshold_info"] = [
                threshold_for_group(v, thresholds, args.threshold) for v in summary[group_col].astype(str)
            ]
        else:
            summary["threshold_info"] = ";".join(f"{k}={v:.6g}" for k, v in thresholds.items())
    if getattr(args, "call_top_fraction", None) is not None:
        summary["call_method"] = f"top_fraction_{args.call_top_fraction:g}"
    summary.to_csv(outdir / "doublet_summary.tsv", sep="\t", index=False)


def write_plots(adata: ad.AnnData, outdir: Path, batch_key: Optional[str], threshold: Optional[float]) -> None:
    plot_dir = outdir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    scores = pd.to_numeric(adata.obs["doublet_score"], errors="coerce")
    predicted = adata.obs["predicted_doublet"].astype(bool)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(scores[~predicted].dropna(), bins=50, alpha=0.7, label="called singlet")
    if predicted.any():
        ax.hist(scores[predicted].dropna(), bins=50, alpha=0.7, label="called doublet")
    if threshold is not None:
        ax.axvline(threshold, linestyle="--", label=f"threshold={threshold:g}")
    ax.set_xlabel("Scrublet doublet score")
    ax.set_ylabel("Number of cells")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(plot_dir / "doublet_score_histogram.png", dpi=180)
    plt.close(fig)

    group_col = batch_key if batch_key is not None else ("sample" if "sample" in adata.obs else None)
    if group_col is not None and adata.obs[group_col].nunique(dropna=True) > 1:
        groups = []
        labels = []
        for label, idx in adata.obs.groupby(group_col, dropna=False).indices.items():
            vals = scores.iloc[list(idx)].dropna().values
            if len(vals) > 0:
                groups.append(vals)
                labels.append(str(label))
        if groups:
            width = max(7, min(24, 0.4 * len(groups) + 4))
            fig, ax = plt.subplots(figsize=(width, 4))
            try:
                ax.boxplot(groups, tick_labels=labels, showfliers=False)
            except TypeError:
                ax.boxplot(groups, labels=labels, showfliers=False)
            ax.set_ylabel("Scrublet doublet score")
            ax.set_xlabel(group_col)
            ax.tick_params(axis="x", rotation=90)
            fig.tight_layout()
            fig.savefig(plot_dir / "doublet_scores_by_batch.png", dpi=180)
            plt.close(fig)


def run_analysis(args: argparse.Namespace, job_name: Optional[str] = None) -> Path:
    """Run one doublet-detection analysis and return the output directory."""
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if job_name is not None:
        LOGGER.info("=" * 80)
        LOGGER.info("Starting job: %s", job_name)
        LOGGER.info("Output: %s", outdir)
        LOGGER.info("Threshold: %s", args.threshold)
        LOGGER.info("Expected doublet rate: %s", args.expected_doublet_rate)
        LOGGER.info("=" * 80)

    adata = load_all_inputs(args)
    add_metadata(adata, args)
    add_qc_metrics(adata)
    batch_key = choose_batch_key(adata, args)
    adata = run_scrublet(adata, args, batch_key=batch_key)
    apply_top_fraction_calls(adata, batch_key=batch_key, fraction=getattr(args, "call_top_fraction", None))

    write_tables(adata, outdir, batch_key=batch_key, args=args)
    write_plots(adata, outdir, batch_key=batch_key, threshold=args.threshold)

    if not args.skip_h5ad:
        adata.write_h5ad(outdir / "adata_with_doublets.h5ad", compression="gzip")
    if args.write_singlets:
        singlets = adata[~adata.obs["predicted_doublet"].astype(bool)].copy()
        singlets.write_h5ad(outdir / "singlets_only.h5ad", compression="gzip")

    n_cells = adata.n_obs
    n_doublets = int(adata.obs["predicted_doublet"].sum())
    pct = 100 * n_doublets / n_cells if n_cells else 0
    LOGGER.info("Done%s. Predicted doublets: %d/%d (%.2f%%)", f" [{job_name}]" if job_name else "", n_doublets, n_cells, pct)
    LOGGER.info("Wrote: %s", outdir)

    # Explicitly free memory before the next job.
    del adata
    gc.collect()
    return outdir


def job_to_args(base_args: argparse.Namespace, job: dict) -> argparse.Namespace:
    """Create an argparse.Namespace for one JOBS entry."""
    job_args = copy.copy(base_args)
    aliases = {
        "inputs": "input",
        "input": "input",
        "sample_ids": "sample_id",
        "sample_id": "sample_id",
        "outdir": "outdir",
        "expected_doublet_rate": "expected_doublet_rate",
        "threshold": "threshold",
        "call_top_fraction": "call_top_fraction",
        "batch_key": "batch_key",
        "counts_layer": "counts_layer",
        "metadata": "metadata",
        "metadata_barcode_col": "metadata_barcode_col",
        "sample_col": "sample_col",
        "matrix": "matrix",
        "features": "features",
        "barcodes": "barcodes",
        "var_names": "var_names",
        "no_gex_only": "no_gex_only",
        "sim_doublet_ratio": "sim_doublet_ratio",
        "n_neighbors": "n_neighbors",
        "n_prin_comps": "n_prin_comps",
        "random_state": "random_state",
        "min_counts": "min_counts",
        "min_cells": "min_cells",
        "min_gene_variability_pctl": "min_gene_variability_pctl",
        "skip_h5ad": "skip_h5ad",
        "write_singlets": "write_singlets",
        "prefix_barcodes": "prefix_barcodes",
        "verbose": "verbose",
    }
    for key, value in job.items():
        if key == "name":
            continue
        if key not in aliases:
            raise KeyError(f"Unknown JOBS key '{key}' in job '{job.get('name', '<unnamed>')}'.")
        setattr(job_args, aliases[key], value)

    if not getattr(job_args, "input", None):
        raise ValueError(f"Job '{job.get('name', '<unnamed>')}' has no inputs.")
    return job_args


def run_jobs(args: argparse.Namespace) -> int:
    if not JOBS:
        raise ValueError("RUN_JOBS=True, but JOBS is empty.")

    requested = set(args.only_job or [])
    selected_jobs = [job for job in JOBS if not requested or job.get("name") in requested]

    missing = requested.difference({job.get("name") for job in JOBS})
    if missing:
        raise ValueError(f"Requested --only-job entries not found in JOBS: {sorted(missing)}")
    if not selected_jobs:
        raise ValueError("No jobs selected.")

    LOGGER.info("Running %d job(s): %s", len(selected_jobs), ", ".join(str(j.get("name")) for j in selected_jobs))

    summary_frames = []
    for job in selected_jobs:
        name = str(job.get("name", "unnamed_job"))
        job_args = job_to_args(args, job)
        outdir = run_analysis(job_args, job_name=name)

        summary_path = outdir / "doublet_summary.tsv"
        if summary_path.exists():
            summary = pd.read_csv(summary_path, sep="\t")
            summary.insert(0, "job", name)
            summary.insert(1, "outdir", str(outdir))
            summary.insert(2, "expected_doublet_rate", job_args.expected_doublet_rate)
            summary.insert(3, "threshold_setting", "auto" if job_args.threshold is None else job_args.threshold)
            summary_frames.append(summary)

    if summary_frames:
        combined = pd.concat(summary_frames, ignore_index=True)
        if args.jobs_summary_tsv is not None:
            combined_path = Path(args.jobs_summary_tsv)
        else:
            combined_path = Path(selected_jobs[0]["outdir"]).parent / "doublet_detection_jobs_summary.tsv"
        combined_path.parent.mkdir(parents=True, exist_ok=True)
        combined.to_csv(combined_path, sep="\t", index=False)
        LOGGER.info("Wrote combined job summary: %s", combined_path)

    return 0


def main() -> int:
    args = parse_args()
    setup_logging(args.verbose)
    sc.settings.verbosity = 2 if args.verbose else 1

    if args.run_jobs:
        return run_jobs(args)

    run_analysis(args)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        LOGGER.error("Interrupted")
        raise SystemExit(130)
    except Exception as exc:
        LOGGER.error("Failed: %s", exc)
        if "--verbose" in sys.argv or VERBOSE:
            raise
        raise SystemExit(1)
