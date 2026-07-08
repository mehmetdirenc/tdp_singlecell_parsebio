#!/usr/bin/env python3
import os
import inspect
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
import celltypist
from celltypist import models

ADATA = "/mnt/storage1/projects/research/24035_1345_TDP43/scanpy_analysis/adata_files"
REPORTS = "/mnt/storage1/projects/research/24035_1345_TDP43/scanpy_analysis/reports_DE_raw"
FIGDIR = "/mnt/storage1/projects/research/24035_1345_TDP43/scanpy_analysis/figures_MMR_raw"

IN_H5AD = os.path.join(ADATA, "mouse_adata_RAW.h5ad")
OUT_H5AD = os.path.join(ADATA, "mouse_adata_RAW_celltypist.h5ad")

os.makedirs(REPORTS, exist_ok=True)
os.makedirs(FIGDIR, exist_ok=True)

CELLTYPIST_MODEL = "Mouse_Whole_Brain.pkl"
CONF_CUT = 0.50
BC1_COL = "bc1_well"

MIN_CELLS_FOR_TOP = 20
MIN_CELLS_PER_CONDITION = 10
LOWCONF_LABEL = "Unknown_lowConf"

target_genes = ["Mlh1", "Msh2", "Msh3", "Msh6", "Pms1", "Pms2"]

print("[CellTypist] version:", getattr(celltypist, "__version__", "unknown"))
try:
    print("[CellTypist] module file:", inspect.getfile(celltypist))
except Exception:
    pass


def _as_series(x, preferred=None, index=None):
    if isinstance(x, pd.Series):
        return x.reindex(index)
    if isinstance(x, pd.DataFrame):
        if preferred and preferred in x.columns:
            s = x[preferred]
        elif "predicted_labels" in x.columns:
            s = x["predicted_labels"]
        elif "majority_voting" in x.columns:
            s = x["majority_voting"]
        elif "labels" in x.columns:
            s = x["labels"]
        elif x.shape[1] == 1:
            s = x.iloc[:, 0]
        else:
            s = x.iloc[:, 0]
        return s.reindex(index)
    return pd.Series(np.asarray(x).reshape(-1), index=index)


def ensure_model_available(model_name):
    try:
        if hasattr(models, "download_model"):
            models.download_model(model_name)
        else:
            raise AttributeError("module 'celltypist.models' has no attribute 'download_model'")
    except Exception as e:
        print(f"[CellTypist] download_model failed: {e}. Trying full catalog...")
        try:
            models.download_models()
        except Exception as e2:
            print(f"[CellTypist] download_models failed: {e2}")


def save_dotplot(adata_plot, genes, groupby, out_png):
    if groupby not in adata_plot.obs.columns:
        print(f"[WARN] {groupby} not in adata.obs, skipping {out_png}")
        return
    sc.pl.dotplot(
        adata_plot,
        var_names=genes,
        groupby=groupby,
        use_raw=False,
        show=False
    )
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close()


# --------------------------------------------------
# load data
# --------------------------------------------------
adata = sc.read_h5ad(IN_H5AD)
print("Loaded:", adata.shape)

missing = [g for g in target_genes if g not in adata.var_names]
if missing:
    raise ValueError(f"Missing target genes in adata.var_names: {missing}")
print("All target genes found.")

# --------------------------------------------------
# define condition from bc1_well (no extra QC filtering)
# --------------------------------------------------
if BC1_COL in adata.obs.columns:
    control_wells = {f"A{i}" for i in range(1, 13)}
    tdp43_wells = {f"B{i}" for i in range(1, 13)}

    bc1 = adata.obs[BC1_COL].astype(str)
    adata.obs["condition"] = pd.Series(index=adata.obs.index, dtype="object")
    adata.obs.loc[bc1.isin(control_wells), "condition"] = "control"
    adata.obs.loc[bc1.isin(tdp43_wells), "condition"] = "tdp43"

    print("[Condition split]")
    print("Control A1-A12 cells:", int((adata.obs["condition"] == "control").sum()))
    print("TDP43 B1-B12 cells:", int((adata.obs["condition"] == "tdp43").sum()))
    print("Unassigned cells:", int(adata.obs["condition"].isna().sum()))
else:
    print(f"[WARN] {BC1_COL} not found, no condition column added.")

# --------------------------------------------------
# normalized copy for CellTypist / plotting only
# --------------------------------------------------
adata_ct = adata.copy()
sc.pp.normalize_total(adata_ct, target_sum=1e4)
sc.pp.log1p(adata_ct)

# --------------------------------------------------
# run CellTypist
# --------------------------------------------------
ensure_model_available(CELLTYPIST_MODEL)
pred = celltypist.annotate(adata_ct, model=CELLTYPIST_MODEL, majority_voting=True)

adata.obs["ct_label"] = _as_series(
    getattr(pred, "predicted_labels", None),
    preferred="predicted_labels",
    index=adata.obs_names
).astype("object")

mv_obj = getattr(pred, "majority_voting", None)
if mv_obj is not None:
    adata.obs["ct_label_mv"] = _as_series(
        mv_obj,
        preferred="majority_voting",
        index=adata.obs_names
    ).astype("object")
else:
    adata.obs["ct_label_mv"] = adata.obs["ct_label"].astype("object")

conf_obj = getattr(pred, "confidence", None)
if conf_obj is not None:
    adata.obs["ct_conf"] = _as_series(
        conf_obj,
        preferred="confidence",
        index=adata.obs_names
    ).astype(float)
else:
    adata.obs["ct_conf"] = np.nan

low = adata.obs["ct_conf"] < CONF_CUT
adata.obs.loc[low, "ct_label_mv"] = LOWCONF_LABEL

# copy annotations to normalized object
adata_ct.obs["ct_label"] = adata.obs["ct_label"].astype(str).values
adata_ct.obs["ct_label_mv"] = adata.obs["ct_label_mv"].astype(str).values
adata_ct.obs["ct_conf"] = adata.obs["ct_conf"].values
if "condition" in adata.obs.columns:
    adata_ct.obs["condition"] = adata.obs["condition"].astype(str).values

# --------------------------------------------------
# save annotated raw object
# --------------------------------------------------
adata.write(OUT_H5AD)
print("[IO] Saved:", OUT_H5AD)

# --------------------------------------------------
# expression summaries
# --------------------------------------------------
expr_raw = adata[:, target_genes].to_df()
expr_norm = adata_ct[:, target_genes].to_df()

label_col = "ct_label_mv"

summary_rows = []
for ct in sorted(adata.obs[label_col].astype(str).dropna().unique()):
    mask = adata.obs[label_col].astype(str) == ct
    n_cells = int(mask.sum())

    for gene in target_genes:
        vals_raw = expr_raw.loc[mask, gene]
        vals_norm = expr_norm.loc[mask, gene]

        summary_rows.append({
            "cell_type": ct,
            "gene": gene,
            "n_cells": n_cells,
            "pct_expr": float((vals_raw > 0).mean()),
            "mean_raw_expr": float(vals_raw.mean()),
            "mean_log1p_norm_expr": float(vals_norm.mean()),
            "median_raw_expr": float(vals_raw.median()),
        })

summary_df = pd.DataFrame(summary_rows)
summary_out = os.path.join(REPORTS, "MMR_expression_by_ct_label_mv.tsv")
summary_df.to_csv(summary_out, sep="\t", index=False)

# filtered expression summary for interpretable cell types
summary_df_min = summary_df[
    (summary_df["cell_type"] != LOWCONF_LABEL) &
    (summary_df["n_cells"] >= MIN_CELLS_FOR_TOP)
].copy()
summary_min_out = os.path.join(REPORTS, f"MMR_expression_by_ct_label_mv_min{MIN_CELLS_FOR_TOP}.tsv")
summary_df_min.to_csv(summary_min_out, sep="\t", index=False)

# --------------------------------------------------
# condition-aware summary
# --------------------------------------------------
summary_cond_out = os.path.join(REPORTS, "MMR_expression_by_ct_label_mv_condition.tsv")
summary_compare_out = os.path.join(REPORTS, "MMR_expression_compare_ct_label_mv_control_vs_tdp43.tsv")
summary_compare_min_out = os.path.join(
    REPORTS,
    f"MMR_expression_compare_ct_label_mv_control_vs_tdp43_min{MIN_CELLS_PER_CONDITION}perCondition.tsv"
)

if "condition" in adata.obs.columns:
    summary_rows_cond = []

    valid = adata.obs["condition"].notna()
    for ct in sorted(adata.obs.loc[valid, label_col].astype(str).dropna().unique()):
        for cond in ["control", "tdp43"]:
            mask = (
                (adata.obs[label_col].astype(str) == ct) &
                (adata.obs["condition"].astype(str) == cond)
            )
            n_cells = int(mask.sum())
            if n_cells == 0:
                continue

            for gene in target_genes:
                vals_raw = expr_raw.loc[mask, gene]
                vals_norm = expr_norm.loc[mask, gene]

                summary_rows_cond.append({
                    "cell_type": ct,
                    "condition": cond,
                    "gene": gene,
                    "n_cells": n_cells,
                    "pct_expr": float((vals_raw > 0).mean()),
                    "mean_raw_expr": float(vals_raw.mean()),
                    "mean_log1p_norm_expr": float(vals_norm.mean()),
                    "median_raw_expr": float(vals_raw.median()),
                })

    summary_df_cond = pd.DataFrame(summary_rows_cond)
    summary_df_cond.to_csv(summary_cond_out, sep="\t", index=False)

    # build side-by-side control vs tdp43 comparison
    comp = summary_df_cond.pivot_table(
        index=["cell_type", "gene"],
        columns="condition",
        values=["n_cells", "pct_expr", "mean_raw_expr", "mean_log1p_norm_expr", "median_raw_expr"],
        aggfunc="first"
    )

    comp.columns = [f"{a}_{b}" for a, b in comp.columns]
    comp = comp.reset_index()

    # ensure expected columns exist
    for c in [
        "n_cells_control", "n_cells_tdp43",
        "pct_expr_control", "pct_expr_tdp43",
        "mean_raw_expr_control", "mean_raw_expr_tdp43",
        "mean_log1p_norm_expr_control", "mean_log1p_norm_expr_tdp43",
        "median_raw_expr_control", "median_raw_expr_tdp43"
    ]:
        if c not in comp.columns:
            comp[c] = np.nan

    comp["delta_pct_expr_tdp43_minus_control"] = comp["pct_expr_tdp43"] - comp["pct_expr_control"]
    comp["delta_mean_log1p_norm_expr_tdp43_minus_control"] = (
        comp["mean_log1p_norm_expr_tdp43"] - comp["mean_log1p_norm_expr_control"]
    )
    comp["delta_mean_raw_expr_tdp43_minus_control"] = (
        comp["mean_raw_expr_tdp43"] - comp["mean_raw_expr_control"]
    )

    comp.to_csv(summary_compare_out, sep="\t", index=False)

    # filtered comparison table
    comp_min = comp[
        (comp["cell_type"] != LOWCONF_LABEL) &
        (comp["n_cells_control"].fillna(0) >= MIN_CELLS_PER_CONDITION) &
        (comp["n_cells_tdp43"].fillna(0) >= MIN_CELLS_PER_CONDITION)
    ].copy()

    comp_min.to_csv(summary_compare_min_out, sep="\t", index=False)
else:
    pd.DataFrame().to_csv(summary_cond_out, sep="\t", index=False)
    pd.DataFrame().to_csv(summary_compare_out, sep="\t", index=False)
    pd.DataFrame().to_csv(summary_compare_min_out, sep="\t", index=False)

# --------------------------------------------------
# top expressing cell types per gene
# --------------------------------------------------
top_df = (
    summary_df
    .sort_values(["gene", "pct_expr", "mean_log1p_norm_expr"], ascending=[True, False, False])
    .groupby("gene", as_index=False)
    .head(10)
)
top_out = os.path.join(REPORTS, "MMR_top_celltypes_per_gene.tsv")
top_df.to_csv(top_out, sep="\t", index=False)

top_df_min = (
    summary_df[
        (summary_df["cell_type"] != LOWCONF_LABEL) &
        (summary_df["n_cells"] >= MIN_CELLS_FOR_TOP)
    ]
    .sort_values(["gene", "pct_expr", "mean_log1p_norm_expr"], ascending=[True, False, False])
    .groupby("gene", as_index=False)
    .head(10)
)
top_min_out = os.path.join(REPORTS, f"MMR_top_celltypes_per_gene_min{MIN_CELLS_FOR_TOP}.tsv")
top_df_min.to_csv(top_min_out, sep="\t", index=False)

# --------------------------------------------------
# plots
# --------------------------------------------------
dotplot_ct_out = os.path.join(FIGDIR, "dotplot_MMR_by_ct_label_mv.png")
save_dotplot(adata_ct, target_genes, "ct_label_mv", dotplot_ct_out)

dotplot_ct_cond_out = os.path.join(FIGDIR, "dotplot_MMR_by_ct_label_mv_condition.png")
if "condition" in adata_ct.obs.columns:
    adata_ct_sub = adata_ct[adata_ct.obs["condition"].isin(["control", "tdp43"])].copy()
    adata_ct_sub.obs["ct_condition"] = (
        adata_ct_sub.obs["ct_label_mv"].astype(str) + " | " + adata_ct_sub.obs["condition"].astype(str)
    )
    save_dotplot(adata_ct_sub, target_genes, "ct_condition", dotplot_ct_cond_out)

# --------------------------------------------------
# print useful summaries to stdout
# --------------------------------------------------
print(f"\nTop expressing cell types per gene (all labels):")
for gene in target_genes:
    print(f"\nTop cell types for {gene}:")
    print(
        top_df[top_df["gene"] == gene][
            ["cell_type", "n_cells", "pct_expr", "mean_log1p_norm_expr"]
        ].head(5).to_string(index=False)
    )

print(f"\nTop expressing cell types per gene (excluding {LOWCONF_LABEL}, min {MIN_CELLS_FOR_TOP} cells):")
for gene in target_genes:
    sub = top_df_min[top_df_min["gene"] == gene][
        ["cell_type", "n_cells", "pct_expr", "mean_log1p_norm_expr"]
    ].head(5)
    print(f"\nTop filtered cell types for {gene}:")
    if len(sub) == 0:
        print("No cell types passed filter.")
    else:
        print(sub.to_string(index=False))

print("\n[IO] Saved:")
print("   ", OUT_H5AD)
print("   ", summary_out)
print("   ", summary_min_out)
print("   ", summary_cond_out)
print("   ", summary_compare_out)
print("   ", summary_compare_min_out)
print("   ", top_out)
print("   ", top_min_out)
print("   ", dotplot_ct_out)
if "condition" in adata_ct.obs.columns:
    print("   ", dotplot_ct_cond_out)