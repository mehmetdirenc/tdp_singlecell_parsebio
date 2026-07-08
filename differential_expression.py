import os
import scanpy as sc
import pandas as pd
import numpy as np

ADATA = '/mnt/storage1/projects/research/24035_1345_TDP43/scanpy_analysis/adata_files'
reports = '/mnt/storage1/projects/research/24035_1345_TDP43/scanpy_analysis/reports'
os.makedirs(reports, exist_ok=True)

adata = sc.read_h5ad("/mnt/storage1/projects/research/24035_1345_TDP43/scanpy_analysis/adata_files/mouse_adata_MAIN_postDoublet_processed.h5ad")

control_samples = [f"A{i}" for i in range(1, 13)]
tdp43_samples   = [f"B{i}" for i in range(1, 13)]

adata.obs["condition"] = pd.Series(index=adata.obs.index, dtype="object")
adata.obs.loc[adata.obs["sample"].isin(control_samples), "condition"] = "control"
adata.obs.loc[adata.obs["sample"].isin(tdp43_samples), "condition"] = "tdp43"

adata = adata[adata.obs["condition"].notna()].copy()

adata.obs["condition"] = pd.Categorical(
    adata.obs["condition"],
    categories=["control", "tdp43"]
)

# --------------------------------------------------
# 2) Make sure expression matrix is appropriate
# rank_genes_groups expects log-transformed data
# --------------------------------------------------
# Usually use adata.X if it already contains log1p-normalized values.
# If your raw counts are in adata.raw, you can use use_raw=True below instead.

# --------------------------------------------------
# 3) Differential expression - all cells together
# --------------------------------------------------
sc.tl.rank_genes_groups(
    adata,
    groupby="condition",
    groups=["tdp43"],
    reference="control",
    method="wilcoxon",
    pts=True,
    key_added="de_tdp43_vs_control"
)

# --------------------------------------------------
# 4) Plot top genes - all cells together
# --------------------------------------------------
sc.pl.rank_genes_groups(
    adata,
    key="de_tdp43_vs_control",
    groups=["tdp43"],
    n_genes=25,
    sharey=False,
    show=False
)

# --------------------------------------------------
# 5) Extract results into a dataframe - all cells together
# --------------------------------------------------
res = sc.get.rank_genes_groups_df(
    adata,
    group="tdp43",
    key="de_tdp43_vs_control"
)

# Add percent expressing if available
pts = adata.uns["de_tdp43_vs_control"].get("pts")
if pts is not None:
    pts = pts.copy()
    if "tdp43" in pts.columns:
        res["pct_expr_tdp43"] = res["names"].map(pts["tdp43"])
    if "control" in pts.columns:
        res["pct_expr_control"] = res["names"].map(pts["control"])

# --------------------------------------------------
# 6) Simple filtering for more sensible candidates - all cells together
# --------------------------------------------------
res_filt = res.copy()

if "pct_expr_tdp43" in res_filt.columns and "pct_expr_control" in res_filt.columns:
    res_filt = res_filt[
        ((res_filt["pct_expr_tdp43"] >= 0.10) | (res_filt["pct_expr_control"] >= 0.10))
    ]

if "logfoldchanges" in res_filt.columns:
    res_filt = res_filt[res_filt["logfoldchanges"].abs() >= 0.25]

res_filt = res_filt.sort_values(["pvals_adj", "logfoldchanges"], ascending=[True, False])

de_path = os.path.join(reports, "DE_tdp43_vs_control_all.csv")
filtered_de_path = os.path.join(reports, "DE_tdp43_vs_control_filtered.csv")

res.to_csv(de_path, index=False)
res_filt.to_csv(filtered_de_path, index=False)

print("Top filtered DE genes across all cells:")
print(res_filt.head(30))

# --------------------------------------------------
# 7) Cell type based DE comparison
# --------------------------------------------------
# Change this if your cell type annotation column has another name
CELLTYPE_COL = "cell_type"

if CELLTYPE_COL not in adata.obs.columns:
    raise ValueError(
        f"'{CELLTYPE_COL}' not found in adata.obs. Available columns include: {list(adata.obs.columns)}"
    )

celltype_tables_all = []
celltype_tables_filt = []
celltype_summary = []

for ct in sorted(adata.obs[CELLTYPE_COL].dropna().unique()):
    sub = adata[adata.obs[CELLTYPE_COL] == ct].copy()

    counts = sub.obs["condition"].value_counts()
    n_control = int(counts.get("control", 0))
    n_tdp43 = int(counts.get("tdp43", 0))

    # keep track of summary even for skipped groups
    summary_row = {
        "cell_type": ct,
        "n_control_cells": n_control,
        "n_tdp43_cells": n_tdp43,
        "status": "tested"
    }

    # skip if one condition missing
    if n_control == 0 or n_tdp43 == 0:
        summary_row["status"] = "skipped_missing_condition"
        celltype_summary.append(summary_row)
        continue

    # skip tiny groups
    if n_control < 20 or n_tdp43 < 20:
        summary_row["status"] = "skipped_too_few_cells"
        celltype_summary.append(summary_row)
        continue

    key_name = f"de_tdp43_vs_control_{str(ct).replace(' ', '_').replace('/', '_')}"

    sc.tl.rank_genes_groups(
        sub,
        groupby="condition",
        groups=["tdp43"],
        reference="control",
        method="wilcoxon",
        pts=True,
        key_added=key_name
    )

    df = sc.get.rank_genes_groups_df(
        sub,
        group="tdp43",
        key=key_name
    )
    df["cell_type"] = ct
    df["n_control_cells"] = n_control
    df["n_tdp43_cells"] = n_tdp43

    # Add percent expressing if available
    pts_sub = sub.uns[key_name].get("pts")
    if pts_sub is not None:
        pts_sub = pts_sub.copy()
        if "tdp43" in pts_sub.columns:
            df["pct_expr_tdp43"] = df["names"].map(pts_sub["tdp43"])
        if "control" in pts_sub.columns:
            df["pct_expr_control"] = df["names"].map(pts_sub["control"])

    # Filter within each cell type
    df_filt = df.copy()

    if "pct_expr_tdp43" in df_filt.columns and "pct_expr_control" in df_filt.columns:
        df_filt = df_filt[
            ((df_filt["pct_expr_tdp43"] >= 0.10) | (df_filt["pct_expr_control"] >= 0.10))
        ]

    if "logfoldchanges" in df_filt.columns:
        df_filt = df_filt[df_filt["logfoldchanges"].abs() >= 0.25]

    df = df.sort_values(["pvals_adj", "logfoldchanges"], ascending=[True, False])
    df_filt = df_filt.sort_values(["pvals_adj", "logfoldchanges"], ascending=[True, False])

    # save per-celltype files
    safe_ct = str(ct).replace(" ", "_").replace("/", "_")
    per_ct_all_path = os.path.join(reports, f"DE_tdp43_vs_control_{safe_ct}_all.csv")
    per_ct_filt_path = os.path.join(reports, f"DE_tdp43_vs_control_{safe_ct}_filtered.csv")

    df.to_csv(per_ct_all_path, index=False)
    df_filt.to_csv(per_ct_filt_path, index=False)

    # store combined tables
    celltype_tables_all.append(df)
    celltype_tables_filt.append(df_filt)

    summary_row["status"] = "tested"
    summary_row["n_genes_all"] = df.shape[0]
    summary_row["n_genes_filtered"] = df_filt.shape[0]
    celltype_summary.append(summary_row)

# --------------------------------------------------
# 8) Save combined cell type DE tables
# --------------------------------------------------
summary_df = pd.DataFrame(celltype_summary)
summary_path = os.path.join(reports, "DE_tdp43_vs_control_by_celltype_summary.csv")
summary_df.to_csv(summary_path, index=False)

if len(celltype_tables_all) > 0:
    de_by_celltype_all = pd.concat(celltype_tables_all, ignore_index=True)
    de_by_celltype_all_path = os.path.join(reports, "DE_tdp43_vs_control_by_celltype_all.csv")
    de_by_celltype_all.to_csv(de_by_celltype_all_path, index=False)

if len(celltype_tables_filt) > 0:
    de_by_celltype_filt = pd.concat(celltype_tables_filt, ignore_index=True)
    de_by_celltype_filt_path = os.path.join(reports, "DE_tdp43_vs_control_by_celltype_filtered.csv")
    de_by_celltype_filt.to_csv(de_by_celltype_filt_path, index=False)

print("\nCell type DE summary:")
print(summary_df)