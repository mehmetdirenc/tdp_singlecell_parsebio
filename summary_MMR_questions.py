#!/usr/bin/env python3
import os
import pandas as pd
import numpy as np

REPORTS = "/mnt/storage1/projects/research/24035_1345_TDP43/scanpy_analysis/reports_DE_raw"

TOP_FILE = os.path.join(REPORTS, "MMR_top_celltypes_per_gene_min20.tsv")
COMPARE_FILE = os.path.join(REPORTS, "MMR_expression_compare_ct_label_mv_control_vs_tdp43_min10perCondition.tsv")
DE_FILE = os.path.join(REPORTS, "DE_tdp43_vs_control_MMR_only.csv")

OUT_Q1 = os.path.join(REPORTS, "MMR_Q1_top_expressing_celltypes_summary.tsv")
OUT_Q2_OVERALL = os.path.join(REPORTS, "MMR_Q2_overall_DE_summary.tsv")
OUT_Q2_CELLTYPE = os.path.join(REPORTS, "MMR_Q2_celltype_shift_summary.tsv")
OUT_COMBINED = os.path.join(REPORTS, "MMR_Q1_Q2_combined_summary.tsv")

target_genes = ["Mlh1", "Msh2", "Msh3", "Msh6", "Pms1", "Pms2"]

# --------------------------------------------------
# load files
# --------------------------------------------------
top_df = pd.read_csv(TOP_FILE, sep="\t")
comp_df = pd.read_csv(COMPARE_FILE, sep="\t")
de_df = pd.read_csv(DE_FILE)

# keep expected genes only
top_df = top_df[top_df["gene"].isin(target_genes)].copy()
comp_df = comp_df[comp_df["gene"].isin(target_genes)].copy()
de_df = de_df[de_df["names"].isin(target_genes)].copy()

# --------------------------------------------------
# Q1: which cell types express these genes?
# Take top 3 expressing cell types per gene
# --------------------------------------------------
q1_rows = []
for gene in target_genes:
    sub = top_df[top_df["gene"] == gene].copy()
    sub = sub.sort_values(
        ["pct_expr", "mean_log1p_norm_expr", "n_cells"],
        ascending=[False, False, False]
    ).head(3)

    row = {"gene": gene}
    for i, (_, r) in enumerate(sub.iterrows(), start=1):
        row[f"top{i}_cell_type"] = r["cell_type"]
        row[f"top{i}_n_cells"] = int(r["n_cells"])
        row[f"top{i}_pct_expr"] = float(r["pct_expr"])
        row[f"top{i}_mean_log1p_norm_expr"] = float(r["mean_log1p_norm_expr"])
    q1_rows.append(row)

q1_df = pd.DataFrame(q1_rows)
q1_df.to_csv(OUT_Q1, sep="\t", index=False)

# --------------------------------------------------
# Q2 overall: more or less after virus?
# From DE_tdp43_vs_control_MMR_only.csv
# --------------------------------------------------
q2_overall = de_df.copy()
q2_overall["direction_tdp43_vs_control"] = np.where(
    q2_overall["logfoldchanges"] > 0, "higher_in_tdp43",
    np.where(q2_overall["logfoldchanges"] < 0, "lower_in_tdp43", "no_change")
)

q2_overall = q2_overall[[
    "names",
    "direction_tdp43_vs_control",
    "logfoldchanges",
    "pvals_adj",
    "pct_expr_tdp43",
    "pct_expr_control",
    "mean_expr_tdp43",
    "mean_expr_control",
    "n_tdp43_cells",
    "n_control_cells",
]].rename(columns={"names": "gene"})

q2_overall.to_csv(OUT_Q2_OVERALL, sep="\t", index=False)

# --------------------------------------------------
# Q2 by cell type: strongest positive/negative shifts
# Based on descriptive cell-type comparison file
# --------------------------------------------------
q2_ct_rows = []
for gene in target_genes:
    sub = comp_df[comp_df["gene"] == gene].copy()
    if sub.empty:
        continue

    sub_pos = sub.sort_values(
        ["delta_mean_log1p_norm_expr_tdp43_minus_control", "delta_pct_expr_tdp43_minus_control"],
        ascending=[False, False]
    ).head(1)

    sub_neg = sub.sort_values(
        ["delta_mean_log1p_norm_expr_tdp43_minus_control", "delta_pct_expr_tdp43_minus_control"],
        ascending=[True, True]
    ).head(1)

    row = {"gene": gene}

    if not sub_pos.empty:
        r = sub_pos.iloc[0]
        row["most_higher_in_tdp43_cell_type"] = r["cell_type"]
        row["most_higher_in_tdp43_n_control"] = r["n_cells_control"]
        row["most_higher_in_tdp43_n_tdp43"] = r["n_cells_tdp43"]
        row["most_higher_in_tdp43_delta_expr"] = r["delta_mean_log1p_norm_expr_tdp43_minus_control"]
        row["most_higher_in_tdp43_delta_pct"] = r["delta_pct_expr_tdp43_minus_control"]

    if not sub_neg.empty:
        r = sub_neg.iloc[0]
        row["most_lower_in_tdp43_cell_type"] = r["cell_type"]
        row["most_lower_in_tdp43_n_control"] = r["n_cells_control"]
        row["most_lower_in_tdp43_n_tdp43"] = r["n_cells_tdp43"]
        row["most_lower_in_tdp43_delta_expr"] = r["delta_mean_log1p_norm_expr_tdp43_minus_control"]
        row["most_lower_in_tdp43_delta_pct"] = r["delta_pct_expr_tdp43_minus_control"]

    q2_ct_rows.append(row)

q2_ct_df = pd.DataFrame(q2_ct_rows)
q2_ct_df.to_csv(OUT_Q2_CELLTYPE, sep="\t", index=False)

# --------------------------------------------------
# Combined summary
# --------------------------------------------------
combined = q1_df.merge(q2_overall, on="gene", how="outer").merge(q2_ct_df, on="gene", how="outer")
combined.to_csv(OUT_COMBINED, sep="\t", index=False)

# --------------------------------------------------
# print quick text summary
# --------------------------------------------------
print("\n=== Q1: Which cell types express these genes? ===")
for _, row in q1_df.iterrows():
    print(f"\n{row['gene']}:")
    for i in [1, 2, 3]:
        ct = row.get(f"top{i}_cell_type", None)
        if pd.notna(ct):
            print(
                f"  Top {i}: {ct} | n_cells={row[f'top{i}_n_cells']} | "
                f"pct_expr={row[f'top{i}_pct_expr']:.3f} | "
                f"mean_log1p_norm_expr={row[f'top{i}_mean_log1p_norm_expr']:.3f}"
            )

print("\n=== Q2: Overall direction after virus (TDP43 vs control) ===")
for _, row in q2_overall.iterrows():
    print(
        f"{row['gene']}: {row['direction_tdp43_vs_control']} | "
        f"logFC={row['logfoldchanges']:.3f} | "
        f"pct_expr_tdp43={row['pct_expr_tdp43']:.3f} | "
        f"pct_expr_control={row['pct_expr_control']:.3f}"
    )

print("\n=== Q2: Strongest cell-type-level shifts (descriptive) ===")
for _, row in q2_ct_df.iterrows():
    print(f"\n{row['gene']}:")
    print(
        f"  Most higher in TDP43: {row.get('most_higher_in_tdp43_cell_type', 'NA')} | "
        f"delta_expr={row.get('most_higher_in_tdp43_delta_expr', np.nan):.3f} | "
        f"delta_pct={row.get('most_higher_in_tdp43_delta_pct', np.nan):.3f}"
    )
    print(
        f"  Most lower in TDP43: {row.get('most_lower_in_tdp43_cell_type', 'NA')} | "
        f"delta_expr={row.get('most_lower_in_tdp43_delta_expr', np.nan):.3f} | "
        f"delta_pct={row.get('most_lower_in_tdp43_delta_pct', np.nan):.3f}"
    )

print("\n[IO] Saved:")
print("  ", OUT_Q1)
print("  ", OUT_Q2_OVERALL)
print("  ", OUT_Q2_CELLTYPE)
print("  ", OUT_COMBINED)