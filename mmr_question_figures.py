#!/usr/bin/env python3
import os
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt

from scipy import sparse
from scipy.stats import mannwhitneyu, spearmanr
from statsmodels.stats.multitest import multipletests

# ============================================================
# User settings
# ============================================================
ADATA = "/mnt/storage1/projects/research/24035_1345_TDP43/scanpy_analysis/adata_files"

# NEW folder so v2 results stay untouched
OUTDIR = "/mnt/storage1/projects/research/24035_1345_TDP43/scanpy_analysis/figures_tdp43_question_panels_v3"
TABLEDIR = os.path.join(OUTDIR, "tables")
FIGDIR = os.path.join(OUTDIR, "figures")

IN_H5AD = os.path.join(ADATA, "mouse_adata_RAW_celltypist.h5ad")

LABEL_COL = "ct_label_mv"
BC1_COL = "bc1_well"
LOWCONF_LABEL = "Unknown_lowConf"

TARGET_GENES = ["Mlh1", "Msh2", "Msh3", "Msh6", "Pms1", "Pms2"]

CONTROL_WELLS = {f"A{i}" for i in range(1, 13)}
TDP43_WELLS = {f"B{i}" for i in range(1, 13)}

# abundance filters
MIN_TOTAL_CELLS = 20
MIN_CONTROL_CELLS = 10
MIN_TDP43_CELLS = 10

# stricter filter for the "enriched" panel to avoid 0 vs 23-type artifacts
Q2_MIN_CONTROL_CELLS = 10
Q2_MIN_TDP43_CELLS = 10

# DE plotting filters for Q4
Q4_MIN_PCT_EXPR = 0.10
Q4_MIN_ABS_LOGFC = 0.25
Q4_TOP_EACH_SIDE = 8

EPS = 1e-6

# optional manual overrides
DEPLETED_CELLTYPE_OVERRIDE = None
ENRICHED_CELLTYPE_OVERRIDE = None

# prefer these for enriched glia/stromal-like candidate search
GLIA_KEYWORDS = [
    "ASTRO", "OLIGO", "OPC", "MICRO", "EPEN", "TANY", "VLMC",
    "AQP4", "MOG", "PLP1", "PDGFRA", "CLDN5", "RGS5",
    "SMC", "LYMPHOID", "BAM", "CHOR", "ENDO", "PERI"
]

os.makedirs(TABLEDIR, exist_ok=True)
os.makedirs(FIGDIR, exist_ok=True)


# ============================================================
# Helpers
# ============================================================
def ensure_condition(adata, bc1_col=BC1_COL):
    if "condition" in adata.obs.columns and adata.obs["condition"].notna().any():
        adata.obs["condition"] = adata.obs["condition"].astype("object")
        return adata

    if bc1_col not in adata.obs.columns:
        raise ValueError(f"{bc1_col} not found in adata.obs")

    bc1 = adata.obs[bc1_col].astype(str)
    adata.obs["condition"] = pd.Series(index=adata.obs.index, dtype="object")
    adata.obs.loc[bc1.isin(CONTROL_WELLS), "condition"] = "control"
    adata.obs.loc[bc1.isin(TDP43_WELLS), "condition"] = "tdp43"
    return adata


def make_log1p_copy(adata):
    a = adata.copy()
    sc.pp.normalize_total(a, target_sum=1e4)
    sc.pp.log1p(a)
    return a


def get_dense_matrix(a):
    X = a.X
    if sparse.issparse(X):
        return X.toarray()
    return np.asarray(X)


def clean_label(s):
    return str(s).replace("/", "_").replace(" ", "_")


def save_table(df, filename, sep="\t"):
    path = os.path.join(TABLEDIR, filename)
    df.to_csv(path, sep=sep, index=False)
    return path


def is_glia_like(label):
    s = str(label).upper()
    return any(k in s for k in GLIA_KEYWORDS)


def classify_celltype_class(label):
    s = str(label)
    u = s.upper()

    if " NN" in u or u.endswith("NN") or is_glia_like(s):
        return "non_neuronal"

    neuronal_tokens = [" GABA", " GLUT", "DOPA-GABA", "HIST-GABA", "GABA-CHOL"]
    if any(tok in u for tok in neuronal_tokens):
        return "neuronal"

    return "other"


def filter_de_table_for_plot(df, min_pct_expr=Q4_MIN_PCT_EXPR, min_abs_logfc=Q4_MIN_ABS_LOGFC):
    out = df.copy()
    out = out[
        (out["pct_expr_tdp43"] >= min_pct_expr) |
        (out["pct_expr_control"] >= min_pct_expr)
    ].copy()
    out = out[out["logfoldchanges"].abs() >= min_abs_logfc].copy()
    out = out.sort_values(["pvals_adj", "logfoldchanges"], ascending=[True, False]).reset_index(drop=True)
    return out


# ============================================================
# Shared computation: cell-type abundance changes
# ============================================================
def compute_celltype_abundance_shift(
    adata,
    label_col=LABEL_COL,
    lowconf_label=LOWCONF_LABEL,
    min_total_cells=MIN_TOTAL_CELLS,
    eps=EPS
):
    df = adata.obs.copy()

    if label_col not in df.columns:
        raise ValueError(f"{label_col} not found in adata.obs")
    if "condition" not in df.columns:
        raise ValueError("condition not found in adata.obs")

    df = df[df["condition"].isin(["control", "tdp43"])].copy()
    df[label_col] = df[label_col].astype(str)
    df = df[df[label_col] != lowconf_label].copy()

    counts = pd.crosstab(df[label_col], df["condition"])
    for c in ["control", "tdp43"]:
        if c not in counts.columns:
            counts[c] = 0
    counts = counts[["control", "tdp43"]].copy()

    counts["total_cells"] = counts["control"] + counts["tdp43"]
    counts = counts[counts["total_cells"] >= min_total_cells].copy()

    total_control = counts["control"].sum()
    total_tdp43 = counts["tdp43"].sum()

    counts["frac_control"] = counts["control"] / total_control if total_control > 0 else np.nan
    counts["frac_tdp43"] = counts["tdp43"] / total_tdp43 if total_tdp43 > 0 else np.nan
    counts["delta_fraction_tdp43_minus_control"] = counts["frac_tdp43"] - counts["frac_control"]
    counts["log2fc_fraction_tdp43_vs_control"] = np.log2(
        (counts["frac_tdp43"] + eps) / (counts["frac_control"] + eps)
    )

    counts = counts.reset_index().rename(columns={label_col: "cell_type"})
    counts["cell_class"] = counts["cell_type"].apply(classify_celltype_class)
    counts = counts.sort_values("log2fc_fraction_tdp43_vs_control")

    save_table(counts, "celltype_abundance_shift_control_vs_tdp43.tsv")
    return counts


# ============================================================
# Q1: depleted in TDP43
# ============================================================
def question_1_depleted_celltypes(abundance_df, top_n=15):
    q1 = abundance_df[
        (abundance_df["control"] >= MIN_CONTROL_CELLS) &
        (abundance_df["tdp43"] >= MIN_TDP43_CELLS)
    ].copy()
    q1 = q1.sort_values("log2fc_fraction_tdp43_vs_control", ascending=True).head(top_n)

    save_table(q1, "Q1_depleted_celltypes.tsv")

    plt.figure(figsize=(11, max(5, 0.42 * len(q1))))
    plt.barh(q1["cell_type"], q1["log2fc_fraction_tdp43_vs_control"])
    plt.axvline(0, linestyle="--", linewidth=1)
    plt.xlabel("log2 fraction shift (TDP43 / control)")
    plt.ylabel("Cell type")
    plt.title("Q1: Cell types depleted in TDP43")
    plt.tight_layout()
    out_png = os.path.join(FIGDIR, "Q1_depleted_celltypes.png")
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close()

    return q1, out_png


# ============================================================
# Q2: enriched in TDP43, with stricter baseline filtering
# ============================================================
def question_2_enriched_celltypes(abundance_df, top_n=15):
    q2 = abundance_df[
        (abundance_df["control"] >= Q2_MIN_CONTROL_CELLS) &
        (abundance_df["tdp43"] >= Q2_MIN_TDP43_CELLS)
    ].copy()
    q2 = q2.sort_values("log2fc_fraction_tdp43_vs_control", ascending=False).head(top_n)

    save_table(q2, "Q2_enriched_celltypes.tsv")

    plt.figure(figsize=(11, max(5, 0.42 * len(q2))))
    plt.barh(q2["cell_type"], q2["log2fc_fraction_tdp43_vs_control"])
    plt.axvline(0, linestyle="--", linewidth=1)
    plt.xlabel("log2 fraction shift (TDP43 / control)")
    plt.ylabel("Cell type")
    plt.title("Q2: Cell types enriched in TDP43")
    plt.tight_layout()
    out_png = os.path.join(FIGDIR, "Q2_enriched_celltypes.png")
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close()

    return q2, out_png


# ============================================================
# Q3: baseline MMR expression vs depletion
# ============================================================
def build_mmr_condition_summary(adata_log1p, label_col=LABEL_COL, genes=TARGET_GENES):
    obs = adata_log1p.obs.copy()
    obs = obs[obs["condition"].isin(["control", "tdp43"])].copy()
    obs[label_col] = obs[label_col].astype(str)
    obs = obs[obs[label_col] != LOWCONF_LABEL].copy()

    expr = adata_log1p[:, genes].to_df()
    expr.index = adata_log1p.obs_names

    rows = []
    for ct in sorted(obs[label_col].dropna().unique()):
        row = {"cell_type": ct, "cell_class": classify_celltype_class(ct)}

        for cond in ["control", "tdp43"]:
            idx = obs.index[
                (obs[label_col] == ct) &
                (obs["condition"].astype(str) == cond)
            ]
            row[f"n_{cond}"] = len(idx)

            if len(idx) > 0:
                vals = expr.loc[idx, genes]
                for g in genes:
                    row[f"{g}_mean_{cond}"] = float(vals[g].mean())
                row[f"MMR_mean_{cond}"] = float(vals.mean(axis=1).mean())
            else:
                for g in genes:
                    row[f"{g}_mean_{cond}"] = np.nan
                row[f"MMR_mean_{cond}"] = np.nan

        row["delta_MMR_mean_tdp43_minus_control"] = row["MMR_mean_tdp43"] - row["MMR_mean_control"]
        rows.append(row)

    mmr_df = pd.DataFrame(rows)
    save_table(mmr_df, "Q3_MMR_condition_summary_by_celltype.tsv")
    return mmr_df


def question_3_mmr_vs_depletion(adata_log1p, abundance_df, genes=TARGET_GENES):
    mmr_df = build_mmr_condition_summary(adata_log1p, genes=genes)

    merged = abundance_df.merge(mmr_df, on=["cell_type", "cell_class"], how="inner").copy()

    merged = merged[
        (merged["control"] >= MIN_CONTROL_CELLS) &
        (merged["tdp43"] >= MIN_TDP43_CELLS)
    ].copy()

    save_table(merged, "Q3_MMR_vs_abundance_shift_table.tsv")

    # -----------------------------
    # correlation table: baseline CONTROL only
    # -----------------------------
    corr_rows = []
    y = merged["log2fc_fraction_tdp43_vs_control"].astype(float).values

    for g in genes:
        x = merged[f"{g}_mean_control"].astype(float).values
        rho, p = spearmanr(x, y) if len(x) >= 3 else (np.nan, np.nan)
        corr_rows.append({
            "feature": g,
            "spearman_rho_vs_log2fc_fraction_tdp43_vs_control": rho,
            "spearman_pvalue": p,
            "n_celltypes": len(x)
        })

    x = merged["MMR_mean_control"].astype(float).values
    rho, p = spearmanr(x, y) if len(x) >= 3 else (np.nan, np.nan)
    corr_rows.append({
        "feature": "MMR_mean_control",
        "spearman_rho_vs_log2fc_fraction_tdp43_vs_control": rho,
        "spearman_pvalue": p,
        "n_celltypes": len(x)
    })

    corr_df = pd.DataFrame(corr_rows)
    save_table(corr_df, "Q3_MMR_gene_correlations.tsv")

    # -----------------------------
    # OLD Q3 figure kept
    # -----------------------------
    plt.figure(figsize=(8, 6))
    plt.scatter(
        merged["MMR_mean_control"],
        merged["log2fc_fraction_tdp43_vs_control"]
    )
    plt.axhline(0, linestyle="--", linewidth=1)
    plt.xlabel("Baseline mean MMR expression in control (log1p normalized)")
    plt.ylabel("Cell-type abundance shift in TDP43\nlog2(fraction TDP43 / fraction control)")
    plt.title("Q3: Baseline MMR expression vs TDP43-associated depletion")

    label_df = pd.concat([
        merged.nsmallest(5, "log2fc_fraction_tdp43_vs_control"),
        merged.nlargest(5, "log2fc_fraction_tdp43_vs_control")
    ]).drop_duplicates("cell_type")

    for _, r in label_df.iterrows():
        plt.text(
            r["MMR_mean_control"],
            r["log2fc_fraction_tdp43_vs_control"],
            str(r["cell_type"]),
            fontsize=7
        )

    plt.tight_layout()
    out_png_old = os.path.join(FIGDIR, "Q3_MMR_vs_depletion_scatter.png")
    plt.savefig(out_png_old, dpi=300, bbox_inches="tight")
    plt.close()

    # -----------------------------
    # NEW Q3 figure: colored by neuronal / non-neuronal
    # -----------------------------
    color_map = {
        "neuronal": "tab:orange",
        "non_neuronal": "tab:blue",
        "other": "tab:gray",
    }

    plt.figure(figsize=(8, 6))
    for cls in ["neuronal", "non_neuronal", "other"]:
        sub = merged[merged["cell_class"] == cls].copy()
        if len(sub) == 0:
            continue
        plt.scatter(
            sub["MMR_mean_control"],
            sub["log2fc_fraction_tdp43_vs_control"],
            label=cls.replace("_", " "),
            color=color_map[cls],
            alpha=0.9
        )

    plt.axhline(0, linestyle="--", linewidth=1)
    plt.xlabel("Baseline mean MMR expression in control (log1p normalized)")
    plt.ylabel("Cell-type abundance shift in TDP43\nlog2(fraction TDP43 / fraction control)")
    plt.title("Q3: Baseline MMR expression vs depletion\ncolored by cell-type class")

    label_df2 = pd.concat([
        merged.nsmallest(5, "log2fc_fraction_tdp43_vs_control"),
        merged.nlargest(5, "log2fc_fraction_tdp43_vs_control")
    ]).drop_duplicates("cell_type")

    for _, r in label_df2.iterrows():
        plt.text(
            r["MMR_mean_control"],
            r["log2fc_fraction_tdp43_vs_control"],
            str(r["cell_type"]),
            fontsize=7
        )

    plt.legend(frameon=False)
    plt.tight_layout()
    out_png_colored = os.path.join(FIGDIR, "Q3_MMR_vs_depletion_scatter_by_class.png")
    plt.savefig(out_png_colored, dpi=300, bbox_inches="tight")
    plt.close()

    # -----------------------------
    # NEW Q3 figure: control vs TDP43 mean MMR
    # This directly answers the "control only?" question
    # -----------------------------
    comp = merged.copy()
    comp = comp[
        comp["MMR_mean_control"].notna() &
        comp["MMR_mean_tdp43"].notna()
    ].copy()

    plt.figure(figsize=(8, 6))
    for cls in ["neuronal", "non_neuronal", "other"]:
        sub = comp[comp["cell_class"] == cls].copy()
        if len(sub) == 0:
            continue
        plt.scatter(
            sub["MMR_mean_control"],
            sub["MMR_mean_tdp43"],
            label=cls.replace("_", " "),
            color=color_map[cls],
            alpha=0.9
        )

    all_vals = np.concatenate([
        comp["MMR_mean_control"].values,
        comp["MMR_mean_tdp43"].values
    ])
    vmin = float(np.nanmin(all_vals))
    vmax = float(np.nanmax(all_vals))
    plt.plot([vmin, vmax], [vmin, vmax], linestyle="--", linewidth=1)

    # label largest positive/negative delta cell types
    label_df3 = pd.concat([
        comp.nsmallest(5, "delta_MMR_mean_tdp43_minus_control"),
        comp.nlargest(5, "delta_MMR_mean_tdp43_minus_control")
    ]).drop_duplicates("cell_type")

    for _, r in label_df3.iterrows():
        plt.text(
            r["MMR_mean_control"],
            r["MMR_mean_tdp43"],
            str(r["cell_type"]),
            fontsize=7
        )

    plt.xlabel("Mean MMR expression in control (log1p normalized)")
    plt.ylabel("Mean MMR expression in TDP43 (log1p normalized)")
    plt.title("Q3: Mean MMR per cell type\ncontrol vs TDP43")
    plt.legend(frameon=False)
    plt.tight_layout()
    out_png_compare = os.path.join(FIGDIR, "Q3_MMR_control_vs_tdp43_scatter_by_class.png")
    plt.savefig(out_png_compare, dpi=300, bbox_inches="tight")
    plt.close()

    return merged, corr_df, out_png_old, out_png_colored, out_png_compare


# ============================================================
# DE helpers for Q4
# ============================================================
def calc_de_two_groups(subadata, group_col="condition", group1="tdp43", group0="control"):
    mask1 = (subadata.obs[group_col].astype(str) == group1).values
    mask0 = (subadata.obs[group_col].astype(str) == group0).values

    n1 = int(mask1.sum())
    n0 = int(mask0.sum())

    if n1 == 0 or n0 == 0:
        raise ValueError(f"Missing one group: {group1}={n1}, {group0}={n0}")

    X = get_dense_matrix(subadata)
    X1 = X[mask1, :]
    X0 = X[mask0, :]

    gene_names = subadata.var_names.to_numpy()

    mean1 = X1.mean(axis=0)
    mean0 = X0.mean(axis=0)
    logfc = np.log2((mean1 + 1e-9) / (mean0 + 1e-9))

    pct1 = (X1 > 0).mean(axis=0)
    pct0 = (X0 > 0).mean(axis=0)

    pvals = np.ones(X.shape[1], dtype=float)
    scores = np.zeros(X.shape[1], dtype=float)

    for i in range(X.shape[1]):
        v1 = X1[:, i]
        v0 = X0[:, i]
        if np.all(v1 == v1[0]) and np.all(v0 == v0[0]) and v1[0] == v0[0]:
            pvals[i] = 1.0
            scores[i] = 0.0
            continue
        try:
            stat, p = mannwhitneyu(v1, v0, alternative="two-sided")
            pvals[i] = p
            scores[i] = stat
        except Exception:
            pvals[i] = 1.0
            scores[i] = 0.0

    pvals_adj = multipletests(pvals, method="fdr_bh")[1]

    res = pd.DataFrame({
        "names": gene_names,
        "scores": scores,
        "logfoldchanges": logfc,
        "pvals": pvals,
        "pvals_adj": pvals_adj,
        "pct_expr_tdp43": pct1,
        "pct_expr_control": pct0,
        "mean_expr_tdp43": mean1,
        "mean_expr_control": mean0,
        "n_tdp43_cells": n1,
        "n_control_cells": n0,
    })

    res = res.sort_values(["pvals_adj", "logfoldchanges"], ascending=[True, False]).reset_index(drop=True)
    return res


def select_candidate_celltypes(abundance_df):
    if DEPLETED_CELLTYPE_OVERRIDE is not None:
        depleted = DEPLETED_CELLTYPE_OVERRIDE
    else:
        dep_sub = abundance_df[
            (abundance_df["control"] >= MIN_CONTROL_CELLS) &
            (abundance_df["tdp43"] >= MIN_TDP43_CELLS)
        ].copy()
        depleted = dep_sub.sort_values("log2fc_fraction_tdp43_vs_control", ascending=True).iloc[0]["cell_type"]

    if ENRICHED_CELLTYPE_OVERRIDE is not None:
        enriched = ENRICHED_CELLTYPE_OVERRIDE
    else:
        enr_sub = abundance_df[
            (abundance_df["control"] >= Q2_MIN_CONTROL_CELLS) &
            (abundance_df["tdp43"] >= Q2_MIN_TDP43_CELLS)
        ].copy()
        glia_sub = enr_sub[enr_sub["cell_type"].apply(is_glia_like)].copy()
        if len(glia_sub) > 0:
            enriched = glia_sub.sort_values("log2fc_fraction_tdp43_vs_control", ascending=False).iloc[0]["cell_type"]
        else:
            enriched = enr_sub.sort_values("log2fc_fraction_tdp43_vs_control", ascending=False).iloc[0]["cell_type"]

    return depleted, enriched


def plot_top_dysregulated_genes_filtered(de_full, title, out_png, top_each_side=Q4_TOP_EACH_SIDE):
    de = filter_de_table_for_plot(de_full)

    if de.empty:
        print(f"[WARN] No genes passed Q4 filters for {title}")
        return de

    up = de[de["logfoldchanges"] > 0].nsmallest(top_each_side, "pvals_adj").copy()
    down = de[de["logfoldchanges"] < 0].nsmallest(top_each_side, "pvals_adj").copy()
    plot_df = pd.concat([down, up], ignore_index=True)

    if plot_df.empty:
        print(f"[WARN] No genes to plot for {title}")
        return de

    plot_df = plot_df.sort_values("logfoldchanges")

    plt.figure(figsize=(8, max(5, 0.35 * len(plot_df))))
    plt.barh(plot_df["names"], plot_df["logfoldchanges"])
    plt.axvline(0, linestyle="--", linewidth=1)
    plt.xlabel("log2 fold change (TDP43 / control)")
    plt.ylabel("Gene")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close()

    return de


# ============================================================
# Q4: genes dysregulated in depleted / enriched cell types
# ============================================================
def question_4_dysregulated_genes(adata, abundance_df):
    depleted_ct, enriched_ct = select_candidate_celltypes(abundance_df)

    summary = pd.DataFrame({
        "candidate_type": ["depleted_candidate", "enriched_candidate"],
        "cell_type": [depleted_ct, enriched_ct]
    })
    save_table(summary, "Q4_selected_candidate_celltypes.tsv")

    outputs = {}

    for kind, ct in [("depleted", depleted_ct), ("enriched", enriched_ct)]:
        sub = adata[
            (adata.obs[LABEL_COL].astype(str) == str(ct)) &
            (adata.obs["condition"].isin(["control", "tdp43"]))
        ].copy()

        n_control = int((sub.obs["condition"].astype(str) == "control").sum())
        n_tdp43 = int((sub.obs["condition"].astype(str) == "tdp43").sum())

        if n_control < MIN_CONTROL_CELLS or n_tdp43 < MIN_TDP43_CELLS:
            print(f"[WARN] Skipping {kind} cell type {ct}: too few cells (control={n_control}, tdp43={n_tdp43})")
            continue

        de_full = calc_de_two_groups(sub, group_col="condition", group1="tdp43", group0="control")
        de_full["cell_type"] = ct

        full_tsv = save_table(de_full, f"Q4_{kind}_{clean_label(ct)}_DE_full.tsv")
        de_filt = filter_de_table_for_plot(de_full)
        filt_tsv = save_table(de_filt, f"Q4_{kind}_{clean_label(ct)}_DE_filtered.tsv")

        out_png = os.path.join(FIGDIR, f"Q4_{kind}_{clean_label(ct)}_DE_filtered_topgenes.png")
        plot_top_dysregulated_genes_filtered(
            de_full,
            title=f"Q4: Top filtered dysregulated genes in {kind} cell type\n{ct}",
            out_png=out_png,
            top_each_side=Q4_TOP_EACH_SIDE
        )

        outputs[kind] = {
            "cell_type": ct,
            "n_control": n_control,
            "n_tdp43": n_tdp43,
            "table_full": full_tsv,
            "table_filtered": filt_tsv,
            "figure": out_png
        }

    return outputs


# ============================================================
# Main
# ============================================================
def main():
    print(f"[IO] Loading {IN_H5AD}")
    adata = sc.read_h5ad(IN_H5AD)
    adata = ensure_condition(adata)

    if LABEL_COL not in adata.obs.columns:
        raise ValueError(f"{LABEL_COL} not found in adata.obs")

    abundance_df = compute_celltype_abundance_shift(adata)

    q1_df, q1_fig = question_1_depleted_celltypes(abundance_df)
    q2_df, q2_fig = question_2_enriched_celltypes(abundance_df)

    adata_log1p = make_log1p_copy(adata)
    q3_table, q3_corr, q3_fig_old, q3_fig_colored, q3_fig_compare = question_3_mmr_vs_depletion(
        adata_log1p, abundance_df
    )

    q4_outputs = question_4_dysregulated_genes(adata, abundance_df)

    summary_txt = os.path.join(OUTDIR, "summary.txt")
    with open(summary_txt, "w") as fh:
        fh.write("TDP43 question panels summary (v3)\n")
        fh.write("=================================\n\n")
        fh.write(f"Q1 depleted cell types figure: {q1_fig}\n")
        fh.write(f"Q2 enriched cell types figure: {q2_fig}\n")
        fh.write(f"Q3 old scatter figure: {q3_fig_old}\n")
        fh.write(f"Q3 colored scatter figure: {q3_fig_colored}\n")
        fh.write(f"Q3 control-vs-TDP43 MMR figure: {q3_fig_compare}\n\n")

        fh.write("Q1 top depleted cell types:\n")
        fh.write(q1_df[["cell_type", "control", "tdp43", "log2fc_fraction_tdp43_vs_control"]].to_string(index=False))
        fh.write("\n\n")

        fh.write("Q2 top enriched cell types:\n")
        fh.write(q2_df[["cell_type", "control", "tdp43", "log2fc_fraction_tdp43_vs_control"]].to_string(index=False))
        fh.write("\n\n")

        fh.write("Q3 gene-wise correlations:\n")
        fh.write(q3_corr.to_string(index=False))
        fh.write("\n\n")

        fh.write("Q4 selected cell types:\n")
        for kind, d in q4_outputs.items():
            fh.write(
                f"{kind}: {d['cell_type']} | control={d['n_control']} | tdp43={d['n_tdp43']} | "
                f"full={d['table_full']} | filtered={d['table_filtered']} | figure={d['figure']}\n"
            )

    print("\n[Done] Outputs written to:")
    print("  Figures:", FIGDIR)
    print("  Tables :", TABLEDIR)
    print("  Summary:", summary_txt)


if __name__ == "__main__":
    main()