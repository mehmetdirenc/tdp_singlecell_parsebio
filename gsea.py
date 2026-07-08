#!/usr/bin/env python3
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gseapy as gp


def select_genes_for_enrichment(
    de_file,
    direction="up",
    gene_col="names",
    logfc_col="logfoldchanges",
    padj_col="pvals_adj",
    pct_tdp43_col="pct_expr_tdp43",
    pct_control_col="pct_expr_control",
    logfc_cut=0.5,
    min_pct_expr=0.10,
    use_padj=False,
    padj_cut=0.05,
    max_genes=20,
    sort_by="effect"
):
    sep = "\t" if de_file.endswith(".tsv") else ","
    df = pd.read_csv(de_file, sep=sep)

    # 1) keep genes that are expressed in at least one condition
    if pct_tdp43_col in df.columns and pct_control_col in df.columns:
        df = df[
            (df[pct_tdp43_col] >= min_pct_expr) |
            (df[pct_control_col] >= min_pct_expr)
        ].copy()

    # 2) optional adjusted p-value filter
    if use_padj and padj_col in df.columns:
        df = df[df[padj_col] < padj_cut].copy()

    # 3) keep genes in requested direction
    if direction == "up":
        df = df[df[logfc_col] >= logfc_cut].copy()
    elif direction == "down":
        df = df[df[logfc_col] <= -logfc_cut].copy()
    else:
        raise ValueError("direction must be 'up' or 'down'")

    # 4) sort genes
    if sort_by == "effect":
        df["abs_logfc"] = df[logfc_col].abs()
        if padj_col in df.columns:
            df = df.sort_values(["abs_logfc", padj_col], ascending=[False, True])
        else:
            df = df.sort_values(["abs_logfc"], ascending=[False])
    elif sort_by == "padj":
        if padj_col not in df.columns:
            raise ValueError("sort_by='padj' requested but pvals_adj column not found")
        if direction == "up":
            df = df.sort_values([padj_col, logfc_col], ascending=[True, False])
        else:
            df = df.sort_values([padj_col, logfc_col], ascending=[True, True])
    else:
        raise ValueError("sort_by must be 'effect' or 'padj'")

    df = df.head(max_genes).copy()
    genes = df[gene_col].dropna().astype(str).tolist()

    return df, genes


def run_enrichr_gene_list(
    genes,
    outdir,
    gene_sets=None,
    organism="mouse"
):
    if gene_sets is None:
        gene_sets = [
            "Reactome_2022",
            "GO_Biological_Process_2023"
        ]

    os.makedirs(outdir, exist_ok=True)

    print(f"[INFO] gene count for enrichment: {len(genes)}")
    if len(genes) < 3:
        print(f"[WARN] Too few genes for enrichment: {len(genes)}")
        return None

    enr = gp.enrichr(
        gene_list=genes,
        gene_sets=gene_sets,
        organism=organism,
        outdir=outdir,
        no_plot=True
    )

    return enr.results


def make_enrichr_like_barplot(
    enrichr_df,
    library_name,
    out_png,
    title=None,
    top_n=10,
    rank_by="Adjusted P-value"
):
    df = enrichr_df.copy()
    df = df[df["Gene_set"] == library_name].copy()

    if df.empty:
        print(f"[WARN] No results for library: {library_name}")
        return

    if rank_by == "Adjusted P-value":
        df = df.sort_values("Adjusted P-value", ascending=True)
        score_col = "minus_log10_adj_p"
        df[score_col] = -np.log10(df["Adjusted P-value"].clip(lower=1e-300))
        xlabel = "-log10(adjusted p-value)"
    elif rank_by == "Combined Score":
        df = df.sort_values("Combined Score", ascending=False)
        score_col = "Combined Score"
        xlabel = "Combined score"
    else:
        raise ValueError("rank_by must be 'Adjusted P-value' or 'Combined Score'")

    df = df.head(top_n).copy()
    df = df.iloc[::-1]

    plt.figure(figsize=(10, max(5, 0.45 * len(df))))
    bars = plt.barh(df["Term"], df[score_col])

    vals = df[score_col].values
    if len(vals) > 1:
        norm = (vals - vals.min()) / (vals.max() - vals.min() + 1e-9)
    else:
        norm = np.array([1.0])

    for b, n in zip(bars, norm):
        b.set_color((1.0, 0.45 + 0.25 * (1 - n), 0.45 + 0.25 * (1 - n)))

    plt.xlabel(xlabel)
    plt.ylabel("")
    if title is None:
        title = f"{library_name} enrichment"
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close()


def run_one_direction_enrichment_and_plot(
    de_file,
    outdir,
    direction,
    label,
    logfc_cut=0.5,
    min_pct_expr=0.10,
    use_padj=False,
    padj_cut=0.05,
    max_genes=20,
    sort_by="effect",
    library_for_plot="GO_Biological_Process_2023"
):
    os.makedirs(outdir, exist_ok=True)

    gene_df, genes = select_genes_for_enrichment(
        de_file=de_file,
        direction=direction,
        logfc_cut=logfc_cut,
        min_pct_expr=min_pct_expr,
        use_padj=use_padj,
        padj_cut=padj_cut,
        max_genes=max_genes,
        sort_by=sort_by
    )

    gene_df.to_csv(os.path.join(outdir, f"{label}_{direction}_genes.tsv"), sep="\t", index=False)

    with open(os.path.join(outdir, f"{label}_{direction}_genes.txt"), "w") as fh:
        for g in genes:
            fh.write(g + "\n")

    print(f"[INFO] {label} {direction}: {len(genes)} genes selected")
    if len(genes) < 3:
        print(f"[WARN] {label} {direction}: too few genes")
        return None

    enr_df = run_enrichr_gene_list(
        genes=genes,
        outdir=os.path.join(outdir, f"{label}_{direction}_enrichr"),
        gene_sets=["GO_Biological_Process_2023"],
        organism="mouse"
    )

    if enr_df is None or enr_df.empty:
        print(f"[WARN] No enrichment results for {label} {direction}")
        return None

    enr_df.to_csv(os.path.join(outdir, f"{label}_{direction}_enrichr_results.tsv"), sep="\t", index=False)

    make_enrichr_like_barplot(
        enrichr_df=enr_df,
        library_name=library_for_plot,
        out_png=os.path.join(outdir, f"{label}_{direction}_{library_for_plot}_barplot.png"),
        title=f"{label} | {direction} genes | {library_for_plot}",
        top_n=10,
        # rank_by="Adjusted P-value"
        rank_by="Combined Score"
    )

    return enr_df


def main():
    base_out = "/mnt/storage1/projects/research/24035_1345_TDP43/scanpy_analysis/enrichr_q4"

    # change these to v3 tables if you want to use the newest outputs
    depleted_de = "/mnt/storage1/projects/research/24035_1345_TDP43/scanpy_analysis/figures_tdp43_question_panels_v2/tables/Q4_depleted_113_MEA-COA-BMA_Ccdc42_Glut_DE_filtered.tsv"
    enriched_de = "/mnt/storage1/projects/research/24035_1345_TDP43/scanpy_analysis/figures_tdp43_question_panels_v2/tables/Q4_enriched_330_VLMC_NN_DE_filtered.tsv"

    # depleted cell type
    run_one_direction_enrichment_and_plot(
        de_file=depleted_de,
        outdir=os.path.join(base_out, "depleted"),
        direction="up",
        label="113_MEA-COA-BMA_Ccdc42_Glut",
        logfc_cut=0.5,
        min_pct_expr=0.10,
        use_padj=False,
        padj_cut=0.05,
        max_genes=20,
        sort_by="effect",
        library_for_plot="GO_Biological_Process_2023"
    )

    run_one_direction_enrichment_and_plot(
        de_file=depleted_de,
        outdir=os.path.join(base_out, "depleted"),
        direction="down",
        label="113_MEA-COA-BMA_Ccdc42_Glut",
        logfc_cut=0.5,
        min_pct_expr=0.10,
        use_padj=False,
        padj_cut=0.05,
        max_genes=20,
        sort_by="effect",
        library_for_plot="GO_Biological_Process_2023"
    )

    # enriched cell type
    run_one_direction_enrichment_and_plot(
        de_file=enriched_de,
        outdir=os.path.join(base_out, "enriched"),
        direction="up",
        label="330_VLMC_NN",
        logfc_cut=0.5,
        min_pct_expr=0.10,
        use_padj=False,
        padj_cut=0.05,
        max_genes=20,
        sort_by="effect",
        library_for_plot="GO_Biological_Process_2023"
    )

    run_one_direction_enrichment_and_plot(
        de_file=enriched_de,
        outdir=os.path.join(base_out, "enriched"),
        direction="down",
        label="330_VLMC_NN",
        logfc_cut=0.5,
        min_pct_expr=0.10,
        use_padj=False,
        padj_cut=0.05,
        max_genes=20,
        sort_by="effect",
        library_for_plot="GO_Biological_Process_2023"
    )

    print("Done.")


if __name__ == "__main__":
    main()