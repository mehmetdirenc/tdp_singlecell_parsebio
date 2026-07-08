from pathlib import Path

import pandas as pd
import scanpy as sc
from scipy.io import mmread
import matplotlib.pyplot as plt


HUMAN_MT_ENSEMBL = [
    "ENSG00000198888",  # MT-ND1
    "ENSG00000198763",  # MT-ND2
    "ENSG00000198804",  # MT-CO1
    "ENSG00000198712",  # MT-CO2
    "ENSG00000228253",  # MT-ATP8
    "ENSG00000198899",  # MT-ATP6
    "ENSG00000198938",  # MT-CO3
    "ENSG00000198840",  # MT-ND3
    "ENSG00000212907",  # MT-ND4L
    "ENSG00000198886",  # MT-ND4
    "ENSG00000198786",  # MT-ND5
    "ENSG00000198695",  # MT-ND6
    "ENSG00000198727",  # MT-CYB
]


def check_dge_folder(dge_dir):
    dge_dir = Path(dge_dir)
    return all([
        (dge_dir / "all_genes.csv").exists(),
        (dge_dir / "cell_metadata.csv").exists(),
        (dge_dir / "count_matrix.mtx").exists(),
    ])


def find_existing_h5ad(outdir):
    outdir = Path(outdir)

    if not outdir.exists():
        return None

    h5ad_files = sorted(outdir.glob("*.h5ad"))

    if len(h5ad_files) == 0:
        return None

    # Prefer already QC/celltyped files if present
    preferred_keywords = [
        "celltyped",
        "mito_qc",
        "raw_mito_qc",
    ]

    for keyword in preferred_keywords:
        for f in h5ad_files:
            if keyword in f.name:
                return f

    return h5ad_files[0]


def load_parsebio_as_adata(dge_dir):
    dge_dir = Path(dge_dir)

    genes = pd.read_csv(dge_dir / "all_genes.csv")
    cells = pd.read_csv(dge_dir / "cell_metadata.csv")
    X = mmread(dge_dir / "count_matrix.mtx").tocsr()

    print("\nLoading ParseBio files:", dge_dir)
    print("Matrix:", X.shape)
    print("Genes:", genes.shape)
    print("Cells:", cells.shape)

    if X.shape[0] == genes.shape[0] and X.shape[1] == cells.shape[0]:
        X = X.T
    elif X.shape[0] == cells.shape[0] and X.shape[1] == genes.shape[0]:
        pass
    else:
        raise ValueError(
            f"Matrix dimensions do not match: X={X.shape}, "
            f"genes={genes.shape}, cells={cells.shape}"
        )

    adata = sc.AnnData(X=X)
    adata.obs = cells.copy()
    adata.var = genes.copy()

    if "cell_barcode" in adata.obs.columns:
        adata.obs_names = adata.obs["cell_barcode"].astype(str).values
    elif "bc_wells" in adata.obs.columns:
        adata.obs_names = adata.obs["bc_wells"].astype(str).values
    else:
        adata.obs_names = adata.obs.iloc[:, 0].astype(str).values

    adata.var_names = adata.var["gene_name"].astype(str).values

    adata.obs_names_make_unique()
    adata.var_names_make_unique()

    return adata


def ensure_gene_metadata(adata, dge_dir):
    """
    If the loaded h5ad does not contain gene_id/gene_name/genome,
    recover them from all_genes.csv.
    """

    dge_dir = Path(dge_dir)

    required_cols = {"gene_id", "gene_name", "genome"}

    if required_cols.issubset(set(adata.var.columns)):
        return adata

    genes = pd.read_csv(dge_dir / "all_genes.csv")

    if adata.n_vars != genes.shape[0]:
        raise ValueError(
            "Cannot attach gene metadata: h5ad n_vars does not match all_genes.csv."
        )

    for col in genes.columns:
        adata.var[col] = genes[col].values

    if "gene_name" in adata.var.columns:
        adata.var_names = adata.var["gene_name"].astype(str).values
        adata.var_names_make_unique()

    return adata


def add_human_mito_qc_if_needed(adata):
    needed_cols = {
        "n_genes_by_counts",
        "total_counts",
        "mt_counts_human",
        "pct_mt_human",
    }

    if needed_cols.issubset(set(adata.obs.columns)):
        print("Mito QC already present. Skipping mito QC calculation.")
        return adata

    adata.var["mt_human"] = adata.var["gene_id"].isin(HUMAN_MT_ENSEMBL)

    detected_mt = adata.var.loc[
        adata.var["mt_human"],
        ["gene_id", "gene_name", "genome"],
    ]

    print("\nDetected canonical human mitochondrial genes:")
    print(detected_mt)

    missing = set(HUMAN_MT_ENSEMBL) - set(detected_mt["gene_id"])
    print("\nMissing canonical human mitochondrial Ensembl IDs:")
    print(sorted(missing))

    sc.pp.calculate_qc_metrics(
        adata,
        qc_vars=["mt_human"],
        percent_top=None,
        log1p=False,
        inplace=True,
    )

    adata.obs["mt_counts_human"] = adata.obs["total_counts_mt_human"]
    adata.obs["pct_mt_human"] = adata.obs["pct_counts_mt_human"]

    return adata


def load_or_create_initial_h5ad(dge_dir, outdir, label):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    existing_h5ad = find_existing_h5ad(outdir)

    if existing_h5ad is not None:
        print(f"\nLoading existing h5ad for {label}: {existing_h5ad}")
        adata = sc.read_h5ad(existing_h5ad)
        adata = ensure_gene_metadata(adata, dge_dir)
        adata = add_human_mito_qc_if_needed(adata)
        return adata

    print(f"\nNo h5ad found for {label}. Creating from ParseBio files.")
    adata = load_parsebio_as_adata(dge_dir)
    adata = add_human_mito_qc_if_needed(adata)

    h5ad_path = outdir / f"{label}_initial_human_mito_qc.h5ad"
    print(f"Writing initial h5ad: {h5ad_path}")
    adata.write(h5ad_path)

    return adata


def has_processed_results(adata):
    return (
        "X_umap" in adata.obsm
        and "leiden_cluster" in adata.obs.columns
        and "celltype_or_cluster" in adata.obs.columns
    )


def make_processed_adata_if_needed(adata, celltypist_model=None, n_top_genes=2000):
    if has_processed_results(adata):
        print("UMAP/clustering/celltype_or_cluster already present. Skipping processing.")
        return adata

    adata_proc = adata.copy()

    sc.pp.normalize_total(adata_proc, target_sum=1e4)
    sc.pp.log1p(adata_proc)
    sc.pp.highly_variable_genes(adata_proc, n_top_genes=n_top_genes)
    sc.pp.pca(adata_proc)
    sc.pp.neighbors(adata_proc)
    sc.tl.umap(adata_proc)
    sc.tl.leiden(adata_proc, resolution=0.6, key_added="leiden_cluster")

    adata_proc.obs["celltype_or_cluster"] = (
        "cluster_" + adata_proc.obs["leiden_cluster"].astype(str)
    )

    if celltypist_model is not None:
        try:
            import celltypist

            predictions = celltypist.annotate(
                adata_proc,
                model=celltypist_model,
                majority_voting=True,
            )

            pred_adata = predictions.to_adata()

            if "majority_voting" in pred_adata.obs.columns:
                adata_proc.obs["celltypist_celltype"] = pred_adata.obs["majority_voting"].astype(str)
            elif "predicted_labels" in pred_adata.obs.columns:
                adata_proc.obs["celltypist_celltype"] = pred_adata.obs["predicted_labels"].astype(str)
            else:
                raise ValueError("Could not find CellTypist prediction columns.")

            adata_proc.obs["celltype_or_cluster"] = adata_proc.obs["celltypist_celltype"]

        except Exception as e:
            print("\nCellTypist failed. Using Leiden clusters.")
            print("Reason:", e)

    return adata_proc


def transfer_processed_to_raw(adata, adata_proc):
    adata.obs["leiden_cluster"] = adata_proc.obs["leiden_cluster"].reindex(adata.obs_names)
    adata.obs["celltype_or_cluster"] = adata_proc.obs["celltype_or_cluster"].reindex(adata.obs_names)

    if "celltypist_celltype" in adata_proc.obs.columns:
        adata.obs["celltypist_celltype"] = adata_proc.obs["celltypist_celltype"].reindex(adata.obs_names)

    adata.obsm["X_umap"] = adata_proc.obsm["X_umap"].copy()

    return adata


def export_small_tables(adata, outdir):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    qc_cols = [
        "n_genes_by_counts",
        "total_counts",
        "mt_counts_human",
        "pct_mt_human",
    ]

    summary = adata.obs[qc_cols].describe()
    summary.to_csv(outdir / "human_mito_qc_summary.csv")

    celltype_summary = (
        adata.obs
        .groupby("celltype_or_cluster")[qc_cols]
        .agg(["count", "mean", "median", "min", "max"])
    )
    celltype_summary.to_csv(outdir / "human_mito_qc_by_celltype_or_cluster.csv")

    mt_gene_counts = pd.DataFrame({
        "gene_id": adata.var["gene_id"].values,
        "gene_name": adata.var["gene_name"].values,
        "genome": adata.var["genome"].values,
        "total_counts": adata.X.sum(axis=0).A1,
    })

    mt_gene_counts = mt_gene_counts[
        mt_gene_counts["gene_id"].isin(HUMAN_MT_ENSEMBL)
    ]

    mt_gene_counts.to_csv(outdir / "human_mito_gene_total_counts.csv", index=False)

    return summary, celltype_summary


def make_basic_plots(adata, outdir):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    sc.settings.figdir = str(outdir)

    sc.pl.violin(
        adata,
        ["n_genes_by_counts", "total_counts", "mt_counts_human", "pct_mt_human"],
        jitter=0.4,
        multi_panel=True,
        save="_human_mito_qc.png",
    )

    sc.pl.violin(
        adata,
        ["mt_counts_human", "pct_mt_human"],
        groupby="celltype_or_cluster",
        rotation=90,
        jitter=0.2,
        save="_human_mito_by_celltype_or_cluster.png",
    )

    sc.pl.umap(
        adata,
        color=[
            "celltype_or_cluster",
            "n_genes_by_counts",
            "total_counts",
            "mt_counts_human",
            "pct_mt_human",
        ],
        save="_human_mito_signal_with_celltypes.png",
    )


def make_celltype_barplots(adata, outdir):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    summary = (
        adata.obs
        .groupby("celltype_or_cluster")
        .agg(
            n_cells=("celltype_or_cluster", "size"),
            mean_mt_counts=("mt_counts_human", "mean"),
            median_mt_counts=("mt_counts_human", "median"),
            mean_pct_mt=("pct_mt_human", "mean"),
            median_pct_mt=("pct_mt_human", "median"),
            mean_genes_per_cell=("n_genes_by_counts", "mean"),
        )
        .sort_values("mean_pct_mt", ascending=False)
    )

    summary.to_csv(outdir / "celltype_or_cluster_mito_barplot_values.csv")

    plt.figure(figsize=(8, max(4, 0.35 * summary.shape[0])))
    summary["mean_pct_mt"].sort_values().plot(kind="barh")
    plt.xlabel("Mean % mitochondrial counts")
    plt.ylabel("Cell type / cluster")
    plt.tight_layout()
    plt.savefig(outdir / "mean_pct_mt_by_celltype_or_cluster.png", dpi=300)
    plt.close()

    plt.figure(figsize=(8, max(4, 0.35 * summary.shape[0])))
    summary["mean_mt_counts"].sort_values().plot(kind="barh")
    plt.xlabel("Mean mitochondrial counts")
    plt.ylabel("Cell type / cluster")
    plt.tight_layout()
    plt.savefig(outdir / "mean_mt_counts_by_celltype_or_cluster.png", dpi=300)
    plt.close()


def run_analysis(dge_dir, outdir, label, celltypist_model=None, save_processed_h5ad=True):
    print("\n" + "=" * 80)
    print(f"Running analysis for: {label}")
    print("=" * 80)

    dge_dir = Path(dge_dir)
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if not check_dge_folder(dge_dir):
        print(f"Skipping {label}: missing required files in {dge_dir}")
        return None

    adata = load_or_create_initial_h5ad(dge_dir, outdir, label)

    if has_processed_results(adata):
        adata_final = adata
    else:
        adata_proc = make_processed_adata_if_needed(
            adata,
            celltypist_model=celltypist_model,
        )
        adata_final = transfer_processed_to_raw(adata, adata_proc)

    if save_processed_h5ad:
        processed_h5ad = outdir / f"{label}_processed_human_mito_qc_celltyped.h5ad"

        if processed_h5ad.exists():
            print(f"Processed h5ad already exists, not overwriting: {processed_h5ad}")
        else:
            print(f"Writing processed h5ad: {processed_h5ad}")
            adata_final.write(processed_h5ad)

    summary, celltype_summary = export_small_tables(adata_final, outdir)

    print(f"\nSummary for {label}:")
    print(summary)

    print(f"\nCell type / cluster summary for {label}:")
    print(celltype_summary)

    make_basic_plots(adata_final, outdir)
    make_celltype_barplots(adata_final, outdir)
    quartile_summary = make_gene_count_quartile_plots(adata_final, outdir)

    print(f"\nGene-count quartile summary for {label}:")
    print(quartile_summary)
    return summary


def main():
    parsebio_dir = Path(
        "/mnt/storage1/projects/research/23067N_SEQ4218_hypoxia-EPO/"
        "all-sample_Sample_23067SubLib_comb_results"
    )

    filtered_dge_dir = parsebio_dir / "DGE_filtered"
    unfiltered_dge_dir = parsebio_dir / "DGE_unfiltered"

    filtered_outdir = parsebio_dir / "parsebio_human_mito_analysis_filtered"
    unfiltered_outdir = parsebio_dir / "parsebio_human_mito_analysis_unfiltered"

    celltypist_model = None
    # Example:
    # celltypist_model = "Immune_All_Low.pkl"
    # celltypist_model = "/path/to/model.pkl"

    filtered_summary = run_analysis(
        dge_dir=filtered_dge_dir,
        outdir=filtered_outdir,
        label="DGE_filtered",
        celltypist_model=celltypist_model,
        save_processed_h5ad=True,
    )

    unfiltered_summary = run_analysis(
        dge_dir=unfiltered_dge_dir,
        outdir=unfiltered_outdir,
        label="DGE_unfiltered",
        celltypist_model=celltypist_model,
        save_processed_h5ad=True,
    )

    if filtered_summary is not None and unfiltered_summary is not None:
        comparison = pd.concat(
            {
                "filtered": filtered_summary,
                "unfiltered": unfiltered_summary,
            },
            axis=1,
        )

        comparison.to_csv(
            parsebio_dir / "human_mito_filtered_vs_unfiltered_summary.csv"
        )

        print("\nFiltered vs unfiltered comparison:")
        print(comparison)

def make_gene_count_quartile_plots(adata, outdir):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = adata.obs[
        ["n_genes_by_counts", "mt_counts_human", "pct_mt_human"]
    ].copy()

    df["gene_count_quartile"] = pd.qcut(
        df["n_genes_by_counts"],
        q=4,
        labels=["Q1 lowest genes", "Q2", "Q3", "Q4 highest genes"],
        duplicates="drop",
    )

    quartile_summary = (
        df.groupby("gene_count_quartile", observed=True)
        .agg(
            n_cells=("pct_mt_human", "size"),
            mean_genes=("n_genes_by_counts", "mean"),
            median_genes=("n_genes_by_counts", "median"),
            mean_mt_counts=("mt_counts_human", "mean"),
            median_mt_counts=("mt_counts_human", "median"),
            mean_pct_mt=("pct_mt_human", "mean"),
            median_pct_mt=("pct_mt_human", "median"),
        )
    )

    quartile_summary.to_csv(outdir / "mito_by_gene_count_quartile.csv")

    plt.figure(figsize=(7, 4))
    quartile_summary["mean_pct_mt"].plot(kind="bar")
    plt.ylabel("Mean % mitochondrial counts")
    plt.xlabel("Gene-count quartile")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(outdir / "mean_pct_mt_by_gene_count_quartile.png", dpi=300)
    plt.close()

    plt.figure(figsize=(7, 4))
    quartile_summary["median_pct_mt"].plot(kind="bar")
    plt.ylabel("Median % mitochondrial counts")
    plt.xlabel("Gene-count quartile")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(outdir / "median_pct_mt_by_gene_count_quartile.png", dpi=300)
    plt.close()

    plt.figure(figsize=(7, 4))
    quartile_summary["mean_mt_counts"].plot(kind="bar")
    plt.ylabel("Mean mitochondrial counts")
    plt.xlabel("Gene-count quartile")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(outdir / "mean_mt_counts_by_gene_count_quartile.png", dpi=300)
    plt.close()

    return quartile_summary


if __name__ == "__main__":
    main()