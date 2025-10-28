# results_quality_check.py
# Quantitative quality checks for sc/snRNA clustering results (AnnData).
# No CellTypist required.

from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict, Any, Iterable, Optional, Tuple
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
from scipy import sparse
from scipy.stats import chi2_contingency
import warnings

# ----------------- internal helpers -----------------

def _get_layer(adata):
    """Return (X_sparse, var_names), preferring adata.raw when available."""
    if getattr(adata, "raw", None) is not None:
        X = adata.raw.X
        var_names = adata.raw.var_names
    else:
        X = adata.X
        var_names = adata.var_names
    if not sparse.issparse(X):
        X = sparse.csr_matrix(X)
    return X, pd.Index(var_names)

def _ensure_pca(adata, n_pcs=20) -> int:
    """Make sure X_pca exists; return number of PCs available (<= n_pcs)."""
    if 'X_pca' not in adata.obsm:
        import scanpy as sc
        sc.tl.pca(adata, svd_solver='arpack')
    return min(n_pcs, adata.obsm['X_pca'].shape[1])

def _neighbors_params(adata, default_npcs=20, default_k=20) -> Tuple[int, int]:
    """Read neighbors params if present; otherwise fall back to defaults."""
    npcs, k = default_npcs, default_k
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pars = adata.uns.get('neighbors', {}).get('params', {})
        npcs = int(pars.get('n_pcs', npcs))
        k    = int(pars.get('n_neighbors', k))
    return npcs, k

def _percent_expressing(adata, genes, mask):
    """Mean % expressing within cluster and delta(%in - %out) for a gene set."""
    X, var_names = _get_layer(adata)
    genes = [g for g in genes if g in var_names]
    if not genes:
        return np.nan, np.nan
    idx = var_names.get_indexer(genes)
    Xin  = X[mask][:, idx]
    Xout = X[~mask][:, idx]
    pin   = (Xin  > 0).mean(axis=0).A1
    pout  = (Xout > 0).mean(axis=0).A1 if Xout.shape[0] > 0 else np.zeros_like(pin)
    return float(np.nanmean(pin)), float(np.nanmean(pin - np.maximum(pout, 0)))

# ----------------- metrics -----------------

def stability_ari(adata, seeds=(1,2,3), resolution=0.6, min_dist=0.3) -> Tuple[float, float]:
    """Leiden stability across seeds (recomputes neighbors/UMAP/Leiden)."""
    import scanpy as sc
    npcs, k = _neighbors_params(adata)
    labels = []
    for s in seeds:
        C = adata.copy()
        sc.pp.neighbors(C, n_neighbors=k, n_pcs=npcs, random_state=s)
        sc.tl.umap(C, min_dist=min_dist, random_state=s)
        sc.tl.leiden(C, resolution=resolution, key_added=f'leiden_tmp_{s}')
        labels.append(C.obs[f'leiden_tmp_{s}'])
    aris, nmis = [], []
    for i in range(len(labels)):
        for j in range(i+1, len(labels)):
            aris.append(adjusted_rand_score(labels[i], labels[j]))
            nmis.append(normalized_mutual_info_score(labels[i], labels[j]))
    return float(np.mean(aris)), float(np.mean(nmis))

def marker_sharpness(adata, cluster_key: str, n_top=20) -> Tuple[float, float]:
    """Average within-cluster % expressing and delta(%in − %out) over top markers."""
    import scanpy as sc
    if 'rank_genes_groups' not in adata.uns or adata.uns['rank_genes_groups'] is None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sc.tl.rank_genes_groups(adata, groupby=cluster_key, method='wilcoxon', use_raw=True, pts=True)
    df = sc.get.rank_genes_groups_df(adata, None)
    if df is None or df.empty:
        return np.nan, np.nan
    pe_in, pe_delta = [], []
    for clust, sub in df.groupby('group'):
        topg = sub.sort_values('logfoldchanges', ascending=False)['names'].head(n_top).tolist()
        mask = (adata.obs[cluster_key].astype(str) == str(clust)).values
        pin, pdiff = _percent_expressing(adata, topg, mask)
        pe_in.append(pin); pe_delta.append(pdiff)
    return float(np.nanmean(pe_in)), float(np.nanmean(pe_delta))

def pc_technical_leakage(adata, covars=('total_counts','pct_counts_mt','n_genes_by_counts'), n_pcs=20) -> Dict[str,float]:
    """Max absolute correlation of PCs (first n_pcs) with technical covariates."""
    used = _ensure_pca(adata, n_pcs=n_pcs)
    pcs = pd.DataFrame(adata.obsm['X_pca'][:, :used], index=adata.obs_names)
    leaks = {}
    for c in covars:
        if c in adata.obs:
            leaks[c] = float(pcs.corrwith(pd.to_numeric(adata.obs[c], errors='coerce')).abs().max())
    return leaks

def batch_mixing(adata, cluster_key: str, sample_key: str = 'sample') -> Tuple[float, float, Optional[pd.DataFrame]]:
    """Normalized entropy (higher=better mixing) and chi² p-value across clusters."""
    if sample_key not in adata.obs:
        return np.nan, np.nan, None
    tab = pd.crosstab(adata.obs[cluster_key], adata.obs[sample_key])
    if tab.shape[1] <= 1:
        return np.nan, np.nan, tab
    probs = tab.div(tab.sum(1), 0)
    with np.errstate(divide='ignore', invalid='ignore'):
        ent = -(probs*np.log(probs+1e-12)).sum(1)
        ent_norm = ent / np.log(probs.shape[1])
    chi2, p, _, _ = chi2_contingency(tab.values)
    return float(ent_norm.mean()), float(p), tab

def doublet_enrichment(adata, cluster_key: str):
    """Summaries if 'doublet' or 'doublet_score' present in adata.obs."""
    if 'doublet' not in adata.obs and 'doublet_score' not in adata.obs:
        return None
    out = {}
    if 'doublet' in adata.obs:
        tab = pd.crosstab(adata.obs[cluster_key], adata.obs['doublet'])
        frac = tab.div(tab.sum(1), 0)[True] if True in tab.columns else pd.Series(0, index=tab.index)
        out['doublet_overall'] = float(adata.obs['doublet'].mean())
        out['doublet_frac_by_cluster'] = frac.sort_values(ascending=False)
    if 'doublet_score' in adata.obs:
        out['doublet_score_by_cluster'] = adata.obs.groupby(cluster_key)['doublet_score'].mean().sort_values(ascending=False)
    return out

# ----------------- main entrypoint -----------------

def evaluate_pipeline(
    adata,
    cluster_key: str,
    sample_key: str = 'sample',
    n_pcs_for_sil: int = 20,
    stability_resolution: float = 0.6,
    stability_min_dist: float = 0.3,
    seeds: Iterable[int] = (1,2,3),
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Compute quantitative metrics for a clustering stored in `adata.obs[cluster_key]`.

    Returns a dict with:
      - n_cells, n_genes, n_clusters
      - silhouette_pca
      - stability_ARI_mean, stability_NMI_mean
      - marker_percent_incluster_mean, marker_delta_percent_in_vs_out_mean
      - pc_technical_leakage_max_abs_corr, pc_technical_leakage_by_covariate
      - batch_entropy_mean_norm, batch_chi2_pvalue
    """
    # Silhouette in PCA space
    used = _ensure_pca(adata, n_pcs=n_pcs_for_sil)
    sil = silhouette_score(adata.obsm['X_pca'][:, :used], adata.obs[cluster_key]) if adata.n_obs > 100 else np.nan

    # Stability across seeds (neighbors/UMAP/Leiden only)
    ari, nmi = stability_ari(adata, seeds=tuple(seeds), resolution=stability_resolution, min_dist=stability_min_dist)

    # Marker sharpness
    pe_in, pe_delta = marker_sharpness(adata, cluster_key, n_top=20)

    # Technical leakage
    leaks = pc_technical_leakage(adata, n_pcs=n_pcs_for_sil)

    # Batch mixing
    ent, chi_p, mix_tab = batch_mixing(adata, cluster_key, sample_key)

    # Doublets
    dblet = doublet_enrichment(adata, cluster_key)

    report = {
        'n_cells': int(adata.n_obs),
        'n_genes': int(adata.n_vars),
        'cluster_key': cluster_key,
        'n_clusters': int(adata.obs[cluster_key].nunique()),
        'silhouette_pca': float(sil) if sil == sil else np.nan,
        'stability_ARI_mean': float(ari),
        'stability_NMI_mean': float(nmi),
        'marker_percent_incluster_mean': float(pe_in),
        'marker_delta_percent_in_vs_out_mean': float(pe_delta),
        'pc_technical_leakage_max_abs_corr': float(max(leaks.values())) if len(leaks) else np.nan,
        'pc_technical_leakage_by_covariate': leaks,
        'batch_entropy_mean_norm': float(ent) if ent == ent else np.nan,
        'batch_chi2_pvalue': float(chi_p) if chi_p == chi_p else np.nan,
        'doublet_summaries_present': bool(dblet),
    }

    if verbose:
        print("\n=== PIPELINE EVALUATION ===")
        for k,v in report.items():
            if k=='pc_technical_leakage_by_covariate': continue
            print(f"{k:40s}: {v}")
        if leaks:
            print("pc_technical_leakage_by_covariate     :", leaks)
        if mix_tab is not None:
            print("\n[Batch composition per cluster] (rows=clusters, cols=samples)")
            print(mix_tab)
        if dblet:
            if 'doublet_overall' in dblet:
                print(f"\n[Doublets] overall fraction: {dblet['doublet_overall']:.3f}")
            if 'doublet_frac_by_cluster' in dblet:
                print("[Doublets] frac by cluster (top 10):")
                print(dblet['doublet_frac_by_cluster'].head(10))
    return report

def compare_labelings(
    adata,
    key_a: str,
    key_b: str,
) -> Dict[str, float]:
    """Compare two clusterings on the SAME AnnData via ARI/NMI."""
    ari = adjusted_rand_score(adata.obs[key_a], adata.obs[key_b])
    nmi = normalized_mutual_info_score(adata.obs[key_a], adata.obs[key_b])
    return {'ARI': float(ari), 'NMI': float(nmi)}


# ----------------- report saving (append-only) -----------------
import os
from datetime import datetime

def _format_report_text(report: dict, leaks: dict, mix_tab, dblet) -> str:
    lines = []
    lines.append("=== PIPELINE EVALUATION ===")
    for k, v in report.items():
        if k == 'pc_technical_leakage_by_covariate':
            continue
        lines.append(f"{k:40s}: {v}")
    if leaks:
        lines.append("pc_technical_leakage_by_covariate     : " + str(leaks))
    if mix_tab is not None:
        lines.append("\n[Batch composition per cluster] (rows=clusters, cols=samples)")
        lines.append(mix_tab.to_string())
    if dblet:
        if 'doublet_overall' in dblet:
            lines.append(f"\n[Doublets] overall fraction: {dblet['doublet_overall']:.3f}")
        if 'doublet_frac_by_cluster' in dblet:
            lines.append("[Doublets] frac by cluster (top 10):")
            lines.append(dblet['doublet_frac_by_cluster'].head(10).to_string())
    lines.append("=== END EVALUATION ===")
    return "\n".join(lines)

def save_evaluation(
    adata,
    cluster_key: str,
    outfolder: str,
    pipeline_tag: str,       # e.g. "large" or "basic"
    organism: str,           # e.g. "rat" or "mouse"
    sample_key: str = 'sample',
    n_pcs_for_sil: int = 20,
    stability_resolution: float = 0.6,
    stability_min_dist: float = 0.3,
    seeds = (1,2,3),
    write_tables: bool = True,
    verbose: bool = True,
) -> str:
    """
    Run the same metrics as evaluate_pipeline and write:
      {outfolder}/{pipeline_tag}_{organism}_report.txt
    Also writes TSVs for batch composition and doublets if available.
    """
    import numpy as np
    from sklearn.metrics import silhouette_score

    os.makedirs(outfolder, exist_ok=True)

    # Recompute internals (same logic as evaluate_pipeline, no changes elsewhere)
    used = _ensure_pca(adata, n_pcs=n_pcs_for_sil)
    sil = silhouette_score(adata.obsm['X_pca'][:, :used], adata.obs[cluster_key]) if adata.n_obs > 100 else np.nan
    ari, nmi = stability_ari(adata, seeds=tuple(seeds), resolution=stability_resolution, min_dist=stability_min_dist)
    pe_in, pe_delta = marker_sharpness(adata, cluster_key, n_top=20)
    leaks = pc_technical_leakage(adata, n_pcs=n_pcs_for_sil)
    ent, chi_p, mix_tab = batch_mixing(adata, cluster_key, sample_key)
    dblet = doublet_enrichment(adata, cluster_key)

    report = {
        'timestamp': datetime.now().isoformat(timespec='seconds'),
        'n_cells': int(adata.n_obs),
        'n_genes': int(adata.n_vars),
        'cluster_key': cluster_key,
        'n_clusters': int(adata.obs[cluster_key].nunique()),
        'silhouette_pca': float(sil) if sil == sil else np.nan,
        'stability_ARI_mean': float(ari),
        'stability_NMI_mean': float(nmi),
        'marker_percent_incluster_mean': float(pe_in),
        'marker_delta_percent_in_vs_out_mean': float(pe_delta),
        'pc_technical_leakage_max_abs_corr': float(max(leaks.values())) if len(leaks) else np.nan,
        'pc_technical_leakage_by_covariate': leaks,
        'batch_entropy_mean_norm': float(ent) if ent == ent else np.nan,
        'batch_chi2_pvalue': float(chi_p) if chi_p == chi_p else np.nan,
    }

    text = _format_report_text(report, leaks, mix_tab, dblet)
    stem = f"{pipeline_tag.lower()}_{organism.lower()}"
    report_path = os.path.join(outfolder, f"{stem}_report.txt")
    with open(report_path, "w") as f:
        f.write(text + "\n")

    if write_tables:
        if mix_tab is not None:
            mix_path = os.path.join(outfolder, f"{stem}_batch_composition.tsv")
            mix_tab.to_csv(mix_path, sep="\t")
        if dblet:
            if 'doublet_frac_by_cluster' in dblet:
                dbl_path = os.path.join(outfolder, f"{stem}_doublet_frac_by_cluster.tsv")
                dblet['doublet_frac_by_cluster'].to_csv(dbl_path, sep="\t", header=True)
            if 'doublet_score_by_cluster' in dblet:
                ds_path = os.path.join(outfolder, f"{stem}_doublet_score_by_cluster.tsv")
                dblet['doublet_score_by_cluster'].to_csv(ds_path, sep="\t", header=True)

    if verbose:
        print(f"[results_quality_check] wrote report -> {report_path}")
    return report_path
