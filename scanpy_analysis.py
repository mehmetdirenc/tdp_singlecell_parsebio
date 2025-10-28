# === TDP-43 rat/mouse hypothalamus sc/snRNA (Parse Bio) end-to-end QC + compare ===

import os, sys, numpy as np, pandas as pd
import scanpy as sc
from scipy import sparse
import scrublet as scr
import matplotlib
import matplotlib.pyplot as plt
from results_quality_check import evaluate_pipeline, save_evaluation

orgs = ["mouse", "rat"]
for org in orgs:

    # Optional: CellTypist
    USE_CELLTYPIST = True
    CELLTYPIST_MODEL = 'Mouse_Whole_Brain'
    if USE_CELLTYPIST:
        import celltyping
        from celltyping import models

    # Print lib versions (helps debug plotting quirks)
    try:
        import seaborn as sns
        print("[versions]", "scanpy", sc.__version__, "| seaborn", sns.__version__, "| matplotlib", matplotlib.__version__)
    except Exception:
        pass

    # Paths & prefs
    sc.settings.verbosity = 1
    sc.settings.set_figure_params(dpi=110, fontsize=10, dpi_save=300, figsize=(5,4), format='png')
    if org == 'mouse':
        FIGDIR = '/home/ubuntu/volume_750gb/results/tdp_project/figures_AB_mouse'
        DATA = '/home/ubuntu/volume_750gb/results/tdp_project/tdp_combined_analysis_files_AB_mouse/DGE_filtered/'
    elif org == 'rat':
        FIGDIR = '/home/ubuntu/volume_750gb/results/tdp_project/figures_CD_rat'
        DATA = '/home/ubuntu/volume_750gb/results/tdp_project/tdp_combined_analysis_files_CD_rat/DGE_filtered/'
    ADATA = '/home/ubuntu/volume_750gb/results/tdp_project/adata_files'
    reports = '/home/ubuntu/volume_750gb/results/tdp_project/reports'
    sc.settings.figdir = FIGDIR
    os.makedirs(FIGDIR, exist_ok=True)

    # ---------- plotting helpers (robust to old scanpy/mpl) ----------
    def _save_with_caption(fig, basename, caption, tight=True):
        try: fig.suptitle(caption, fontsize=11)
        except Exception: pass
        if tight:
            try: fig.tight_layout(rect=[0, 0, 1, 0.95])
            except Exception: pass
        out = os.path.join(FIGDIR, f'{basename}.png')
        fig.savefig(out, dpi=300)
        plt.close(fig)

    def _call_plot(fn, *args, **kwargs):
        """Try return_fig=True; on ANY exception, retry without it and grab gcf()."""
        try:
            obj = fn(*args, show=False, return_fig=True, **kwargs)
            return obj if hasattr(obj, 'savefig') else plt.gcf()
        except Exception:
            fn(*args, show=False, **kwargs)
            return plt.gcf()

    def vplot(A, keys, basename, caption, **kwargs):
        fig = _call_plot(sc.pl.violin, A, keys, **kwargs)
        _save_with_caption(fig, basename, caption)

    def splot(A, x, y, basename, caption, **kwargs):
        fig = _call_plot(sc.pl.scatter, A, x=x, y=y, **kwargs)
        _save_with_caption(fig, basename, caption)

    def umapplot(A, color, basename, caption, **kwargs):
        fig = _call_plot(sc.pl.umap, A, color=color, **kwargs)
        _save_with_caption(fig, basename, caption)

    def hvgsplot(A, basename, caption):
        fig = _call_plot(sc.pl.highly_variable_genes, A)
        _save_with_caption(fig, basename, caption)

    def pca_varplot(A, n_pcs, basename, caption):
        fig = _call_plot(sc.pl.pca_variance_ratio, A, log=True, n_pcs=n_pcs)
        _save_with_caption(fig, basename, caption)

    def rankgenesplot(A, basename, caption, **kwargs):
        fig = _call_plot(sc.pl.rank_genes_groups, A, **kwargs)
        _save_with_caption(fig, basename, caption)

    # ---------- load ----------
    adata = sc.read_mtx(os.path.join(DATA, "count_matrix.mtx"))
    gene_data = pd.read_csv(os.path.join(DATA, "all_genes.csv"))
    cell_meta = pd.read_csv(os.path.join(DATA, "cell_metadata.csv"))

    adata.var_names = gene_data["gene_name"].astype(str).values
    adata.obs = cell_meta.copy()
    adata.obs_names = adata.obs["bc_wells"].astype(str).values
    adata.var_names_make_unique()
    print("Loaded:", adata.shape)

    # ---------- QC flags ----------
    adata.var['mt'] = adata.var_names.str.upper().str.startswith('MT-')
    rps_rpl_num = adata.var_names.str.match(r'^(RPS|RPL)\d{1,3}[A-Z]?$', case=False)
    rplp = adata.var_names.str.match(r'^RPLP[0-9]+$', case=False)
    adata.var['ribo'] = rps_rpl_num | rplp
    _sym_norm = pd.Index(adata.var_names.astype(str)).str.upper().str.replace(r'[-_.]', '', regex=True)
    HB_SET = {'HBA','HBA1','HBA2','HBD','HBG1','HBG2','HBM','HBQ1','HBQ2','HBZ','HBB','HBBBS','HBBBT','HBBBH1','HBBBH2','HBBY'}
    HB_BLACKLIST = {'HBS1L','HBEGF','HBP1'}
    adata.var['hb'] = _sym_norm.isin(HB_SET) & ~_sym_norm.isin(HB_BLACKLIST)
    sc.pp.calculate_qc_metrics(adata, qc_vars=['mt','ribo','hb'], percent_top=None, log1p=False, inplace=True)
    print("QC gene counts:", "mt:", int(adata.var['mt'].sum()), "ribo:", int(adata.var['ribo'].sum()), "hb:", int(adata.var['hb'].sum()))

    # ----- RAW figs -----
    vplot(adata, ['n_genes_by_counts'], '_QC_RAW_n_genes', 'RAW: QC after load — n_genes_by_counts')
    vplot(adata, ['total_counts'], '_QC_RAW_total_counts', 'RAW: QC after load — total_counts')
    vplot(adata, ['pct_counts_mt'], '_QC_RAW_pct_mt', 'RAW: QC after load — % mitochondrial counts')
    vplot(adata, ['pct_counts_hb'], '_QC_RAW_pct_hb', 'RAW: QC after load — % hemoglobin counts')
    splot(adata, 'total_counts', 'n_genes_by_counts', '_QC_RAW_counts_vs_genes', 'RAW: total_counts vs n_genes_by_counts')

    adata0 = adata.copy()

    # ---------- data-driven QC ----------
    group = 'sample' if 'sample' in adata.obs.columns else None

    def robust_filter(A, group_key=None, mt_cut=10.0, hb_cut=None):
        df = A.obs
        if group_key is None:
            keep = (df['n_genes_by_counts'] < df['n_genes_by_counts'].quantile(0.99)) & \
                   (df['total_counts']      < df['total_counts'].quantile(0.99)) & \
                   (df['pct_counts_mt']     < mt_cut)
            if 'pct_counts_hb' in df and hb_cut is not None:
                keep &= df['pct_counts_hb'] < hb_cut
            return keep.values
        idxs = []
        for s, sub in df.groupby(group_key):
            k = (sub['n_genes_by_counts'] < sub['n_genes_by_counts'].quantile(0.99)) & \
                (sub['total_counts']      < sub['total_counts'].quantile(0.99)) & \
                (sub['pct_counts_mt']     < mt_cut)
            if 'pct_counts_hb' in sub and hb_cut is not None:
                k &= sub['pct_counts_hb'] < hb_cut
            idxs.append(k)
        keep = pd.concat(idxs).loc[df.index]
        return keep.values

    keep_qc = robust_filter(adata, group_key=group, mt_cut=10.0, hb_cut=None)
    print(f"[QC_dataDriven] keep {keep_qc.sum():,}/{adata.n_obs:,} cells")
    adata1 = adata[keep_qc, :].copy()

    vplot(adata1, ['n_genes_by_counts'], '_QC_dataDriven_n_genes', 'QC_dataDriven: post-filter — n_genes_by_counts')
    vplot(adata1, ['total_counts'], '_QC_dataDriven_total_counts', 'QC_dataDriven: post-filter — total_counts')
    vplot(adata1, ['pct_counts_mt'], '_QC_dataDriven_pct_mt', 'QC_dataDriven: post-filter — % mitochondrial counts')
    vplot(adata1, ['pct_counts_hb'], '_QC_dataDriven_pct_hb', 'QC_dataDriven: post-filter — % hemoglobin counts')
    splot(adata1, 'total_counts', 'n_genes_by_counts', '_QC_dataDriven_counts_vs_genes', 'QC_dataDriven: total_counts vs n_genes_by_counts')

    # ---------- Scrublet ----------
    X = adata1.X if sparse.issparse(adata1.X) else sparse.csr_matrix(adata1.X)
    scrubber = scr.Scrublet(X, expected_doublet_rate=0.06)
    doublet_scores, doublet_preds = scrubber.scrub_doublets()
    adata1.obs['doublet_score'] = doublet_scores
    adata1.obs['doublet'] = doublet_preds
    adata1_preD = adata1.copy()
    print("[Scrublet] predicted doublets:", int(adata1.obs['doublet'].sum()), f"({adata1.obs['doublet'].mean()*100:.1f}%)")
    vplot(adata1, ['doublet_score'], '_preDoublet_doublet_score', 'preDoublet: Scrublet doublet_score distribution', jitter=0.3)

    # ---------- Embeddings (pre vs post) ----------
    def embed_and_cluster(A, tag, n_pcs=20, n_neighbors=20, res=0.6, seed=42):
        B = A.copy()
        sc.pp.normalize_total(B, 1e4); sc.pp.log1p(B)
        sc.pp.highly_variable_genes(B, flavor='seurat_v3', n_top_genes=3000,
                                    batch_key='sample' if 'sample' in B.obs.columns else None)
        B = B[:, B.var['highly_variable']].copy()
        covars = ['total_counts']
        if 'pct_counts_mt' in B.obs: covars.append('pct_counts_mt')
        sc.pp.regress_out(B, covars)
        sc.pp.scale(B, max_value=10)
        sc.tl.pca(B, svd_solver='arpack')
        sc.pp.neighbors(B, n_neighbors=n_neighbors, n_pcs=n_pcs, random_state=seed)
        sc.tl.umap(B, min_dist=0.3, random_state=seed)
        sc.tl.leiden(B, resolution=res, key_added='leiden')
        umapplot(B, ['leiden'], f'_{tag}_leiden', f'{tag}: UMAP — Leiden {res}', legend_fontsize=8)
        for c in ['n_genes_by_counts','total_counts','pct_counts_mt','pct_counts_hb']:
            if c in B.obs:
                umapplot(B, [c], f'_{tag}_UMAP_{c}', f'{tag}: UMAP colored by {c}')
        return B

    B_pre = embed_and_cluster(adata1_preD, tag='preDoublet', n_pcs=20, n_neighbors=20, res=0.6, seed=42)
    adata2 = adata1[~adata1.obs['doublet']].copy()
    print(f"[Scrublet] removed {adata1.n_obs - adata2.n_obs:,} cells (postDoublet n={adata2.n_obs:,})")
    B_post = embed_and_cluster(adata2, tag='postDoublet', n_pcs=20, n_neighbors=20, res=0.6, seed=42)

    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
    common = B_pre.obs_names.intersection(B_post.obs_names)
    ari = adjusted_rand_score(B_pre.obs.loc[common,'leiden'], B_post.obs.loc[common,'leiden'])
    nmi = normalized_mutual_info_score(B_pre.obs.loc[common,'leiden'], B_post.obs.loc[common,'leiden'])
    print(f"[Compare] Cluster agreement on shared cells: ARI={ari:.3f}, NMI={nmi:.3f}")

    tab = pd.crosstab(B_pre.obs['leiden'], adata1_preD.obs.loc[B_pre.obs_names, 'doublet'])
    tab.to_csv(os.path.join(FIGDIR, 'preDoublet_doublet_enrichment_by_cluster.tsv'), sep='\t')
    umapplot(B_pre, ['doublet_score'], '_preDoublet_UMAP_doubletScore', 'preDoublet: UMAP colored by Scrublet doublet_score', vmax='p99')

    for col in ['n_genes_by_counts','total_counts','pct_counts_mt','pct_counts_hb']:
        if col in adata1_preD.obs and col in adata2.obs:
            print(f"[QC shift] {col}: median pre={adata1_preD.obs[col].median():.1f} post={adata2.obs[col].median():.1f}")

    # ---------- MAIN pipeline ----------
    adata_main = adata2.copy()
    sc.pp.normalize_total(adata_main, 1e4); sc.pp.log1p(adata_main)
    sc.pp.highly_variable_genes(adata_main, flavor='seurat_v3', n_top_genes=3000,
                                batch_key='sample' if 'sample' in adata_main.obs.columns else None)
    hvgsplot(adata_main, '_MAIN_seuratv3_HVGs', 'MAIN: Highly variable genes (Seurat v3)')
    adata_main.raw = adata_main
    adata_main = adata_main[:, adata_main.var['highly_variable']].copy()
    covars = ['total_counts']
    if 'pct_counts_mt' in adata_main.obs: covars.append('pct_counts_mt')
    sc.pp.regress_out(adata_main, covars)
    sc.pp.scale(adata_main, max_value=10)
    sc.tl.pca(adata_main, svd_solver='arpack')
    pca_varplot(adata_main, n_pcs=50, basename='_MAIN_pca_elbow', caption='MAIN: PCA variance ratio (choose ~20 PCs)')
    sc.pp.neighbors(adata_main, n_neighbors=20, n_pcs=20, random_state=42)
    sc.tl.umap(adata_main, min_dist=0.3, random_state=42)
    sc.tl.leiden(adata_main, resolution=0.6, key_added='leiden_0p6')
    umapplot(adata_main, ['leiden_0p6'], '_MAIN_leiden_0p6', 'MAIN: UMAP — Leiden 0.6', legend_fontsize=8)

    # ---------- DE ----------
    sc.tl.rank_genes_groups(adata_main, groupby='leiden_0p6', method='wilcoxon', use_raw=True, pts=True)
    rankgenesplot(adata_main, '_MAIN_wilcoxon', 'MAIN: Top markers per cluster (Wilcoxon)', n_genes=25, sharey=False)
    sc.get.rank_genes_groups_df(adata_main, None).to_csv(os.path.join(FIGDIR, 'MAIN_markers_leiden0p6_wilcoxon.csv'), index=False)

    # ---------- CellTypist (v1.7.1-safe) ----------
    CELLTYPIST_MODEL = "Mouse_Whole_Brain.pkl"  # <- set to an actual .pkl in ~/.celltypist/models

    if USE_CELLTYPIST:
        print("[CellTypist] version:", getattr(celltypist, "__version__", "unknown"))

        # Ensure model is present
        try:
            models.download_model(CELLTYPIST_MODEL)
        except Exception as e:
            print(f"[CellTypist] download_model failed: {e}. Trying full catalog...")
            try:
                models.download_models()
            except Exception as e2:
                print(f"[CellTypist] download_models failed: {e2}")

        def run_celltypist(A, model=CELLTYPIST_MODEL, mv=True):
            pred = celltypist.annotate(A, model=model, majority_voting=mv)
            # add .confidence if missing (max prob per cell)
            if not hasattr(pred, "confidence") or pred.confidence is None:
                if hasattr(pred, "probability") and pred.probability is not None:
                    import numpy as np, pandas as pd
                    conf = np.asarray(pred.probability).max(axis=1).astype(float)
                    pred.confidence = pd.Series(conf, index=A.obs_names)
            return pred

        # ---- main object ----
        pred_main = run_celltypist(adata_main)
        adata_main.obs["ct_label"] = pred_main.predicted_labels
        adata_main.obs["ct_label_mv"] = getattr(pred_main, "majority_voting", adata_main.obs["ct_label"])
        if hasattr(pred_main, "confidence") and pred_main.confidence is not None:
            adata_main.obs["ct_conf"] = pd.Series(pred_main.confidence, index=adata_main.obs_names)

        # optional post-threshold
        conf_cut = 0.5
        if "ct_conf" in adata_main.obs:
            low = adata_main.obs["ct_conf"] < conf_cut
            adata_main.obs.loc[low, "ct_label_mv"] = "Unknown_lowConf"

        # ---- if you also annotate B_pre (optional) ----
        if 'B_pre' in globals():
            pred_pre = run_celltypist(B_pre)
            B_pre.obs["ct_label_mv"] = getattr(pred_pre, "majority_voting", pred_pre.predicted_labels)
            if hasattr(pred_pre, "confidence") and pred_pre.confidence is not None:
                B_pre.obs["ct_conf"] = pd.Series(pred_pre.confidence, index=B_pre.obs_names)
                low2 = B_pre.obs["ct_conf"] < conf_cut
                B_pre.obs.loc[low2, "ct_label_mv"] = "Unknown_lowConf"

            # agreement audit (only if both exist)
            common2 = B_pre.obs_names.intersection(adata_main.obs_names)
            if len(common2) > 0:
                agree = (B_pre.obs.loc[common2, "ct_label_mv"] == adata_main.obs.loc[common2, "ct_label_mv"]).mean()
                print(f"[CellTypist] label agreement preDoublet vs MAIN on shared cells: {agree*100:.1f}%")

    # ---------- Save snapshots ----------
    adata0.write(os.path.join(ADATA, org + '_adata_RAW.h5ad'))
    adata1_preD.write(os.path.join(ADATA, org + '_adata_QC_dataDriven_preDoublet.h5ad'))
    adata2.write(os.path.join(ADATA, org + '_adata_postDoublet.h5ad'))
    adata_main.write(os.path.join(ADATA, org + '_adata_MAIN_postDoublet_processed.h5ad'))




    report_main = evaluate_pipeline(
        adata_main,
        cluster_key='leiden_0p6',
        sample_key='sample',          # or whatever your batch/sample column is
        n_pcs_for_sil=20,
        stability_resolution=0.6,
        stability_min_dist=0.3,
        seeds=(1,2,3),
        verbose=True,
    )
    save_evaluation(
        adata_main,
        cluster_key="leiden_0p6",
        outfolder=reports,
        pipeline_tag="large",
        organism=org,
        sample_key="one_sample_name",
    )


print("Done.")

