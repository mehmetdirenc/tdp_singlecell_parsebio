import sys
import scrublet as scr
import numpy as np
import pandas as pd
import scanpy as sc
import scipy, os
import celltyping
from celltyping import models
import scipy.io as sio
from results_quality_check import evaluate_pipeline

## General scanpy settings ##
sc.settings.verbosity = 1 # verbosity: errors (0), warnings (1), info (2), hints (3)
sc.settings.set_figure_params(dpi=100, fontsize=10, dpi_save=300, figsize=(5,4), format='png')
# sc.settings.figdir = '/home/ubuntu/volume_750gb/results/tdp_project/figures_AB_mouse'
# data_path = '/home/ubuntu/volume_750gb/results/tdp_project/tdp_combined_analysis_files_AB_mouse/DGE_filtered/'
# sc.settings.figdir = '/home/ubuntu/volume_750gb/results/tdp_project/figures_CD_rat'
data_path = '/home/ubuntu/volume_750gb/results/tdp_project/tdp_combined_analysis_files_CD_rat/DGE_filtered/'


adata = sc.read_mtx(data_path + "count_matrix.mtx")
gene_data = pd.read_csv(data_path + "all_genes.csv")
cell_meta = pd.read_csv(data_path + "cell_metadata.csv")

adata.var_names = gene_data["gene_name"]
adata.obs = cell_meta
adata.obs_names = adata.obs["bc_wells"]

# (Optional)
sc.pp.filter_genes(adata, min_cells=5)
sc.pp.filter_cells(adata, min_genes=300)

adata.var_names_make_unique() ## Complains later on if the variable names are not unique

# Check dimensions of the expression matrix (cells, genes)
print(adata.shape)

# --- QC feature flags ---
# mtDNA genes (case-insensitive): mt-Nd1, mt-Co1, mt-Cytb, etc.
adata.var['mt'] = adata.var_names.str.upper().str.startswith('MT-')

# ribosomal (nuclear) genes: RPS*, RPL*
adata.var['ribo'] = adata.var_names.str.startswith(("RPS", "RPL"))

# hemoglobin (blood contamination)
adata.var['hb'] = adata.var_names.str.match(r'^(HBA|HBB|HB.*)$', case=False)

# Compute QC metrics using the mt flag
sc.pp.calculate_qc_metrics(adata, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)

# do we have non-empty lists
print("mt genes found (first 10), count:", list(adata.var_names[adata.var['mt']])[:10], len(list(adata.var_names[adata.var['mt']])))
print("ribo genes found (first 10), count:", list(adata.var_names[adata.var['ribo']])[:10], len(list(adata.var_names[adata.var['ribo']])))
print("hb genes found (first 10), count:", list(adata.var_names[adata.var['hb']])[:10], len(list(adata.var_names[adata.var['hb']])))
# Scanpy will prepend the string in the save argument with "violin"
# and save it to our figure directory defined in the first step.
# sc.pl.violin(adata, ['n_genes_by_counts'], save='_n_genes', jitter=0.4)
# sc.pl.violin(adata, ['total_counts'], save='_total_counts', jitter=0.4)
# sc.pl.violin(adata, ['pct_counts_mt'], save='_mito_pct', jitter=0.4)

# Example robust filters (global or per batch if adata.obs has 'sample'):
q = adata.obs['n_genes_by_counts'].quantile
high_genes = q(0.99)         # or use median + 3*MAD
high_counts = adata.obs['total_counts'].quantile(0.99)
adata = adata[adata.obs['n_genes_by_counts'] < high_genes, :]
adata = adata[adata.obs['total_counts'] < high_counts, :]
adata = adata[adata.obs['pct_counts_mt'] < 15, :]  # 10–15% typical; adjust after fixing mt


# counts = adata.X.copy()
# if not scipy.sparse.issparse(counts):
#     counts = scipy.sparse.csr_matrix(counts)
# scrub = scr.Scrublet(counts, expected_doublet_rate=0.06)  # ParseBio ~5–10% often
# doublet_scores, preds = scrub.scrub_doublets()
# adata.obs['doublet_score'] = doublet_scores
# adata.obs['doublet'] = preds
# adata = adata[~adata.obs['doublet'], :]

# sys.exit()
# Scanpy will prepend the string in the save argument with "violin"
# and save it to our figure directory defined in the first step.
# sc.pl.violin(adata, ['n_genes_by_counts'], save='_after_quantile_doublet_n_genes', jitter=0.4)
# sc.pl.violin(adata, ['total_counts'], save='_after_quantile_doublet_total_counts', jitter=0.4)
# sc.pl.violin(adata, ['pct_counts_mt'], save='_after_quantile_doublet_mito_pct', jitter=0.4)

##TODO CONSIDER CHANGING THE THRESHOLDING STRATEGY TO DATA-DRIVEN ##

adata = adata[adata.obs.n_genes_by_counts < 5000,:]
adata = adata[adata.obs.total_counts < 20000,:]
adata = adata[adata.obs.pct_counts_mt < 15,:]
print(adata.shape) # Checking number of cells remaining

# sc.pl.scatter(adata, x='total_counts', y='n_genes_by_counts', save='_gene_vs_transcript_counts')
print('median transcript count per cell: ' + str(adata.obs['tscp_count'].median(0)))
print('median gene count per cell: ' + str(adata.obs['gene_count'].median(0)))
# print(sc.settings.figdir, os.getcwd())

sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)

sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.25)
# sc.pl.highly_variable_genes(adata, save='') # scanpy generates the filename automatically

# Save raw expression values before variable gene subset
adata.raw = adata

adata = adata[:, adata.var.highly_variable]
sc.pp.regress_out(adata, ['total_counts'])
sc.pp.scale(adata, max_value=10)

sc.tl.pca(adata, svd_solver='arpack')
# sc.pl.pca_variance_ratio(adata, log=True, n_pcs=50, save='') # scanpy generates the filename automatically

sc.pp.neighbors(adata, n_neighbors=10, n_pcs=30)
sc.tl.umap(adata)
sc.tl.leiden(adata, resolution=0.5)
# sc.pl.umap(adata, color=['leiden'], legend_fontsize=8, save='_leiden')
sc.pl.umap(adata, color=['leiden'], legend_fontsize=8)

sc.tl.rank_genes_groups(adata, 'leiden', method='t-test')

# The head function returns the top n genes per cluster
top_markers = pd.DataFrame(adata.uns['rank_genes_groups']['names']).head(5)
print(top_markers)

adata.write(data_path + 'adata_after_diffexp.h5ad')
adata = sc.read(data_path + 'adata_after_diffexp.h5ad')


# after you create `adata` and 'leiden'
report_basic = evaluate_pipeline(adata, cluster_key='leiden', sample_key='sample')
