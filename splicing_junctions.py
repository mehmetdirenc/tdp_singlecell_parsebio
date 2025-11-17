import os, re
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse
import matplotlib.pyplot as plt

# ---------------- USER CONFIG ----------------
ADATA_DIR     = " /home/ubuntu/volume_750gb/results/tdp_project/adata_files"
FIGDIR_MOUSE  = "/home/ubuntu/volume_750gb/results/tdp_project/figures_AB_mouse"
RESULTS_MOUSE = "/home/ubuntu/volume_750gb/results/tdp_project/splicing_mouse"  # <— define

os.makedirs(FIGDIR_MOUSE, exist_ok=True)
os.makedirs(RESULTS_MOUSE, exist_ok=True)

# Target genes (anchors first, rest optional)
GENE_ALIASES = [
    "Mrpl45", "Ddi2", "Psmd14",
    "Hjurp","Cops4","Tiam1","Creld1","Inpp4a","Shank1",
    "Arhgap44","Psme3","Dgkq","Arhgef11","Osbpl6","Pik3cb","Pdzdc1","Dsel",
    "Herc2","Golga4","Stx24","Ube3c","C530008M17Rik","Tub","Atp2b1",
    "Lhx1","Lhx1os","Onecut1","Slc32a1","Gad1os","Gad2","Dlx6","Npas1","Pou6f2",
]

# Grouping column
GROUP_KEY = "ct_label_mv"  # change to 'leiden_0p6' if you prefer

# Mouse only (choose RAW or MAIN)
ADATAS = [
    ("mouse", os.path.join(ADATA_DIR, "mouse_adata_RAW.h5ad"), RESULTS_MOUSE),
    # ("mouse", os.path.join(ADATA_DIR, "mouse_adata_MAIN_postDoublet_processed.h5ad"), RESULTS_MOUSE),
]

# --- positivity threshold (recommended ≥2 UMIs) ---
MIN_UMI = 2
LOG2_CUT = np.log2(1 + MIN_UMI)

# ---------- HELPERS ----------
def _to_dense(x):
    if sparse.issparse(x): return x.toarray()
    toarr = getattr(x, "toarray", None)
    return toarr() if callable(toarr) else np.asarray(x)

def _var_index(A):
    """Prefer .raw.var_names if available (so non-HVG genes are visible)."""
    if A.raw is not None:
        return pd.Index(A.raw.var_names).astype(str), True
    return pd.Index(A.var_names).astype(str), False

def map_gene_keys(adata, gene_list):
    var, _ = _var_index(adata)
    var_upper = var.str.upper()
    var_stripped = var_upper.str.replace(r'[-_.]', '', regex=True)
    mapping = {}
    for g in gene_list:
        gu = g.upper()
        gs = re.sub(r'[-_.]', '', gu)
        if g in var: mapping[g] = g; continue
        m = var[var.str.match(fr"^{re.escape(g)}(_.*)?$", case=False, na=False)]
        if len(m): mapping[g] = m[0]; continue
        m = var[var_upper == gu]
        if len(m): mapping[g] = m[0]; continue
        m = var[var_stripped == gs]
        if len(m): mapping[g] = m[0]; continue
        m = var[var_upper.str.contains(gu)]
        mapping[g] = m[0] if len(m) else None
    return mapping

def _get_expr_log2(A, var_key):
    """Return Series of log2(1+x) expression for a single gene, preferring A.raw."""
    if A.raw is not None and var_key in A.raw.var_names:
        X = A.raw[:, var_key].X
    else:
        X = A[:, var_key].X
    vals = _to_dense(X).ravel()
    # auto-detect: if it looks like counts -> log1p; else assume ln -> convert to log2
    p99 = float(np.nanpercentile(vals, 99)) if np.isfinite(vals).any() else 0.0
    if p99 > 15:   # raw counts
        vals = np.log1p(vals) / np.log(2)
    else:          # likely ln(1+x)
        vals = vals / np.log(2)
    return pd.Series(vals, index=A.obs_names, name=var_key)

def summarize_expression(A, var_key, group_key=None):
    """
    Returns: (overall_dict, per_group_df)
      overall_dict keys: n_cells, n_detected_cells, pct_detected, mean, median, mean_in_detected, median_in_detected
      per_group_df: one row per group with the same metrics (+ 'group')
    """
    x = _get_expr_log2(A, var_key)
    n = x.size
    pos = x > LOG2_CUT  # STRICTER: ≥ MIN_UMI UMIs

    def stats(v):
        if v.size == 0: return {"mean": np.nan, "median": np.nan}
        return {"mean": float(np.nanmean(v)), "median": float(np.nanmedian(v))}

    overall = {
        "n_cells": int(n),
        "n_detected_cells": int(pos.sum()),
        "pct_detected": float(pos.mean() * 100.0),
        **stats(x),
        **{f"{k}_in_detected": v for k, v in stats(x[pos]).items()},
    }

    rows = []
    if group_key and group_key in A.obs.columns:
        for g, idx in A.obs.groupby(group_key).indices.items():
            xi = x.iloc[idx]
            pi = xi > LOG2_CUT
            row = {
                "group": str(g),
                "n_cells": int(xi.size),
                "n_detected_cells": int(pi.sum()),
                "pct_detected": float(pi.mean() * 100.0),
                **stats(xi),
                **{f"{k}_in_detected": v for k, v in stats(xi[pi]).items()},
            }
            rows.append(row)
    return overall, pd.DataFrame(rows)

def ensure_umap(A):
    if "X_umap" in A.obsm_keys(): return
    B = A.copy()
    sc.pp.normalize_total(B, 1e4); sc.pp.log1p(B)
    sc.pp.pca(B); sc.pp.neighbors(B, n_neighbors=15, n_pcs=30); sc.tl.umap(B)
    A.obsm["X_umap"] = B.obsm["X_umap"]

def plot_gene_umap(A, gene_name, var_key, out_png, vmax=4.0):
    """One PNG per gene; zeros in grey; OrRd; 0–4 log2 scale."""
    vals = _get_expr_log2(A, var_key)
    tmp = "_tmp_expr_"
    A.obs[tmp] = vals.where(vals > 0, np.nan)
    ensure_umap(A)
    fig = sc.pl.umap(A, color=[tmp], na_color="lightgray", cmap="OrRd",
                     vmin=0, vmax=vmax, frameon=True, show=False, return_fig=True)
    if not hasattr(fig, "savefig"): fig = plt.gcf()
    fig.axes[0].set_title(gene_name)
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    A.obs.drop(columns=[tmp], inplace=True, errors="ignore")

# ---------- MAIN ----------
def main():
    for org, h5, outdir in ADATAS:
        if not os.path.exists(h5):
            print(f"[{org}] missing: {h5} — skipping"); continue

        A = sc.read_h5ad(h5)
        print(f"[{org}] loaded {h5}  | cells={A.n_obs:,}  genes={A.n_vars:,}")
        os.makedirs(outdir, exist_ok=True)

        # Map aliases -> actual var_names present
        gene_map = map_gene_keys(A, GENE_ALIASES)
        found = {k: v for k, v in gene_map.items() if v}
        missing = [k for k, v in gene_map.items() if v is None]
        print(f"[{org}] found {len(found)}/{len(GENE_ALIASES)} genes.")
        if missing:
            print(f"[{org}] missing: {missing}")

        # Summaries
        overall_rows = []
        per_group_rows = []
        for alias, var_key in found.items():
            overall, per_group = summarize_expression(A, var_key, group_key=GROUP_KEY)
            overall_rows.append({"organism": org, "alias": alias, "var_name": var_key, **overall})
            if len(per_group):
                per_group.insert(0, "organism", org)
                per_group.insert(1, "alias", alias)
                per_group.insert(2, "var_name", var_key)
                per_group.insert(3, "group_key", GROUP_KEY)
                per_group_rows.append(per_group)

            # UMAP per gene
            png = os.path.join(outdir, f"{org}_UMAP_{alias}.png")
            plot_gene_umap(A, alias, var_key, png)

        # Write CSVs
        df_overall = pd.DataFrame(overall_rows).sort_values(["organism", "alias"])
        out_overall = os.path.join(outdir, f"{org}_gene_expression_overall.csv")
        df_overall.to_csv(out_overall, index=False)
        print(f"[{org}] wrote: {out_overall}")

        if per_group_rows:
            df_groups = pd.concat(per_group_rows, ignore_index=True)
            out_groups = os.path.join(outdir, f"{org}_gene_expression_by_{GROUP_KEY}.csv")
            df_groups.to_csv(out_groups, index=False)
            print(f"[{org}] wrote: {out_groups}")

        # (Optional) a single 3-panel anchor plot
        anchors = [g for g in ["Mrpl45", "Ddi2", "Psmd14"] if found.get(g)]
        if anchors:
            ensure_umap(A)
            cols = [gene_map[g] for g in anchors]
            fig = sc.pl.umap(A, color=cols, na_color="lightgray", cmap="OrRd",
                             vmin=0, vmax=4, ncols=min(3,len(cols)),
                             show=False, return_fig=True)
            if not hasattr(fig, "savefig"): fig = plt.gcf()
            fig.savefig(os.path.join(outdir, f"{org}_UMAP_anchor_genes.png"), dpi=300, bbox_inches="tight")
            plt.close(fig)

if __name__ == "__main__":
    main()
