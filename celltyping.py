#!/usr/bin/env python3
import os
import sys
import inspect
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
from textwrap import shorten
from matplotlib.gridspec import GridSpec

# --- import celltypist safely (avoid local shadowing) ---
try:
    import celltypist
    from celltypist import models
except Exception as e:
    print(f"[ERROR] Could not import celltypist: {e}")
    sys.exit(1)

print("[CellTypist] version:", getattr(celltypist, "__version__", "unknown"))
try:
    print("[CellTypist] module file:", inspect.getfile(celltypist))
except Exception:
    pass

# -------------------- USER SETTINGS --------------------
CELLTYPIST_MODEL = "Mouse_Whole_Brain.pkl"   # must exist in ~/.celltypist/models or downloadable
ADATA_DIR = "/home/ubuntu/volume_750gb/results/tdp_project/adata_files"
# IN_H5AD   = os.path.join(ADATA_DIR, "rat_adata_MAIN_postDoublet_processed.h5ad")
# OUT_H5AD  = os.path.join(ADATA_DIR, "rat_adata_MAIN_postDoublet_celltypist.h5ad")
# OUT_H5AD_VIZ = os.path.join(ADATA_DIR, "rat_adata_MAIN_postDoublet_celltypist+viz.h5ad")
# FIGDIR = "/home/ubuntu/volume_750gb/results/tdp_project/figures_CD_rat"
IN_H5AD   = os.path.join(ADATA_DIR, "mouse_adata_MAIN_postDoublet_processed.h5ad")
OUT_H5AD  = os.path.join(ADATA_DIR, "mouse_adata_MAIN_postDoublet_celltypist.h5ad")
OUT_H5AD_VIZ = os.path.join(ADATA_DIR, "mouse_adata_MAIN_postDoublet_celltypist+viz.h5ad")
FIGDIR = "/home/ubuntu/volume_750gb/results/tdp_project/figures_AB_mouse"

CONF_CUT  = 0.50        # confidence threshold for Unknown_lowConf; set None to disable
MAJORITY_VOTING = True
CLUSTER_KEY = "leiden_0p6"   # change if your leiden key is different, or set to None to skip
os.makedirs(FIGDIR, exist_ok=True)
# -------------------------------------------------------

# --------- helpers: robustly coerce outputs to Series ---------

import numpy as np
import matplotlib.pyplot as plt
from textwrap import shorten
from matplotlib.gridspec import GridSpec

def plot_leiden_with_ct_panel(
    adata,
    cluster_key="leiden_0p6",
    label_key="ct_label_mv",
    out_png="umap_leiden_with_celltypist_panel.png",
    point_size=2.0,
    width=16,
    height=10.5,
    show_top_k=1,        # 1 or 2 labels per cluster
    max_label_width=32,  # truncate long labels for readability
    panel_cols="1",   # "auto", 2, or 3
    panel_fontsize=9,
):
    """Left: UMAP colored by Leiden. Right: text panel mapping cluster -> cell type (dominant, purity, n)."""
    if "X_umap" not in adata.obsm_keys():
        if "X_pca" not in adata.obsm:
            sc.pp.pca(adata)
        sc.pp.neighbors(adata, n_neighbors=15, n_pcs=min(50, adata.obsm["X_pca"].shape[1]))
        sc.tl.umap(adata)

    if cluster_key not in adata.obs or label_key not in adata.obs:
        print(f"[Viz] Missing {cluster_key=} or {label_key=}; skipping.")
        return

    # Composition
    tab = pd.crosstab(adata.obs[cluster_key], adata.obs[label_key])
    totals = tab.sum(axis=1).replace(0, np.nan)
    prop = tab.div(totals, axis=0).fillna(0)

    # Build lines: keep (cluster_id, text_without_cluster_number)
    lines = []
    for cl in tab.index.astype(str):
        row = prop.loc[cl].sort_values(ascending=False)
        entries = []
        for lab, frac in row.head(show_top_k).items():
            n = int(tab.loc[cl, lab])
            lab_print = shorten(str(lab), width=max_label_width, placeholder="…")
            entries.append(f"{lab_print} ({frac*100:.1f}%, n={n})")
        text = " ;  ".join(entries) if entries else "—"
        lines.append((cl, text))

    # Categorical palette
    cats = adata.obs[cluster_key].astype("category")
    adata.obs[cluster_key] = cats
    palette = sc.pl.palettes.default_102
    pal = {c: palette[i % len(palette)] for i, c in enumerate(cats.cat.categories)}

    # Figure
    fig = plt.figure(figsize=(width, height))
    gs = GridSpec(1, 2, width_ratios=[3.2, 1.8], wspace=0.06, figure=fig)

    # Left: UMAP
    ax0 = fig.add_subplot(gs[0, 0])
    sc.pl.umap(
        adata,
        color=cluster_key,
        ax=ax0,
        show=False,
        legend_loc=None,
        frameon=False,
        size=point_size,
        palette=[pal[c] for c in cats.cat.categories],
        use_raw=False,
    )
    ax0.set_title(f"MAIN: UMAP — {cluster_key.replace('_',' ').upper()}")

    # Right: neat non-overlapping text panel
    ax1 = fig.add_subplot(gs[0, 1])
    ax1.axis("off")
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)

    ax1.text(0.02, 0.975, "Leiden → CellTypist", fontsize=12, fontweight="bold", va="top")

    # Sort clusters numerically if possible
    def _key(c):
        try:
            return int(c)
        except Exception:
            return c
    lines_sorted = sorted(lines, key=lambda x: _key(x[0]))
    n = len(lines_sorted)

    # Decide number of columns
    if panel_cols == "auto":
        ncols = 2 if n <= 28 else 3
    else:
        ncols = int(panel_cols)
    n_per_col = (n + ncols - 1) // ncols

    # Column blocks and evenly spaced y to avoid overlap
    col_width = 0.96 / ncols  # small margin on right
    for col in range(ncols):
        block = lines_sorted[col*n_per_col : (col+1)*n_per_col]
        if not block:
            continue
        x0 = 0.02 + col * col_width
        # Title per column
        first_idx = _key(block[0][0])
        last_idx  = _key(block[-1][0])
        ax1.text(x0, 0.94, f"Clusters {first_idx}–{last_idx}", fontsize=11, fontweight="bold", va="top")

        ys = np.linspace(0.90, 0.06, num=len(block))  # evenly spaced, top→bottom
        for (cl, text), y in zip(block, ys):
            # colored square (cluster color)
            try:
                color = pal[cl if cl in pal else cats.cat.categories[int(cl)]]
            except Exception:
                color = "black"
            ax1.add_patch(plt.Rectangle((x0, y-0.012), 0.012, 0.012, color=color,
                                        transform=ax1.transAxes, clip_on=False))
            # text WITHOUT the numeric cluster id (as requested)
            ax1.text(x0 + 0.018, y, text, fontsize=panel_fontsize, va="top")

    out_path = os.path.join(FIGDIR, out_png) if "FIGDIR" in globals() else out_png
    plt.savefig(out_path, dpi=250, bbox_inches="tight")
    plt.close(fig)
    print(f"[Viz] Saved {out_path}")



def _as_series(x, preferred=None, index=None):
    """Coerce celltypist outputs (Series/DataFrame/np.array) to a 1D Series aligned to `index`."""
    if isinstance(x, pd.Series):
        return x.reindex(index)
    if isinstance(x, pd.DataFrame):
        if preferred and preferred in x.columns:
            s = x[preferred]
        elif 'predicted_labels' in x.columns:
            s = x['predicted_labels']
        elif 'majority_voting' in x.columns:
            s = x['majority_voting']
        elif 'labels' in x.columns:
            s = x['labels']
        elif x.shape[1] == 1:
            s = x.iloc[:, 0]
        else:
            s = x.iloc[:, 0]  # fallback
        return s.reindex(index)
    try:
        arr = np.asarray(x).reshape(-1)
        return pd.Series(arr, index=index)
    except Exception:
        return pd.Series(pd.NA, index=index)

# ----------------- core: run celltypist -----------------
def run_celltypist(A, model=CELLTYPIST_MODEL, mv=MAJORITY_VOTING):
    """
    Run celltypist.annotate and return the prediction object.
    Ensures the model is available; falls back to catalog download if needed.
    """
    # Ensure model exists or download
    try:
        models.download_model(model)
    except Exception as e:
        print(f"[CellTypist] download_model('{model}') failed: {e}. Trying full catalog...")
        try:
            models.download_models()
        except Exception as e2:
            print(f"[CellTypist] download_models() failed: {e2}")
            print("[CellTypist] Proceeding anyway; if the model is missing, annotate will fail.")
    # Annotate
    pred = celltypist.annotate(A, model=model, majority_voting=mv)
    # If .confidence missing, derive from probability
    if not hasattr(pred, "confidence") or pred.confidence is None:
        prob = getattr(pred, "probability", None)
        if prob is not None:
            try:
                if isinstance(prob, pd.DataFrame):
                    conf = prob.max(axis=1).astype(float)
                else:
                    conf = pd.Series(np.asarray(prob).max(axis=1), index=A.obs_names).astype(float)
                pred.confidence = conf
            except Exception:
                pass
    return pred

# --------------------- visualization --------------------
def ensure_umap(adata):
    """Compute neighbors/UMAP if missing (non-destructive if present)."""
    has_umap = "X_umap" in adata.obsm_keys()
    has_nn = "neighbors" in adata.uns and "distances" in adata.obsp and "connectivities" in adata.obsp
    if not has_nn:
        # Try a reasonable default; assumes PCA already exists; if not, make one.
        if "X_pca" not in adata.obsm_keys():
            sc.pp.pca(adata)
        sc.pp.neighbors(adata, n_neighbors=15, n_pcs=min(50, adata.obsm.get("X_pca", np.empty((0,0))).shape[1] if "X_pca" in adata.obsm_keys() else 50))
    if not has_umap:
        sc.tl.umap(adata)

# def plot_umap(adata, keys, fname):
#     sc.pl.umap(adata, color=keys, wspace=0.35, legend_loc="on data", frameon=False, ncols=2, show=False)
#     plt.savefig(os.path.join(FIGDIR, fname), dpi=200, bbox_inches="tight")
#     plt.close()



def plot_umap(adata, keys, fname):
    # keep only keys that exist in obs or var names
    valid = [k for k in keys if (k in adata.obs.columns) or (k in adata.var_names)]
    missing = [k for k in keys if k not in valid]
    for k in missing:
        print(f"[Viz] Skipping missing color key: {k}")
    if not valid:
        print(f"[Viz] No valid color keys for {fname}; skipping.")
        return
    sc.pl.umap(
        adata,
        color=valid,
        wspace=0.35,
        legend_loc="on data",
        frameon=False,
        ncols=2,
        show=False,
        use_raw=False,   # <- important: avoid falling back to .raw
    )
    plt.savefig(os.path.join(FIGDIR, fname), dpi=200, bbox_inches="tight")
    plt.close()




def plot_stacked_composition(adata, cluster_key, label_key="ct_label_mv", top_k=12, fname="stacked_comp.png"):
    tab = pd.crosstab(adata.obs[cluster_key], adata.obs[label_key])
    comp = tab.div(tab.sum(axis=1), axis=0)  # per-cluster proportions
    top_labels = comp.sum(0).sort_values(ascending=False).head(top_k).index
    comp_plot = comp[top_labels].copy()
    comp_plot["Other"] = (1 - comp_plot.sum(axis=1)).clip(lower=0)
    ax = comp_plot.sort_index().plot(kind="bar", stacked=True, figsize=(12, 5))
    ax.set_ylabel("Proportion")
    ax.set_xlabel(cluster_key)
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", title=label_key)
    plt.tight_layout()
    out = os.path.join(FIGDIR, fname)
    plt.savefig(out, dpi=200)
    plt.close()

# ----------------------- main --------------------------
def main():
    # Safety: prevent running if script name shadows the package
    this_file = os.path.basename(__file__)
    if this_file == "celltypist.py":
        print("[ERROR] Please rename this script (e.g., run_celltyping_and_viz.py); "
              "current name shadows the 'celltypist' package.")
        sys.exit(3)

    # Load AnnData
    if not os.path.exists(IN_H5AD):
        print(f"[ERROR] Input h5ad not found: {IN_H5AD}")
        sys.exit(2)
    adata = sc.read_h5ad(IN_H5AD)
    cells_index = adata.obs_names
    print(f"[IO] Loaded: {IN_H5AD}  | cells: {adata.n_obs:,}  genes: {adata.n_vars:,}")

    # Run celltypist
    pred = run_celltypist(adata)

    # Normalize outputs into obs
    adata.obs["ct_label"] = _as_series(getattr(pred, "predicted_labels", None),
                                       preferred="predicted_labels",
                                       index=cells_index)

    mv_obj = getattr(pred, "majority_voting", None)
    if mv_obj is not None:
        adata.obs["ct_label_mv"] = _as_series(mv_obj, preferred="majority_voting", index=cells_index)
    else:
        adata.obs["ct_label_mv"] = adata.obs["ct_label"]

    conf_obj = getattr(pred, "confidence", None)
    if conf_obj is not None:
        adata.obs["ct_conf"] = _as_series(conf_obj, preferred="confidence", index=cells_index).astype(float)
    else:
        prob = getattr(pred, "probability", None)
        if prob is not None:
            if isinstance(prob, pd.DataFrame):
                adata.obs["ct_conf"] = prob.max(axis=1).reindex(cells_index).astype(float)
            else:
                adata.obs["ct_conf"] = pd.Series(np.asarray(prob).max(axis=1), index=cells_index).astype(float)

    # Optional confidence threshold
    if CONF_CUT is not None and "ct_conf" in adata.obs:
        low = adata.obs["ct_conf"] < float(CONF_CUT)
        adata.obs.loc[low, "ct_label_mv"] = "Unknown_lowConf"
    elif "ct_conf" not in adata.obs:
        print("[Note] ct_conf not available (no confidence/probability in prediction); "
              "skipping confidence thresholding and confidence plots.")

    # Also keep full raw tables if they were DataFrames (useful for debugging/audits)
    if isinstance(getattr(pred, "predicted_labels", None), pd.DataFrame):
        for c in pred.predicted_labels.columns:
            adata.obs[f"ct_pred_{c}"] = pred.predicted_labels[c].reindex(cells_index)
    if isinstance(getattr(pred, "majority_voting", None), pd.DataFrame):
        for c in pred.majority_voting.columns:
            adata.obs[f"ct_mv_{c}"] = pred.majority_voting[c].reindex(cells_index)

    # Quick summary
    print("[Summary] ct_label (top 10):")
    print(adata.obs["ct_label"].value_counts(dropna=False).head(100))
    print("[Summary] ct_label_mv (top 10):")
    print(adata.obs["ct_label_mv"].value_counts(dropna=False).head(10))
    if "ct_conf" in adata.obs:
        print("[Summary] ct_conf: min={:.3f} median={:.3f} max={:.3f}".format(
            float(adata.obs["ct_conf"].min()),
            float(adata.obs["ct_conf"].median()),
            float(adata.obs["ct_conf"].max()),
        ))

    # Save annotated object (pre-viz)
    adata.write(OUT_H5AD)
    print(f"[IO] Saved annotated AnnData → {OUT_H5AD}")

    # ------------- Visualization & Leiden integration -------------
    # Ensure neighbors/UMAP exist for plotting
    ensure_umap(adata)
    plot_leiden_with_ct_panel(adata,
                              cluster_key=CLUSTER_KEY,  # e.g. "leiden_0p6"
                              label_key="ct_label_mv",  # CellTypist majority-vote labels
                              out_png="umap_leiden_with_celltypist_panel.png",
                              point_size=1.8,  # tweak if you want smaller/larger points
                              show_top_k=1,  # set 2 to list top-2 labels per cluster
                              )
    # If you have a Leiden key, analyze & plot against it
    if CLUSTER_KEY and CLUSTER_KEY in adata.obs:
        # Dominant CellTypist label per Leiden cluster
        ctab = pd.crosstab(adata.obs[CLUSTER_KEY], adata.obs["ct_label_mv"])
        dominant = ctab.idxmax(axis=1)
        adata.obs["ct_major_by_leiden"] = adata.obs[CLUSTER_KEY].map(dominant)

        # Pretty cluster name for quick browsing
        adata.obs[f"{CLUSTER_KEY}_pretty"] = (
            adata.obs[CLUSTER_KEY].astype(str) + " | " + adata.obs["ct_major_by_leiden"].astype(str)
        ).astype("category")

        # UMAPs (these calls now auto-skip missing keys and won’t touch .raw)
        plot_umap(adata, [CLUSTER_KEY, "ct_label_mv"], "umap_leiden_and_celltypistMV.png")
        plot_umap(adata, ["ct_label", "ct_conf"], "umap_celltypist_and_conf.png")  # will skip ct_conf if missing
        plot_umap(adata, [f"{CLUSTER_KEY}_pretty"], "umap_leiden_pretty.png")

        # Composition per cluster
        plot_stacked_composition(
            adata, cluster_key=CLUSTER_KEY, label_key="ct_label_mv",
            top_k=12, fname="stacked_comp_leiden_vs_celltypistMV.png"
        )
    else:
        print(f"[Viz] Skipping Leiden-based plots: CLUSTER_KEY={CLUSTER_KEY} not present in adata.obs")

    # Save with viz additions
    adata.write(OUT_H5AD_VIZ)
    print(f"[IO] Saved annotated+viz AnnData → {OUT_H5AD_VIZ}")
    print(f"[IO] Figures saved in: {FIGDIR}")

if __name__ == "__main__":
    main()
