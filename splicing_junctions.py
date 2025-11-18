#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Mouse TDP-43 splicing pipeline (ParseBio)

Parts:
 A) Expression audit + Split-Pipe-like UMAPs
 B) PSI: count inclusion/skip junction UMIs per cell across multiple SubLib BAMs
 C) PSI aggregation by cell type (+ quick pairwise z-test)

Assumptions:
 - AnnData has .raw with counts (or ln(1+x)); we'll coerce to log2(1+x) for comparability.
 - Cell types live in obs['ct_label_mv'] (change CELLTYPE_COL if needed).
 - BAMs are STAR-aligned Split-Pipe outputs with XC/XM or CB/UB tags and are indexed (.bai next to .bam).
 - events_mouse.csv provides exon coordinates for Psmd14 Ex5, Ddi2 Ex9, Mrpl45 Ex7 (mouse).
"""

import os, re, csv, math, collections, warnings
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse
import matplotlib.pyplot as plt

# ---------- USER CONFIG (edit paths if needed) ----------
BASE = "/home/ubuntu/volume_750gb/results/tdp_project"

# AnnData (mouse)
ADATA = os.path.join(BASE, "adata_files", "mouse_adata_RAW.h5ad")   # or MAIN_postDoublet if you prefer

# Expression/UMAP outputs
OUT_EXPR = os.path.join(BASE, "splicing_mouse")
os.makedirs(OUT_EXPR, exist_ok=True)

# BAMs: put each SubLib here as bam_files_mouse/<Sample_...>/barcode_headAligned_anno.bam
BAM_ROOT = os.path.join(BASE, "bam_files_mouse")

# PSI working folder
OUT_PSI_DIR = os.path.join(BASE, "psi_mouse")
os.makedirs(os.path.join(OUT_PSI_DIR, "outputs"), exist_ok=True)

# Exon events (MOUSE!) CSV with columns: event_id,gene,chr,strand,up_end,t_start,t_end,down_start
EVENTS_CSV = os.path.join(OUT_PSI_DIR, "events_mouse.csv")

# Cell type column
CELLTYPE_COL = "ct_label_mv"

# Anchors + optional additional genes to audit
GENE_ALIASES = [
    "Mrpl45", "Ddi2", "Psmd14",
    "Hjurp","Cops4","Tiam1","Creld1","Inpp4a","Shank1",
    "Arhgap44","Psme3","Dgkq","Arhgef11","Osbpl6","Pik3cb","Pdzdc1","Dsel",
    "Herc2","Golga4","Stx24","Ube3c","C530008M17Rik","Tub","Atp2b1",
    "Lhx1","Lhx1os","Onecut1","Slc32a1","Gad1os","Gad2","Dlx6","Npas1","Pou6f2",
]

# Positivity threshold for “detected” at per-cell level (recommended ≥2 UMIs)
MIN_UMI = 2
LOG2_CUT = float(np.log2(1 + MIN_UMI))

# PSI coverage filter per event×celltype (UMIs)
MIN_PSI_COV = 30

# Try these tags in order (Split-Pipe usually XC/XM)
CB_TAGS = ["CB", "XC"]
UMI_TAGS = ["UB", "XM"]
XS_TAGS = ["XS"]  # STAR writes XS for strand on spliced alignments


# ========================== UTILITIES ==========================

def _to_dense(x):
    if sparse.issparse(x): return x.toarray()
    toarr = getattr(x, "toarray", None)
    return toarr() if callable(toarr) else np.asarray(x)

def _var_index(A):
    if A.raw is not None:
        return pd.Index(A.raw.var_names).astype(str), True
    return pd.Index(A.var_names).astype(str), False

def map_gene_keys(adata, gene_list):
    var, _ = _var_index(adata)
    vv = pd.Series(var, index=var)
    var_upper = vv.str.upper()
    var_stripped = var_upper.str.replace(r'[-_.]', '', regex=True)

    mapping = {}
    for g in gene_list:
        gu = g.upper()
        gs = re.sub(r'[-_.]', '', gu)

        if g in vv.index: mapping[g] = g; continue
        m = vv[vv.str.match(fr"^{re.escape(g)}(_.*)?$", case=False, na=False)]
        if len(m): mapping[g] = m.iloc[0]; continue
        m = vv[var_upper == gu]
        if len(m): mapping[g] = m.iloc[0]; continue
        m = vv[var_stripped == gs]
        if len(m): mapping[g] = m.iloc[0]; continue
        m = vv[var_upper.str.contains(gu)]
        mapping[g] = m.iloc[0] if len(m) else None
    return mapping

def _get_expr_log2(A, var_key):
    """Return Series of log2(1+x) for a gene; prefer A.raw."""
    if A.raw is not None and var_key in A.raw.var_names:
        X = A.raw[:, var_key].X
    else:
        X = A[:, var_key].X
    vals = _to_dense(X).ravel()
    # Auto-detect transform: if counts, log1p; else assume ln(1+x) and convert to log2
    p99 = float(np.nanpercentile(vals, 99)) if np.isfinite(vals).any() else 0.0
    if p99 > 15:
        vals = np.log1p(vals) / np.log(2)
    else:
        vals = vals / np.log(2)
    return pd.Series(vals, index=A.obs_names, name=var_key)

def ensure_umap(A):
    if "X_umap" in A.obsm_keys(): return
    B = A.copy()
    sc.pp.normalize_total(B, 1e4); sc.pp.log1p(B)
    sc.pp.pca(B); sc.pp.neighbors(B, n_neighbors=15, n_pcs=30); sc.tl.umap(B)
    A.obsm["X_umap"] = B.obsm["X_umap"]

def summarize_series_on_cut(x, cut):
    n = x.size
    pos = x > cut
    def stats(v):
        if v.size == 0: return {"mean": np.nan, "median": np.nan}
        return {"mean": float(np.nanmean(v)), "median": float(np.nanmedian(v))}
    return {
        "n_cells": int(n),
        "n_detected_cells": int(pos.sum()),
        "pct_detected": float(pos.mean() * 100.0),
        **stats(x),
        **{f"{k}_in_detected": v for k, v in stats(x[pos]).items()},
    }

def first_present_tag(aln, tags):
    for t in tags:
        try:
            return aln.get_tag(t)
        except KeyError:
            continue
    return None


# ========================= PART A: Expression audit =========================

def partA_expression_and_umaps():
    A = sc.read_h5ad(ADATA)
    print(f"[A] Loaded {ADATA} | cells={A.n_obs:,} genes={A.n_vars:,}")

    gene_map = map_gene_keys(A, GENE_ALIASES)
    found = {k: v for k, v in gene_map.items() if v}
    missing = [k for k, v in gene_map.items() if v is None]
    print(f"[A] Found {len(found)}/{len(GENE_ALIASES)} genes. Missing: {missing or '—'}")

    # Summaries
    overall_rows, per_group_rows = [], []
    for alias, var_key in found.items():
        x = _get_expr_log2(A, var_key)
        overall = summarize_series_on_cut(x, LOG2_CUT)
        overall_rows.append({"alias": alias, "var_name": var_key, **overall})

        if CELLTYPE_COL in A.obs.columns:
            for ct, idx in A.obs.groupby(CELLTYPE_COL).indices.items():
                xi = x.iloc[idx]
                row = summarize_series_on_cut(xi, LOG2_CUT)
                per_group_rows.append({
                    "alias": alias, "var_name": var_key,
                    "group_key": CELLTYPE_COL, "group": str(ct),
                    **row
                })

        # Per-gene UMAP (zeros grey)
        ensure_umap(A)
        tmp = "_tmp_expr_"
        A.obs[tmp] = x.where(x > 0, np.nan)
        fig = sc.pl.umap(A, color=[tmp], na_color="lightgray", cmap="OrRd",
                         vmin=0, vmax=4, frameon=True, show=False, return_fig=True)
        if not hasattr(fig, "savefig"): fig = plt.gcf()
        fig.axes[0].set_title(alias)
        fig.savefig(os.path.join(OUT_EXPR, f"mouse_UMAP_{alias}.png"), dpi=300, bbox_inches="tight")
        plt.close(fig)
        A.obs.drop(columns=[tmp], inplace=True, errors="ignore")

    pd.DataFrame(overall_rows).sort_values("alias").to_csv(
        os.path.join(OUT_EXPR, "mouse_gene_expression_overall.csv"), index=False
    )
    if per_group_rows:
        pd.DataFrame(per_group_rows).to_csv(
            os.path.join(OUT_EXPR, f"mouse_gene_expression_by_{CELLTYPE_COL}.csv"), index=False
        )
    print("[A] Wrote expression CSVs and per-gene UMAPs to:", OUT_EXPR)

    # Optional: 3-panel UMAP for anchors
    anchors = [g for g in ["Mrpl45", "Ddi2", "Psmd14"] if g in found]
    if anchors:
        cols = [found[g] for g in anchors]
        ensure_umap(A)
        fig = sc.pl.umap(A, color=cols, na_color="lightgray", cmap="OrRd",
                         vmin=0, vmax=4, ncols=min(3, len(cols)), show=False, return_fig=True)
        if not hasattr(fig, "savefig"): fig = plt.gcf()
        fig.savefig(os.path.join(OUT_EXPR, "mouse_UMAP_anchor_genes.png"), dpi=300, bbox_inches="tight")
        plt.close(fig)
        print("[A] Wrote mouse_UMAP_anchor_genes.png")


# ========================= PART B: Junction counting =========================

def discover_bams(root):
    """Return list of (bam_path, sublib_id). Expects bam_files_mouse/Sample_.../barcode_headAligned_anno.bam"""
    hits = []
    for dirpath, dirnames, filenames in os.walk(root):
        for fn in filenames:
            if fn.endswith(".bam"):
                sublib = os.path.basename(dirpath)  # e.g., Sample_0001APSubLib01_01
                hits.append((os.path.join(dirpath, fn), sublib))
    return hits

def load_events(events_csv):
    evs = []
    with open(events_csv) as f:
        r = csv.DictReader(f)
        for row in r:
            # required columns: event_id,gene,chr,strand,up_end,t_start,t_end,down_start
            for k in ("up_end","t_start","t_end","down_start"):
                row[k] = int(row[k])
            if row["strand"] not in ("+","-"):
                raise ValueError(f"Invalid strand for {row.get('event_id')}: {row['strand']}")
            evs.append(row)
    if not evs:
        raise ValueError("No events loaded; check events_mouse.csv")
    return evs

def event_to_junctions(ev):
    # define (chr, donor, acceptor, strand, label)
    chr, strand = ev["chr"], ev["strand"]
    ue, ts, te, ds = ev["up_end"], ev["t_start"], ev["t_end"], ev["down_start"]
    return [
        (chr, ue, ts, strand, "incl_up_target"),
        (chr, te, ds, strand, "incl_target_down"),
        (chr, ue, ds, strand, "skip_up_down"),
    ]

def partB_count_junctions():
    try:
        import pysam
    except Exception as e:
        raise SystemExit(f"[B] ERROR: pysam not available: {e}")

    events = load_events(EVENTS_CSV)
    # reverse index: (chr,donor,acceptor,strand) -> [(event_id,label), ...]
    rev = collections.defaultdict(list)
    for ev in events:
        for chr, d, a, strand, label in event_to_junctions(ev):
            rev[(chr, int(d), int(a), strand)].append((ev["event_id"], label))

    bams = discover_bams(BAM_ROOT)
    if not bams:
        raise SystemExit(f"[B] No BAMs discovered under {BAM_ROOT}")

    print(f"[B] Found {len(bams)} BAMs. Counting junction UMIs per cell...")

    acc = collections.Counter()    # (cell,event_id,label) -> UMI count
    for bam_path, sublib in bams:
        bam = pysam.AlignmentFile(bam_path, "rb")
        umi_seen = set()  # (cell,event,label,UMI)
        n_aln = 0
        for aln in bam.fetch(until_eof=True):
            n_aln += 1
            if aln.is_unmapped or aln.is_secondary or aln.is_supplementary:
                continue

            cb = first_present_tag(aln, CB_TAGS)
            umi = first_present_tag(aln, UMI_TAGS)
            if cb is None or umi is None:
                continue

            # Prefix barcodes with SubLib to avoid collisions across libraries
            cb = f"{sublib}:{cb}"

            # Strand: prefer XS if present; else fall back to flag (approximate)
            xs = first_present_tag(aln, XS_TAGS)
            if xs in ("+","-"):
                strand = xs
            else:
                strand = "-" if aln.is_reverse else "+"

            ref = bam.get_reference_name(aln.reference_id)
            # walk CIGAR; N = splice jumps on reference
            pos = aln.reference_start + 1  # 1-based
            for op, length in (aln.cigartuples or []):
                if op == 3:  # N (splice)
                    donor = pos
                    acceptor = pos + length
                    key = (ref, donor, acceptor, strand)
                    for ev_id, label in rev.get(key, []):
                        sig = (cb, ev_id, label, umi)
                        if sig not in umi_seen:
                            umi_seen.add(sig)
                            acc[(cb, ev_id, label)] += 1
                    pos += length
                elif op in (0,7,8,2):  # M,=,X,D advance reference
                    pos += length
                # I,S,H do not advance reference
        bam.close()
        print(f"[B] {os.path.basename(bam_path)} | processed alignments: {n_aln:,}")

    out_counts = os.path.join(OUT_PSI_DIR, "outputs", "junction_counts_per_cell_mouse.csv")
    with open(out_counts, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["cell","event_id","label","count"])
        for (cell, ev_id, label), c in acc.items():
            w.writerow([cell, ev_id, label, c])
    print("[B] Wrote:", out_counts)


# ========================= PART C: PSI aggregation =========================

def partC_aggregate_psi():
    from statsmodels.stats.proportion import proportions_ztest

    counts_csv = os.path.join(OUT_PSI_DIR, "outputs", "junction_counts_per_cell_mouse.csv")
    if not os.path.exists(counts_csv):
        raise SystemExit("[C] Missing junction counts — run Part B first.")

    A = sc.read_h5ad(ADATA)
    if CELLTYPE_COL not in A.obs.columns:
        raise SystemExit(f"[C] Missing obs['{CELLTYPE_COL}'] in AnnData.")
    meta = A.obs[[CELLTYPE_COL]].copy()
    meta.index.name = "cell"

    df = pd.read_csv(counts_csv)
    # merge cell types; cells not in AnnData drop out
    df = df.merge(meta, left_on="cell", right_index=True, how="inner")
    if df.empty:
        raise SystemExit("[C] After merging with AnnData, no cells remain. Check barcode prefixes match AnnData.")

    pivot = df.pivot_table(index=[CELLTYPE_COL,"event_id","label"], values="count", aggfunc="sum").reset_index()

    def summarize(group):
        incl = group.loc[group["label"].isin(["incl_up_target","incl_target_down"]), "count"].sum()
        skip = group.loc[group["label"]=="skip_up_down", "count"].sum()
        cov = incl + skip
        psi = (incl / cov) if cov > 0 else np.nan
        return pd.Series({"incl": int(incl), "skip": int(skip), "coverage": int(cov), "PSI": psi})

    PSI = pivot.groupby([CELLTYPE_COL,"event_id"]).apply(summarize).reset_index()
    PSI = PSI[PSI["coverage"] >= MIN_PSI_COV].copy()

    out_psi = os.path.join(OUT_PSI_DIR, "outputs", "PSI_by_celltype_mouse.csv")
    PSI.to_csv(out_psi, index=False)
    print("[C] Wrote:", out_psi)

    # Quick pairwise test between two highest-coverage cell types per event
    rows = []
    for ev, sub in PSI.groupby("event_id"):
        sub = sub.sort_values("coverage", ascending=False)
        if len(sub) < 2: continue
        a, b = sub.iloc[0], sub.iloc[1]
        count = np.array([a["incl"], b["incl"]], dtype=float)
        nobs  = np.array([a["coverage"], b["coverage"]], dtype=float)
        try:
            stat, p = proportions_ztest(count, nobs)
        except Exception:
            stat, p = (np.nan, np.nan)
        rows.append(dict(
            event_id=ev,
            ct1=str(a[CELLTYPE_COL]), ct2=str(b[CELLTYPE_COL]),
            PSI1=float(a["PSI"]), PSI2=float(b["PSI"]),
            dPSI=float(a["PSI"]-b["PSI"]),
            cov1=int(a["coverage"]), cov2=int(b["coverage"]),
            z=float(stat) if not math.isnan(stat) else np.nan,
            pvalue=float(p) if not math.isnan(p) else np.nan
        ))
    if rows:
        out_pair = out_psi.replace(".csv", "_pairwise_top2.csv")
        pd.DataFrame(rows).to_csv(out_pair, index=False)
        print("[C] Wrote:", out_pair)


# ========================= MAIN =========================

def main():
    print("=== PART A: Expression audit + UMAPs ===")
    partA_expression_and_umaps()
    print("=== PART B: PSI junction counting (BAMs) ===")
    partB_count_junctions()
    print("=== PART C: PSI aggregation by cell type ===")
    partC_aggregate_psi()
    print("All done.")

if __name__ == "__main__":
    main()
