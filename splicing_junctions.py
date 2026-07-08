#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Mouse TDP-43 splicing pipeline (ParseBio)

Parts:
 A) Expression audit + Split-Pipe-like UMAPs
 B) PSI: count inclusion/skip junction UMIs per cell across multiple SubLib BAMs
 C) PSI aggregation by cell type (+ quick pairwise z-test), for multiple
    coverage thresholds (e.g. >=30 and >=10).

Key design points (this version):
 - We can run multiple AnnData configurations in one go (e.g. RAW + MAIN).
 - BAM-based counting is **independent** of which AnnData is used; per-BAM
   caches are shared across configs.
 - Cell IDs used for junction counts are constructed as:

      <bc_wells>__sX   (optionally with "Sample_...:" prefix if desired)

   where X is inferred directly from the SubLib number in the BAM path:
   SubLib01 -> __s1, SubLib02 -> __s2, ... SubLib08 -> __s8

   This matches Split-Pipe combine-mode output like "01_01_01__s1" etc.
"""

import os, re, csv, math, collections, hashlib, warnings, shutil, subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse
import matplotlib.pyplot as plt
import pysam

# ---------- GLOBAL PROJECT CONFIG ----------
BASE = "/home/ubuntu/volume_750gb/results/tdp_project"

# Where the BAMs and event definitions live (shared for all runs)
BAM_ROOT   = os.path.join(BASE, "bam_files_mouse")
EVENTS_CSV = os.path.join(BASE, "reference_files", "events_mouse.csv")

CELLTYPE_COL = "ct_label_mv"

# Whether to prefix cell IDs with the SubLib sample name (e.g. "Sample_...:")
USE_SUBLIB_PREFIX = False

# Global per-BAM cache root (shared for RAW + MAIN runs)
CACHE_ROOT = os.path.join(BASE, "mouse", "per_bam_cache_global")

# Cache/version bump
CODE_VERSION = "C.0"

# Anchors + optional additional genes to audit
# GENE_ALIASES = [
#     "Mrpl45", "Ddi2", "Psmd14",
#     "Hjurp","Cops4","Tiam1","Creld1","Inpp4a","Shank1",
#     "Arhgap44","Psme3","Dgkq","Arhgef11","Osbpl6","Pik3cb","Pdzdc1","Dsel",
#     "Herc2","Golga4","Stx24","Ube3c","C530008M17Rik","Tub","Atp2b1",
#     "Lhx1","Lhx1os","Onecut1","Slc32a1","Gad1os","Gad2","Dlx6","Npas1","Pou6f2",
# ]

GENE_ALIASES = [
    "Psmd14","Ddi2","Mrpl45", "Ap3b2", "Camk1g", "Adnp2", "Bud23","Poldip3", "Sort1",
    "Hjurp","Cops4","Tiam1","Creld1","Inpp4a","Shank1",
    "Arhgap44","Psme3","Dgkq","Arhgef11","Osbpl6","Pik3cb","Pdzd4","Dsel",
    "Herc2","Golga4","Stk24","Ube3c","Cracd","Tub","Atp2b1",
    "Lhx1","Lhx1os","Onecut1","Slc32a1","Gad1os","Gad2","Dlx6","Npas1","Pou6f2",
]

# Expression thresholds (gene expression)
MIN_UMI  = 2
LOG2_CUT = float(np.log2(1 + MIN_UMI))

# PSI coverage thresholds: we'll produce outputs for each of these
PSI_COVERAGE_THRESHOLDS = [30, 10]

# Split-Pipe/STAR tags
CB_TAGS  = ["CB", "XC"]           # cell barcode tags
UMI_TAGS = ["UB", "XM", "pN"]     # UMI tags (10x + ParseBio)
XS_TAGS  = ["XS"]                 # STAR XS strand tag on spliced reads

# ---------- PER-RUN CONFIGS (RAW / MAIN) ----------
RUN_CONFIGS = [
    # dict(
    #     name    = "RAW",
    #     enabled = True,
    #     adata   = os.path.join(BASE, "adata_files", "mouse_adata_RAW.h5ad"),
    #     out_expr= os.path.join(BASE, "mouse", "splicing_mouse_raw"),
    # ),
    dict(
        name    = "MAIN_postDoublet",
        enabled = True,
        adata   = os.path.join(
            BASE, "adata_files", "mouse_adata_MAIN_postDoublet_celltypist+viz.h5ad"
        ),
        out_expr= os.path.join(
            BASE, "mouse", "splicing_mouse_main_postDoublet_celltypist_viz"
        ),
    ),
]

# ---------- GLOBALS THAT CHANGE PER RUN ----------
ADATA_PATH: str  = None
OUT_EXPR: str    = None
OUT_PSI_DIR: str = None
CACHE_DIR: str   = None

# ========================== UTILITIES ==========================

def set_run_context(cfg: dict):
    """
    Set global paths for a specific run (RAW / MAIN etc.).
    """
    global ADATA_PATH, OUT_EXPR, OUT_PSI_DIR, CACHE_DIR

    ADATA_PATH = cfg["adata"]
    OUT_EXPR   = cfg["out_expr"]
    OUT_PSI_DIR = OUT_EXPR  # counts + debug; PSI results go into subfolders

    os.makedirs(OUT_EXPR, exist_ok=True)

    # per-BAM cache shared across runs (so BAMs only scanned once)
    CACHE_DIR = os.path.join(CACHE_ROOT, f"cache_{CODE_VERSION}")
    os.makedirs(CACHE_DIR, exist_ok=True)

    print(f"[CTX] Run '{cfg['name']}'")
    print(f"[CTX]   ADATA_PATH = {ADATA_PATH}")
    print(f"[CTX]   OUT_EXPR   = {OUT_EXPR}")
    print(f"[CTX]   OUT_PSI_DIR= {OUT_PSI_DIR}")
    print(f"[CTX]   CACHE_DIR  = {CACHE_DIR}")


def _to_dense(x):
    if sparse.issparse(x):
        return x.toarray()
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
    var_stripped = var_upper.str.replace(r"[-_.]", "", regex=True)
    mapping = {}
    for g in gene_list:
        gu = g.upper()
        gs = re.sub(r'[-_.]', '', gu)
        if g in vv.index:
            mapping[g] = g
            continue
        m = vv[vv.str.match(fr"^{re.escape(g)}(_.*)?$", case=False, na=False)]
        if len(m):
            mapping[g] = m.iloc[0]
            continue
        m = vv[var_upper == gu]
        if len(m):
            mapping[g] = m.iloc[0]
            continue
        m = vv[var_stripped == gs]
        if len(m):
            mapping[g] = m.iloc[0]
            continue
        m = vv[var_upper.str.contains(gu)]
        mapping[g] = m.iloc[0] if len(m) else None
    return mapping


def _get_expr_log2(A, var_key):
    if A.raw is not None and var_key in A.raw.var_names:
        X = A.raw[:, var_key].X
    else:
        X = A[:, var_key].X
    vals = _to_dense(X).ravel()
    p99 = float(np.nanpercentile(vals, 99)) if np.isfinite(vals).any() else 0.0
    if p99 > 15:
        vals = np.log1p(vals) / np.log(2)
    else:
        vals = vals / np.log(2)
    return pd.Series(vals, index=A.obs_names, name=var_key)


def ensure_umap(A):
    if "X_umap" in A.obsm_keys():
        return
    B = A.copy()
    sc.pp.normalize_total(B, target_sum=1e4)
    sc.pp.log1p(B)
    sc.pp.pca(B)
    sc.pp.neighbors(B, n_neighbors=15, n_pcs=30)
    sc.tl.umap(B)
    A.obsm["X_umap"] = B.obsm["X_umap"]


def summarize_series_on_cut(x, cut):
    n = x.size
    pos = x > cut

    def stats(v):
        if v.size == 0:
            return {"mean": np.nan, "median": np.nan}
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


def _bam_is_coord_sorted(bam_path: str) -> bool:
    try:
        with pysam.AlignmentFile(bam_path, "rb") as bam:
            so = (bam.header.get("HD") or {}).get("SO", "").lower()
        return so in ("coordinate", "sorted")
    except Exception:
        return False


def _sha1_file(path, block=1 << 20):
    h = hashlib.sha1()
    with open(path, "rb") as f:
        while True:
            b = f.read(block)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _events_sig():
    return _sha1_file(EVENTS_CSV)


def _file_sig(path):
    st = os.stat(path)
    return f"{st.st_size}-{int(st.st_mtime)}"


def _bam_cache_path(bam_path, sublib, events_sig):
    sig = _file_sig(bam_path)
    base = os.path.basename(bam_path).replace(".bam", "")
    fn = f"{CODE_VERSION}.{sublib}.{base}.{sig}.{events_sig}.counts.csv"
    return os.path.join(CACHE_DIR, fn)


def _known_samtools_paths():
    cands = []
    p = shutil.which("samtools")
    if p:
        cands.append(p)
    for cand in [
        "/home/ubuntu/micromamba/envs/single_cell/bin/samtools",
        "/home/ubuntu/micromamba/bin/samtools",
        "/usr/bin/samtools",
        "/usr/local/bin/samtools",
    ]:
        if os.path.isfile(cand) and os.access(cand, os.X_OK):
            cands.append(cand)
    seen, out = set(), []
    for c in cands:
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out


def _try_external_samtools_index(bam_path: str, threads: int = 8) -> bool:
    for exe in _known_samtools_paths():
        try:
            subprocess.run(
                [exe, "index", "-@", str(threads), bam_path],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
            )
            print(f"[B] Indexed via samtools: {exe} :: {os.path.basename(bam_path)}")
            return True
        except subprocess.CalledProcessError as e:
            msg = e.stderr.decode("utf-8", "ignore") if e.stderr else str(e)
            print(f"[B] samtools index failed with {exe}: {msg.strip()}")
    activate = [
        'if [ -f "$HOME/micromamba/etc/profile.d/micromamba.sh" ]; then source "$HOME/micromamba/etc/profile.d/micromamba.sh"; micromamba activate single_cell 2>/dev/null || true; fi',
        'if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then source "$HOME/miniconda3/etc/profile.d/conda.sh"; conda activate single_cell 2>/dev/null || true; fi',
        'source "$HOME/.bashrc" 2>/dev/null || true',
    ]
    bash_cmd = (
        "set -euo pipefail; "
        + " ; ".join(activate)
        + f'; command -v samtools >/dev/null && samtools index -@ {threads} "{bam_path}"'
    )
    try:
        subprocess.run(
            ["bash", "-lc", bash_cmd],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        print(f"[B] Indexed via shell-activated samtools: {os.path.basename(bam_path)}")
        return True
    except subprocess.CalledProcessError as e:
        msg = e.stderr.decode("utf-8", "ignore") if e.stderr else str(e)
        print(
            f"[B] samtools index via shell activation failed; continuing without index: {msg.strip()}"
        )
        return False


def ensure_bam_index(bam_path: str, threads: int = 8) -> None:
    bai, csi = bam_path + ".bai", bam_path + ".csi"
    if os.path.exists(bai) or os.path.exists(csi):
        return
    if not os.access(os.path.dirname(bam_path) or ".", os.W_OK):
        print(f"[B] Skip indexing (no write permission): {bam_path}")
        return
    if not _bam_is_coord_sorted(bam_path):
        print(f"[B] Skip indexing (BAM not coordinate-sorted): {bam_path}")
        return
    try:
        print(f"[B] Indexing via pysam: {os.path.basename(bam_path)}")
        pysam.index(bam_path)
    except Exception as e:
        print(f"[B] pysam.index failed ({e}); trying external samtools...")
        _ = _try_external_samtools_index(bam_path, threads=threads)


def norm_ref_name(s: str) -> str:
    """
    Normalize ref/chrom names to a compact form:
      "grcm39_4" / "chr4" / "4" -> "4"
      "chrX" / "X" / "grcm39_x" -> "X"
      mitochondrion: "MT","M","chrM","chrMT" -> "MT"
    """
    if not s:
        return s
    t = s.strip()
    t = re.sub(r"grcm39[_\-]?", "", t, flags=re.IGNORECASE)
    t = re.sub(r"mm39[_\-]?", "", t, flags=re.IGNORECASE)
    t = re.sub(r"^chr", "", t, flags=re.IGNORECASE)
    t = t.upper()
    if t in ("M", "CHRMT", "CHRM", "MTT", "MTDNA"):
        t = "MT"
    return t


# ========================= PART A =========================
def partA_expression_and_umaps():
    A = sc.read_h5ad(ADATA_PATH)
    print(f"[A] Loaded {ADATA_PATH} | cells={A.n_obs:,} genes={A.n_vars:,}")

    # ----- 1) Alias genes: expression + per-CT + UMAPs -----
    gene_map = map_gene_keys(A, GENE_ALIASES)
    found = {k: v for k, v in gene_map.items() if v}
    missing = [k for k, v in gene_map.items() if v is None]
    print(f"[A] Found {len(found)}/{len(GENE_ALIASES)} genes. Missing: {missing or '—'}")

    overall_rows, per_group_rows = [], []
    ensure_umap(A)

    for alias, var_key in found.items():
        x = _get_expr_log2(A, var_key)
        overall = summarize_series_on_cut(x, LOG2_CUT)
        overall_rows.append({"alias": alias, "var_name": var_key, **overall})

        if CELLTYPE_COL in A.obs.columns:
            for ct, idx in A.obs.groupby(CELLTYPE_COL, observed=False).indices.items():
                xi = x.iloc[idx]
                row = summarize_series_on_cut(xi, LOG2_CUT)
                per_group_rows.append(
                    {
                        "alias": alias,
                        "var_name": var_key,
                        "group_key": CELLTYPE_COL,
                        "group": str(ct),
                        **row,
                    }
                )

        # per-gene UMAP (only for alias genes)
        tmp = "_tmp_expr_"
        A.obs[tmp] = x.where(x > 0, np.nan)
        fig = sc.pl.umap(
            A,
            color=[tmp],
            na_color="lightgray",
            cmap="OrRd",
            vmin=0,
            vmax=4,
            frameon=True,
            show=False,
            return_fig=True,
        )
        if not hasattr(fig, "savefig"):
            fig = plt.gcf()
        fig.axes[0].set_title(alias)
        fig.savefig(
            os.path.join(OUT_EXPR, f"mouse_UMAP_{alias}.png"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close(fig)
        A.obs.drop(columns=[tmp], inplace=True, errors="ignore")

    # Alias-only summary (what you already had)
    pd.DataFrame(overall_rows).sort_values("alias").to_csv(
        os.path.join(OUT_EXPR, "mouse_gene_expression_overall.csv"), index=False
    )
    if per_group_rows:
        pd.DataFrame(per_group_rows).to_csv(
            os.path.join(OUT_EXPR, f"mouse_gene_expression_by_{CELLTYPE_COL}.csv"),
            index=False,
        )
    print("[A] Wrote expression CSVs and per-gene UMAPs to:", OUT_EXPR)

    # Anchor UMAPs (unchanged)
    anchors = [g for g in ["Mrpl45", "Ddi2", "Psmd14"] if g in found]
    if anchors:
        cols = [found[g] for g in anchors]
        fig = sc.pl.umap(
            A,
            color=cols,
            na_color="lightgray",
            cmap="OrRd",
            vmin=0,
            vmax=4,
            ncols=min(3, len(cols)),
            show=False,
            return_fig=True,
        )
        if not hasattr(fig, "savefig"):
            fig = plt.gcf()
        fig.savefig(
            os.path.join(OUT_EXPR, "mouse_UMAP_anchor_genes.png"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close(fig)
        print("[A] Wrote mouse_UMAP_anchor_genes.png")

    # # ----- 2) NEW: summary for *all* genes -----
    # # Use the same log2 expression + LOG2_CUT threshold,
    # # but no aliases / no per-CT / no UMAPs.
    # print("[A] Computing overall expression summary for ALL genes...")
    # var_all, _ = _var_index(A)  # raw.var_names if present, else var_names
    # all_rows = []
    # for var_key in var_all:
    #     x = _get_expr_log2(A, var_key)
    #     overall = summarize_series_on_cut(x, LOG2_CUT)
    #     all_rows.append({"var_name": var_key, **overall})
    #
    # df_all = pd.DataFrame(all_rows)
    # df_all.to_csv(
    #     os.path.join(OUT_EXPR, "mouse_gene_expression_overall_all_genes.csv"),
    #     index=False,
    # )
    # print(
    #     "[A] Wrote overall expression summary for all genes to "
    #     "mouse_gene_expression_overall_all_genes.csv"
    # )



# ========================= PART B =========================
def discover_bams(root):
    hits = []
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            if fn.endswith(".bam"):
                sublib = os.path.basename(dirpath)  # e.g., Sample_0001APSubLib01_01
                hits.append((os.path.join(dirpath, fn), sublib))
    return hits


def sublib_to_suffix(sublib: str) -> str:
    m = re.search(r"SubLib(\d+)", sublib)
    if m:
        idx = int(m.group(1))
        return f"__s{idx}"
    warnings.warn(
        f"[B] Could not parse SubLib index from '{sublib}', defaulting to '__s1'"
    )
    return "__s1"


def load_events(events_csv):
    evs = []
    with open(events_csv) as f:
        r = csv.DictReader(f)
        for row in r:
            for k in ("up_end", "t_start", "t_end", "down_start"):
                row[k] = int(row[k])
            if row["strand"] not in ("+", "-"):
                raise ValueError(
                    f"Invalid strand for {row.get('event_id')}: {row['strand']}"
                )
            row["chr_norm"] = norm_ref_name(row["chr"])
            evs.append(row)
    if not evs:
        raise ValueError("No events loaded; check events_mouse.csv")
    return evs


def event_to_junctions(ev):
    chrn, strand = ev["chr_norm"], ev["strand"]
    ue, ts, te, ds = ev["up_end"], ev["t_start"], ev["t_end"], ev["down_start"]
    return [
        (chrn, ue, ts, strand, "incl_up_target"),
        (chrn, te, ds, strand, "incl_target_down"),
        (chrn, ue, ds, strand, "skip_up_down"),
    ]


def partB_count_junctions():
    events = load_events(EVENTS_CSV)
    rev = collections.defaultdict(list)
    for ev in events:
        for chrn, d, a, strand, label in event_to_junctions(ev):
            rev[(chrn, int(d), int(a), strand)].append((ev["event_id"], label))

    bams = discover_bams(BAM_ROOT)
    if not bams:
        raise SystemExit(f"[B] No BAMs discovered under {BAM_ROOT}")

    evsig = _events_sig()
    print(f"[B] Found {len(bams)} BAMs. Counting junction UMIs per cell (with cache)...")

    per_bam_outs = []

    for bam_path, sublib in bams:
        suffix = sublib_to_suffix(sublib)
        cache_csv = _bam_cache_path(bam_path, sublib, evsig)
        if os.path.exists(cache_csv):
            print(f"[B] Using cache: {os.path.basename(cache_csv)} (SubLib {sublib}, suffix {suffix})")
            per_bam_outs.append(cache_csv)
            continue

        ensure_bam_index(bam_path)
        bam = pysam.AlignmentFile(bam_path, "rb")

        umi_seen = set()
        acc = collections.Counter()
        n_aln = 0
        n_splice_ops = 0

        obs_junc = collections.Counter()

        for aln in bam.fetch(until_eof=True):
            n_aln += 1
            if aln.is_unmapped or aln.is_secondary or aln.is_supplementary:
                continue

            vbc = first_present_tag(aln, CB_TAGS)
            umi = first_present_tag(aln, UMI_TAGS)
            if vbc is None or umi is None:
                continue

            cell_id = f"{vbc}{suffix}"
            if USE_SUBLIB_PREFIX:
                cell_id = f"{sublib}:{cell_id}"

            xs = first_present_tag(aln, XS_TAGS)
            strand = xs if xs in ("+", "-") else ("-" if aln.is_reverse else "+")

            ref_raw = bam.get_reference_name(aln.reference_id)
            ref = norm_ref_name(ref_raw)

            pos = aln.reference_start + 1
            for op, length in (aln.cigartuples or []):
                if op == 3:  # N (splice)
                    n_splice_ops += 1
                    donor = pos - 1
                    acceptor = pos + length
                    key = (ref, donor, acceptor, strand)

                    if len(obs_junc) < 10000:
                        obs_junc[key] += 1

                    hits = rev.get(key)
                    if hits:
                        for ev_id, label in hits:
                            sig = (cell_id, ev_id, label, umi)
                            if sig not in umi_seen:
                                umi_seen.add(sig)
                                acc[(cell_id, ev_id, label)] += 1
                    pos += length
                elif op in (0, 7, 8, 2):
                    pos += length

        bam.close()
        print(
            f"[B] {os.path.basename(bam_path)} (SubLib {sublib}, suffix {suffix}) | "
            f"processed alignments: {n_aln:,} | spliced_ops: {n_splice_ops:,}"
        )

        with open(cache_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["cell", "event_id", "label", "count"])
            for (cell, ev_id, label), c in acc.items():
                w.writerow([cell, ev_id, label, c])

        if not acc and obs_junc:
            dbg = os.path.join(OUT_PSI_DIR, "debug_observed_junction_keys.tsv")
            if not os.path.exists(dbg):
                with open(dbg, "w") as g:
                    g.write("chr\tdonor\tacceptor\tstrand\tseen\n")
                    for (r, d, a, s), n in list(obs_junc.items())[:5000]:
                        g.write(f"{r}\t{d}\t{a}\t{s}\t{n}\n")
                print(
                    f"[B][WARN] No event matches in {os.path.basename(bam_path)}. "
                    f"Wrote observed junction sample to {dbg}"
                )

        per_bam_outs.append(cache_csv)

    out_counts = os.path.join(OUT_PSI_DIR, "junction_counts_per_cell_mouse.csv")
    with open(out_counts, "w", newline="") as out:
        w = csv.writer(out)
        w.writerow(["cell", "event_id", "label", "count"])
        for part in per_bam_outs:
            with open(part) as f:
                r = csv.reader(f)
                _ = next(r, None)
                for row in r:
                    w.writerow(row)
    print("[B] Wrote:", out_counts)

    try:
        sz = os.path.getsize(out_counts)
        if sz <= 25:
            print(
                "[B][WARN] Counts file has header only (no junctions). "
                "If this persists, check events vs BAM naming/coords."
            )
    except Exception:
        pass


# ========================= PART C =========================
def partC_aggregate_psi():
    """
    Aggregate junction counts to PSI per cell type, for multiple coverage
    thresholds defined in PSI_COVERAGE_THRESHOLDS.
    """
    from statsmodels.stats.proportion import proportions_ztest

    counts_csv = os.path.join(OUT_PSI_DIR, "junction_counts_per_cell_mouse.csv")
    if not os.path.exists(counts_csv):
        raise SystemExit("[C] Missing junction counts — run Part B first.")

    A = sc.read_h5ad(ADATA_PATH)
    if CELLTYPE_COL not in A.obs.columns:
        raise SystemExit(f"[C] Missing obs['{CELLTYPE_COL}'] in AnnData.")
    meta = A.obs[[CELLTYPE_COL]].copy()
    meta.index.name = "cell"

    df = pd.read_csv(counts_csv)
    if df.empty or len(df) == 0:
        raise SystemExit("[C] Counts file has no rows. See Part B warnings.")

    df = df.merge(meta, left_on="cell", right_index=True, how="inner")
    if df.empty:
        print(
            "[C] After merging with AnnData, no cells remain. "
            "Example obs_names head:\n",
            meta.index.to_series().head(10).to_string(index=False),
        )
        uniq_cells = pd.read_csv(counts_csv).cell.dropna().unique()[:10]
        print(
            "Example counted cell ids head:\n",
            "\n".join(map(str, uniq_cells)),
        )
        raise SystemExit(
            "[C] No overlapping cell IDs. Check SubLib→__sX mapping or USE_SUBLIB_PREFIX."
        )

    pivot = df.pivot_table(
        index=[CELLTYPE_COL, "event_id", "label"], values="count", aggfunc="sum"
    ).reset_index()

    def summarize(group):
        incl = group.loc[
            group["label"].isin(["incl_up_target", "incl_target_down"]), "count"
        ].sum()
        skip = group.loc[group["label"] == "skip_up_down", "count"].sum()
        cov = incl + skip
        psi = (incl / cov) if cov > 0 else np.nan
        return pd.Series(
            {"incl": int(incl), "skip": int(skip), "coverage": int(cov), "PSI": psi}
        )

    # Unfiltered PSI (per cell type, event)
    PSI_full = pivot.groupby([CELLTYPE_COL, "event_id"]).apply(summarize).reset_index()

    # For each coverage threshold, write separate PSI + pairwise files
    for cov_thr in PSI_COVERAGE_THRESHOLDS:
        psi_dir = os.path.join(OUT_EXPR, f"psi_cov{cov_thr}")
        os.makedirs(psi_dir, exist_ok=True)

        PSI = PSI_full[PSI_full["coverage"] >= cov_thr].copy()

        out_psi = os.path.join(psi_dir, "PSI_by_celltype_mouse.csv")
        PSI.to_csv(out_psi, index=False)
        print(f"[C] Wrote PSI (coverage >= {cov_thr}) to:", out_psi)

        # Quick pairwise z-test between top-coverage two cell types per event
        rows = []
        for ev, sub in PSI.groupby("event_id"):
            sub = sub.sort_values("coverage", ascending=False)
            if len(sub) < 2:
                continue
            a, b = sub.iloc[0], sub.iloc[1]
            count = np.array([a["incl"], b["incl"]], dtype=float)
            nobs = np.array([a["coverage"], b["coverage"]], dtype=float)
            try:
                stat, p = proportions_ztest(count, nobs)
            except Exception:
                stat, p = (np.nan, np.nan)
            rows.append(
                dict(
                    event_id=ev,
                    ct1=str(a[CELLTYPE_COL]),
                    ct2=str(b[CELLTYPE_COL]),
                    PSI1=float(a["PSI"]),
                    PSI2=float(b["PSI"]),
                    dPSI=float(a["PSI"] - b["PSI"]),
                    cov1=int(a["coverage"]),
                    cov2=int(b["coverage"]),
                    z=float(stat) if not math.isnan(stat) else np.nan,
                    pvalue=float(p) if not math.isnan(p) else np.nan,
                )
            )

        if rows:
            out_pair = os.path.join(psi_dir, "PSI_by_celltype_mouse_pairwise_top2.csv")
            pd.DataFrame(rows).to_csv(out_pair, index=False)
            print(f"[C] Wrote pairwise z-tests (coverage >= {cov_thr}) to:", out_pair)


# ========================= MAIN =========================
def main():
    for cfg in RUN_CONFIGS:
        if not cfg.get("enabled", True):
            continue

        print("\n" + "=" * 80)
        set_run_context(cfg)
        print("=== PART A: Expression audit + UMAPs ===")
        partA_expression_and_umaps()
        # return
        print("=== PART B: PSI junction counting (BAMs) ===")
        partB_count_junctions()
        print("=== PART C: PSI aggregation by cell type ===")
        partC_aggregate_psi()
        print(f"[DONE] Run '{cfg['name']}' finished.")
    print("\nAll runs done.")


if __name__ == "__main__":
    main()
