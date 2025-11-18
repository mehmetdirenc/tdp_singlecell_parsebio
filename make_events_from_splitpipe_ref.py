#!/usr/bin/env python3
import os, csv, re, sys
import pandas as pd

# --------- CONFIG: set this to your Split-Pipe reference directory ----------
REF_DIR = "/mnt/storage2/users/ahmungm1/genomes/spipe_mus_musculus_grcm39_ref"

GENE_LIST = [
    "Psmd14","Ddi2","Mrpl45",
    "Hjurp","Cops4","Tiam1","Creld1","Inpp4a","Shank1",
    "Arhgap44","Psme3","Dgkq","Arhgef11","Osbpl6","Pik3cb","Pdzdc1","Dsel",
    "Herc2","Golga4","Stx24","Ube3c","C530008M17Rik","Tub","Atp2b1",
    "Lhx1","Lhx1os","Onecut1","Slc32a1","Gad1os","Gad2","Dlx6","Npas1","Pou6f2",
]

OUT_CSV = "/home/ubuntu/volume_750gb/results/tdp_project/psi_mouse/events_mouse.csv"

# --------- load Split-Pipe tables ----------
# Expected columns (typical Split-Pipe/STAR prep; slight variants are OK):
# geneInfo.tab:         gene_id, gene_name, gene_biotype (sometimes), chr, strand, start, end
# transcriptInfo.tab:   transcript_id, gene_id, length (and sometimes exon_count)
# exonInfo.tab:         exon_id, chr, start, end, strand
# exonGeTrInfo.tab:     exon_id, transcript_id, gene_id, exon_number (or order)

def read_tab(path):
    # auto header if present; otherwise assign generic names and rename later
    df = pd.read_csv(path, sep="\t", header=None, comment="#", dtype=str)
    return df

# Try to detect schema heuristically
def load_gene_info(path):
    df = read_tab(path)
    # heuristics for columns
    if df.shape[1] >= 7:
        df = df.iloc[:, :7]
        df.columns = ["gene_id","gene_name","biotype","chr","strand","start","end"]
    elif df.shape[1] >= 5:
        df = df.iloc[:, :5]
        df.columns = ["gene_id","gene_name","chr","strand","dummy"]
        df["biotype"] = ""
        df["start"] = pd.NA
        df["end"] = pd.NA
    else:
        df.columns = [f"c{i}" for i in range(df.shape[1])]
        df["gene_name"] = df.iloc[:,0]
        df["gene_id"] = df.iloc[:,0]
        df["biotype"] = ""
    return df

def load_transcript_info(path):
    df = read_tab(path)
    # common: transcript_id, gene_id, length, exon_count?
    if df.shape[1] >= 3:
        df = df.iloc[:, :3]
        df.columns = ["transcript_id","gene_id","length"]
    else:
        df.columns = ["transcript_id","gene_id"]
        df["length"] = "0"
    df["length"] = pd.to_numeric(df["length"], errors="coerce").fillna(0).astype(int)
    return df

def load_exon_info(path):
    df = read_tab(path)
    # common: exon_id, chr, start, end, strand
    if df.shape[1] >= 5:
        df = df.iloc[:, :5]
        df.columns = ["exon_id","chr","start","end","strand"]
    else:
        raise ValueError("exonInfo.tab has unexpected format")
    df["start"] = pd.to_numeric(df["start"], errors="coerce").astype(int)
    df["end"]   = pd.to_numeric(df["end"], errors="coerce").astype(int)
    return df

def load_exon_getr(path):
    df = read_tab(path)
    # common: exon_id, transcript_id, gene_id, exon_number
    if df.shape[1] >= 4:
        df = df.iloc[:, :4]
        df.columns = ["exon_id","transcript_id","gene_id","exon_number"]
    else:
        df.columns = ["exon_id","transcript_id","gene_id"]
        df["exon_number"] = pd.NA
    # try to coerce exon_number to int; fall back to order-by-genomic position
    df["exon_number"] = pd.to_numeric(df["exon_number"], errors="coerce")
    return df

def main():
    geneInfo_path = os.path.join(REF_DIR, "geneInfo.tab")
    transcriptInfo_path = os.path.join(REF_DIR, "transcriptInfo.tab")
    exonInfo_path = os.path.join(REF_DIR, "exonInfo.tab")
    exonGeTrInfo_path = os.path.join(REF_DIR, "exonGeTrInfo.tab")

    g = load_gene_info(geneInfo_path)
    t = load_transcript_info(transcriptInfo_path)
    e = load_exon_info(exonInfo_path)
    x = load_exon_getr(exonGeTrInfo_path)

    # keep only genes in GENE_LIST (by name)
    g_keep = g[g["gene_name"].isin(GENE_LIST)].copy()
    if g_keep.empty:
        print("[make_events] WARNING: none of the target gene names found in geneInfo.tab; check naming.")
    gid_by_name = dict(zip(g_keep["gene_name"], g_keep["gene_id"]))
    name_by_gid = {v:k for k,v in gid_by_name.items()}

    # join transcript->gene and keep transcripts of target genes
    t2 = t[t["gene_id"].isin(gid_by_name.values())].copy()

    # build exon table with coordinates + mapping to transcript/gene
    xt = x.merge(e, on="exon_id", how="left") \
          .merge(t2[["transcript_id","gene_id","length"]], on="transcript_id", how="inner") \
          .merge(g[["gene_id","gene_name"]], on="gene_id", how="left")

    # choose canonical transcript per gene: prefer largest exon count, then longest length
    can_tx = {}
    for gene_id, sub in xt.groupby("gene_id"):
        # exon count per transcript
        cnt = sub.groupby("transcript_id")["exon_id"].nunique().reset_index(name="n_exons")
        cnt = cnt.merge(t[["transcript_id","length"]], on="transcript_id", how="left")
        cnt["length"] = cnt["length"].fillna(0).astype(int)
        # (no biotype column assured here—gene_info.json could refine if needed)
        # pick max n_exons, then max length
        cnt = cnt.sort_values(["n_exons","length"], ascending=[False, False])
        can_tx[gene_id] = cnt.iloc[0]["transcript_id"]

    rows = []
    for gene_id, tx_id in can_tx.items():
        gene_name = name_by_gid.get(gene_id, gene_id)
        sub = xt[(xt["gene_id"]==gene_id) & (xt["transcript_id"]==tx_id)].copy()
        if sub.empty:
            continue
        # order exons by genomic coordinate (left->right), regardless of strand
        sub = sub.drop_duplicates("exon_id")
        sub = sub.sort_values(["chr","start","end"]).reset_index(drop=True)
        # keep strand/chr from the first exon (all should match)
        strand_vals = sub["strand"].dropna().unique()
        strand = strand_vals[0] if len(strand_vals) else "+"
        chrom = sub["chr"].iloc[0]

        if sub.shape[0] < 3:
            continue  # need at least 3 exons to define internal cassette events

        for i in range(1, sub.shape[0]-1):
            up    = sub.iloc[i-1]
            cur   = sub.iloc[i]
            down  = sub.iloc[i+1]
            up_end      = int(up["end"])
            t_start     = int(cur["start"])
            t_end       = int(cur["end"])
            down_start  = int(down["start"])
            event_id = f"{gene_name}_ex{i+1}"  # position within this transcript’s exon chain

            rows.append({
                "event_id": event_id,
                "gene": gene_name,
                "chr": chrom,
                "strand": strand,
                "up_end": up_end,
                "t_start": t_start,
                "t_end": t_end,
                "down_start": down_start,
            })

    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["event_id","gene","chr","strand","up_end","t_start","t_end","down_start"])
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"[make_events_from_splitpipe_ref] wrote {OUT_CSV} with {len(rows)} events across {len(can_tx)} genes.")

if __name__ == "__main__":
    sys.exit(main())
