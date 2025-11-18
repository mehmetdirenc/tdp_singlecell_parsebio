#!/usr/bin/env python3
import csv, re, sys, os
from collections import defaultdict, namedtuple
print("ASDASDAsd")
# --- CONFIG ---
GTF_PATH = "/mnt/storage2/users/ahmungm1/genomes/mus_musculus/mm_grcm39.gtf"  # <-- set this
OUT_CSV  = "/mnt/storage1/projects/research/24035_1345_TDP43/events_mouse.csv"

# Only build events for these genes (edit/extend as you like)
GENE_LIST = [
    "Psmd14","Ddi2","Mrpl45",
    "Hjurp","Cops4","Tiam1","Creld1","Inpp4a","Shank1",
    "Arhgap44","Psme3","Dgkq","Arhgef11","Osbpl6","Pik3cb","Pdzdc1","Dsel",
    "Herc2","Golga4","Stx24","Ube3c","C530008M17Rik","Tub","Atp2b1",
    "Lhx1","Lhx1os","Onecut1","Slc32a1","Gad1os","Gad2","Dlx6","Npas1","Pou6f2",
]

# --- helpers to parse attributes ---
def parse_attr(s):
    # works for GTF key "gene_name" etc.
    out = {}
    for m in re.finditer(r'(\w+)\s+"([^"]+)"', s):
        out[m.group(1)] = m.group(2)
    return out

Exon = namedtuple("Exon","start end exon_number")

def main():
    # Collect exons per transcript per gene
    tx_by_gene = defaultdict(lambda: defaultdict(list))   # gene -> transcript -> [Exon,...]
    tx_meta    = defaultdict(dict)                        # gene -> transcript -> meta dict (chr,strand,biotype,length)

    with open(GTF_PATH) as f:
        for line in f:
            if not line or line.startswith("#"): continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 9: continue
            chrom, source, feature, start, end, score, strand, frame, attr = fields
            if feature not in ("exon","transcript"): continue
            a = parse_attr(attr)
            gene_name = a.get("gene_name") or a.get("gene_id")
            if gene_name not in GENE_LIST:  # limit to your targets
                continue
            tx_id = a.get("transcript_id")
            if not tx_id:
                continue
            start, end = int(start), int(end)

            if feature == "exon":
                # exon_number may be missing; we can infer later by transcript order
                exon_num = a.get("exon_number")
                exon_num = int(exon_num) if exon_num and exon_num.isdigit() else None
                tx_by_gene[gene_name][tx_id].append(Exon(start, end, exon_num))
                # store meta
                meta = tx_meta[gene_name].setdefault(tx_id, {"chr":chrom, "strand":strand, "biotype":a.get("transcript_biotype") or a.get("transcript_type"), "len":0})
                meta["len"] += (end - start + 1)

    # choose canonical transcript per gene
    can_tx = {}
    for gene, txs in tx_by_gene.items():
        # prefer protein_coding
        candidates = []
        for tid, exons in txs.items():
            meta = tx_meta[gene].get(tid, {})
            biotype = (meta.get("biotype") or "").lower()
            n_exons = len(exons)
            length  = meta.get("len", 0)
            pc = 1 if "protein_coding" in biotype else 0
            candidates.append(( -pc, -n_exons, -length, tid ))  # sort ascending -> pc first, more exons, longer
        if not candidates:
            continue
        candidates.sort()
        can_tid = candidates[0][3]
        can_tx[gene] = can_tid

    # build events from canonical transcript exons (sorted by genomic coordinate)
    rows = []
    for gene, tid in can_tx.items():
        meta = tx_meta[gene][tid]
        chrom, strand = meta["chr"], meta["strand"]
        exons = tx_by_gene[gene][tid]
        # sort by genomic coordinate (start); regardless of strand
        exons_sorted = sorted(exons, key=lambda e: (e.start, e.end))
        # internal exons only (exclude first and last)
        for i in range(1, len(exons_sorted)-1):
            up    = exons_sorted[i-1]
            cur   = exons_sorted[i]
            down  = exons_sorted[i+1]
            # event_id: gene_ex<ordinal>_tx<short>
            exon_idx = i+1  # 1-based position in this sorted list (not necessarily same as GTF exon_number)
            event_id = f"{gene}_ex{exon_idx}"

            # junction endpoints use genomic (left->right) coordinates
            up_end      = up.end
            t_start     = cur.start
            t_end       = cur.end
            down_start  = down.start

            rows.append({
                "event_id": event_id,
                "gene": gene,
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

    print(f"[make_events_from_gtf] wrote {OUT_CSV} with {len(rows)} events from {len(can_tx)} genes.")

if __name__ == "__main__":
    sys.exit(main())
