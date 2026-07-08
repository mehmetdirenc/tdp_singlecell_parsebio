#!/usr/bin/env python3
import os
import pandas as pd

# ============================================================
# paths
# ============================================================
BASE = "/mnt/storage1/projects/research/24035_1345_TDP43/scanpy_analysis"

# use the newest tables you want
TABLEDIR = os.path.join(BASE, "figures_tdp43_question_panels_v3", "tables")
OUTDIR = os.path.join(BASE, "enrichr_gene_lists_top20")

os.makedirs(OUTDIR, exist_ok=True)

# DE tables
DE_FILES = {
    "113_MEA-COA-BMA_Ccdc42_Glut": os.path.join(
        TABLEDIR, "Q4_depleted_113_MEA-COA-BMA_Ccdc42_Glut_DE_full.tsv"
    ),
    "330_VLMC_NN": os.path.join(
        TABLEDIR, "Q4_enriched_330_VLMC_NN_DE_full.tsv"
    ),
}

# ============================================================
# settings
# ============================================================
GENE_COL = "names"
LOGFC_COL = "logfoldchanges"
PCT_TDP43_COL = "pct_expr_tdp43"
PCT_CONTROL_COL = "pct_expr_control"

MIN_PCT_EXPR = 0.10
LOGFC_CUT = 0.5
TOP_N = 20

# if you want to also require pvals_adj filter, switch this to True
USE_PADJ = False
PADJ_COL = "pvals_adj"
PADJ_CUT = 0.05


# ============================================================
# helpers
# ============================================================
def load_table(path):
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    sep = "\t" if path.endswith(".tsv") else ","
    return pd.read_csv(path, sep=sep)


def filter_base(df):
    out = df.copy()

    # expressed in at least one condition
    if PCT_TDP43_COL in out.columns and PCT_CONTROL_COL in out.columns:
        out = out[
            (out[PCT_TDP43_COL] >= MIN_PCT_EXPR) |
            (out[PCT_CONTROL_COL] >= MIN_PCT_EXPR)
        ].copy()

    # optional padj filter
    if USE_PADJ and PADJ_COL in out.columns:
        out = out[out[PADJ_COL] < PADJ_CUT].copy()

    return out


def select_direction(df, direction="up", top_n=20):
    out = df.copy()

    if direction == "up":
        out = out[out[LOGFC_COL] >= LOGFC_CUT].copy()
    elif direction == "down":
        out = out[out[LOGFC_COL] <= -LOGFC_CUT].copy()
    else:
        raise ValueError("direction must be 'up' or 'down'")

    out["abs_logfc"] = out[LOGFC_COL].abs()

    # rank by effect size first; tie-break with padj if available
    if PADJ_COL in out.columns:
        out = out.sort_values(["abs_logfc", PADJ_COL], ascending=[False, True])
    else:
        out = out.sort_values(["abs_logfc"], ascending=[False])

    out = out.head(top_n).copy()
    return out


def save_gene_outputs(df, label, direction):
    prefix = f"{label}_{direction}_top{TOP_N}"

    tsv_path = os.path.join(OUTDIR, f"{prefix}.tsv")
    txt_path = os.path.join(OUTDIR, f"{prefix}.txt")

    df.to_csv(tsv_path, sep="\t", index=False)

    genes = df[GENE_COL].dropna().astype(str).tolist()
    with open(txt_path, "w") as fh:
        for g in genes:
            fh.write(g + "\n")

    return tsv_path, txt_path, genes


# ============================================================
# main
# ============================================================
def main():
    summary_rows = []

    for label, path in DE_FILES.items():
        print(f"\n[INFO] Processing: {label}")
        df = load_table(path)
        df = filter_base(df)

        for direction in ["up", "down"]:
            sel = select_direction(df, direction=direction, top_n=TOP_N)
            tsv_path, txt_path, genes = save_gene_outputs(sel, label, direction)

            print(f"[INFO] {label} {direction}: {len(genes)} genes")
            print(f"[IO]  TSV: {tsv_path}")
            print(f"[IO]  TXT: {txt_path}")

            if len(genes) > 0:
                print("[GENES]")
                print("\n".join(genes))
            else:
                print("[WARN] no genes passed filters")

            summary_rows.append({
                "label": label,
                "direction": direction,
                "n_genes": len(genes),
                "tsv_path": tsv_path,
                "txt_path": txt_path,
                "genes_joined": ",".join(genes),
            })

    summary_df = pd.DataFrame(summary_rows)
    summary_path = os.path.join(OUTDIR, f"summary_top{TOP_N}.tsv")
    summary_df.to_csv(summary_path, sep="\t", index=False)

    print(f"\n[IO] summary: {summary_path}")


if __name__ == "__main__":
    main()