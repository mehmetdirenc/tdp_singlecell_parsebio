#!/usr/bin/env python3

import os
import re
import warnings
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt

from scipy import sparse
from scipy.stats import fisher_exact
from statsmodels.stats.multitest import multipletests

try:
    import gseapy as gp
except ImportError:
    gp = None


# =============================================================================
# USER SETTINGS
# =============================================================================

BASE = "/mnt/storage1/projects/research/24035_1345_TDP43/scanpy_analysis"

IN_H5AD = os.path.join(
    BASE,
    "adata_files",
    "mouse_adata_RAW_celltypist.h5ad"
)

OUTDIR = os.path.join(BASE, "pathway_analysis_interest_celltypes")
TABLEDIR = os.path.join(OUTDIR, "tables")
FIGDIR = os.path.join(OUTDIR, "figures")

os.makedirs(TABLEDIR, exist_ok=True)
os.makedirs(FIGDIR, exist_ok=True)

LABEL_COL = "ct_label_mv"
BC1_COL = "bc1_well"
LOWCONF_LABEL = "Unknown_lowConf"

CONTROL_WELLS = {f"A{i}" for i in range(1, 13)}
TDP43_WELLS = {f"B{i}" for i in range(1, 13)}

MIN_TOTAL_CELLS = 20
MIN_CONTROL_CELLS = 10
MIN_TDP43_CELLS = 10

ABUND_EPS = 0.10

# Gene-list selection for Enrichr/ORA
MIN_PCT_EXPR = 0.10
LOGFC_CUT = 0.25
USE_PADJ_FOR_GENE_LIST = False
PADJ_CUT = 0.05
MAX_GENES_FOR_ENRICHR = 200

# Enrichr needs internet.
RUN_ENRICHR = True
ENRICHR_GENE_SETS = [
    "GO_Biological_Process_2023",
    "Reactome_2022",
]

# Optional custom gene set.
# Put one gene per line in a txt file, or use a CSV/TSV with a column called gene/names/symbol.
# Leave empty for now if you do not have it yet.
CUSTOM_GENE_LIST = ""
CUSTOM_GENE_SET_NAME = "custom_gene_set"

# Run ranked GSEA for the custom gene set.
RUN_CUSTOM_PRERANK = True
CUSTOM_GSEA_PERMUTATIONS = 1000

# These are the groups from your screenshot.
# The regex matching is intentionally flexible because CellTypist labels include extra tokens.
TARGET_GROUPS = {
    "decreased_central_amygdala_stress_CEA_BST_CEA_AAA": {
        "expected": "decreased",
        "patterns": [
            r"CEA[-_ ]?BST",
            r"CEA[-_ ]?AAA",
        ],
    },
    "decreased_medial_amygdala_social_olfactory_MEA_BST_MEA_COA_BMA": {
        "expected": "decreased",
        "patterns": [
            r"MEA[-_ ]?BST",
            r"MEA[-_ ]?COA[-_ ]?BMA",
        ],
    },
    "decreased_striatal_GABA_STR_D2_STR_PAL": {
        "expected": "decreased",
        "patterns": [
            r"STR.*D2",
            r"STR[-_ ]?PAL",
        ],
    },
    "increased_microglia": {
        "expected": "increased",
        "patterns": [
            r"MICRO",
            r"MICROGLIA",
        ],
    },
    "increased_VLMC": {
        "expected": "increased",
        "patterns": [
            r"VLMC",
        ],
    },
    "increased_lymphoid": {
        "expected": "increased",
        "patterns": [
            r"LYMPHOID",
        ],
    },
    "increased_BAM": {
        "expected": "increased",
        "patterns": [
            r"\bBAM\b",
            r"BORDER.*MACRO",
            r"MACROPHAGE",
        ],
    },
}


# =============================================================================
# HELPERS
# =============================================================================

def clean_label(x):
    x = str(x)
    x = re.sub(r"[^\w.\-]+", "_", x)
    x = re.sub(r"_+", "_", x)
    return x.strip("_")


def save_table(df, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, sep="\t", index=False)
    print(f"[IO] {path}")


def ensure_condition(adata):
    if "condition" in adata.obs.columns and adata.obs["condition"].notna().any():
        adata.obs["condition"] = adata.obs["condition"].astype("object")
        return adata

    if BC1_COL not in adata.obs.columns:
        raise ValueError(f"{BC1_COL} not found in adata.obs")

    bc1 = adata.obs[BC1_COL].astype(str)
    adata.obs["condition"] = pd.Series(index=adata.obs.index, dtype="object")
    adata.obs.loc[bc1.isin(CONTROL_WELLS), "condition"] = "control"
    adata.obs.loc[bc1.isin(TDP43_WELLS), "condition"] = "tdp43"

    return adata


def make_log1p_copy(adata):
    a = adata.copy()
    sc.pp.normalize_total(a, target_sum=1e4)
    sc.pp.log1p(a)
    return a


def matrix_mean_pct(X):
    if sparse.issparse(X):
        mean = np.asarray(X.mean(axis=0)).ravel()
        pct = np.asarray((X > 0).mean(axis=0)).ravel()
    else:
        X = np.asarray(X)
        mean = X.mean(axis=0)
        pct = (X > 0).mean(axis=0)
    return mean, pct


def classify_celltype_class(label):
    s = str(label).upper()

    if any(x in s for x in ["GABA", "GLUT", "DOPA-GABA", "HIST-GABA", "GABA-CHOL"]):
        return "neuronal"

    if any(x in s for x in [
        "ASTRO", "OLIGO", "OPC", "MICRO", "EPEN", "TANY", "VLMC",
        "AQP4", "MOG", "PLP1", "PDGFRA", "CLDN5", "RGS5",
        "SMC", "LYMPHOID", "BAM", "CHOR", "ENDO", "PERI", "MACRO"
    ]):
        return "non_neuronal"

    return "other"


def call_abundance_direction(x):
    if pd.isna(x):
        return "unknown"
    if x > ABUND_EPS:
        return "increased_in_TDP43"
    if x < -ABUND_EPS:
        return "decreased_in_TDP43"
    return "similar_abundance"


# =============================================================================
# CELL-TYPE ABUNDANCE
# =============================================================================

def compute_celltype_abundance_shift(adata):
    obs = adata.obs.copy()

    if LABEL_COL not in obs.columns:
        raise ValueError(f"{LABEL_COL} not found in adata.obs")

    obs = obs[obs["condition"].isin(["control", "tdp43"])].copy()
    obs[LABEL_COL] = obs[LABEL_COL].astype(str)
    obs = obs[obs[LABEL_COL] != LOWCONF_LABEL].copy()

    counts = pd.crosstab(obs[LABEL_COL], obs["condition"])

    for col in ["control", "tdp43"]:
        if col not in counts.columns:
            counts[col] = 0

    counts = counts[["control", "tdp43"]].copy()
    counts["total_cells"] = counts["control"] + counts["tdp43"]
    counts = counts[counts["total_cells"] >= MIN_TOTAL_CELLS].copy()

    total_control = counts["control"].sum()
    total_tdp43 = counts["tdp43"].sum()

    counts["frac_control"] = counts["control"] / total_control
    counts["frac_tdp43"] = counts["tdp43"] / total_tdp43
    counts["delta_fraction_tdp43_minus_control"] = counts["frac_tdp43"] - counts["frac_control"]
    counts["log2fc_fraction_tdp43_vs_control"] = np.log2(
        (counts["frac_tdp43"] + 1e-9) / (counts["frac_control"] + 1e-9)
    )

    out = counts.reset_index().rename(columns={LABEL_COL: "cell_type"})
    out["cell_class"] = out["cell_type"].apply(classify_celltype_class)
    out["abundance_direction"] = out["log2fc_fraction_tdp43_vs_control"].apply(call_abundance_direction)

    out = out.sort_values("log2fc_fraction_tdp43_vs_control").reset_index(drop=True)

    save_table(
        out,
        os.path.join(TABLEDIR, "celltype_abundance_shift_control_vs_tdp43.tsv")
    )

    return out


def find_target_celltypes(abundance_df):
    all_labels = abundance_df["cell_type"].astype(str).tolist()
    rows = []

    for group_name, cfg in TARGET_GROUPS.items():
        expected = cfg["expected"]
        patterns = cfg["patterns"]

        matched = []
        for label in all_labels:
            for pat in patterns:
                if re.search(pat, label, flags=re.IGNORECASE):
                    matched.append(label)
                    break

        matched = sorted(set(matched))

        if not matched:
            rows.append({
                "target_group": group_name,
                "expected": expected,
                "cell_type": "NO_MATCH_FOUND",
                "control": np.nan,
                "tdp43": np.nan,
                "log2fc_fraction_tdp43_vs_control": np.nan,
                "abundance_direction": "missing",
                "use_for_pathway": False,
            })
            continue

        for ct in matched:
            r = abundance_df[abundance_df["cell_type"] == ct].iloc[0].to_dict()
            use = (
                r["control"] >= MIN_CONTROL_CELLS and
                r["tdp43"] >= MIN_TDP43_CELLS
            )

            rows.append({
                "target_group": group_name,
                "expected": expected,
                "cell_type": ct,
                "control": int(r["control"]),
                "tdp43": int(r["tdp43"]),
                "log2fc_fraction_tdp43_vs_control": float(r["log2fc_fraction_tdp43_vs_control"]),
                "abundance_direction": r["abundance_direction"],
                "use_for_pathway": bool(use),
            })

    selected = pd.DataFrame(rows)

    save_table(
        selected,
        os.path.join(TABLEDIR, "selected_celltypes_for_pathway_analysis.tsv")
    )

    usable = selected[
        (selected["use_for_pathway"]) &
        (selected["cell_type"] != "NO_MATCH_FOUND")
    ].copy()

    return selected, usable


def find_gaba_glut_not_decreased(abundance_df):
    gg = abundance_df[
        abundance_df["cell_type"].astype(str).str.contains("GABA|GLUT", case=False, regex=True)
    ].copy()

    gg = gg[
        (gg["control"] >= MIN_CONTROL_CELLS) &
        (gg["tdp43"] >= MIN_TDP43_CELLS)
    ].copy()

    gg["not_decreased"] = gg["log2fc_fraction_tdp43_vs_control"] >= -ABUND_EPS

    not_decreased = gg[gg["not_decreased"]].copy()
    not_decreased = not_decreased.sort_values(
        "log2fc_fraction_tdp43_vs_control",
        ascending=False
    )

    save_table(
        gg.sort_values("log2fc_fraction_tdp43_vs_control"),
        os.path.join(TABLEDIR, "all_GABA_GLUT_celltypes_abundance.tsv")
    )

    save_table(
        not_decreased,
        os.path.join(TABLEDIR, "GABA_GLUT_not_decreased_or_similar.tsv")
    )

    return not_decreased


# =============================================================================
# DIFFERENTIAL EXPRESSION
# =============================================================================

def run_de_for_celltype(adata_log1p, cell_type):
    sub = adata_log1p[
        (adata_log1p.obs[LABEL_COL].astype(str) == str(cell_type)) &
        (adata_log1p.obs["condition"].isin(["control", "tdp43"]))
    ].copy()

    n_control = int((sub.obs["condition"].astype(str) == "control").sum())
    n_tdp43 = int((sub.obs["condition"].astype(str) == "tdp43").sum())

    if n_control < MIN_CONTROL_CELLS or n_tdp43 < MIN_TDP43_CELLS:
        print(
            f"[WARN] Skipping {cell_type}: too few cells "
            f"(control={n_control}, tdp43={n_tdp43})"
        )
        return None

    sc.tl.rank_genes_groups(
        sub,
        groupby="condition",
        groups=["tdp43"],
        reference="control",
        method="wilcoxon",
        corr_method="benjamini-hochberg",
        use_raw=False,
        pts=False,
    )

    de = sc.get.rank_genes_groups_df(sub, group="tdp43")
    de = de.rename(columns={
        "names": "names",
        "scores": "scores",
        "logfoldchanges": "scanpy_logfoldchanges",
        "pvals": "pvals",
        "pvals_adj": "pvals_adj",
    })

    de["names"] = de["names"].astype(str)

    mask_t = sub.obs["condition"].astype(str).values == "tdp43"
    mask_c = sub.obs["condition"].astype(str).values == "control"

    mean_t, pct_t = matrix_mean_pct(sub.X[mask_t, :])
    mean_c, pct_c = matrix_mean_pct(sub.X[mask_c, :])

    gene_info = pd.DataFrame({
        "names": sub.var_names.astype(str),
        "mean_expr_tdp43": mean_t,
        "mean_expr_control": mean_c,
        "pct_expr_tdp43": pct_t,
        "pct_expr_control": pct_c,
    })

    de = de.merge(gene_info, on="names", how="left")

    de["logfoldchanges"] = np.log2(
        (de["mean_expr_tdp43"] + 1e-9) /
        (de["mean_expr_control"] + 1e-9)
    )

    de["n_tdp43_cells"] = n_tdp43
    de["n_control_cells"] = n_control
    de["cell_type"] = cell_type

    p_for_rank = de["pvals"].fillna(1.0).clip(lower=1e-300)
    de["rank_score"] = np.sign(de["logfoldchanges"]) * (-np.log10(p_for_rank))

    de = de.sort_values(["pvals_adj", "logfoldchanges"], ascending=[True, False])
    de = de.reset_index(drop=True)

    return de


def select_genes_for_ora(de, direction):
    df = de.copy()

    df = df[
        (df["pct_expr_tdp43"] >= MIN_PCT_EXPR) |
        (df["pct_expr_control"] >= MIN_PCT_EXPR)
    ].copy()

    if USE_PADJ_FOR_GENE_LIST:
        df = df[df["pvals_adj"] < PADJ_CUT].copy()

    if direction == "up":
        df = df[df["logfoldchanges"] >= LOGFC_CUT].copy()
        df = df.sort_values(["rank_score", "logfoldchanges"], ascending=[False, False])
    elif direction == "down":
        df = df[df["logfoldchanges"] <= -LOGFC_CUT].copy()
        df = df.sort_values(["rank_score", "logfoldchanges"], ascending=[True, True])
    else:
        raise ValueError("direction must be up or down")

    df["abs_rank_score"] = df["rank_score"].abs()
    df = df.sort_values(["abs_rank_score", "pvals_adj"], ascending=[False, True])
    df = df.head(MAX_GENES_FOR_ENRICHR).copy()

    genes = df["names"].dropna().astype(str).tolist()
    return df, genes


def write_rank_file(de, path):
    rnk = de[["names", "rank_score"]].copy()
    rnk = rnk.replace([np.inf, -np.inf], np.nan).dropna()
    rnk = rnk.groupby("names", as_index=False)["rank_score"].max()
    rnk = rnk.sort_values("rank_score", ascending=False)
    rnk.to_csv(path, sep="\t", index=False, header=False)
    return rnk


def plot_top_de_genes(de, out_png, title, top_each_side=10):
    df = de.copy()
    df = df[
        (df["pct_expr_tdp43"] >= MIN_PCT_EXPR) |
        (df["pct_expr_control"] >= MIN_PCT_EXPR)
    ].copy()

    up = df[df["logfoldchanges"] > 0].nsmallest(top_each_side, "pvals_adj")
    down = df[df["logfoldchanges"] < 0].nsmallest(top_each_side, "pvals_adj")

    plot_df = pd.concat([down, up], ignore_index=True)
    if plot_df.empty:
        return

    plot_df = plot_df.sort_values("logfoldchanges")

    plt.figure(figsize=(8, max(5, 0.35 * len(plot_df))))
    plt.barh(plot_df["names"], plot_df["logfoldchanges"])
    plt.axvline(0, linestyle="--", linewidth=1)
    plt.xlabel("log2 fold change, TDP43 / control")
    plt.ylabel("Gene")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close()


# =============================================================================
# ENRICHR / PATHWAY ANALYSIS
# =============================================================================

def run_enrichr_for_genes(genes, outdir, label):
    if not RUN_ENRICHR:
        return None

    if gp is None:
        print("[WARN] gseapy is not installed. Skipping Enrichr.")
        return None

    if len(genes) < 3:
        print(f"[WARN] {label}: too few genes for Enrichr: {len(genes)}")
        return None

    os.makedirs(outdir, exist_ok=True)

    try:
        enr = gp.enrichr(
            gene_list=genes,
            gene_sets=ENRICHR_GENE_SETS,
            organism="mouse",
            outdir=outdir,
            no_plot=True,
        )
    except Exception as e:
        print(f"[WARN] Enrichr failed for {label}: {e}")
        return None

    if enr is None or enr.results is None or enr.results.empty:
        print(f"[WARN] No Enrichr results for {label}")
        return None

    res = enr.results.copy()
    res["analysis_label"] = label

    return res


def plot_enrichr_barplot(enr_df, out_png, title, top_n=15):
    if enr_df is None or enr_df.empty:
        return

    df = enr_df.copy()

    if "Adjusted P-value" in df.columns:
        df = df.sort_values("Adjusted P-value", ascending=True)
        df["score_for_plot"] = -np.log10(df["Adjusted P-value"].clip(lower=1e-300))
        xlabel = "-log10 adjusted p-value"
    elif "Combined Score" in df.columns:
        df = df.sort_values("Combined Score", ascending=False)
        df["score_for_plot"] = df["Combined Score"]
        xlabel = "Combined score"
    else:
        return

    df = df.head(top_n).copy()
    df["label"] = df["Term"].astype(str) + " [" + df["Gene_set"].astype(str) + "]"
    df = df.iloc[::-1]

    plt.figure(figsize=(10, max(5, 0.42 * len(df))))
    plt.barh(df["label"], df["score_for_plot"])
    plt.xlabel(xlabel)
    plt.ylabel("")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close()


# =============================================================================
# CUSTOM GENE SET
# =============================================================================

def read_custom_gene_list(path):
    if path is None or str(path).strip() == "":
        return []

    path = str(path)

    if not os.path.exists(path):
        raise FileNotFoundError(path)

    if path.endswith(".gmt"):
        genes = []
        with open(path) as fh:
            for line in fh:
                parts = line.rstrip("\n").split("\t")
                if len(parts) >= 3:
                    genes.extend(parts[2:])
        return sorted(set([g.strip() for g in genes if g.strip()]))

    sep = "\t" if path.endswith(".tsv") else ","

    if path.endswith(".txt") or path.endswith(".list"):
        with open(path) as fh:
            genes = [x.strip() for x in fh if x.strip() and not x.startswith("#")]
        return sorted(set(genes))

    df = pd.read_csv(path, sep=sep)

    possible_cols = ["gene", "genes", "names", "symbol", "gene_symbol", "Gene", "GeneSymbol"]
    gene_col = None

    for col in possible_cols:
        if col in df.columns:
            gene_col = col
            break

    if gene_col is None:
        gene_col = df.columns[0]

    genes = df[gene_col].dropna().astype(str).str.strip().tolist()
    return sorted(set([g for g in genes if g]))


def map_genes_to_universe(genes, universe):
    universe = [str(g) for g in universe]
    upper_to_gene = {g.upper(): g for g in universe}

    mapped = []
    missing = []

    for g in genes:
        key = str(g).upper()
        if key in upper_to_gene:
            mapped.append(upper_to_gene[key])
        else:
            missing.append(g)

    return sorted(set(mapped)), sorted(set(missing))


def custom_gene_set_fisher(de, custom_genes, direction):
    universe = set(de["names"].astype(str))

    df = de.copy()
    df = df[
        (df["pct_expr_tdp43"] >= MIN_PCT_EXPR) |
        (df["pct_expr_control"] >= MIN_PCT_EXPR)
    ].copy()

    if USE_PADJ_FOR_GENE_LIST:
        df = df[df["pvals_adj"] < PADJ_CUT].copy()

    if direction == "up":
        selected = set(df[df["logfoldchanges"] >= LOGFC_CUT]["names"].astype(str))
    elif direction == "down":
        selected = set(df[df["logfoldchanges"] <= -LOGFC_CUT]["names"].astype(str))
    else:
        raise ValueError("direction must be up or down")

    custom = set(custom_genes) & universe

    a = len(selected & custom)
    b = len(selected - custom)
    c = len(custom - selected)
    d = len(universe - selected - custom)

    if len(custom) == 0 or len(selected) == 0:
        oddsratio, pvalue = np.nan, np.nan
    else:
        oddsratio, pvalue = fisher_exact([[a, b], [c, d]], alternative="greater")

    return {
        "direction": direction,
        "n_universe": len(universe),
        "n_selected": len(selected),
        "n_custom_in_universe": len(custom),
        "n_overlap": a,
        "oddsratio": oddsratio,
        "pvalue": pvalue,
        "overlap_genes": ",".join(sorted(selected & custom)),
    }


def run_custom_prerank(rnk, custom_genes, outdir, label):
    if not RUN_CUSTOM_PRERANK:
        return None

    if gp is None:
        print("[WARN] gseapy is not installed. Skipping custom prerank GSEA.")
        return None

    if len(custom_genes) < 3:
        print(f"[WARN] {label}: too few custom genes for prerank: {len(custom_genes)}")
        return None

    os.makedirs(outdir, exist_ok=True)

    gene_sets = {
        CUSTOM_GENE_SET_NAME: custom_genes
    }

    try:
        pre = gp.prerank(
            rnk=rnk,
            gene_sets=gene_sets,
            outdir=outdir,
            permutation_num=CUSTOM_GSEA_PERMUTATIONS,
            min_size=2,
            max_size=5000,
            seed=13,
            no_plot=True,
            verbose=False,
        )
    except Exception as e:
        print(f"[WARN] Custom prerank failed for {label}: {e}")
        return None

    res = pre.res2d.copy()
    res["analysis_label"] = label

    return res


# =============================================================================
# MAIN
# =============================================================================

def main():
    warnings.filterwarnings("ignore", category=FutureWarning)

    print(f"[IO] Loading: {IN_H5AD}")
    adata = sc.read_h5ad(IN_H5AD)
    adata = ensure_condition(adata)

    print("[INFO] adata:", adata.shape)
    print("[INFO] condition counts:")
    print(adata.obs["condition"].value_counts(dropna=False).to_string())

    abundance_df = compute_celltype_abundance_shift(adata)

    selected_all, usable_targets = find_target_celltypes(abundance_df)

    not_decreased = find_gaba_glut_not_decreased(abundance_df)

    print("\n=== GABA / GLUT cell types that are not clearly decreased ===")
    if not_decreased.empty:
        print("None found with the current thresholds.")
    else:
        print(
            not_decreased[
                ["cell_type", "control", "tdp43", "log2fc_fraction_tdp43_vs_control", "abundance_direction"]
            ].to_string(index=False)
        )

    print("\n=== Cell types selected for pathway analysis ===")
    if usable_targets.empty:
        print("[WARN] No usable target cell types found.")
        print("Check:", os.path.join(TABLEDIR, "selected_celltypes_for_pathway_analysis.tsv"))
        return

    print(
        usable_targets[
            ["target_group", "cell_type", "control", "tdp43", "log2fc_fraction_tdp43_vs_control"]
        ].to_string(index=False)
    )

    print("\n[INFO] Making log-normalized copy for DE...")
    adata_log1p = make_log1p_copy(adata)

    custom_raw = read_custom_gene_list(CUSTOM_GENE_LIST)
    if custom_raw:
        custom_genes_mapped, custom_missing = map_genes_to_universe(
            custom_raw,
            adata_log1p.var_names.astype(str)
        )

        save_table(
            pd.DataFrame({"custom_gene_input": custom_raw}),
            os.path.join(TABLEDIR, "custom_gene_set_input.tsv")
        )

        save_table(
            pd.DataFrame({"custom_gene_matched": custom_genes_mapped}),
            os.path.join(TABLEDIR, "custom_gene_set_matched_to_adata.tsv")
        )

        save_table(
            pd.DataFrame({"custom_gene_missing": custom_missing}),
            os.path.join(TABLEDIR, "custom_gene_set_missing_from_adata.tsv")
        )

        print(
            f"[INFO] Custom gene set: {len(custom_genes_mapped)} matched, "
            f"{len(custom_missing)} missing"
        )
    else:
        custom_genes_mapped = []
        print("[INFO] No custom gene list provided yet. Skipping custom gene-set tests.")

    all_enrichr = []
    all_custom_fisher = []
    all_custom_gsea = []
    de_summary_rows = []

    processed_celltypes = usable_targets["cell_type"].drop_duplicates().tolist()

    for cell_type in processed_celltypes:
        safe_ct = clean_label(cell_type)

        print("\n" + "=" * 100)
        print(f"[INFO] Processing cell type: {cell_type}")
        print("=" * 100)

        ct_outdir = os.path.join(OUTDIR, "per_celltype", safe_ct)
        ct_tabledir = os.path.join(ct_outdir, "tables")
        ct_figdir = os.path.join(ct_outdir, "figures")
        os.makedirs(ct_tabledir, exist_ok=True)
        os.makedirs(ct_figdir, exist_ok=True)

        de = run_de_for_celltype(adata_log1p, cell_type)
        if de is None:
            continue

        de_path = os.path.join(ct_tabledir, f"{safe_ct}_DE_tdp43_vs_control_full.tsv")
        save_table(de, de_path)

        rnk_path = os.path.join(ct_tabledir, f"{safe_ct}_ranked_genes.rnk")
        rnk = write_rank_file(de, rnk_path)
        print(f"[IO] {rnk_path}")

        plot_top_de_genes(
            de,
            os.path.join(ct_figdir, f"{safe_ct}_top_DE_genes.png"),
            title=f"{cell_type}\nTDP43 vs control",
            top_each_side=10,
        )

        n_up = int(
            (
                ((de["pct_expr_tdp43"] >= MIN_PCT_EXPR) | (de["pct_expr_control"] >= MIN_PCT_EXPR)) &
                (de["logfoldchanges"] >= LOGFC_CUT)
            ).sum()
        )

        n_down = int(
            (
                ((de["pct_expr_tdp43"] >= MIN_PCT_EXPR) | (de["pct_expr_control"] >= MIN_PCT_EXPR)) &
                (de["logfoldchanges"] <= -LOGFC_CUT)
            ).sum()
        )

        de_summary_rows.append({
            "cell_type": cell_type,
            "n_control_cells": int(de["n_control_cells"].iloc[0]),
            "n_tdp43_cells": int(de["n_tdp43_cells"].iloc[0]),
            "n_up_genes_for_ora": n_up,
            "n_down_genes_for_ora": n_down,
            "de_table": de_path,
            "rank_file": rnk_path,
        })

        for direction in ["up", "down"]:
            gene_df, genes = select_genes_for_ora(de, direction=direction)

            gene_df_path = os.path.join(ct_tabledir, f"{safe_ct}_{direction}_genes_for_enrichment.tsv")
            gene_txt_path = os.path.join(ct_tabledir, f"{safe_ct}_{direction}_genes_for_enrichment.txt")

            save_table(gene_df, gene_df_path)

            with open(gene_txt_path, "w") as fh:
                for g in genes:
                    fh.write(g + "\n")
            print(f"[IO] {gene_txt_path}")

            enrichr_outdir = os.path.join(ct_outdir, "enrichr", direction)
            enr_df = run_enrichr_for_genes(
                genes=genes,
                outdir=enrichr_outdir,
                label=f"{safe_ct}_{direction}"
            )

            if enr_df is not None and not enr_df.empty:
                enr_path = os.path.join(ct_tabledir, f"{safe_ct}_{direction}_enrichr_results.tsv")
                save_table(enr_df, enr_path)

                plot_enrichr_barplot(
                    enr_df,
                    os.path.join(ct_figdir, f"{safe_ct}_{direction}_enrichr_top_terms.png"),
                    title=f"{cell_type}\n{direction} genes pathway enrichment",
                    top_n=15,
                )

                tmp = enr_df.copy()
                tmp["cell_type"] = cell_type
                tmp["direction"] = direction
                all_enrichr.append(tmp)

            if custom_genes_mapped:
                custom_res = custom_gene_set_fisher(
                    de=de,
                    custom_genes=custom_genes_mapped,
                    direction=direction,
                )
                custom_res["cell_type"] = cell_type
                all_custom_fisher.append(custom_res)

        if custom_genes_mapped:
            gsea_outdir = os.path.join(ct_outdir, "custom_gene_set_prerank")
            custom_gsea = run_custom_prerank(
                rnk=rnk,
                custom_genes=custom_genes_mapped,
                outdir=gsea_outdir,
                label=safe_ct,
            )

            if custom_gsea is not None and not custom_gsea.empty:
                custom_gsea["cell_type"] = cell_type
                all_custom_gsea.append(custom_gsea)

                save_table(
                    custom_gsea,
                    os.path.join(ct_tabledir, f"{safe_ct}_custom_gene_set_prerank.tsv")
                )

    de_summary = pd.DataFrame(de_summary_rows)
    save_table(
        de_summary,
        os.path.join(TABLEDIR, "DE_and_pathway_input_summary.tsv")
    )

    if all_enrichr:
        combined_enrichr = pd.concat(all_enrichr, ignore_index=True)

        save_table(
            combined_enrichr,
            os.path.join(TABLEDIR, "combined_enrichr_all_results.tsv")
        )

        top_enrichr = (
            combined_enrichr
            .sort_values(["Adjusted P-value", "Combined Score"], ascending=[True, False])
            .groupby(["cell_type", "direction", "Gene_set"], as_index=False)
            .head(10)
        )

        save_table(
            top_enrichr,
            os.path.join(TABLEDIR, "combined_enrichr_top10_per_celltype_direction_library.tsv")
        )

    if all_custom_fisher:
        custom_fisher_df = pd.DataFrame(all_custom_fisher)
        custom_fisher_df["pvalue_adj"] = multipletests(
            custom_fisher_df["pvalue"].fillna(1.0),
            method="fdr_bh"
        )[1]

        custom_fisher_df = custom_fisher_df.sort_values(
            ["pvalue_adj", "pvalue", "oddsratio"],
            ascending=[True, True, False]
        )

        save_table(
            custom_fisher_df,
            os.path.join(TABLEDIR, "custom_gene_set_ORA_fisher_up_down_summary.tsv")
        )

    if all_custom_gsea:
        custom_gsea_df = pd.concat(all_custom_gsea, ignore_index=True)

        save_table(
            custom_gsea_df,
            os.path.join(TABLEDIR, "custom_gene_set_prerank_GSEA_summary.tsv")
        )

    summary_txt = os.path.join(OUTDIR, "summary.txt")
    with open(summary_txt, "w") as fh:
        fh.write("Pathway analysis summary\n")
        fh.write("========================\n\n")
        fh.write(f"Input h5ad: {IN_H5AD}\n")
        fh.write(f"Label column: {LABEL_COL}\n")
        fh.write(f"Condition column: condition from {BC1_COL}\n")
        fh.write(f"Control wells: A1-A12\n")
        fh.write(f"TDP43 wells: B1-B12\n\n")
        fh.write("Main outputs:\n")
        fh.write(f"- {os.path.join(TABLEDIR, 'selected_celltypes_for_pathway_analysis.tsv')}\n")
        fh.write(f"- {os.path.join(TABLEDIR, 'GABA_GLUT_not_decreased_or_similar.tsv')}\n")
        fh.write(f"- {os.path.join(TABLEDIR, 'DE_and_pathway_input_summary.tsv')}\n")
        fh.write(f"- {os.path.join(TABLEDIR, 'combined_enrichr_top10_per_celltype_direction_library.tsv')}\n")
        fh.write(f"- {os.path.join(TABLEDIR, 'custom_gene_set_ORA_fisher_up_down_summary.tsv')}\n")
        fh.write(f"- {os.path.join(TABLEDIR, 'custom_gene_set_prerank_GSEA_summary.tsv')}\n")

    print("\n[Done]")
    print(f"[IO] Output folder: {OUTDIR}")
    print(f"[IO] Summary: {summary_txt}")


if __name__ == "__main__":
    main()