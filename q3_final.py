#!/usr/bin/env python3
import os
import pandas as pd
import numpy as np

BASE = "/mnt/storage1/projects/research/24035_1345_TDP43/scanpy_analysis/figures_tdp43_question_panels_v3"
TABLE = os.path.join(BASE, "tables", "Q3_MMR_vs_abundance_shift_table.tsv")
OUT = os.path.join(BASE, "tables", "Q3_celltype_direction_summary.tsv")

# optional thresholds so tiny differences are called "similar"
ABUND_EPS = 0.10   # in log2 fraction shift
MMR_EPS = 0.01     # in mean log1p-normalized MMR


def call_abundance_direction(x, eps=ABUND_EPS):
    if pd.isna(x):
        return "unknown"
    if x > eps:
        return "enriched_in_TDP43"
    if x < -eps:
        return "depleted_in_TDP43"
    return "similar_abundance"


def call_mmr_direction(delta, eps=MMR_EPS):
    if pd.isna(delta):
        return "unknown"
    if delta > eps:
        return "higher_MMR_in_TDP43"
    if delta < -eps:
        return "higher_MMR_in_control"
    return "similar_MMR"


def main():
    df = pd.read_csv(TABLE, sep="\t")

    # expected columns from your v3 table:
    # cell_type
    # cell_class
    # log2fc_fraction_tdp43_vs_control
    # MMR_mean_control
    # MMR_mean_tdp43
    # delta_MMR_mean_tdp43_minus_control
    # control / tdp43 counts

    required = [
        "cell_type",
        "cell_class",
        "log2fc_fraction_tdp43_vs_control",
        "MMR_mean_control",
        "MMR_mean_tdp43",
        "delta_MMR_mean_tdp43_minus_control",
        "control",
        "tdp43",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {TABLE}: {missing}")

    df["abundance_direction"] = df["log2fc_fraction_tdp43_vs_control"].apply(call_abundance_direction)
    df["MMR_direction"] = df["delta_MMR_mean_tdp43_minus_control"].apply(call_mmr_direction)

    # combined label = easiest for presentation
    df["combined_pattern"] = (
        df["abundance_direction"] + " | " + df["MMR_direction"]
    )

    # nicer numeric columns
    df["MMR_ratio_tdp43_over_control"] = (
        (df["MMR_mean_tdp43"] + 1e-9) / (df["MMR_mean_control"] + 1e-9)
    )

    # save full summary
    out_cols = [
        "cell_type",
        "cell_class",
        "control",
        "tdp43",
        "log2fc_fraction_tdp43_vs_control",
        "abundance_direction",
        "MMR_mean_control",
        "MMR_mean_tdp43",
        "delta_MMR_mean_tdp43_minus_control",
        "MMR_ratio_tdp43_over_control",
        "MMR_direction",
        "combined_pattern",
    ]
    df[out_cols].sort_values(
        ["abundance_direction", "MMR_direction", "log2fc_fraction_tdp43_vs_control"],
        ascending=[True, True, True]
    ).to_csv(OUT, sep="\t", index=False)

    print(f"[IO] wrote: {OUT}\n")

    # print easy summaries
    print("Counts by abundance direction:")
    print(df["abundance_direction"].value_counts(dropna=False).to_string())
    print()

    print("Counts by MMR direction:")
    print(df["MMR_direction"].value_counts(dropna=False).to_string())
    print()

    print("Counts by combined pattern:")
    print(df["combined_pattern"].value_counts(dropna=False).to_string())
    print()

    # show examples for each class
    patterns = [
        "enriched_in_TDP43 | higher_MMR_in_TDP43",
        "enriched_in_TDP43 | higher_MMR_in_control",
        "depleted_in_TDP43 | higher_MMR_in_TDP43",
        "depleted_in_TDP43 | higher_MMR_in_control",
    ]

    for p in patterns:
        sub = df[df["combined_pattern"] == p].copy()
        if sub.empty:
            continue

        # for enriched: show largest positive abundance shift first
        # for depleted: show largest negative abundance shift first
        asc = "depleted" in p
        sub = sub.sort_values("log2fc_fraction_tdp43_vs_control", ascending=asc)

        print(f"Top examples: {p}")
        print(
            sub[
                [
                    "cell_type",
                    "cell_class",
                    "control",
                    "tdp43",
                    "log2fc_fraction_tdp43_vs_control",
                    "MMR_mean_control",
                    "MMR_mean_tdp43",
                    "delta_MMR_mean_tdp43_minus_control",
                ]
            ]
            .head(10)
            .to_string(index=False)
        )
        print()


if __name__ == "__main__":
    main()