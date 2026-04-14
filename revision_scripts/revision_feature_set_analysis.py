"""Re-cluster Perturb-seq data using Tsankov 2015 lineage panels and run
permutation-based ligand enrichment per Leiden cluster.

One mini-analysis per (day, panel). For each: subset to detected panel genes,
log-normalize on full adata (before subset), per-gene z-score on subset, PCA
(n_comps=min(20, n_genes)), neighbors (n=15), leiden (res=1.0), UMAP, then
permutation enrichment (10k perms, BH-FDR alpha=0.05, NTC null).

Usage:
    python revision_feature_set_analysis.py \
        --h5ad <path> --day {Day4,Day6} \
        --panel {PL,EC,ME,EN,MS,all} \
        --output-dir <path> [--n-permutations 10000]
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp


TSANKOV_PANELS: dict[str, list[str]] = {
    "PL": ["HESX1", "DNMT3B", "IDO1", "LCK", "POU5F1", "TRIM22", "NANOG", "CXCL5"],
    "EC": [
        "CDH9", "DMBX1", "EN1", "LMX1A", "TRPM8", "MYO3B", "OLFM3", "PAX3", "ZBTB16",
        "SOX1", "NOS2", "WNT1", "NR2F1", "NR2F2", "PAX6", "COL2A1", "MAP2", "POU4F1",
        "SDC2", "DRD4", "PAPLN",
    ],
    "ME": [
        "ODAM", "HAND1", "ABCA4", "FOXA1", "CDX2", "SNAI2", "FOXF1", "PDGFRA", "TBX3",
        "BMP10", "HOPX", "HEY1", "FCN3", "RGS4", "HAND2", "ESM1", "CDH5", "PLVAP",
        "COLEC10", "ALOX15", "IL6ST", "SST", "NKX2-5", "KLF5", "TM4SF1", "GATA4",
    ],
    "EN": [
        "CABP7", "CDH20", "FOXA1", "RXRG", "PHOX2B", "POU3F3", "HNF4A", "HMP19",
        "FOXP2", "ELAVL3", "SOX17", "CPLX2", "GATA6", "CLDN1", "HHEX", "FOXA2",
        "NODAL", "LEFTY1", "EOMES", "LEFTY2",
    ],
    "MS": ["FGF4", "T", "GDF3", "NPPB", "NR5A2"],
}

RANDOM_STATE = 42


# ---------------------------------------------------------------------------
# permutation_cluster_enrichment — copied verbatim from
# Perturb_Seq_Analysis/analysis_scripts/Day4_Analysis.ipynb (cell 35).
# ---------------------------------------------------------------------------
def permutation_cluster_enrichment(
    adata,
    cluster_number,
    perturbation_key: str = "gene_target",
    control_label: str = "NTC",
    leiden_key: str = "leiden",
    n_permutations: int = 10000,
    alpha: float = 0.05,
    min_cells_per_perturb: int = 20,
    random_state: int | None = 0,
):
    """Permutation test for perturbation enrichment/depletion in a Leiden cluster.

    For a given Leiden cluster, compares each perturbation against the NTC population:
      - Observed: proportion of that perturbation's cells in the cluster
      - Null: draw n cells *with replacement* from NTC cells, repeat n_permutations times,
        compute the proportion of sampled cells in the cluster.
      - p_enrich: fraction of null proportions >= observed proportion
      - p_deplete: fraction of null proportions <= observed proportion

    For this cluster, applies BH (FDR) correction separately to enrichment and depletion p-values
    across perturbations, then plots significantly enriched/depleted perturbations.
    """
    from statsmodels.stats.multitest import multipletests

    if leiden_key not in adata.obs.columns:
        raise KeyError(f"'{leiden_key}' not found in adata.obs")
    if perturbation_key not in adata.obs.columns:
        raise KeyError(f"'{perturbation_key}' not found in adata.obs")

    cluster_str = str(cluster_number)
    leiden = adata.obs[leiden_key].astype(str)
    pert = adata.obs[perturbation_key].astype(str)

    cluster_mask = leiden == cluster_str
    is_ntc = pert == control_label

    n_ntc = int(is_ntc.sum())
    if n_ntc == 0:
        raise ValueError(f"No NTC cells found where {perturbation_key} == '{control_label}'")

    ntc_in_cluster = int((cluster_mask & is_ntc).sum())
    p_ntc_cluster = ntc_in_cluster / n_ntc

    perturbations = (
        pert[~is_ntc]
        .value_counts()
        .loc[lambda s: s >= min_cells_per_perturb]
        .index.to_list()
    )

    if len(perturbations) == 0:
        print("No perturbations with sufficient cells for testing.")
        return None, pd.DataFrame()

    rng = np.random.default_rng(random_state)
    ntc_indices = np.where(is_ntc)[0]
    cluster_mask_arr = cluster_mask.values

    rows = []
    for p in perturbations:
        mask_p = pert == p
        n_p = int(mask_p.sum())
        if n_p < min_cells_per_perturb:
            continue

        obs_in_cluster = int((cluster_mask & mask_p).sum())
        obs_prop = obs_in_cluster / n_p

        null_props = []
        for _ in range(n_permutations):
            sample_idx = rng.choice(ntc_indices, size=n_p, replace=True)
            sample_in_cluster = np.sum(cluster_mask_arr[sample_idx])
            null_props.append(sample_in_cluster / n_p)
        null_props = np.asarray(null_props, dtype=float)

        p_enrich = (np.sum(null_props >= obs_prop) + 1) / (n_permutations + 1)
        p_deplete = (np.sum(null_props <= obs_prop) + 1) / (n_permutations + 1)

        rows.append(
            {
                "cluster": cluster_str,
                "perturbation": p,
                "n_cells": n_p,
                "prop_pert": obs_prop,
                "prop_ntc": p_ntc_cluster,
                "delta_prop": obs_prop - p_ntc_cluster,
                "p_enrich": p_enrich,
                "p_deplete": p_deplete,
            }
        )

    stats_df = pd.DataFrame(rows)
    if stats_df.empty:
        print("No perturbations passed min_cells_per_perturb filter.")
        return None, stats_df

    for col_p, col_adj in [("p_enrich", "p_enrich_adj"), ("p_deplete", "p_deplete_adj")]:
        mask = stats_df[col_p].notna()
        if mask.any():
            _, p_adj, _, _ = multipletests(
                stats_df.loc[mask, col_p].values, alpha=alpha, method="fdr_bh"
            )
            stats_df.loc[mask, col_adj] = p_adj
        else:
            stats_df[col_adj] = np.nan

    stats_df["direction"] = "none"
    stats_df.loc[stats_df["p_enrich_adj"] < alpha, "direction"] = "enriched"
    stats_df.loc[stats_df["p_deplete_adj"] < alpha, "direction"] = "depleted"

    sig_df = stats_df[stats_df["direction"] != "none"].copy()
    if sig_df.empty:
        print(f"No significant perturbation shifts in cluster {cluster_str} at FDR {alpha}.")
        return None, stats_df

    sig_df = sig_df.sort_values("delta_prop", ascending=False)

    fig, ax = plt.subplots(figsize=(9, 5), dpi=130)
    x = np.arange(len(sig_df))
    colors = np.where(sig_df["direction"] == "enriched", "#1982C4", "#FF595E")

    delta_pct = sig_df["delta_prop"] * 100.0
    ax.bar(x, delta_pct, color=colors, alpha=0.9)
    ax.axhline(0, color="black", linewidth=1.0, alpha=0.7)

    ax.set_xticks(x)
    ax.set_xticklabels(sig_df["perturbation"], rotation=90, fontsize=9)
    ax.set_ylabel("Δ% (perturbation − NTC)", fontsize=14)
    ax.set_xlabel("Perturbation", fontsize=14)
    ax.set_title(
        f"Cluster {cluster_str}: perturbations enriched/depleted vs NTC", fontsize=16
    )
    ax.tick_params(labelsize=12)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(2.0)
    ax.spines["bottom"].set_linewidth(2.0)
    ax.grid(False)

    plt.tight_layout()

    return fig, stats_df


# ---------------------------------------------------------------------------
# Panel helpers
# ---------------------------------------------------------------------------
def resolve_panel_genes(adata, symbols: list[str]) -> tuple[list[str], pd.DataFrame]:
    """Return (detected panel genes, dropped_df).

    A gene is kept if it appears in var_names AND has count > 0 in >=1% of cells
    (computed on layers['counts']).
    """
    n_cells = adata.n_obs
    threshold = 0.01 * n_cells
    var_set = set(adata.var_names)

    rows = []
    detected = []

    if "counts" in adata.layers:
        counts = adata.layers["counts"]
    else:
        counts = adata.X

    # Precompute nonzero-cell counts per gene once (over whole matrix is expensive;
    # instead compute only for the panel symbols).
    for g in symbols:
        if g not in var_set:
            rows.append({"gene": g, "reason": "missing"})
            continue
        col = counts[:, adata.var_names.get_loc(g)]
        if sp.issparse(col):
            n_expr = col.getnnz()
        else:
            n_expr = int(np.count_nonzero(np.asarray(col).ravel()))
        if n_expr < threshold:
            rows.append(
                {"gene": g, "reason": f"below_1pct ({n_expr}/{n_cells} cells)"}
            )
        else:
            detected.append(g)
            rows.append({"gene": g, "reason": "kept"})

    dropped_df = pd.DataFrame(rows)
    return detected, dropped_df


def prepare_normalized_adata(adata):
    """Copy adata; set X = counts; normalize_total(1e4); log1p. Returns new AnnData."""
    out = adata.copy()
    if "counts" in out.layers:
        out.X = out.layers["counts"].copy()
    sc.pp.normalize_total(out, target_sum=1e4)
    sc.pp.log1p(out)
    return out


# ---------------------------------------------------------------------------
# Per-panel runner
# ---------------------------------------------------------------------------
def run_panel(
    adata_norm,
    panel_name: str,
    symbols: list[str],
    day: str,
    out_root: Path,
    n_perms: int,
) -> dict:
    panel_dir = out_root / day / panel_name
    panel_dir.mkdir(parents=True, exist_ok=True)

    n_input = len(symbols)
    n_cells = adata_norm.n_obs

    detected, dropped_df = resolve_panel_genes(adata_norm, symbols)
    dropped_df.to_csv(panel_dir / "dropped_genes.csv", index=False)

    summary_row = {
        "day": day,
        "panel": panel_name,
        "n_input_genes": n_input,
        "n_detected_genes": len(detected),
        "n_cells": n_cells,
        "n_clusters": 0,
        "n_clusters_with_sig_hits": 0,
        "total_sig_ligands": 0,
        "notable_ligands": "",
        "skip_reason": "",
    }

    if len(detected) < 3:
        msg = f"<3 detected panel genes ({len(detected)}); skipping."
        print(f"[{day}/{panel_name}] {msg}")
        summary_row["skip_reason"] = msg
        return summary_row

    print(
        f"[{day}/{panel_name}] {len(detected)}/{n_input} genes detected; "
        f"clustering on {n_cells} cells"
    )

    adata_panel = adata_norm[:, detected].copy()
    sc.pp.scale(adata_panel)

    n_comps = min(20, len(detected) - 1, adata_panel.n_obs - 1)
    sc.pp.pca(adata_panel, n_comps=n_comps, random_state=RANDOM_STATE)
    sc.pp.neighbors(adata_panel, n_neighbors=15, random_state=RANDOM_STATE)
    sc.tl.leiden(
        adata_panel,
        resolution=1.0,
        n_iterations=2,
        flavor="igraph",
        directed=False,
        random_state=RANDOM_STATE,
    )
    sc.tl.umap(adata_panel, random_state=RANDOM_STATE)

    X_scaled = adata_panel.X
    if sp.issparse(X_scaled):
        X_scaled = X_scaled.toarray()
    adata_panel.obs["panel_score"] = np.asarray(X_scaled).mean(axis=1)

    clusters = sorted(adata_panel.obs["leiden"].unique(), key=lambda c: int(c))
    cluster_sizes = (
        adata_panel.obs["leiden"].value_counts().rename_axis("cluster").reset_index(name="n_cells")
    )
    cluster_sizes["cluster"] = cluster_sizes["cluster"].astype(int)
    cluster_sizes = cluster_sizes.sort_values("cluster").reset_index(drop=True)
    cluster_sizes.to_csv(panel_dir / "cluster_sizes.csv", index=False)

    # UMAP colored by Leiden
    fig, ax = plt.subplots(figsize=(6, 6))
    sc.pl.umap(
        adata_panel,
        color=["leiden"],
        frameon=False,
        legend_loc="on data",
        size=20,
        palette="Spectral",
        show=False,
        ax=ax,
        legend_fontoutline=4,
        legend_fontsize="medium",
    )
    fig.savefig(panel_dir / "umap_clusters.pdf", bbox_inches="tight")
    plt.close(fig)

    # UMAP colored by panel score
    fig, ax = plt.subplots(figsize=(6, 6))
    sc.pl.umap(
        adata_panel,
        color=["panel_score"],
        frameon=False,
        size=20,
        cmap="viridis",
        show=False,
        ax=ax,
    )
    fig.savefig(panel_dir / "umap_score.pdf", bbox_inches="tight")
    plt.close(fig)

    # Dotplot of panel genes x leiden, using log-normalized (not scaled) expression.
    adata_norm_local = adata_norm[:, detected].copy()
    adata_norm_local.obs["leiden"] = adata_panel.obs["leiden"].values
    fig = sc.pl.dotplot(
        adata_norm_local,
        var_names=detected,
        groupby="leiden",
        standard_scale="var",
        show=False,
        return_fig=True,
    )
    fig.savefig(panel_dir / "panel_dotplot.pdf", bbox_inches="tight")
    plt.close("all")

    # Permutation enrichment per cluster
    all_stats = []
    n_clusters_with_sig = 0
    for c in clusters:
        print(f"[{day}/{panel_name}] cluster {c}...", flush=True)
        fig, stats_df = permutation_cluster_enrichment(
            adata_panel,
            cluster_number=c,
            perturbation_key="perturbation",
            control_label="NTC",
            leiden_key="leiden",
            n_permutations=n_perms,
            alpha=0.05,
            min_cells_per_perturb=20,
            random_state=RANDOM_STATE,
        )
        if not stats_df.empty:
            all_stats.append(stats_df)
        if fig is not None:
            fig.savefig(
                panel_dir / f"cluster_{c}_significant_ligands.pdf",
                bbox_inches="tight",
            )
            plt.close(fig)
            n_clusters_with_sig += 1
        plt.close("all")

    if all_stats:
        full_stats = pd.concat(all_stats, ignore_index=True)
        sig_stats = full_stats[full_stats["direction"] != "none"].copy()
        out_df = pd.DataFrame(
            {
                "perturbation": sig_stats["perturbation"],
                "cluster": sig_stats["cluster"],
                "delta_pct": sig_stats["delta_prop"] * 100.0,
                "p_enrich": sig_stats["p_enrich"],
                "p_deplete": sig_stats["p_deplete"],
                "q_enrich": sig_stats["p_enrich_adj"],
                "q_deplete": sig_stats["p_deplete_adj"],
                "n_cells": sig_stats["n_cells"],
                "direction": sig_stats["direction"],
            }
        ).sort_values(["cluster", "delta_pct"], ascending=[True, False])
        out_df.to_csv(panel_dir / "significant_hits.csv", index=False)

        notable = (
            out_df.reindex(out_df["delta_pct"].abs().sort_values(ascending=False).index)
            .head(5)["perturbation"]
            .tolist()
        )
        summary_row["total_sig_ligands"] = int(len(out_df))
        summary_row["notable_ligands"] = ";".join(notable)
    else:
        pd.DataFrame(
            columns=[
                "perturbation", "cluster", "delta_pct", "p_enrich", "p_deplete",
                "q_enrich", "q_deplete", "n_cells", "direction",
            ]
        ).to_csv(panel_dir / "significant_hits.csv", index=False)

    summary_row["n_clusters"] = len(clusters)
    summary_row["n_clusters_with_sig_hits"] = n_clusters_with_sig
    return summary_row


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--h5ad", required=True, type=Path)
    p.add_argument("--day", required=True, choices=["Day4", "Day6"])
    p.add_argument(
        "--panel",
        required=True,
        choices=["PL", "EC", "ME", "EN", "MS", "all"],
    )
    p.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Root output dir; per-day/per-panel subdirs created underneath.",
    )
    p.add_argument("--n-permutations", type=int, default=10000)
    return p.parse_args()


def main():
    args = parse_args()
    sc.settings.verbosity = 1
    np.random.seed(RANDOM_STATE)

    print(f"Loading {args.h5ad}")
    adata = sc.read_h5ad(args.h5ad)
    print(f"  shape: {adata.shape}")

    adata_norm = prepare_normalized_adata(adata)
    del adata

    panels_to_run = (
        ["PL", "EC", "ME", "EN", "MS"] if args.panel == "all" else [args.panel]
    )

    day_dir = args.output_dir / args.day
    day_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for panel in panels_to_run:
        row = run_panel(
            adata_norm=adata_norm,
            panel_name=panel,
            symbols=TSANKOV_PANELS[panel],
            day=args.day,
            out_root=args.output_dir,
            n_perms=args.n_permutations,
        )
        rows.append(row)

    summary = pd.DataFrame(rows)
    summary_path = day_dir / "summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"\nWrote {summary_path}\n")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
