"""Diagnose why Day 4 has 76 leiden clusters with many "pale" signatures:
- Is leiden partitioning on channel/batch?
- What fraction of cells have any marker z > 1 (meaningful commitment)?
- SDC2: is it a genuine lineage marker here or a noise-driven pseudo-signal?
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad

sys.path.insert(0, str(Path(__file__).resolve().parent))
from revision_feature_set_analysis import (  # noqa: E402
    TSANKOV_PANELS,
    NTC_LABEL,
    PERTURBATION_KEY,
    RANDOM_STATE,
    detected_panel_genes,
    prepare_log_adata,
    ntc_zscore,
)

H5AD = "/tscc/projects/ps-malilab/ydoctor/Ligandome_Screens/Perturb_Seq_Analysis/anndata_objects/Day4_Ligandome_Aggregate_Post_Energy_Test.h5ad"
OUT = Path("/tscc/projects/ps-malilab/ydoctor/Ligandome_Screens/revision_figures/feature_set_analysis/Day4")


def main():
    adata = sc.read_h5ad(H5AD)
    print(f"Loaded {adata.shape}")

    per_panel_detected = {}
    for panel_name, symbols in TSANKOV_PANELS.items():
        det, _ = detected_panel_genes(adata, symbols)
        per_panel_detected[panel_name] = det

    union_genes = []
    for det in per_panel_detected.values():
        for g in det:
            if g not in union_genes:
                union_genes.append(g)

    adata_log = prepare_log_adata(adata)
    Z, kept_union = ntc_zscore(adata_log, union_genes)

    adata_panel = ad.AnnData(X=Z, obs=adata_log.obs.copy(), var=pd.DataFrame(index=kept_union))
    sc.pp.neighbors(adata_panel, n_neighbors=30, use_rep="X", random_state=RANDOM_STATE)
    sc.tl.leiden(adata_panel, resolution=1.0, n_iterations=2, flavor="igraph", directed=False, random_state=RANDOM_STATE)
    sc.tl.umap(adata_panel, random_state=RANDOM_STATE, min_dist=0.8)

    # --- 1. UMAP by channel (batch confound check) ---
    fig, ax = plt.subplots(figsize=(7, 7))
    sc.pl.umap(adata_panel, color=["channel"], frameon=False, size=6, show=False, ax=ax, legend_fontsize="small")
    fig.savefig(OUT / "diag_umap_by_channel.pdf", bbox_inches="tight")
    plt.close(fig)

    # Contingency: channel x leiden (how much does channel determine cluster?)
    ct = pd.crosstab(adata_panel.obs["channel"], adata_panel.obs["leiden"])
    ct.to_csv(OUT / "diag_channel_vs_cluster.csv")
    # Per-cluster: fraction from single most-represented channel
    per_cluster_top_channel_frac = (ct.max(axis=0) / ct.sum(axis=0)).sort_values(ascending=False)
    per_cluster_top_channel_frac.to_csv(OUT / "diag_channel_dominance_per_cluster.csv", header=["top_channel_frac"])
    print(f"Cluster-channel dominance distribution:")
    print(f"  fraction of clusters >90% single channel: {(per_cluster_top_channel_frac > 0.9).mean():.2%}")
    print(f"  fraction >75%: {(per_cluster_top_channel_frac > 0.75).mean():.2%}")
    print(f"  fraction >50%: {(per_cluster_top_channel_frac > 0.5).mean():.2%}")
    print(f"  median dominance: {per_cluster_top_channel_frac.median():.2%}")

    # --- 2. Commitment depth ---
    max_z_per_cell = Z.max(axis=1)
    n_committed = int((max_z_per_cell > 1).sum())
    n_strongly = int((max_z_per_cell > 2).sum())
    print(f"\nCommitment depth (Day 4, {Z.shape[0]} cells):")
    print(f"  max_z > 1 in any marker: {n_committed} ({100*n_committed/Z.shape[0]:.1f}%)")
    print(f"  max_z > 2 in any marker: {n_strongly} ({100*n_strongly/Z.shape[0]:.1f}%)")

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(max_z_per_cell, bins=80, color="#1982C4", alpha=0.8)
    ax.axvline(1, color="black", linestyle="--", alpha=0.5, label="z=1")
    ax.axvline(2, color="red", linestyle="--", alpha=0.5, label="z=2")
    ax.set_xlabel("max NTC-z across panel genes, per cell")
    ax.set_ylabel("cells")
    ax.set_title("Day 4: commitment-depth distribution")
    ax.legend()
    fig.savefig(OUT / "diag_max_z_histogram.pdf", bbox_inches="tight")
    plt.close(fig)

    # --- 3. Per-cluster mean max_z (are pale clusters really pale?) ---
    adata_panel.obs["max_z"] = max_z_per_cell
    cluster_max_z = adata_panel.obs.groupby("leiden")["max_z"].agg(["mean", "median", "std", "count"])
    cluster_max_z = cluster_max_z.sort_values("mean")
    cluster_max_z.to_csv(OUT / "diag_cluster_commitment.csv")
    print(f"\nPer-cluster commitment (sorted by mean max_z):")
    print(cluster_max_z.head(10).round(3))
    print(f"...")
    print(cluster_max_z.tail(5).round(3))

    # --- 4. SDC2 investigation ---
    if "SDC2" in kept_union:
        sdc2_idx = kept_union.index("SDC2")
        sdc2_z = Z[:, sdc2_idx]
        adata_panel.obs["SDC2_z"] = sdc2_z
        print(f"\nSDC2 overall: mean {sdc2_z.mean():.3f}, std {sdc2_z.std():.3f}, max {sdc2_z.max():.3f}")
        # How many cells have SDC2 z > 1?
        sdc2_high = (sdc2_z > 1).sum()
        print(f"  SDC2 z > 1: {sdc2_high} cells ({100*sdc2_high/len(sdc2_z):.1f}%)")
        # Per-cluster SDC2 mean
        cluster_sdc2 = adata_panel.obs.groupby("leiden")["SDC2_z"].mean().sort_values(ascending=False)
        print(f"  Top 5 clusters by SDC2 mean: {cluster_sdc2.head().to_dict()}")
        print(f"  Bot 5: {cluster_sdc2.tail().to_dict()}")

    # --- 5. UMAP colored by max_z so "pale" clusters visible ---
    fig, ax = plt.subplots(figsize=(7, 7))
    sc.pl.umap(adata_panel, color=["max_z"], frameon=False, size=6, show=False, ax=ax, cmap="viridis", vmin=0, vmax=3)
    fig.savefig(OUT / "diag_umap_by_max_z.pdf", bbox_inches="tight")
    plt.close(fig)

    print("\nDone.")


if __name__ == "__main__":
    main()
