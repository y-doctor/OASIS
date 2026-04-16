"""Re-cluster Perturb-seq data on the UNION of Tsankov 2015 lineage panels
(PL/EC/ME/EN/MS), then run permutation-based ligand enrichment per Leiden
cluster. One unified clustering + 5 per-cell panel-score overlays.

Z-scoring is anchored to NTC cells (mu/sigma computed on NTC only), so the
embedding reads as deviation-from-unperturbed-baseline. Per-cell panel
scores are means of NTC-z-scored values across each panel's detected genes.

Permutation test (10k perms, BH-FDR alpha=0.05, NTC null) copied verbatim
from Day{4,6}_Analysis.ipynb. Vectorized inner loop.

Usage:
    python revision_feature_set_analysis.py \
        --h5ad <path> --day {Day4,Day6} \
        --output-dir <path> [--n-permutations 10000]
"""

from __future__ import annotations

import argparse
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

NTC_LABEL = "NTC"
PERTURBATION_KEY = "perturbation"
RANDOM_STATE = 42


# ---------------------------------------------------------------------------
# permutation_cluster_enrichment — copied from Day{4,6}_Analysis.ipynb cell 35,
# with a vectorized null draw (one rng.choice for all permutations per pert).
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
    """Permutation test for perturbation enrichment/depletion in a Leiden cluster."""
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

        sample_idx = rng.choice(ntc_indices, size=(n_permutations, n_p), replace=True)
        null_counts = cluster_mask_arr[sample_idx].sum(axis=1)
        null_props = null_counts / n_p

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
    ax.set_title(f"Cluster {cluster_str}: perturbations enriched/depleted vs NTC", fontsize=16)
    ax.tick_params(labelsize=12)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(2.0)
    ax.spines["bottom"].set_linewidth(2.0)
    ax.grid(False)
    plt.tight_layout()

    return fig, stats_df


# ---------------------------------------------------------------------------
# Panel + normalization helpers
# ---------------------------------------------------------------------------
def detected_panel_genes(adata, symbols: list[str]) -> tuple[list[str], pd.DataFrame]:
    """Return panel genes present in var_names AND counts > 0 in >=1% of cells."""
    n_cells = adata.n_obs
    threshold = 0.01 * n_cells
    var_set = set(adata.var_names)
    counts = adata.layers["counts"] if "counts" in adata.layers else adata.X

    rows = []
    detected = []
    for g in symbols:
        if g not in var_set:
            rows.append({"gene": g, "reason": "missing"})
            continue
        col = counts[:, adata.var_names.get_loc(g)]
        n_expr = col.getnnz() if sp.issparse(col) else int(np.count_nonzero(np.asarray(col).ravel()))
        if n_expr < threshold:
            rows.append({"gene": g, "reason": f"below_1pct ({n_expr}/{n_cells} cells)"})
        else:
            detected.append(g)
            rows.append({"gene": g, "reason": "kept"})
    return detected, pd.DataFrame(rows)


def prepare_log_adata(adata):
    """Copy adata, set X = counts, normalize_total(1e4), log1p. Returns new AnnData."""
    out = adata.copy()
    if "counts" in out.layers:
        out.X = out.layers["counts"].copy()
    sc.pp.normalize_total(out, target_sum=1e4)
    sc.pp.log1p(out)
    return out


def ntc_zscore(adata_log, union_genes: list[str], clip: float = 10.0) -> tuple[np.ndarray, list[str]]:
    """Compute (X - mu_NTC) / sigma_NTC on union_genes; clip to ±`clip`.

    Returns (matrix [n_cells x n_kept_genes], list of kept gene symbols after
    dropping any gene with sigma_NTC == 0).
    """
    pert = adata_log.obs[PERTURBATION_KEY].astype(str)
    ntc_mask = (pert == NTC_LABEL).values
    if ntc_mask.sum() == 0:
        raise ValueError("No NTC cells found")

    idx = [adata_log.var_names.get_loc(g) for g in union_genes]
    X_panel = adata_log.X[:, idx]
    if sp.issparse(X_panel):
        X_panel = X_panel.toarray()
    X_panel = np.asarray(X_panel, dtype=np.float32)

    mu = X_panel[ntc_mask].mean(axis=0)
    sigma = X_panel[ntc_mask].std(axis=0)

    keep = sigma > 0
    if not keep.all():
        dropped = [g for g, k in zip(union_genes, keep) if not k]
        print(f"  dropped {len(dropped)} genes with zero NTC std: {dropped}")

    X_panel = X_panel[:, keep]
    mu = mu[keep]
    sigma = sigma[keep]
    Z = (X_panel - mu) / sigma
    np.clip(Z, -clip, clip, out=Z)
    kept_genes = [g for g, k in zip(union_genes, keep) if k]
    return Z, kept_genes


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------
def run(adata, day: str, out_dir: Path, n_perms: int) -> pd.DataFrame:
    """Run union-mode analysis. Returns a one-row summary DataFrame."""
    out_dir.mkdir(parents=True, exist_ok=True)

    # Per-panel detection (on raw counts)
    per_panel_detected: dict[str, list[str]] = {}
    dropped_rows = []
    for panel_name, symbols in TSANKOV_PANELS.items():
        det, drop_df = detected_panel_genes(adata, symbols)
        per_panel_detected[panel_name] = det
        drop_df.insert(0, "panel", panel_name)
        dropped_rows.append(drop_df)
        print(f"[{day}] panel {panel_name}: {len(det)}/{len(symbols)} detected")
    dropped_df = pd.concat(dropped_rows, ignore_index=True)
    dropped_df.to_csv(out_dir / "dropped_genes.csv", index=False)

    # Union of detected genes (preserve panel mapping for reporting)
    union_genes: list[str] = []
    panel_membership: dict[str, list[str]] = {}
    for panel_name, det in per_panel_detected.items():
        for g in det:
            if g not in union_genes:
                union_genes.append(g)
        panel_membership[panel_name] = det
    print(f"[{day}] union: {len(union_genes)} genes")

    # log-normalize full adata, then z-score against NTC
    adata_log = prepare_log_adata(adata)
    Z, kept_union = ntc_zscore(adata_log, union_genes)

    # Update per-panel membership to only kept (post-NTC-std-filter) genes
    kept_set = set(kept_union)
    panel_membership = {p: [g for g in genes if g in kept_set] for p, genes in panel_membership.items()}

    # Write union_genes.csv with panel membership
    rows = []
    for g in kept_union:
        panels_for_g = [p for p, genes in panel_membership.items() if g in genes]
        rows.append({"gene": g, "panels": ",".join(panels_for_g)})
    pd.DataFrame(rows).to_csv(out_dir / "union_genes.csv", index=False)

    # Build a panel-gene AnnData using the NTC-z-scored matrix as X
    import anndata as ad
    adata_panel = ad.AnnData(
        X=Z,
        obs=adata_log.obs.copy(),
        var=pd.DataFrame(index=kept_union),
    )

    sc.pp.neighbors(
        adata_panel,
        n_neighbors=30,
        use_rep="X",
        random_state=RANDOM_STATE,
    )
    sc.tl.leiden(
        adata_panel,
        resolution=1.0,
        n_iterations=2,
        flavor="igraph",
        directed=False,
        random_state=RANDOM_STATE,
    )
    sc.tl.umap(adata_panel, random_state=RANDOM_STATE, min_dist=0.8)

    # Per-panel scores: mean of NTC-z-scored expression across panel genes (kept).
    for panel_name, genes in panel_membership.items():
        if not genes:
            adata_panel.obs[f"score_{panel_name}"] = np.nan
            continue
        idx = [kept_union.index(g) for g in genes]
        adata_panel.obs[f"score_{panel_name}"] = Z[:, idx].mean(axis=1)

    # Cluster sizes
    clusters = sorted(adata_panel.obs["leiden"].unique(), key=lambda c: int(c))
    cs = adata_panel.obs["leiden"].value_counts().rename_axis("cluster").reset_index(name="n_cells")
    cs["cluster"] = cs["cluster"].astype(int)
    cs.sort_values("cluster").to_csv(out_dir / "cluster_sizes.csv", index=False)

    # UMAP: clusters
    fig, ax = plt.subplots(figsize=(7, 7))
    sc.pl.umap(
        adata_panel,
        color=["leiden"],
        frameon=False,
        legend_loc="on data",
        size=15,
        palette="Spectral",
        show=False,
        ax=ax,
        legend_fontoutline=4,
        legend_fontsize="small",
    )
    fig.savefig(out_dir / "umap_clusters.pdf", bbox_inches="tight")
    plt.close(fig)

    # UMAPs: per-panel scores (one PDF each)
    for panel_name in TSANKOV_PANELS:
        fig, ax = plt.subplots(figsize=(6, 6))
        sc.pl.umap(
            adata_panel,
            color=[f"score_{panel_name}"],
            frameon=False,
            size=15,
            cmap="viridis",
            show=False,
            ax=ax,
            title=f"{panel_name} score (mean NTC-z, n={len(panel_membership[panel_name])} genes)",
        )
        fig.savefig(out_dir / f"umap_score_{panel_name}.pdf", bbox_inches="tight")
        plt.close(fig)

    # Dotplot: union genes grouped by panel × leiden, log-normalized expression.
    adata_norm_local = adata_log[:, kept_union].copy()
    adata_norm_local.obs["leiden"] = adata_panel.obs["leiden"].values
    var_group = {p: panel_membership[p] for p in TSANKOV_PANELS if panel_membership[p]}
    fig = sc.pl.dotplot(
        adata_norm_local,
        var_names=var_group,
        groupby="leiden",
        standard_scale="var",
        show=False,
        return_fig=True,
    )
    fig.savefig(out_dir / "panel_dotplot.pdf", bbox_inches="tight")
    plt.close("all")

    # Permutation enrichment per cluster
    all_stats = []
    n_clusters_with_sig = 0
    for c in clusters:
        print(f"[{day}] cluster {c}/{len(clusters)-1}...", flush=True)
        fig, stats_df = permutation_cluster_enrichment(
            adata_panel,
            cluster_number=c,
            perturbation_key=PERTURBATION_KEY,
            control_label=NTC_LABEL,
            leiden_key="leiden",
            n_permutations=n_perms,
            alpha=0.05,
            min_cells_per_perturb=20,
            random_state=RANDOM_STATE,
        )
        if not stats_df.empty:
            all_stats.append(stats_df)
        if fig is not None:
            fig.savefig(out_dir / f"cluster_{c}_significant_ligands.pdf", bbox_inches="tight")
            plt.close(fig)
            n_clusters_with_sig += 1
        plt.close("all")

    # significant_hits.csv + full enrichment table (every perturbation tested)
    if all_stats:
        full_stats = pd.concat(all_stats, ignore_index=True)
        full_out = pd.DataFrame(
            {
                "perturbation": full_stats["perturbation"],
                "cluster": full_stats["cluster"],
                "delta_pct": full_stats["delta_prop"] * 100.0,
                "p_enrich": full_stats["p_enrich"],
                "p_deplete": full_stats["p_deplete"],
                "q_enrich": full_stats["p_enrich_adj"],
                "q_deplete": full_stats["p_deplete_adj"],
                "n_cells": full_stats["n_cells"],
                "prop_pert": full_stats["prop_pert"],
                "prop_ntc": full_stats["prop_ntc"],
                "direction": full_stats["direction"],
            }
        ).sort_values(["cluster", "delta_pct"], ascending=[True, False])
        full_out.to_csv(out_dir / "all_enrichment_stats.csv", index=False)

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
        out_df.to_csv(out_dir / "significant_hits.csv", index=False)
        total_sig = int(len(out_df))
        notable = (
            out_df.reindex(out_df["delta_pct"].abs().sort_values(ascending=False).index)
            .head(10)["perturbation"]
            .tolist()
        )
    else:
        pd.DataFrame(
            columns=[
                "perturbation", "cluster", "delta_pct", "p_enrich", "p_deplete",
                "q_enrich", "q_deplete", "n_cells", "direction",
            ]
        ).to_csv(out_dir / "significant_hits.csv", index=False)
        total_sig = 0
        notable = []

    summary = pd.DataFrame(
        [
            {
                "day": day,
                "n_input_total": sum(len(s) for s in TSANKOV_PANELS.values()),
                "n_union_input": len(set(g for s in TSANKOV_PANELS.values() for g in s)),
                "n_union_detected": len(kept_union),
                "n_cells": adata_panel.n_obs,
                "n_clusters": len(clusters),
                "n_clusters_with_sig_hits": n_clusters_with_sig,
                "total_sig_ligands": total_sig,
                "notable_ligands": ";".join(notable),
            }
        ]
    )
    return summary


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--h5ad", required=True, type=Path)
    p.add_argument("--day", required=True, choices=["Day4", "Day6"])
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--n-permutations", type=int, default=10000)
    return p.parse_args()


def main():
    args = parse_args()
    sc.settings.verbosity = 1
    np.random.seed(RANDOM_STATE)

    print(f"Loading {args.h5ad}")
    adata = sc.read_h5ad(args.h5ad)
    print(f"  shape: {adata.shape}")

    out_dir = args.output_dir / args.day
    summary = run(adata, day=args.day, out_dir=out_dir, n_perms=args.n_permutations)

    summary_path = out_dir / "summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"\nWrote {summary_path}\n")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
