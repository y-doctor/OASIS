"""Post-processing: produce a matrixplot of mean NTC z-score per (cluster, gene)
with a diverging colormap centered at 0 (= NTC baseline). Makes it visually
obvious which clusters are differentiated relative to NTC and which are not.

Reuses the same clustering logic as revision_feature_set_analysis.py
(deterministic with seed=42), so cluster IDs match.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
import anndata as ad

# Pull in panel definitions and helpers from the main script.
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


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--h5ad", required=True, type=Path)
    p.add_argument("--day", required=True, choices=["Day4", "Day6"])
    p.add_argument("--output-dir", required=True, type=Path)
    args = p.parse_args()

    out_dir = args.output_dir / args.day
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.h5ad}")
    adata = sc.read_h5ad(args.h5ad)
    print(f"  shape: {adata.shape}")

    # Per-panel detection on raw counts
    per_panel_detected: dict[str, list[str]] = {}
    for panel_name, symbols in TSANKOV_PANELS.items():
        det, _ = detected_panel_genes(adata, symbols)
        per_panel_detected[panel_name] = det

    union_genes: list[str] = []
    for det in per_panel_detected.values():
        for g in det:
            if g not in union_genes:
                union_genes.append(g)

    adata_log = prepare_log_adata(adata)
    Z, kept_union = ntc_zscore(adata_log, union_genes)
    kept_set = set(kept_union)
    panel_membership = {p: [g for g in genes if g in kept_set] for p, genes in per_panel_detected.items()}

    adata_panel = ad.AnnData(
        X=Z,
        obs=adata_log.obs.copy(),
        var=pd.DataFrame(index=kept_union),
    )

    sc.pp.neighbors(adata_panel, n_neighbors=30, use_rep="X", random_state=RANDOM_STATE)
    sc.tl.leiden(
        adata_panel,
        resolution=1.0,
        n_iterations=2,
        flavor="igraph",
        directed=False,
        random_state=RANDOM_STATE,
    )

    var_group = {p: panel_membership[p] for p in TSANKOV_PANELS if panel_membership[p]}

    # Matrixplot of mean NTC-z per cluster x gene; diverging cmap centered at 0.
    mp = sc.pl.matrixplot(
        adata_panel,
        var_names=var_group,
        groupby="leiden",
        cmap="RdBu_r",
        vcenter=0,
        vmin=-2,
        vmax=2,
        show=False,
        return_fig=True,
        colorbar_title="mean NTC z-score",
    )
    out_path = out_dir / "panel_matrixplot_ntcz.pdf"
    mp.savefig(out_path, bbox_inches="tight")
    plt.close("all")
    print(f"Wrote {out_path}")

    # Also a dotplot version: color = mean NTC z, size = % cells with z > 1
    # (= cells that look meaningfully above NTC baseline for that gene).
    # We need to override scanpy's default expression_cutoff for the size metric.
    dp = sc.pl.dotplot(
        adata_panel,
        var_names=var_group,
        groupby="leiden",
        cmap="RdBu_r",
        vcenter=0,
        vmin=-2,
        vmax=2,
        expression_cutoff=1.0,  # "% expressing" = % above 1 SD over NTC mean
        show=False,
        return_fig=True,
        colorbar_title="mean NTC z-score",
        size_title="% cells z > 1",
    )
    out_path = out_dir / "panel_dotplot_ntcz.pdf"
    dp.savefig(out_path, bbox_inches="tight")
    plt.close("all")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
