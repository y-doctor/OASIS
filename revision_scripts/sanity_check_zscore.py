"""Sanity check: prove the matrix fed into sc.pp.neighbors is NTC-z-scored,
not log-normalized."""

import sys
from pathlib import Path
import numpy as np
import scanpy as sc
import anndata as ad
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from revision_feature_set_analysis import (
    TSANKOV_PANELS,
    detected_panel_genes,
    prepare_log_adata,
    ntc_zscore,
    NTC_LABEL,
    PERTURBATION_KEY,
)

H5AD = "/tscc/projects/ps-malilab/ydoctor/Ligandome_Screens/Perturb_Seq_Analysis/anndata_objects/Day6_Ligandome_Aggregate_Post_Energy_Test.h5ad"

adata = sc.read_h5ad(H5AD)

per_panel = {p: detected_panel_genes(adata, s)[0] for p, s in TSANKOV_PANELS.items()}
union_genes = []
for det in per_panel.values():
    for g in det:
        if g not in union_genes:
            union_genes.append(g)

adata_log = prepare_log_adata(adata)

# --- Log-normalized values on union genes (what we would feed if we DIDN'T z-score) ---
idx = [adata_log.var_names.get_loc(g) for g in union_genes]
X_log_panel = adata_log.X[:, idx]
if hasattr(X_log_panel, "toarray"):
    X_log_panel = X_log_panel.toarray()
X_log_panel = np.asarray(X_log_panel, dtype=np.float32)

# --- NTC-z-scored values (what we actually feed) ---
Z, kept_union = ntc_zscore(adata_log, union_genes)

# --- Compare ---
print("LOG-NORMALIZED on union genes:")
print(f"  overall mean = {X_log_panel.mean():.4f}")
print(f"  overall std  = {X_log_panel.std():.4f}")
print(f"  per-gene mean range: [{X_log_panel.mean(axis=0).min():.3f}, {X_log_panel.mean(axis=0).max():.3f}]")
print(f"  per-gene std range:  [{X_log_panel.std(axis=0).min():.3f}, {X_log_panel.std(axis=0).max():.3f}]")

print("\nNTC-Z-SCORED (Z):")
print(f"  overall mean = {Z.mean():.4f}")
print(f"  overall std  = {Z.std():.4f}")
print(f"  per-gene mean range (all cells): [{Z.mean(axis=0).min():.3f}, {Z.mean(axis=0).max():.3f}]")
print(f"  per-gene std range  (all cells): [{Z.std(axis=0).min():.3f}, {Z.std(axis=0).max():.3f}]")

# Most telling: per-gene mean/std over NTC cells specifically
ntc_mask = (adata_log.obs[PERTURBATION_KEY].astype(str) == NTC_LABEL).values
print(f"\nNTC-only per-gene stats on Z (should be ≈0 mean, ≈1 std by construction):")
print(f"  per-gene mean (NTC): [{Z[ntc_mask].mean(axis=0).min():.4f}, {Z[ntc_mask].mean(axis=0).max():.4f}]")
print(f"  per-gene std  (NTC): [{Z[ntc_mask].std(axis=0).min():.4f}, {Z[ntc_mask].std(axis=0).max():.4f}]")

# --- Build adata_panel exactly as the script does, feed to sc.pp.neighbors, inspect what got used ---
adata_panel = ad.AnnData(X=Z, obs=adata_log.obs.copy(), var=pd.DataFrame(index=kept_union))
print(f"\nadata_panel.X stats (this is what sc.pp.neighbors will see with use_rep='X'):")
x = adata_panel.X
if hasattr(x, "toarray"):
    x = x.toarray()
x = np.asarray(x)
print(f"  mean = {x.mean():.4f}, std = {x.std():.4f}")
print(f"  min  = {x.min():.4f}, max = {x.max():.4f}")

# Sanity: are adata_panel.X and Z byte-identical?
print(f"\nadata_panel.X is byte-identical to Z? {np.array_equal(np.asarray(adata_panel.X), Z)}")
