# T-Cell Screen

Pooled OASIS ligandome fitness screen in primary human T cells (PBMC-derived) under **Acute** stimulation.

## Design

- **Library:** 770 barcoded constructs (521 ligands plus controls). Each barcode is an independent construct.
- **Samples:** Acute condition with 2 replicates, compared against the plasmid library pool (`CXL-plasmid`).

## Analysis

`mageck test` was run at the per-construct level (one row per barcode) for **Acute vs plasmid**.

## Files

| Path | Contents |
|---|---|
| `mageck_test/PBMC_Acute_vs_CXL-plasmid.sgrna_summary.xlsx` | Raw MAGeCK sgRNA (= construct) output, Acute vs plasmid |
| `lfc_tables/PBMC_Acute_replicates_combined.xlsx` | LFC vs plasmid for each replicate (`LFC_r1`, `LFC_r2`) |
