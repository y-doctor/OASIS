# T-Cell Screen

Pooled OASIS ligandome fitness screen in primary human T cells (PBMC-derived) under **Acute** stimulation.

## Design

- **Library:** 770 barcoded constructs: 764 ligand constructs (one ligand per construct, derived from 515 unique genes of origin) plus 6 control constructs (AMPR, CLUC, GLUC, HALO, mCherry, SNAP). Each barcode is an independent construct.
- **Samples:** Acute condition with 2 replicates, compared against the plasmid library pool (`CXL-plasmid`).

## Analysis

Barcodes were counted with `mageck count` from read 1, one sample at a time. `mageck test` was then run at the per-construct level (one row per barcode) for **Acute vs plasmid**.

## Files

| Path | Contents |
|---|---|
| `mageck_count/tcell_acute.count.perconstruct.txt` | Raw per-construct counts for `CXL-ligand-plasmid`, `Tcells-A-r1` and `Tcells-A-r2`, merged into one table |
| `mageck_count/per_sample/Read1_count_*.count.txt` | The original per-sample `mageck count` outputs that the merged table was built from |
| `mageck_test/PBMC_Acute_vs_CXL-plasmid.sgrna_summary.xlsx` | Raw MAGeCK sgRNA (= construct) output, Acute vs plasmid |
| `lfc_tables/PBMC_Acute_replicates_combined.xlsx` | LFC vs plasmid for each replicate (`LFC_r1`, `LFC_r2`) |
