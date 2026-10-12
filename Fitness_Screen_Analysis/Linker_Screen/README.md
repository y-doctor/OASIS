# Linker Screen

Pooled OASIS ligandome fitness screen in iPSCs that compares linker variants between the ligand and its anchor: **NoLinker**, **15GS**, **30GS** and **XTEN**.

## Design

- **Library:** 770 barcoded constructs (521 ligands plus controls). Each barcode is an independent construct.
- **Samples (20):** `Day4` and `Day14` for each linker with 2 replicates, plus one `Plasmid` library pool per linker.

## Processing

1. **Counting.** Barcodes were counted with `mageck count` from read 1 in sense orientation (11-nt barcode, 5′ trim 14–16).
2. **Sample labels.** The sequencing core's sample sheet had an off-by-one shift in sample names. Labels were corrected by matching each file's in-read i7/i5 indices to the true index table. All files here use the corrected labels.
3. **CPM filter (per linker).** A construct was dropped if its CPM was below 1 in that linker's `Day4_r1`, `Day4_r2` or `Plasmid` sample. 758–761 of 770 constructs were kept (see `mageck_test/cpm_filter_summary.csv`).
4. **Testing.** 8 `mageck test` comparisons were run, at the **per-construct** level (gene = barcode ID) and median-normalized. Each test was FDR-corrected on its own. For each linker: `Day14 vs Day4` and `Day14 vs Plasmid`.
5. **Hits.** `LFC_zscore = (LFC − mean) / sd` within each comparison. A hit has **|Z| > 2 and FDR < 0.05**.

## Files

| Path | Contents |
|---|---|
| `mageck_count/linker.count.perconstruct.txt` | Raw per-construct counts, all 20 samples |
| `mageck_count/cpm_filtered/linker_<linker>_cpmfilt.count.txt` | Counts after the CPM filter, for each linker (test input) |
| `mageck_test/<comparison>.sgrna_summary.txt` | Raw MAGeCK sgRNA (= construct) output for each of the 8 comparisons |
| `mageck_test/<comparison>.gene_summary.txt` | Raw MAGeCK RRA output for each comparison (gene = construct) |
| `mageck_test/cpm_filter_summary.csv` | Constructs kept and dropped for each linker |
| `lfc_tables/linker_sgrna_all8_CPMfiltered.csv` | Long table of all 8 comparisons: LFC, LFC_zscore, FDR, hit call, ligand and linker annotation |
| `lfc_tables/linker_sgrna_summaries_CPMfiltered.xlsx` | The same data with one sheet per comparison, plus Summary, Methods and CPM_filter sheets |
