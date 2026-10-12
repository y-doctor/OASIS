# E6 vs mTeSR Media Timecourse Screen

Pooled OASIS ligandome fitness screen in iPSCs comparing **mTeSR** (complete) and **E6** (FGF2/TGFβ-free) media over a time course.

## Design

- **Library:** 770 barcoded constructs (521 ligands plus controls). Each barcode is an independent construct.
- **Samples:** a shared `Day4` baseline, then `{mTeSR, E6}` at `Day7`, `Day9` and `Day14`, with 2 replicates each. The `NGS_plasmid` library pool is included as a reference.

## Processing

1. **Counting.** Barcodes were counted with `mageck count` from read 2 in antisense (reverse-complement) orientation.
2. **CPM filter.** A construct was dropped if its CPM was below 1 in `Day4_r1`, `Day4_r2` or `NGS_plasmid`. **744 of 770** constructs were kept.
3. **Testing.** 15 `mageck test` comparisons were run, all at the **per-construct** level (gene = barcode ID, one construct per gene) and median-normalized. Each test was FDR-corrected on its own.
   - E6 and mTeSR at each of D7, D9 and D14 **vs Day4**
   - E6 and mTeSR at each of D7, D9 and D14 **vs Plasmid**
   - **E6 vs mTeSR** at D7, D9 and D14. mTeSR is the control, so a positive LFC means the construct is enriched in E6.
4. **Hits.** `LFC_zscore = (LFC − mean) / sd` within each comparison. A hit has **|Z| > 2 and FDR < 0.05** (sgRNA-level FDR).

## Files

| Path | Contents |
|---|---|
| `mageck_count/e6_mtesr.count.perconstruct.txt` | Raw per-construct counts, all 15 samples |
| `mageck_count/cpm_filtered/e6_mtesr_cpmfilt.perconstruct.count.txt` | Counts after the CPM filter, used as test input |
| `mageck_test/<comparison>.sgrna_summary.txt` | Raw MAGeCK sgRNA (= construct) output for each of the 15 comparisons |
| `mageck_test/<comparison>.gene_summary.txt` | Raw MAGeCK RRA output for each comparison (gene = construct) |
| `mageck_test/cpm_filter_summary.csv` | Number of constructs kept and dropped by the CPM filter |
| `lfc_tables/e6_mtesr_sgrna_all15_CPMfiltered.csv` | Long table of all 15 comparisons: LFC, LFC_zscore, FDR, hit call, ligand annotation |
| `lfc_tables/e6_mtesr_sgrna_summaries_CPMfiltered.xlsx` | The same data with one sheet per comparison, plus Summary, Methods and CPM_filter sheets |
