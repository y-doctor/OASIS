# Obligate Autocrine Stimulation in Situ (OASIS)

Data and analysis for **Multiplexed Pan Soluble Ligandome Screening via OASIS**.

## Repository layout

```
Fitness_Screen_Analysis/
  E6_mTeSR_Timecourse/   iPSC fitness screen, E6 vs mTeSR media over a D4→D7→D9→D14 time course
  Linker_Screen/         iPSC fitness screen comparing NoLinker / 15GS / 30GS / XTEN linkers
  T_Cell_Screen/         primary T-cell fitness screen, Acute stimulation
Perturb_Seq_Analysis/
  analysis_scripts/      Day 4 and Day 6 Perturb-seq analysis notebooks
```

### Fitness screens

Each fitness-screen folder has its own README covering the design, processing and file descriptions, and contains:

- `mageck_count/`: raw MAGeCK per-construct count tables (plus CPM-filtered versions for the E6/mTeSR and Linker screens)
- `mageck_test/`: raw MAGeCK test outputs (`sgrna_summary` / `gene_summary`) for each comparison
- `lfc_tables/`: raw log2 fold-change tables with z-scored LFC and hit calls, compiled across comparisons

All fitness-screen results are reported **per construct**: each barcode is tested as an independent construct and is not aggregated by ligand. Hits are defined as **|LFC z-score| > 2 and FDR < 0.05**.

### Perturb-seq

- `Perturb_Seq_Analysis/analysis_scripts/Day4_Analysis.ipynb` and `Day6_Analysis.ipynb` are the main Perturb-seq analysis notebooks.
- They need `scanpy` / `anndata` (plus `numpy`, `pandas`, `scipy`, `matplotlib`, `seaborn`) and the `psp` module from https://github.com/y-doctor/KOLF2.1J_Perturbation_Cell_Atlas.
- Single-cell data is provided pre-QC at the link given in the manuscript. The notebooks expect precomputed inputs, so adjust paths for your environment.
- To analyze the Perturb-seq datasets from scratch, we recommend 256 GB RAM and 32 CPU cores.
