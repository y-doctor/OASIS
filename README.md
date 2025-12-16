# Ligandome Screens (OASIS)

Data processing and analysis for **Multiplexed Pan Soluble Ligandome Screening via OASIS**, including:

- **Fitness screen** (MaGeCK-based analysis and summary tables)
- **Perturb-seq / single-cell** analysis (Scanpy/AnnData workflows)

## Repository layout

- **`Fitness_Screen_Analysis/`**: Fitness screen analysis and supporting data
  - `analysis.ipynb`: Main analysis notebook
  - `global_r1_vs_r2.csv`: Replicate comparison table
  - `iPSC Ligand Fitness Screen Data.xlsx`: MaGeCK output comparing all conditions to Day 4 mTeSR condition
  - `ALL_vs_NGSplasmid.sgrna_summary.xlsx`: MaGeCK output comparing all conditions to plasmid pool 
- **`Perturb_Seq_Analysis/analysis_scripts/`**: Perturb-seq analysis notebooks
  - `Day4_Analysis.ipynb`, `Day6_Analysis.ipynb`: Main analysis notebooks for Day4/6 respectively

## Requirements / environment notes

This repo mixes notebooks and large binary inputs/outputs. To run the **Perturb-seq** notebooks you’ll need:

- **Python packages**: `scanpy`/`anndata` (+ typical scientific stack: `numpy`, `pandas`, `scipy`, `matplotlib`, `seaborn`)
- **External dependency**: psp module from https://github.com/y-doctor/KOLF2.1J_Perturbation_Cell_Atlas


## How to run

### Fitness screen analysis
- Open and run `Fitness_Screen_Analysis/analysis.ipynb` in Jupyter.

### Perturb-seq analysis (notebooks)
- Open `Perturb_Seq_Analysis/analysis_scripts/Day4_Analysis.ipynb` and/or `Day6_Analysis.ipynb`.
- The notebooks expect precomputed inputs and write intermediate figures locally; adjust paths as needed for your environment.
- Data is provided pre-QC at the link found in the manuscript.  

## Notes
- To analyze perturb-seq datasets from scratch we reccomend use of 256GB RAM and 32 CPU cores
