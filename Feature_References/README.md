# Feature References

Barcode references for the OASIS ligand library: **770 barcoded constructs**, each with a unique 11-nt barcode. The constructs are 764 ligand constructs (one ligand per construct, derived from 515 unique genes of origin) plus 6 controls (AMPR, CLUC, GLUC, HALO, mCherry, SNAP). Both files list the same 770 barcodes in the sense orientation.

| File | Format | Used for |
|---|---|---|
| `mageck_Ligand_Feature_Reference.txt` | MAGeCK `--list-seq` library, tab-separated with no header: `construct_id`, `barcode`, `construct_id` (the third column is the "gene", one per construct) | Fitness screen barcode counting (`mageck count`) |
| `10X_Ligand_Feature_Reference.csv` | 10x Genomics Cell Ranger feature reference: barcode on read 2, pattern `(BC)TTGCTAGGACCGGCCTTAAA` | Perturb-seq ligand-barcode calling |

## Read orientation per fitness screen

| Screen | Read counted | Barcode orientation |
|---|---|---|
| Linker | Read 1 | Sense: use the reference as is |
| E6/mTeSR timecourse | Read 2 | Antisense: reverse-complement the barcodes |

Construct IDs in the 10x reference leave out the `_HUMAN` suffix (for example, `GS0001_A4` in the 10x file is `GS0001_A4_HUMAN` in the MAGeCK file); control IDs are the same in both files. The barcode sequences are identical in both files.
