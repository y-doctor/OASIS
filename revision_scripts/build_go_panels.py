"""Build expanded lineage panels (PL/EC/ME/EN/MS) by combining:
  - Enrichr GO_Biological_Process_2023 terms for each lineage
  - Canonical Tsankov 2015 markers (as seeds to guarantee well-known genes are in)
  - Manually-added canonical pluripotency markers (PL only; GO library lacks good PL coverage)

Writes revision_scripts/go_panels.json.
"""

import json
import re
from pathlib import Path

import requests

ENRICHR_URL = "https://maayanlab.cloud/Enrichr/geneSetLibrary?mode=text&libraryName=GO_Biological_Process_2023"
OUT = Path(__file__).parent / "go_panels.json"

PANELS_GO_IDS: dict[str, list[str]] = {
    "PL": [
        "GO:0048864",  # Stem Cell Development
        "GO:2000737",  # Negative Regulation Of Stem Cell Differentiation
        "GO:2000738",  # Positive Regulation Of Stem Cell Differentiation
        "GO:1902455",  # Negative Regulation Of Stem Cell Population Maintenance
        "GO:1902459",  # Positive Regulation Of Stem Cell Population Maintenance
        "GO:2000648",  # Positive Regulation Of Stem Cell Proliferation
    ],
    "EC": [
        "GO:0007398",  # Ectoderm Development
        "GO:0021915",  # Neural Tube Development
        "GO:0014020",  # Primary Neural Tube Formation
        "GO:0014032",  # Neural Crest Cell Development
        "GO:0014033",  # Neural Crest Cell Differentiation
    ],
    "ME": [
        "GO:0007498",  # Mesoderm Development
        "GO:0001707",  # Mesoderm Formation
        "GO:0048332",  # Mesoderm Morphogenesis
        "GO:0014706",  # Striated Muscle Tissue Development
        "GO:0007507",  # Heart Development
    ],
    "EN": [
        "GO:0007492",  # Endoderm Development
        "GO:0001706",  # Endoderm Formation
        "GO:0048566",  # Embryonic Digestive Tract Development
    ],
    "MS": [
        "GO:0007369",  # Gastrulation
        "GO:0000578",  # Embryonic Axis Specification
        "GO:0010470",  # Regulation Of Gastrulation
    ],
}

# Canonical pluripotency markers — GO has no "pluripotency" term, so seed manually.
CANONICAL_PL_SEEDS = [
    "POU5F1", "NANOG", "SOX2", "KLF4", "MYC", "LIN28A", "LIN28B",
    "DNMT3B", "DPPA3", "DPPA4", "DPPA5", "UTF1", "TDGF1", "ESRRB",
    "ZFP42", "L1TD1", "NODAL", "LEFTY1", "LEFTY2", "FGF2",
    "TRIM22", "HESX1", "IDO1", "LCK", "CXCL5",
]

# Tsankov 2015 scorecard (from the main script) — included as seeds to
# guarantee the canonical lineage TFs are present in each panel.
TSANKOV_SEEDS: dict[str, list[str]] = {
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


def main():
    print("Fetching Enrichr GO_Biological_Process_2023 library...")
    r = requests.get(ENRICHR_URL, timeout=60)
    r.raise_for_status()

    term_to_genes: dict[str, list[str]] = {}
    term_label_by_id: dict[str, str] = {}
    for line in r.text.strip().split("\n"):
        parts = line.split("\t")
        term = parts[0]
        genes = [g for g in parts[1:] if g and g != ""]
        m = re.search(r"\(GO:(\d+)\)", term)
        if not m:
            continue
        goid = f"GO:{m.group(1)}"
        term_to_genes[goid] = genes
        term_label_by_id[goid] = term
    print(f"  {len(term_to_genes)} GO:BP terms loaded")

    panels: dict[str, list[str]] = {}
    panel_provenance: dict[str, dict] = {}
    for panel_name, go_ids in PANELS_GO_IDS.items():
        genes: set[str] = set()
        term_records = []
        for goid in go_ids:
            if goid not in term_to_genes:
                print(f"  [WARN] {panel_name}: {goid} not found in Enrichr library")
                term_records.append({"go_id": goid, "term": "NOT_FOUND", "n_genes": 0})
                continue
            g = term_to_genes[goid]
            genes.update(g)
            term_records.append({"go_id": goid, "term": term_label_by_id[goid], "n_genes": len(g)})
            print(f"  [{panel_name}] {goid} ({term_label_by_id[goid]}): {len(g)} genes")

        # Seed canonical markers
        seeds_added = []
        for seed in TSANKOV_SEEDS.get(panel_name, []):
            if seed not in genes:
                genes.add(seed)
                seeds_added.append(seed)
        if panel_name == "PL":
            for seed in CANONICAL_PL_SEEDS:
                if seed not in genes:
                    genes.add(seed)
                    seeds_added.append(seed)

        panels[panel_name] = sorted(genes)
        panel_provenance[panel_name] = {
            "go_terms": term_records,
            "seed_genes_added": seeds_added,
            "final_size": len(panels[panel_name]),
        }
        print(f"  [{panel_name}] union + seeds: {len(panels[panel_name])} unique genes "
              f"({len(seeds_added)} seeds added on top of GO)")

    out_obj = {
        "source": "Enrichr GO_Biological_Process_2023 + Tsankov 2015 scorecard + canonical PL markers",
        "panels": panels,
        "provenance": panel_provenance,
    }
    OUT.write_text(json.dumps(out_obj, indent=2, sort_keys=False))
    print(f"\nWrote {OUT}")
    print(f"Final panel sizes: { {k: len(v) for k, v in panels.items()} }")


if __name__ == "__main__":
    main()
