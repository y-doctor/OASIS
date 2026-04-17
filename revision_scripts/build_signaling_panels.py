"""Build signaling-pathway panels (WNT, TGFB, NOTCH, MAPK) from KEGG_2021_Human
via Enrichr. These are a fundamentally different lens from lineage markers —
cells cluster by which signaling program is active, not by germ-layer identity.

Writes revision_scripts/signaling_panels.json.
"""

import json
import re
from pathlib import Path

import requests

ENRICHR_URL = "https://maayanlab.cloud/Enrichr/geneSetLibrary?mode=text&libraryName=KEGG_2021_Human"
OUT = Path(__file__).parent / "signaling_panels.json"

# Canonical KEGG signaling-pathway gene sets. Names taken verbatim from
# KEGG_2021_Human (Enrichr).
PANEL_TO_KEGG = {
    "WNT":   "Wnt signaling pathway",
    "TGFB":  "TGF-beta signaling pathway",
    "NOTCH": "Notch signaling pathway",
    "MAPK":  "MAPK signaling pathway",
}


def main():
    print("Fetching Enrichr KEGG_2021_Human library...")
    r = requests.get(ENRICHR_URL, timeout=60)
    r.raise_for_status()

    term_to_genes: dict[str, list[str]] = {}
    for line in r.text.strip().split("\n"):
        parts = line.split("\t")
        term = parts[0]
        genes = [g for g in parts[1:] if g and g != ""]
        term_to_genes[term] = genes
    print(f"  {len(term_to_genes)} KEGG pathways loaded")

    panels: dict[str, list[str]] = {}
    provenance: dict[str, dict] = {}
    for panel_name, kegg_term in PANEL_TO_KEGG.items():
        if kegg_term not in term_to_genes:
            raise KeyError(f"{kegg_term!r} not found in KEGG library")
        genes = sorted(set(term_to_genes[kegg_term]))
        panels[panel_name] = genes
        provenance[panel_name] = {
            "source_library": "KEGG_2021_Human (Enrichr)",
            "pathway_name": kegg_term,
            "n_genes": len(genes),
        }
        print(f"  [{panel_name}] {kegg_term}: {len(genes)} genes")

    out_obj = {
        "source": "Enrichr KEGG_2021_Human canonical signaling pathways",
        "panels": panels,
        "provenance": provenance,
    }
    OUT.write_text(json.dumps(out_obj, indent=2, sort_keys=False))
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
