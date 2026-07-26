#!/usr/bin/env python3
"""Rebuild the OASIS fitness-screen GH-Pages gallery (docs/fitness_screen_v2/) IN PLACE with the
MaGeCK CPM-filtered re-analysis figures. Copies every PNG+SVG into img/ and regenerates index.html
so each figure renders as PNG (click for full size) with both PNG and SVG download buttons."""
import os, shutil, html

REPO = "/tscc/projects/ps-malilab/ydoctor/Ligandome_Screens"
SRC  = f"{REPO}/Fitness_Screen_Analysis/E6_mTeSR_timecourse/figures"
RNA  = f"{REPO}/t_cell_screens/rna_seq_validations"      # T-cell ligand validation RNA-seq
DOCS = f"{REPO}/docs/fitness_screen_v2"
IMG  = f"{DOCS}/img"

# ---- (dir, basename, caption[, dest_basename]) --------------------------------
# `dir` is relative to SRC, or an absolute path (used for the RNA-seq figures, which live in a
# different project tree). `dest_basename` renames on copy — needed because several analyses share
# generic basenames like `repcorr_ALL_grid` and img/ is flat.
def L(d): return d + "_LFC_vs_score"

def rna_go(godir, tags):
    """GO bar-plot entries, in the given comparison order, for whichever ones Enrichr returned."""
    out = []
    for tag in tags:
        for direction in ("up", "down"):
            if os.path.exists(f"{godir}/GO_{tag}_{direction}.png"):
                pretty = tag.replace("_vs_SNAPD4", " vs SNAP D4").replace("_vs_D4", " vs own D4") \
                            .replace("_", " ")
                out.append((godir, f"GO_{tag}_{direction}",
                            f"GO — {pretty} ({direction} DEGs)"))
    return out

WITHIN_TAGS  = ["SNAP_D8_vs_D4", "SNAP_D12_vs_D4", "CCL13_D8_vs_D4", "CCL13_D12_vs_D4",
                "IL18_D8_vs_D4", "IL18_D12_vs_D4"]
VSSNAP_TAGS  = ["SNAP_D8_vs_SNAPD4", "SNAP_D12_vs_SNAPD4", "CCL13_D4_vs_SNAPD4",
                "CCL13_D8_vs_SNAPD4", "CCL13_D12_vs_SNAPD4", "IL18_D4_vs_SNAPD4",
                "IL18_D8_vs_SNAPD4", "IL18_D12_vs_SNAPD4"]
SECTIONS = [
    ("E6 vs mTeSR — the media effect", "media",
     "The headline contrast (mTeSR = control, so <strong>+LFC = enriched in E6</strong>). "
     "LFC z-score (x) vs MaGeCK ligand score (y); each point is a construct. "
     "Enriched hits climb 2&rarr;6&rarr;18 across D7&rarr;D9&rarr;D14, all one-directional.",
     [(None, [("lfc_vs_score", "GRID_E6_vs_mTeSR_LFC_vs_score", "E6 vs mTeSR — all timepoints (grid)"),
              ("lfc_vs_score", L("Day7_E6_vs_mTeSR"),  "E6 vs mTeSR — Day 7"),
              ("lfc_vs_score", L("Day9_E6_vs_mTeSR"),  "E6 vs mTeSR — Day 9"),
              ("lfc_vs_score", L("Day14_E6_vs_mTeSR"), "E6 vs mTeSR — Day 14")])]),

    ("FGF-family rescue", "fgf",
     "E6 (Essential 6) is mTeSR/E8 <strong>minus FGF2 and TGF&beta;</strong>. Ligands that supply "
     "their own FGF autocrine-rescue the FGF starvation, so the FGF family enriches in E6 over time. "
     "The bar panel contrasts the FGF hits against the 6 NTC controls across the timecourse; brackets "
     "show the per-day FGF-vs-NTC Welch <em>p</em>-value (D7 0.42, D9 0.003, D14 7.7e-9). "
     "Two-way ANOVA: Group <em>p</em>=5.8e-14, Day <em>p</em>=8.2e-16, Group&times;Day <em>p</em>=6.4e-12. "
     "CPM-filtered per-construct data, consistent with the rest of this gallery.",
     [(None, [("fgf_trajectory", "E6_vs_mTeSR_FGF_trajectory", "FGF-family LFC trajectory (E6 vs mTeSR, D7&rarr;D14)"),
              ("fgf_hits_barplot", "FGF_Hits_vs_NTCs_LFC_Barplot_CPMfiltered", "FGF hits vs NTCs — LFC across D7/9/14 (mean &plusmn; SD) &middot; standard width"),
              ("fgf_hits_barplot", "FGF_Hits_vs_NTCs_LFC_Barplot_CPMfiltered_wide", "FGF hits vs NTCs — same figure, 1.5&times; wider")])]),

    ("Hit overlap across timepoints", "overlap",
     "Enriched hits (|Z|&gt;2 &amp; FDR&lt;0.05) shared across Day 7 / Day 9 / Day 14 — one Venn per arm, "
     "circles are timepoints. The central lobe is the persistent (all-three-days) set. The E6 and mTeSR "
     "arms are each vs their Day 4 baseline; the third is the E6-vs-mTeSR differential.",
     [(None, [("venn", "venn_timepoints_E6", "E6 (vs Day 4) — D7/D9/D14 overlap"),
              ("venn", "venn_timepoints_mTeSR", "mTeSR (vs Day 4) — D7/D9/D14 overlap"),
              ("venn", "venn_timepoints_E6_vs_mTeSR", "E6 vs mTeSR — D7/D9/D14 overlap")])]),

    ("Fitness vs Day 4 baseline", "day4",
     "Each medium/timepoint versus the shared Day 4 baseline (depletion/enrichment over the timecourse).",
     [("E6 vs Day 4", [("lfc_vs_score", "GRID_E6_vs_Day4_LFC_vs_score", "E6 vs Day 4 — grid"),
                       ("lfc_vs_score", L("Day7_E6_vs_Day4"),  "E6 Day 7 vs Day 4"),
                       ("lfc_vs_score", L("Day9_E6_vs_Day4"),  "E6 Day 9 vs Day 4"),
                       ("lfc_vs_score", L("Day14_E6_vs_Day4"), "E6 Day 14 vs Day 4")]),
      ("mTeSR vs Day 4", [("lfc_vs_score", "GRID_mTeSR_vs_Day4_LFC_vs_score", "mTeSR vs Day 4 — grid"),
                       ("lfc_vs_score", L("Day7_mTeSR_vs_Day4"),  "mTeSR Day 7 vs Day 4"),
                       ("lfc_vs_score", L("Day9_mTeSR_vs_Day4"),  "mTeSR Day 9 vs Day 4"),
                       ("lfc_vs_score", L("Day14_mTeSR_vs_Day4"), "mTeSR Day 14 vs Day 4")])]),

    ("vs Plasmid library", "plasmid",
     "Each medium/timepoint versus the NGS plasmid library (representation / bottleneck).",
     [("E6 vs plasmid", [("lfc_vs_score", "GRID_E6_vs_Plasmid_LFC_vs_score", "E6 vs plasmid — grid"),
                       ("lfc_vs_score", L("Day7_E6_vs_Plasmid"),  "E6 Day 7 vs plasmid"),
                       ("lfc_vs_score", L("Day9_E6_vs_Plasmid"),  "E6 Day 9 vs plasmid"),
                       ("lfc_vs_score", L("Day14_E6_vs_Plasmid"), "E6 Day 14 vs plasmid")]),
      ("mTeSR vs plasmid", [("lfc_vs_score", "GRID_mTeSR_vs_Plasmid_LFC_vs_score", "mTeSR vs plasmid — grid"),
                       ("lfc_vs_score", L("Day7_mTeSR_vs_Plasmid"),  "mTeSR Day 7 vs plasmid"),
                       ("lfc_vs_score", L("Day9_mTeSR_vs_Plasmid"),  "mTeSR Day 9 vs plasmid"),
                       ("lfc_vs_score", L("Day14_mTeSR_vs_Plasmid"), "mTeSR Day 14 vs plasmid")])]),

    ("Replicate reproducibility (QC)", "qc",
     "sgRNA LFC of rep1 vs rep2, CPM-filtered survivors (n=744), against two references. "
     "<strong>vs plasmid</strong> (the deep, stable library reference) the reps are highly reproducible "
     "(Pearson r 0.91–0.93 at every condition, incl. Day 4). <strong>vs the Day 4 mean</strong> the common "
     "representation signal is subtracted out, leaving only small biological fitness differences, so r is "
     "modest early (0.23–0.28) and climbs as real effects emerge (Day 14: E6 0.52, mTeSR 0.40). "
     "Both are correct — they measure different things.",
     [("vs plasmid", [("replicate_corr", "ALL_replicate_corr_vs_plasmid_grid", "vs plasmid — all conditions (grid)"),
              ("replicate_corr", "repcorr_Day4_LFC_vs_plasmid",       "Day 4 — rep1 vs rep2"),
              ("replicate_corr", "repcorr_Day7_E6_LFC_vs_plasmid",    "E6 Day 7 — rep1 vs rep2"),
              ("replicate_corr", "repcorr_Day9_E6_LFC_vs_plasmid",    "E6 Day 9 — rep1 vs rep2"),
              ("replicate_corr", "repcorr_Day14_E6_LFC_vs_plasmid",   "E6 Day 14 — rep1 vs rep2"),
              ("replicate_corr", "repcorr_Day7_mTeSR_LFC_vs_plasmid", "mTeSR Day 7 — rep1 vs rep2"),
              ("replicate_corr", "repcorr_Day9_mTeSR_LFC_vs_plasmid", "mTeSR Day 9 — rep1 vs rep2"),
              ("replicate_corr", "repcorr_Day14_mTeSR_LFC_vs_plasmid","mTeSR Day 14 — rep1 vs rep2")]),
      ("vs Day 4 mean", [("replicate_corr", "ALL_replicate_corr_grid", "vs Day 4 mean — all conditions (grid)"),
              ("replicate_corr", "repcorr_Day7_E6_LFC_vs_Day4mean",  "E6 Day 7 — rep1 vs rep2"),
              ("replicate_corr", "repcorr_Day9_E6_LFC_vs_Day4mean",  "E6 Day 9 — rep1 vs rep2"),
              ("replicate_corr", "repcorr_Day14_E6_LFC_vs_Day4mean", "E6 Day 14 — rep1 vs rep2"),
              ("replicate_corr", "repcorr_Day7_mTeSR_LFC_vs_Day4mean",  "mTeSR Day 7 — rep1 vs rep2"),
              ("replicate_corr", "repcorr_Day9_mTeSR_LFC_vs_Day4mean",  "mTeSR Day 9 — rep1 vs rep2"),
              ("replicate_corr", "repcorr_Day14_mTeSR_LFC_vs_Day4mean", "mTeSR Day 14 — rep1 vs rep2")])]),

    ("CCK8 proliferation validation", "cck8",
     "Independent CCK8 (WST-8) proliferation assays validating fitness-screen hit ligands against an "
     "mCherry control; mean &plusmn; SEM, n=3 replicate wells (dots), Student's t-test (uncorrected p + "
     "stars), each bar vs the earliest-day mCherry baseline (dashed line). "
     "<strong>First panel:</strong> NRG1/BMP8B/CXCL7/DB133 at Day 3 &amp; 5 (vs mCherry Day 3) — all four "
     "significant at both days, DB133 strongest (~2.2&times;). "
     "<strong>Second panel:</strong> FGF16/NRG2/EGF at Day 1 &amp; 3 (vs mCherry Day 1; this set has no "
     "Day 5) — FGF16 is flat at Day 1 then jumps to ~1.6&times; by Day 3.",
     [(None, [("cck8_validation", "val1_cck8_day3day5_bars", "CCK8 — NRG1/BMP8B/CXCL7/DB133 (Day 3 &amp; 5, vs mCherry Day 3)"),
              ("cck8_validation", "val2_fgf_cck8_day1day3_bars", "CCK8 — FGF16/NRG2/EGF (Day 1 &amp; 3, vs mCherry Day 1)")])]),

    ("T-cell ligandome screen (PBMC, Day 10)", "tcell",
     "Original PBMC T-cell ligandome fitness screen, Day-10 endpoint (separate from the iPSC E6/mTeSR "
     "screen above). LFC-vs-ligand-score for the <strong>Acute arm vs CXL-plasmid</strong> "
     "(x = LFC z-score, y = MaGeCK ligand score; blue = enriched, |Z|&gt;2 &amp; FDR&lt;0.05): "
     "top enriched CALC / CCL13 / KLOT / IL23A / PDYN (no significant depletions). The Acute panel is "
     "provided at several y-axis (ligand-score) maxima — 150 / 200 / 250 / 300 — for scale matching. "
     "Replicate reproducibility is the LFC (vs CXL-plasmid) of rep 1 vs rep 2 for each arm (Pearson r "
     "on the panel: acute 0.72, chronic 0.70).",
     [(None, [("tcell_screen", "Acute_LFC_vs_logscore", "Acute vs plasmid — LFC z-score vs ligand score (y 0–300)"),
              ("tcell_screen", "Acute_LFC_vs_logscore_y250", "Acute vs plasmid — y-axis 0–250"),
              ("tcell_screen", "Acute_LFC_vs_logscore_y200", "Acute vs plasmid — y-axis 0–200"),
              ("tcell_screen", "Acute_LFC_vs_logscore_y150", "Acute vs plasmid — y-axis 0–150"),
              ("tcell_screen", "acute_replicate_correlations", "Acute — replicate correlation (LFC vs plasmid, rep1 vs rep2)"),
              ("tcell_screen", "chronic_replicate_correlations", "Chronic — replicate correlation (LFC vs plasmid, rep1 vs rep2)")])]),

    # ---------------- T-cell ligand validation RNA-seq (IGM run 260720) ----------------
    ("T-cell RNA-seq validation — ligand vs SNAP at matched timepoints", "rnaseq-null",
     "Bulk RNA-seq validation of two screen hits (<strong>CCL13</strong>, <strong>IL18</strong>) against the "
     "<strong>SNAP</strong> control in primary T cells at Day 4 / 8 / 12 (IGM run "
     "<code>260720_LH00444_0550_B23MVY3LT4</code>; STAR &rarr; featureCounts &rarr; pydeseq2, n=2 per group). "
     "Because raw depth ranged 32M&ndash;557M pairs across the 18 libraries, every panel in these three "
     "sections is computed on BAMs <strong>randomly downsampled to 20M mapped pairs</strong> (seed 42) so all "
     "samples are depth-matched. "
     "<strong>Result: 0 DEGs</strong> at padj&lt;0.05 &amp; |LFC|&ge;2 in all six ligand-vs-SNAP contrasts. "
     "The replicate panels show why this is a genuine null rather than a technical failure: count-level "
     "replicate correlation is ~0.997, but the replicates' <em>fold-changes</em> vs SNAP agree only weakly "
     "(r 0.19&ndash;0.70) because there is no real fold-change for them to agree on. D8 is the least-null "
     "timepoint in both arms.",
     [("volcanoes (ligand vs SNAP, same day)",
       [(f"{RNA}/figures_ds", f"volcano_{L2}_vs_SNAP_{tp}", f"{L2} vs SNAP — {tp}")
        for L2 in ("CCL13", "IL18") for tp in ("D4", "D8", "D12")]),
      ("replicate reproducibility of LFC vs SNAP",
       [(f"{RNA}/figures_ds/rep_corr", "repcorr_ALL_grid",
         "All six conditions (grid) — rep1 vs rep2 LFC vs SNAP", "rnaseq_null_repcorr_ALL_grid")] +
       [(f"{RNA}/figures_ds/rep_corr", f"repcorr_{L2}_vs_SNAP_{tp}", f"{L2} vs SNAP — {tp}")
        for L2 in ("CCL13", "IL18") for tp in ("D4", "D8", "D12")])]),

    ("T-cell RNA-seq — within-ligand time course (vs its own Day 4)", "rnaseq-within",
     "Each arm compared to <strong>its own Day 4</strong> baseline: SNAP D8/D12 vs SNAP D4, CCL13 D8/D12 vs "
     "CCL13 D4, IL18 D8/D12 vs IL18 D4. This is where the dataset's real signal lives — "
     "<strong>~1,000&ndash;1,700 DEGs per comparison</strong> with replicate LFC correlations of "
     "<strong>0.91&ndash;0.98</strong>, which validates the libraries, the counts and the DESeq2 setup, and "
     "confirms that the flat ligand-vs-SNAP result above is biology and not a depth or normalisation artefact. "
     "About <strong>70%</strong> of the D4&rarr;D8 DEGs are shared by all three arms, i.e. most of the change "
     "is culture/activation rather than ligand-driven. The exception is <strong>IL18 at D8</strong> "
     "(1,008 up / 698 down, the largest set, with 278 up-only and 268 down-only genes beyond SNAP and CCL13). "
     "GO: the shared programme is a coordinated loss of the cytokine/inflammatory and B-cell-receptor "
     "signalling signature with ribosome-biogenesis down, and calcium-channel / synapse-organisation terms up "
     "by D12.",
     [("summary grids",
       [(f"{RNA}/figures_ds_within", "volcano_ALL_grid", "All six comparisons — volcanoes (grid)", "rnaseq_within_volcano_grid"),
        (f"{RNA}/figures_ds_within", "repcorr_ALL_grid", "All six comparisons — replicate LFC reproducibility (grid)", "rnaseq_within_repcorr_grid")]),
      ("volcanoes",
       [(f"{RNA}/figures_ds_within/volcano", f"volcano_{t}", t.replace("_vs_D4", " vs own Day 4").replace("_", " "))
        for t in WITHIN_TAGS]),
      ("replicate reproducibility of LFC",
       [(f"{RNA}/figures_ds_within/rep_corr", f"repcorr_{t}", t.replace("_vs_D4", " vs own Day 4").replace("_", " "))
        for t in WITHIN_TAGS]),
      ("GO enrichment (Enrichr; BP / MF / CC 2023, padj&lt;0.05 &amp; |LFC|&ge;2 DEGs)",
       rna_go(f"{RNA}/figures_ds_within/go", WITHIN_TAGS))]),

    ("T-cell RNA-seq — every group vs SNAP Day 4", "rnaseq-vssnapd4",
     "The same eight groups all referenced to the single <strong>SNAP Day 4</strong> baseline, so time and "
     "ligand effects sit on one common scale. The two SNAP rows are by construction identical to the "
     "within-ligand contrasts above. The pattern is unambiguous: the two same-day contrasts "
     "(<strong>CCL13 D4</strong>, <strong>IL18 D4</strong> vs SNAP D4) return <strong>0 DEGs</strong>, while "
     "every D8 and D12 group returns ~1,000&ndash;1,700 DEGs that closely track the SNAP time course — the "
     "time axis dominates and the ligands contribute little on top of it.",
     [("summary grids",
       [(f"{RNA}/figures_ds_vsSNAPd4", "volcano_ALL_grid", "All eight comparisons — volcanoes (grid)", "rnaseq_vssnap_volcano_grid"),
        (f"{RNA}/figures_ds_vsSNAPd4", "repcorr_ALL_grid", "All eight comparisons — replicate LFC reproducibility (grid)", "rnaseq_vssnap_repcorr_grid")]),
      ("volcanoes",
       [(f"{RNA}/figures_ds_vsSNAPd4/volcano", f"volcano_{t}", t.replace("_vs_SNAPD4", " vs SNAP Day 4").replace("_", " "))
        for t in VSSNAP_TAGS]),
      ("replicate reproducibility of LFC",
       [(f"{RNA}/figures_ds_vsSNAPd4/rep_corr", f"repcorr_{t}", t.replace("_vs_SNAPD4", " vs SNAP Day 4").replace("_", " "))
        for t in VSSNAP_TAGS]),
      ("GO enrichment (Enrichr; BP / MF / CC 2023, padj&lt;0.05 &amp; |LFC|&ge;2 DEGs)",
       rna_go(f"{RNA}/figures_ds_vsSNAPd4/go", VSSNAP_TAGS))]),
]

# ---- copy figures (replace old img/) ------------------------------------------
if os.path.isdir(IMG):
    for f in os.listdir(IMG):
        os.remove(os.path.join(IMG, f))
os.makedirs(IMG, exist_ok=True)
n = 0
has_svg = set()          # bases that also have a vector SVG (SVG is optional per figure)

def unpack(fig):
    d, base, cap = fig[0], fig[1], fig[2]
    dest = fig[3] if len(fig) > 3 else base
    return (d if os.path.isabs(d) else f"{SRC}/{d}"), base, cap, dest

for _, _, _, subs in SECTIONS:
    for _, figs in subs:
        for fig in figs:
            sd, base, _, dest = unpack(fig)
            shutil.copy2(f"{sd}/{base}.png", f"{IMG}/{dest}.png")
            svg = f"{sd}/{base}.svg"
            if os.path.exists(svg):
                shutil.copy2(svg, f"{IMG}/{dest}.svg"); has_svg.add(dest)
            n += 1
print(f"copied {n} figures ({n + len(has_svg)} files; {n - len(has_svg)} PNG-only) into {IMG}")

# ---- emit HTML ----------------------------------------------------------------
def fig_html(base, cap):
    a = html.escape(cap.replace("&rarr;", "->").replace("&beta;", "b"), quote=True)
    svg_btn = f'\n          <a class="dl-btn svg" href="img/{base}.svg" download>SVG</a>' if base in has_svg else ""
    return f'''      <figure>
        <a class="img-link" href="img/{base}.png"><img src="img/{base}.png" alt="{a}"></a>
        <figcaption><span>{cap}</span><span class="dl">
          <a class="dl-btn" href="img/{base}.png" download>PNG</a>{svg_btn}
        </span></figcaption>
      </figure>'''

body = []
for h2, hid, intro, subs in SECTIONS:
    body.append(f'  <h2 id="{hid}">{h2}</h2>')
    body.append(f"  <p>{intro}</p>")
    for h3, figs in subs:
        if not figs:
            continue
        if h3:
            body.append(f"  <h3>{h3}</h3>")
        body.append('  <div class="grid">')
        body += [fig_html(dest, cap) for _, _, cap, dest in (unpack(f) for f in figs)]
        body.append("  </div>")
SECTIONS_HTML = "\n".join(body)

HTML = f'''<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>OASIS — E6 vs mTeSR Fitness Screen (MaGeCK, CPM-filtered)</title>
<style>
  :root {{
    --bg: #fafaf7; --fg: #16181d; --muted: #5b6470; --rule: #e2e2dc;
    --accent: #1a4480; --accent-soft: #e8eef7; --code-bg: #f0efe9;
    --enrich: #1982C4; --deplete: #FF595E;
  }}
  @media (prefers-color-scheme: dark) {{
    :root {{
      --bg: #14151a; --fg: #e8e7e1; --muted: #9aa3ad; --rule: #2a2c33;
      --accent: #6ea3e5; --accent-soft: #1c2a40; --code-bg: #1f2027;
    }}
  }}
  html {{ font-size: 16px; }}
  body {{ margin: 0; background: var(--bg); color: var(--fg);
    font-family: "Charter", "Iowan Old Style", "Source Serif Pro", Georgia, serif; line-height: 1.55; }}
  main {{ max-width: 1200px; margin: 0 auto; padding: 3rem 1.5rem 6rem; }}
  header.lede {{ border-bottom: 1px solid var(--rule); padding-bottom: 1.5rem; margin-bottom: 2rem; }}
  h1 {{ font-size: 2rem; line-height: 1.15; margin: 0 0 .25rem; letter-spacing: -.01em; }}
  .kicker {{ color: var(--muted); font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", system-ui, sans-serif;
    font-size: .82rem; text-transform: uppercase; letter-spacing: .08em; margin: 0 0 .75rem; }}
  h2 {{ font-size: 1.4rem; margin: 3rem 0 .75rem; letter-spacing: -.005em;
    border-top: 1px solid var(--rule); padding-top: 2rem; }}
  h3 {{ font-size: 1.05rem; margin: 2rem 0 .5rem;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", system-ui, sans-serif;
    color: var(--muted); text-transform: uppercase; letter-spacing: .04em; font-weight: 600; }}
  p {{ margin: 0 0 1rem; }}
  a {{ color: var(--accent); text-decoration: none; }}
  a:hover {{ text-decoration: underline; }}
  .note {{ border-left: 3px solid var(--accent); background: var(--accent-soft);
    margin: 1.5rem 0; padding: .9rem 1.1rem; font-size: .95rem; }}
  .note p:last-child {{ margin-bottom: 0; }}
  code {{ font-family: "JetBrains Mono", "SF Mono", Menlo, Consolas, monospace;
    background: var(--code-bg); padding: .1em .35em; border-radius: 3px; font-size: .88em; }}
  .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(340px, 1fr));
    gap: 1.25rem; margin: 1.25rem 0 2rem; }}
  figure {{ margin: 0; border: 1px solid var(--rule); border-radius: 6px; padding: .75rem;
    background: var(--bg); display: flex; flex-direction: column; }}
  figure a.img-link {{ display: block; }}
  figure img {{ max-width: 100%; height: auto; display: block; background: white; border-radius: 4px; }}
  figcaption {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", system-ui, sans-serif;
    font-size: .85rem; color: var(--fg); margin-top: .6rem; line-height: 1.35;
    display: flex; align-items: center; justify-content: space-between; gap: .5rem; }}
  .dl {{ display: flex; gap: .35rem; flex-shrink: 0; }}
  .dl-btn {{ font-size: .72rem; text-transform: uppercase; letter-spacing: .04em;
    background: var(--accent); color: white; padding: .3rem .6rem; border-radius: 3px; white-space: nowrap; }}
  .dl-btn.svg {{ background: transparent; color: var(--accent); border: 1px solid var(--accent); }}
  .dl-btn:hover {{ text-decoration: none; opacity: .85; }}
</style>
</head>
<body>
<main>
  <header class="lede">
    <p class="kicker">OASIS · iPSC Ligandome Fitness Screen</p>
    <h1>E6 vs mTeSR media timecourse — MaGeCK re-analysis</h1>
    <p>Re-analysis of the E6-vs-mTeSR media fitness screen (IGM run <code>251013_LH00444_0422_B22YWHGLT4</code>)
    with MaGeCK on the R2/antiBarcode count matrix. Every figure renders as PNG (click for full size) and is
    downloadable as PNG or vector <strong>SVG</strong>.</p>
  </header>

  <section class="note">
    <p><strong>Methods for this run:</strong></p>
    <p>&bull; <strong>CPM filter</strong> (Day 4 + plasmid only): constructs with CPM&lt;1 in <code>Day4_r1</code>,
    <code>Day4_r2</code> or <code>NGS_plasmid</code> were dropped before testing (<strong>744/770</strong>
    constructs, 504/521 ligands kept).</p>
    <p>&bull; <strong>Per-construct grounding</strong> (gene = barcode id, 1 sgRNA/gene), 15 independent MaGeCK
    tests, each median-normalized and FDR-corrected. E6-vs-mTeSR uses <strong>mTeSR as control</strong>
    (+LFC = enriched in E6).</p>
    <p>&bull; Gene naming: ligand genes carry <code>_HUMAN</code> (stripped for display); NTC references carry
    <code>_CONTROL</code> (<code>AMPR/CLUC/GLUC/HALO/MCHERRY/SNAP_CONTROL</code>).</p>
    <p>&bull; Hit definition: |LFC z-score| &gt; 2 and FDR &lt; 0.05. Blue = enriched, red = depleted.</p>
    <p>&bull; <strong>Headline:</strong> the FGF family enriches in E6 (autocrine rescue of FGF2/TGF&beta;-stripped
    media) — 2&rarr;6&rarr;18 enriched hits across D7&rarr;D9&rarr;D14, one-directional; a built-in positive control.</p>
  </section>

{SECTIONS_HTML}

  <p style="margin-top:3rem; font-size:.85rem; color:var(--muted); font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',system-ui,sans-serif;">
    Source: <code>Fitness_Screen_Analysis/E6_mTeSR_timecourse/</code> ·
    scripts <code>build_and_filter.py</code>, <code>run_mageck_tests.sh</code>, <code>plot_*.py</code> ·
    compiled hits: <code>results/e6_mtesr_sgrna_summaries_CPMfiltered.xlsx</code>. All figures at 300&nbsp;dpi; SVG is vector/editable.
  </p>
</main>
</body>
</html>
'''

with open(f"{DOCS}/index.html", "w") as fh:
    fh.write(HTML)
print(f"wrote {DOCS}/index.html  ({len(SECTIONS_HTML.splitlines())} section lines)")
