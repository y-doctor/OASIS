#!/usr/bin/env python3
"""GO-term enrichment on the up/down DEGs of the time-course comparisons (20M-downsampled counts).

MODE=within    -> deg_ds_within/   (each arm vs its own Day 4)
MODE=vs_snapd4 -> deg_ds_vsSNAPd4/ (everything vs SNAP Day 4)

Unlike the same-timepoint ligand-vs-SNAP contrasts, these have real gene lists, so the study's
standard padj<0.05 & |LFC|>=2 threshold is used. Background = expressed genes (baseMean>0) of that
comparison; gene sets = GO Biological Process / Molecular Function / Cellular Component 2023.

Calls the Enrichr "speedrichr" background-enrichment API directly rather than going through
gseapy.enrichr: Enrichr emits bare `Infinity` odds ratios for terms whose overlap is complete, and
requests' JSON decoding (simplejson, installed in this env) rejects that token, which made gseapy
fail deterministically on ~1/3 of the gene lists. Stdlib json parses `Infinity` fine.
Needs internet — run on a login node."""
import os, glob, re, json, time, requests, pandas as pd, numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt, matplotlib as mpl
mpl.rcParams.update({'svg.fonttype': 'none', 'font.family': 'DejaVu Sans'})

BASE = '/tscc/projects/ps-malilab/ydoctor/Ligandome_Screens/t_cell_screens/rna_seq_validations'
MODE = os.environ.get('MODE', 'within')
DEGD = {'within': f'{BASE}/deg_ds_within', 'vs_snapd4': f'{BASE}/deg_ds_vsSNAPd4'}[MODE]
GOD  = {'within': f'{BASE}/go_ds_within',  'vs_snapd4': f'{BASE}/go_ds_vsSNAPd4'}[MODE]
FIG  = {'within': f'{BASE}/figures_ds_within/go', 'vs_snapd4': f'{BASE}/figures_ds_vsSNAPd4/go'}[MODE]
os.makedirs(GOD, exist_ok=True); os.makedirs(FIG, exist_ok=True)
PADJ, LFC = 0.05, 2.0
GENE_SETS = ['GO_Biological_Process_2023', 'GO_Molecular_Function_2023', 'GO_Cellular_Component_2023']
API = 'https://maayanlab.cloud/speedrichr'
COLS = ['rank', 'Term', 'P-value', 'Odds Ratio', 'Combined Score', 'genes', 'Adjusted P-value',
        'old_p', 'old_adj_p']


def _post(url, **kw):
    """POST + stdlib-json parse, with retries (Enrichr is intermittently slow/flaky)."""
    last = None
    for attempt in range(5):
        try:
            r = requests.post(url, timeout=300, **kw)
            r.raise_for_status()
            return json.loads(r.text)
        except Exception as ex:
            last = ex; time.sleep(10 * (attempt + 1))
    raise RuntimeError(f'{url} failed after 5 attempts: {last}')


def speedrichr(genes, background, gene_set):
    lid = _post(f'{API}/api/addList',
                files=dict(list=(None, '\n'.join(map(str, genes))),
                           description=(None, 'oasis')))['userListId']
    bid = _post(f'{API}/api/addbackground',
                data=dict(background='\n'.join(map(str, background))))['backgroundid']
    res = _post(f'{API}/api/backgroundenrich',
                data=dict(userListId=lid, backgroundid=bid, backgroundType=gene_set))
    df = pd.DataFrame(res[gene_set], columns=COLS)
    df['Genes'] = df['genes'].apply(';'.join)
    df['Overlap'] = df['genes'].apply(len)
    df.insert(0, 'Gene_set', gene_set)
    return df.drop(columns=['genes', 'rank', 'old_p', 'old_adj_p'])


def enrich(genes, background, tag, direction):
    if len(genes) < 5:
        print(f'  {tag} {direction}: only {len(genes)} genes — skipped', flush=True); return None
    parts = []
    for gs in GENE_SETS:
        try:
            parts.append(speedrichr(genes, background, gs))
        except Exception as ex:
            print(f'  {tag} {direction} {gs}: FAILED ({ex})', flush=True)
    if not parts:
        return None
    res = pd.concat(parts, ignore_index=True)
    res.insert(0, 'direction', direction); res.insert(0, 'comparison', tag)
    res.to_csv(f'{GOD}/GO_{tag}_{direction}.csv', index=False)

    sig = res[res['Adjusted P-value'] < 0.05].copy()
    if len(sig):
        sig['nlq'] = -np.log10(sig['Adjusted P-value'].clip(lower=1e-300))
        top = sig.sort_values('nlq', ascending=False).head(12).iloc[::-1]
        col = '#c1121f' if direction == 'up' else '#1d3557'
        fig, ax = plt.subplots(figsize=(9, max(3, 0.45 * len(top))))
        ax.barh(range(len(top)), top['nlq'], color=col, alpha=0.85)
        ax.set_yticks(range(len(top))); ax.set_yticklabels([t[:60] for t in top['Term']], fontsize=8)
        ax.set_xlabel('-log10 adjusted p')
        ax.set_title(f'GO — {tag} ({direction} DEGs, n={len(genes)})', fontsize=11)
        ax.spines[['top', 'right']].set_visible(False)
        fig.tight_layout()
        for ext in ('png', 'svg'):
            fig.savefig(f'{FIG}/GO_{tag}_{direction}.{ext}', dpi=150, bbox_inches='tight')
        plt.close(fig)
    print(f'  {tag} {direction}: {len(genes)} genes -> {len(sig)} sig GO terms', flush=True)
    return res


allres, summ = [], []
for f in sorted(glob.glob(f'{DEGD}/*_deseq2.csv')):
    tag = re.sub(r'_deseq2\.csv$', '', os.path.basename(f))
    res = pd.read_csv(f, index_col=0)
    bg = res.index[res['baseMean'] > 0].tolist()
    up = res.index[(res['padj'] < PADJ) & (res['log2FoldChange'] >= LFC)].tolist()
    dn = res.index[(res['padj'] < PADJ) & (res['log2FoldChange'] <= -LFC)].tolist()
    print(f'{tag}: {len(up)} up / {len(dn)} down (bg {len(bg)})', flush=True)
    for genes, d in [(up, 'up'), (dn, 'down')]:
        r = enrich(genes, bg, tag, d)
        if r is not None:
            allres.append(r)
            s = r[r['Adjusted P-value'] < 0.05]
            summ.append((tag, d, len(genes), len(s),
                         '; '.join(s.sort_values('Adjusted P-value')['Term'].head(3))))

if allres:
    pd.concat(allres, ignore_index=True).to_csv(f'{GOD}/GO_all_comparisons.csv', index=False)
sm = pd.DataFrame(summ, columns=['comparison', 'direction', 'n_genes', 'n_sig_terms', 'top3_terms'])
sm.to_csv(f'{GOD}/GO_summary.csv', index=False)
print(f'\n==== GO SUMMARY ({MODE}) ====')
print(sm.to_string(index=False))
print('GO enrichment done')
