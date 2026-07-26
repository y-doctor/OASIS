#!/usr/bin/env python3
"""Time-course DEG (pydeseq2) + volcano + per-replicate LFC rep-corr on the 20M-DOWNSAMPLED counts.

Two designs (MODE env var):
  MODE=within    -> each ligand vs ITS OWN D4:      SNAP D8/D12 vs SNAP D4,
                                                     CCL13 D8/D12 vs CCL13 D4,
                                                     IL18  D8/D12 vs IL18  D4     (6 comparisons)
  MODE=vs_snapd4 -> everything vs SNAP D4:          SNAP D8, SNAP D12, CCL13 D4/D8/D12,
                                                     IL18 D4/D8/D12  vs SNAP D4    (8 comparisons)

Each comparison is a 2-vs-2 pydeseq2 model on the 4 relevant samples (same as the study's prior
deg_volcano.py), volcano in JB style (padj<0.05 & |LFC|>=2; up #c1121f / down #1d3557), plus a
replicate-reproducibility scatter: LFC_rep_i = log2(CPM_test_rep_i+1) - log2(mean CPM_ref+1).
"""
import os, numpy as np, pandas as pd
from pydeseq2.dds import DeseqDataSet
from pydeseq2.ds import DeseqStats
from pydeseq2.default_inference import DefaultInference
from scipy.stats import pearsonr, spearmanr
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt, matplotlib as mpl
try:
    from adjustText import adjust_text; HAS_AT = True
except Exception:
    HAS_AT = False
mpl.rcParams.update({'svg.fonttype': 'none'})

BASE = '/tscc/projects/ps-malilab/ydoctor/Ligandome_Screens/t_cell_screens/rna_seq_validations'
MAT  = os.environ.get('MAT', f'{BASE}/deg_ds/tcell_gene_counts_matrix.csv')
MODE = os.environ.get('MODE', 'within')
PADJ, LFC, MINCOUNT = 0.05, 2.0, 10

# (label, test_group, ref_group)   group = '<TP>-<LIGAND>'
if MODE == 'within':
    COMPS = [(f'{L}_D{d}_vs_D4', f'D{d}-{L}', f'D4-{L}')
             for L in ('SNAP', 'CCL13', 'IL18') for d in (8, 12)]
    OUTD, FIGD = f'{BASE}/deg_ds_within', f'{BASE}/figures_ds_within'
    SUPT = 'Within-ligand time course (vs own Day 4) — 20M downsampled'
    NROW, NCOL = 3, 2
elif MODE == 'vs_snapd4':
    COMPS = [('SNAP_D8_vs_SNAPD4',   'D8-SNAP',   'D4-SNAP'),
             ('SNAP_D12_vs_SNAPD4',  'D12-SNAP',  'D4-SNAP'),
             ('CCL13_D4_vs_SNAPD4',  'D4-CCL13',  'D4-SNAP'),
             ('CCL13_D8_vs_SNAPD4',  'D8-CCL13',  'D4-SNAP'),
             ('CCL13_D12_vs_SNAPD4', 'D12-CCL13', 'D4-SNAP'),
             ('IL18_D4_vs_SNAPD4',   'D4-IL18',   'D4-SNAP'),
             ('IL18_D8_vs_SNAPD4',   'D8-IL18',   'D4-SNAP'),
             ('IL18_D12_vs_SNAPD4',  'D12-IL18',  'D4-SNAP')]
    OUTD, FIGD = f'{BASE}/deg_ds_vsSNAPd4', f'{BASE}/figures_ds_vsSNAPd4'
    SUPT = 'All groups vs SNAP Day 4 — 20M downsampled'
    NROW, NCOL = 4, 2
else:
    raise SystemExit(f'unknown MODE={MODE}')
os.makedirs(OUTD, exist_ok=True); os.makedirs(f'{FIGD}/volcano', exist_ok=True)
os.makedirs(f'{FIGD}/rep_corr', exist_ok=True)

counts = pd.read_csv(MAT, index_col=0)
cpm = counts / counts.sum() * 1e6
print(f'MODE={MODE}  matrix={counts.shape}  -> {OUTD}', flush=True)


def run_deseq(test, ref, tag):
    samples = [f'{ref}-1', f'{ref}-2', f'{test}-1', f'{test}-2']
    md = pd.DataFrame({'condition': ['ref', 'ref', 'test', 'test']}, index=samples)
    cT = counts[samples].T.astype(int); cT = cT.loc[:, cT.sum(axis=0) > 0]
    dds = DeseqDataSet(counts=cT, metadata=md, design='~condition', refit_cooks=True,
                       inference=DefaultInference(n_cpus=8))
    dds.deseq2()
    st = DeseqStats(dds, contrast=['condition', 'test', 'ref'], inference=DefaultInference(n_cpus=8))
    st.summary()
    res = st.results_df.copy().sort_values('padj')
    res.to_csv(f'{OUTD}/{tag}_deseq2.csv')
    return res


def volcano(ax, res, title, xlab):
    r = res.copy(); r['nlq'] = -np.log10(r['padj'].fillna(1).clip(lower=1e-300))
    x, y = r.log2FoldChange.to_numpy(), r.nlq.to_numpy()
    su = ((r.padj.fillna(1) < PADJ) & (r.log2FoldChange >= LFC)).to_numpy()
    sd = ((r.padj.fillna(1) < PADJ) & (r.log2FoldChange <= -LFC)).to_numpy()
    ns = ~(su | sd); nup, ndn = int(su.sum()), int(sd.sum())
    ax.scatter(x[ns], y[ns], s=8, alpha=.3, color='lightgray', edgecolors='none', rasterized=True)
    ax.scatter(x[sd], y[sd], s=22, alpha=.85, color='#1d3557', edgecolors='none', rasterized=True, label=f'down (n={ndn})')
    ax.scatter(x[su], y[su], s=22, alpha=.85, color='#c1121f', edgecolors='none', rasterized=True, label=f'up (n={nup})')
    ax.axvline(LFC, color='k', ls='--', lw=.7, alpha=.5); ax.axvline(-LFC, color='k', ls='--', lw=.7, alpha=.5)
    ax.axhline(-np.log10(PADJ), color='k', ls='--', lw=.7, alpha=.5)
    texts = []
    for gname in list(r.index[(su | sd)][:20]):
        gr = r.loc[gname]; texts.append(ax.text(gr.log2FoldChange, gr.nlq, gname, fontsize=7))
    if HAS_AT and texts:
        try: adjust_text(texts, ax=ax, arrowprops=dict(arrowstyle='-', color='gray', lw=.5))
        except Exception: pass
    ax.set_xlabel(xlab); ax.set_ylabel('-log10 padj')
    ax.set_title(f'{title}\npadj<{PADJ}, |LFC|>={LFC}: {nup} up / {ndn} down', fontsize=11)
    ax.legend(frameon=False, fontsize=8)
    ax.spines[['top', 'right']].set_visible(False)
    return nup, ndn


def repcorr(ax, test, ref, title):
    r1s, r2s = f'{ref}-1', f'{ref}-2'; t1, t2 = f'{test}-1', f'{test}-2'
    keep = counts[[r1s, r2s, t1, t2]].mean(axis=1) >= MINCOUNT
    refm = (cpm[r1s] + cpm[r2s]) / 2.0
    a = (np.log2(cpm[t1] + 1) - np.log2(refm + 1))[keep].to_numpy()
    b = (np.log2(cpm[t2] + 1) - np.log2(refm + 1))[keep].to_numpy()
    pr = pearsonr(a, b)[0]; rho = spearmanr(a, b).correlation
    lim = max(np.nanpercentile(np.abs(np.r_[a, b]), 99.5), 1)
    ax.axhline(0, color='#ccc', lw=.6); ax.axvline(0, color='#ccc', lw=.6)
    ax.plot([-lim, lim], [-lim, lim], '--', color='gray', lw=1, alpha=.7)
    ax.scatter(a, b, s=10, alpha=.4, color='#1982C4', edgecolors='none')
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_aspect('equal', 'box')
    ax.spines[['top', 'right']].set_visible(False)
    ax.set_title(title, fontsize=12)
    ax.set_xlabel(f'LFC vs {ref} (rep 1)', fontsize=10); ax.set_ylabel(f'LFC vs {ref} (rep 2)', fontsize=10)
    ax.text(.04, .96, f'r = {pr:.3f}\nρ = {rho:.3f}\nn = {len(a):,}', transform=ax.transAxes,
            ha='left', va='top', fontsize=9, color='#222')
    return pr, rho, len(a)


rows, cache = [], {}
for tag, test, ref in COMPS:
    pretty = f'{test} vs {ref}'
    res = run_deseq(test, ref, tag); cache[tag] = res

    fig, ax = plt.subplots(figsize=(8, 7))
    nup, ndn = volcano(ax, res, f'{pretty} — DEG (pydeseq2, 20M downsampled)',
                       f'log2 fold change  ({test} / {ref})')
    fig.tight_layout()
    for ext in ('png', 'svg'):
        fig.savefig(f'{FIGD}/volcano/volcano_{tag}.{ext}', dpi=150, bbox_inches='tight')
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.2, 5.2), dpi=130)
    pr, rho, n = repcorr(ax, test, ref, pretty); fig.tight_layout()
    for ext in ('png', 'svg'):
        fig.savefig(f'{FIGD}/rep_corr/repcorr_{tag}.{ext}', dpi=200, bbox_inches='tight')
    plt.close(fig)

    rows.append((tag, test, ref, len(res), nup, ndn, round(pr, 3), round(rho, 3), n))
    print(f'{tag}: genes={len(res):,} up={nup} down={ndn}  repcorr r={pr:.3f}', flush=True)

# ---- montage grids ----
fig, axes = plt.subplots(NROW, NCOL, figsize=(NCOL * 7.2, NROW * 6.2), dpi=110)
for ax, (tag, test, ref) in zip(np.ravel(axes), COMPS):
    volcano(ax, cache[tag], f'{test} vs {ref}', f'log2FC ({test} / {ref})')
fig.suptitle(f'{SUPT} — volcanoes', fontsize=15, y=1.0); fig.tight_layout()
for ext in ('png', 'svg'):
    fig.savefig(f'{FIGD}/volcano_ALL_grid.{ext}', dpi=150, bbox_inches='tight')
plt.close(fig)

fig, axes = plt.subplots(NROW, NCOL, figsize=(NCOL * 5.4, NROW * 5.4), dpi=110)
for ax, (tag, test, ref) in zip(np.ravel(axes), COMPS):
    repcorr(ax, test, ref, f'{test} vs {ref}')
fig.suptitle(f'{SUPT} — replicate reproducibility of LFC', fontsize=15, y=1.0); fig.tight_layout()
for ext in ('png', 'svg'):
    fig.savefig(f'{FIGD}/repcorr_ALL_grid.{ext}', dpi=170, bbox_inches='tight')
plt.close(fig)

summ = pd.DataFrame(rows, columns=['comparison', 'test', 'ref', 'n_genes', 'n_up', 'n_down',
                                   'repcorr_pearson_r', 'repcorr_spearman_rho', 'n_expressed'])
summ.to_csv(f'{OUTD}/DEG_summary.csv', index=False)
print(f'\n==== {MODE} SUMMARY (padj<{PADJ} & |LFC|>={LFC}) ====')
print(summ.to_string(index=False))
print('done')
