"""
make_figures3.py — fixes:
- Fig 1 & 2: legend top-left of first snapshot panel
- Fig 3: distinct markers (o, s, ^) per line; no log×10^0 on axes;
         wall-time y-axis shows all tick labels; single shared legend
"""

import pickle, numpy as np, os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from matplotlib.ticker import LogFormatter, LogLocator, NullFormatter
import matplotlib.ticker as ticker

OUTDIR = '/home/claude'
os.makedirs(OUTDIR, exist_ok=True)

PAL_SPH  = '#E63946'
PAL_PLT  = '#2196F3'
EXP_COL  = '#d62728'
EVAL_COL = '#2ca02c'
ASP_COL  = '#9467bd'

r1 = pickle.load(open('/mnt/user-data/uploads/AlAl_dx1mm.pkl', 'rb'))
r2 = pickle.load(open('/mnt/user-data/uploads/AlCu_dx1mm.pkl', 'rb'))

def set_rc(scale):
    plt.rcParams.update({
        'text.usetex':      True,
        'font.family':      'serif',
        'font.serif':       ['Computer Modern Roman'],
        'font.size':        13 * scale,
        'axes.titlesize':   13 * scale,
        'axes.labelsize':   13 * scale,
        'xtick.labelsize':  11 * scale,
        'ytick.labelsize':  11 * scale,
        'legend.fontsize':  11 * scale,
        'figure.dpi':       150,
        'lines.linewidth':  2.0 * scale,
        'lines.markersize': 5  * scale,
    })

def plain_log_axis(ax, which='both'):
    """Remove the ×10^0 multiplier; show plain numbers on log axes."""
    fmt = ticker.LogFormatterSciNotation(base=10, labelOnlyBase=False)
    # Use ScalarFormatter so e.g. 100, 200, 500 show as-is, not ×10^n
    scalar = ticker.ScalarFormatter()
    scalar.set_scientific(False)
    if which in ('x', 'both'):
        ax.xaxis.set_major_formatter(scalar)
        ax.xaxis.set_minor_formatter(NullFormatter())
    if which in ('y', 'both'):
        ax.yaxis.set_major_formatter(scalar)
        ax.yaxis.set_minor_formatter(NullFormatter())

# ─────────────────────────────────────────────────────────────────────────────
def make_case_figure(res, exp_crater, exp_aspect, crater_t, aspect_t,
                     snap_times, fname):
    set_rc(2.0)

    snaps  = res['history']['snaps']
    t_arr  = np.array(res['history']['t'])
    cr_arr = np.array(res['history']['crater'])
    ar_arr = np.array(res['history']['aspect'])
    KE_arr = np.array(res['history']['KE'])
    IE_arr = np.array(res['history']['IE'])

    snap_idx = [np.argmin(np.abs(t_arr - tt)) for tt in snap_times]

    fig = plt.figure(figsize=(28, 13))
    gs  = gridspec.GridSpec(2, 5, figure=fig,
                            hspace=0.52, wspace=0.38,
                            left=0.06, right=0.97,
                            top=0.90, bottom=0.09)

    # ── Shared particle legend — anchored above the figure, not inside any axis ──
    legend_handles = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=PAL_SPH,
               markersize=22, label='Sphere'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=PAL_PLT,
               markersize=22, label='Plate'),
    ]

    # Row 0: snapshots 0,1,2
    for col in range(3):
        ax = fig.add_subplot(gs[0, col])
        si = snap_idx[col]
        sn = snaps[si]
        for mv, c in [(0, PAL_SPH), (1, PAL_PLT)]:
            mk = sn['mid'] == mv
            ax.scatter(sn['x'][mk,0]*100, sn['x'][mk,1]*100,
                       s=6, c=c, rasterized=True, linewidths=0)
        ax.set_title(r'$t = %.1f~\mu\mathrm{s}$' % sn['t'], pad=5)
        ax.set_xlabel(r'$x$ (cm)')
        ax.set_ylabel(r'$y$ (cm)')
        ax.set_aspect('equal', 'box')
        ax.axhline(0, color='k', lw=0.6, ls=':', alpha=0.5)

    # Legend placed in the white margin above the top-left subplot — never overlaps data
    fig.legend(handles=legend_handles,
               loc='upper left',
               bbox_to_anchor=(0.01, 1.00),
               bbox_transform=fig.transFigure,
               ncol=1, framealpha=0.92, fontsize=26,
               handletextpad=0.4, borderpad=0.5, labelspacing=0.35)

    # Row 0 col 3: crater
    ax = fig.add_subplot(gs[0, 3])
    ax.plot(t_arr, cr_arr, '-o', color=PAL_SPH, ms=7, lw=2.5, label='SPH')
    ax.axhline(exp_crater, color=EXP_COL, ls='--', lw=2.5,
               label=r'Exp.\ $%.2f$~cm' % exp_crater)
    ax.axvline(crater_t, color=EVAL_COL, ls=':', lw=2)
    i_c = np.argmin(np.abs(t_arr - crater_t))
    ax.scatter([t_arr[i_c]], [cr_arr[i_c]], s=150, c=EVAL_COL, zorder=5)
    err_c = abs(cr_arr[i_c] - exp_crater) / exp_crater * 100
    ax.annotate(r'$%.3f$~cm ($%.1f\%%$)' % (cr_arr[i_c], err_c),
                xy=(t_arr[i_c], cr_arr[i_c]),
                xytext=(t_arr[i_c] + max(t_arr)*0.15, cr_arr[i_c] + 0.3),
                arrowprops=dict(arrowstyle='->', color=EVAL_COL, lw=1.5),
                color=EVAL_COL, fontsize=20)
    ax.set_xlabel(r'$t$ ($\mu$s)')
    ax.set_ylabel(r'Crater diameter (cm)')
    ax.set_title(r'Crater diameter')
    ax.legend(loc='upper left', framealpha=0.7)
    ax.grid(alpha=0.3)

    # Row 0 col 4: aspect ratio
    ax = fig.add_subplot(gs[0, 4])
    valid = [(t, a) for t, a in zip(t_arr, ar_arr) if a > 0.01]
    if valid:
        tv, av = zip(*valid)
        ax.plot(tv, av, '-o', color='#2ca02c', ms=7, lw=2.5, label='SPH')
        ax.axhline(exp_aspect, color=EXP_COL, ls='--', lw=2.5,
                   label=r'Exp.\ $%.2f$' % exp_aspect)
        ax.axvline(aspect_t, color=ASP_COL, ls=':', lw=2)
        i_a = np.argmin(np.abs(np.array(tv) - aspect_t))
        ax.scatter([tv[i_a]], [av[i_a]], s=150, c=ASP_COL, zorder=5)
        err_a = abs(av[i_a] - exp_aspect) / exp_aspect * 100
        ax.annotate(r'$%.3f$ ($%.1f\%%$)' % (av[i_a], err_a),
                    xy=(tv[i_a], av[i_a]),
                    xytext=(tv[i_a] + max(tv)*0.15, av[i_a] + 0.05),
                    arrowprops=dict(arrowstyle='->', color=ASP_COL, lw=1.5),
                    color=ASP_COL, fontsize=20)
        ax.legend(loc='lower right', framealpha=0.7)
    ax.set_xlabel(r'$t$ ($\mu$s)')
    ax.set_ylabel(r'Debris aspect ratio $l/w$')
    ax.set_title(r'Debris cloud shape')
    ax.grid(alpha=0.3)

    # Row 1: snapshots 3,4
    for col in range(2):
        ax = fig.add_subplot(gs[1, col])
        si = snap_idx[3 + col]
        sn = snaps[si]
        for mv, c in [(0, PAL_SPH), (1, PAL_PLT)]:
            mk = sn['mid'] == mv
            ax.scatter(sn['x'][mk,0]*100, sn['x'][mk,1]*100,
                       s=6, c=c, rasterized=True, linewidths=0)
        ax.set_title(r'$t = %.1f~\mu\mathrm{s}$' % sn['t'], pad=5)
        ax.set_xlabel(r'$x$ (cm)')
        ax.set_ylabel(r'$y$ (cm)')
        ax.set_aspect('equal', 'box')
        ax.axhline(0, color='k', lw=0.6, ls=':', alpha=0.5)

    # Row 1 cols 2-3: energy
    ax = fig.add_subplot(gs[1, 2:4])
    TE  = KE_arr + IE_arr
    TE0 = TE[0] if TE[0] != 0 else 1.0
    st  = np.arange(len(TE))
    ax.plot(st, TE/TE0,     'k-',  lw=2.5, label=r'Total $E$')
    ax.plot(st, KE_arr/TE0, '--',  color='#1f77b4', lw=2, label=r'$KE$')
    ax.plot(st, IE_arr/TE0, '--',  color='#d62728', lw=2, label=r'$IE$')
    ax.axhline(1.0, color='gray', ls=':', lw=1)
    ax.set_xlabel(r'Time step index')
    ax.set_ylabel(r'$E\,/\,E_0$')
    ax.set_title(r'Energy history')
    ax.legend(framealpha=0.7)
    ax.grid(alpha=0.3)

    # Row 1 col 4: speed field
    ax = fig.add_subplot(gs[1, 4])
    sn_sp = snaps[np.argmin(np.abs(t_arr - crater_t))]
    spd = np.sqrt((sn_sp['v']**2).sum(axis=1))
    sc  = ax.scatter(sn_sp['x'][:,0]*100, sn_sp['x'][:,1]*100,
                     c=np.clip(spd/1e3, 0, 14), s=6,
                     cmap='hot', rasterized=True)
    cb = plt.colorbar(sc, ax=ax, shrink=0.85)
    cb.set_label(r'Speed (km/s)')
    ax.set_xlabel(r'$x$ (cm)')
    ax.set_ylabel(r'$y$ (cm)')
    ax.set_title(r'Speed at $t=%.0f~\mu\mathrm{s}$' % sn_sp['t'])
    ax.set_aspect('equal', 'box')

    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved {fname}')


make_case_figure(r1, exp_crater=3.10, exp_aspect=1.39,
                 crater_t=4.0, aspect_t=8.0,
                 snap_times=[0.0, 4.0, 8.0, 12.0, 18.0],
                 fname=os.path.join(OUTDIR, 'Fig1_AlAl.png'))

make_case_figure(r2, exp_crater=2.12, exp_aspect=1.39,
                 crater_t=2.0, aspect_t=4.0,
                 snap_times=[0.0, 2.0, 4.0, 10.0, 20.0],
                 fname=os.path.join(OUTDIR, 'Fig2_AlCu.png'))

# ─────────────────────────────────────────────────────────────────────────────
# Figure 3: Convergence  (1.3x fonts)
# Three data points per panel; each point gets its OWN marker shape.
# Shared legend at top: one entry per case (dx / N).
# ─────────────────────────────────────────────────────────────────────────────
set_rc(1.3)

spacings   = [1.0, 0.75, 0.5]
fnames_pkl = [
    '/mnt/user-data/uploads/result_Al_Al_dx1_0mm.pkl',
    '/mnt/user-data/uploads/result_Al_Al_dx0_75mm.pkl',
    '/mnt/user-data/uploads/result_Al_Al_dx0_5mm.pkl',
]
craters6 = []; aspects6 = []; Ns = []; walls = []
for dx, fn in zip(spacings, fnames_pkl):
    r = pickle.load(open(fn, 'rb'))
    h = r['history']
    ta = np.array(h['t']); ca = np.array(h['crater']); aa = np.array(h['aspect'])
    i6 = np.argmin(np.abs(ta - 6.0))
    craters6.append(ca[i6]); aspects6.append(aa[i6])
    Ns.append(r['N']); walls.append(r['wall_s'])

dx_arr = np.array(spacings)
N_arr  = np.array(Ns)
W_arr  = np.array(walls)
p_c = np.polyfit(np.log(dx_arr), np.log(craters6), 1)
p_w = np.polyfit(np.log(N_arr),  np.log(W_arr),    1)

# One marker shape and color per CASE (dx / N)
PT_MARKERS = ['o', 's', '^']           # circle=dx1.0, square=dx0.75, triangle=dx0.5
PT_COLOR   = '#1f77b4'                  # same line color throughout each panel
LINE_COLOR_WALL = '#d62728'
PT_SIZE    = 13

def apply_scalar_fmt(ax, which='both'):
    sf = ticker.ScalarFormatter()
    sf.set_scientific(False)
    if which in ('x','both'):
        ax.xaxis.set_major_formatter(sf)
        ax.xaxis.set_minor_formatter(ticker.NullFormatter())
    if which in ('y','both'):
        sf2 = ticker.ScalarFormatter(); sf2.set_scientific(False)
        ax.yaxis.set_major_formatter(sf2)
        ax.yaxis.set_minor_formatter(ticker.NullFormatter())

fig, axes = plt.subplots(1, 3, figsize=(18, 6),
                         gridspec_kw=dict(left=0.08, right=0.97,
                                          top=0.86, bottom=0.14,
                                          wspace=0.42))

# ── Panel 1: crater vs dx ─────────────────────────────────────────────
ax = axes[0]
# connecting line first (behind markers)
ax.plot(dx_arr, craters6, '-', color=PT_COLOR, lw=2, zorder=1)
# fit line
dx_fit = np.linspace(0.45, 1.1, 80)
ax.plot(dx_fit, np.exp(np.polyval(p_c, np.log(dx_fit))),
        ':', color='gray', lw=1.8, alpha=0.8, zorder=1)
# experimental reference
ax.axhline(3.10, color=EXP_COL, ls='--', lw=2.5, zorder=1)
# each point with its own marker
for dx, c, n, mk in zip(spacings, craters6, Ns, PT_MARKERS):
    ax.scatter([dx], [c], marker=mk, s=PT_SIZE**2,
               color=PT_COLOR, zorder=3, linewidths=1.2, edgecolors='k')
ax.set_xscale('log'); ax.set_yscale('log')
ax.set_xlabel(r'$\Delta x$ (mm)')
ax.set_ylabel(r'Crater diameter (cm)')
ax.set_title(r'Crater diameter at $t=6~\mu\mathrm{s}$')
ax.grid(which='both', alpha=0.3)
apply_scalar_fmt(ax)

# ── Panel 2: aspect ratio vs dx ──────────────────────────────────────
ax = axes[1]
ax.plot(dx_arr, aspects6, '-', color=PT_COLOR, lw=2, zorder=1)
ax.axhline(1.39, color=EXP_COL, ls='--', lw=2.5, zorder=1)
for dx, a, mk in zip(spacings, aspects6, PT_MARKERS):
    ax.scatter([dx], [a], marker=mk, s=PT_SIZE**2,
               color=PT_COLOR, zorder=3, linewidths=1.2, edgecolors='k')
ax.set_xscale('log'); ax.set_yscale('log')
ax.set_xlabel(r'$\Delta x$ (mm)')
ax.set_ylabel(r'Debris aspect ratio $l/w$')
ax.set_title(r'Debris aspect ratio at $t=6~\mu\mathrm{s}$')
ax.grid(which='both', alpha=0.3)
apply_scalar_fmt(ax)

# ── Panel 3: wall time vs N ───────────────────────────────────────────
ax = axes[2]
ax.plot(N_arr, W_arr, '-', color=LINE_COLOR_WALL, lw=2, zorder=1)
N_fit = np.linspace(450, 2300, 100)
ax.plot(N_fit, np.exp(np.polyval(p_w, np.log(N_fit))),
        ':', color='gray', lw=1.8, alpha=0.8, zorder=1,
        label=r'Fit: $t \propto N^{%.2f}$' % p_w[0])
for n, w, mk in zip(Ns, walls, PT_MARKERS):
    ax.scatter([n], [w], marker=mk, s=PT_SIZE**2,
               color=LINE_COLOR_WALL, zorder=3, linewidths=1.2, edgecolors='k')
ax.set_xscale('log'); ax.set_yscale('log')
ax.set_xlabel(r'Number of particles $N$')
ax.set_ylabel(r'Wall-clock time (s)')
ax.set_title(r'Computational cost ($t_\mathrm{wall} \propto N^{%.2f}$)' % p_w[0])
ax.grid(which='both', alpha=0.3)
ax.set_ylim(70, 900)
ax.set_yticks([100, 200, 300, 400, 500, 600, 700, 800])
apply_scalar_fmt(ax)

# ── Single shared legend: one entry per case (marker shape + label) ───
fs = 13 * 1.3
legend_handles = [
    Line2D([0], [0], marker=PT_MARKERS[0], color=PT_COLOR,
           markeredgecolor='k', markeredgewidth=1.0,
           linestyle='None', markersize=fs*0.6,
           label=r'$\Delta x = 1.00$\,mm  ($N = 585$)'),
    Line2D([0], [0], marker=PT_MARKERS[1], color=PT_COLOR,
           markeredgecolor='k', markeredgewidth=1.0,
           linestyle='None', markersize=fs*0.6,
           label=r'$\Delta x = 0.75$\,mm  ($N = 941$)'),
    Line2D([0], [0], marker=PT_MARKERS[2], color=PT_COLOR,
           markeredgecolor='k', markeredgewidth=1.0,
           linestyle='None', markersize=fs*0.6,
           label=r'$\Delta x = 0.50$\,mm  ($N = 2125$)'),
    Line2D([0], [0], linestyle='--', color=EXP_COL, lw=2.2,
           label=r'Experimental reference'),
    Line2D([0], [0], linestyle=':', color='gray', lw=1.8,
           label=r'Power-law fit'),
]
fig.legend(handles=legend_handles, loc='upper center',
           ncol=5, framealpha=0.9,
           bbox_to_anchor=(0.5, 1.01),
           fontsize=fs * 0.78,
           handletextpad=0.4, columnspacing=0.9)

plt.savefig(os.path.join(OUTDIR, 'convergence_Al_Al.png'),
            dpi=150, bbox_inches='tight')
plt.close()
print('Saved convergence_Al_Al.png')
print('All done.')