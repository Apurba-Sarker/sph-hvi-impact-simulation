import pickle, numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt, matplotlib.gridspec as gridspec
import os
os.makedirs('/home/claude/sph_output', exist_ok=True)

r1 = pickle.load(open('/home/claude/sph_output/AlAl_dx1mm.pkl','rb'))
r2 = pickle.load(open('/home/claude/sph_output/AlCu_dx1mm.pkl','rb'))

PAL = ['#E63946','#2196F3']

def plot_case(res, title_str, exp_crater, exp_aspect, crater_t, aspect_t, fname):
    snaps  = res['history']['snaps']
    t_arr  = res['history']['t']
    cr_arr = res['history']['crater']
    ar_arr = res['history']['aspect']
    KE_arr = np.array(res['history']['KE'])
    IE_arr = np.array(res['history']['IE'])
    N      = res['N']

    idxs = np.linspace(0, len(snaps)-1, 5, dtype=int)

    fig = plt.figure(figsize=(20, 10))
    fig.suptitle(title_str, fontsize=12, fontweight='bold', y=0.99)
    gs = gridspec.GridSpec(2, 5, figure=fig, hspace=0.42, wspace=0.34)

    # ── Row 1: particle snapshots ──────────────────────────────────────
    for col, si in enumerate(idxs):
        ax = fig.add_subplot(gs[0, col])
        sn = snaps[si]
        for mv, c, lbl in [(0, PAL[0], 'Sphere'), (1, PAL[1], 'Plate')]:
            mk = sn['mid'] == mv
            ax.scatter(sn['x'][mk, 0]*100, sn['x'][mk, 1]*100,
                       s=3, c=c, label=lbl, rasterized=True, linewidths=0)
        ax.set_title('t = %.1f µs' % sn['t'], fontsize=9, fontweight='bold')
        ax.set_xlabel('x (cm)', fontsize=7); ax.set_ylabel('y (cm)', fontsize=7)
        ax.tick_params(labelsize=6); ax.set_aspect('equal', 'box')
        ax.axhline(0, color='k', lw=0.4, ls=':', alpha=0.5)
        if col == 0:
            ax.legend(fontsize=5, markerscale=3, loc='upper right')

    # ── Crater vs time ─────────────────────────────────────────────────
    ax = fig.add_subplot(gs[1, 0])
    ax.plot(t_arr, cr_arr, 'b-o', ms=5, lw=2, label='SPH')
    ax.axhline(exp_crater, color='r', ls='--', lw=2, label='Exp: %.2f cm' % exp_crater)
    ax.axvline(crater_t, color='g', ls=':', lw=1.5)
    best_c = cr_arr[np.argmin(np.abs(np.array(t_arr) - crater_t))]
    err_c  = abs(best_c - exp_crater) / exp_crater * 100
    ax.scatter([crater_t], [best_c], s=120, c='g', zorder=5)
    ax.annotate('%.3f cm\n(%.1f%%)' % (best_c, err_c),
                xy=(crater_t, best_c),
                xytext=(crater_t + 4, best_c + 0.3),
                arrowprops=dict(arrowstyle='->', color='green'),
                fontsize=8, color='green')
    ax.set_xlabel('t (µs)'); ax.set_ylabel('Crater (cm)')
    ax.set_title('Crater diameter'); ax.legend(fontsize=7); ax.grid(alpha=0.3)

    # ── Debris l/w vs time ─────────────────────────────────────────────
    ax = fig.add_subplot(gs[1, 1])
    valid = [(t, a) for t, a in zip(t_arr, ar_arr) if a > 0.01]
    if valid:
        tv, av = zip(*valid)
        tv, av = list(tv), list(av)
        ax.plot(tv, av, 'g-o', ms=5, lw=2, label='SPH')
        ax.axhline(exp_aspect, color='r', ls='--', lw=2, label='Exp: %.2f' % exp_aspect)
        ax.axvline(aspect_t, color='purple', ls=':', lw=1.5)
        best_a = av[np.argmin(np.abs(np.array(tv) - aspect_t))]
        err_a  = abs(best_a - exp_aspect) / exp_aspect * 100
        ax.scatter([aspect_t], [best_a], s=120, c='purple', zorder=5)
        ax.annotate('%.3f\n(%.1f%%)' % (best_a, err_a),
                    xy=(aspect_t, best_a),
                    xytext=(aspect_t + 3, best_a + 0.05),
                    arrowprops=dict(arrowstyle='->', color='purple'),
                    fontsize=8, color='purple')
    ax.set_xlabel('t (µs)'); ax.set_ylabel('Aspect ratio l/w')
    ax.set_title('Debris cloud shape'); ax.legend(fontsize=7); ax.grid(alpha=0.3)

    # ── Energy history ─────────────────────────────────────────────────
    ax = fig.add_subplot(gs[1, 2])
    TE = KE_arr + IE_arr; TE0 = TE[0] if TE[0] != 0 else 1.0
    st = np.arange(len(TE))
    ax.plot(st, TE/TE0, 'k-', lw=1.5, label='Total')
    ax.plot(st, KE_arr/TE0, 'b--', lw=1, label='KE')
    ax.plot(st, IE_arr/TE0, 'r--', lw=1, label='IE')
    ax.axhline(1.0, color='gray', ls=':', lw=0.8)
    ax.set_xlabel('Snapshot #'); ax.set_ylabel('E / E₀')
    ax.set_title('Energy history\n(drift in debris phase, see notes)')
    ax.legend(fontsize=7); ax.grid(alpha=0.3)

    # ── Speed field at representative snapshot ──────────────────────────
    # Use snapshot near crater_t for Al-Cu, last snapshot for Al-Al
    t_speed = crater_t if 'Cu' in title_str else 8.0
    sn_sp = snaps[np.argmin(np.abs(np.array(t_arr) - t_speed))]
    ax = fig.add_subplot(gs[1, 3])
    spd = np.sqrt((sn_sp['v']**2).sum(axis=1))
    sc  = ax.scatter(sn_sp['x'][:, 0]*100, sn_sp['x'][:, 1]*100,
                     c=np.clip(spd/1e3, 0, 14), s=3, cmap='hot', rasterized=True)
    plt.colorbar(sc, ax=ax, label='Speed (km/s)', shrink=0.8)
    ax.set_xlabel('x (cm)'); ax.set_ylabel('y (cm)')
    ax.set_title('Speed at t=%.0fµs' % sn_sp['t']); ax.set_aspect('equal', 'box')

    # ── Summary table ───────────────────────────────────────────────────
    ax = fig.add_subplot(gs[1, 4]); ax.axis('off')
    if valid:
        best_a2 = av[np.argmin(np.abs(np.array(tv) - aspect_t))]
        err_a2  = abs(best_a2 - exp_aspect) / exp_aspect * 100
    else:
        best_a2, err_a2 = 0, 999.
    best_c2 = cr_arr[np.argmin(np.abs(np.array(t_arr) - crater_t))]
    err_c2  = abs(best_c2 - exp_crater) / exp_crater * 100

    table_data = [
        ['Crater (cm) @ t=%.0fµs' % crater_t,
         '%.3f' % best_c2, '%.2f' % exp_crater, '%.1f%%' % err_c2],
        ['Debris l/w @ t=%.0fµs' % aspect_t,
         '%.3f' % best_a2 if best_a2 > 0 else 'N/A',
         '%.2f' % exp_aspect, '%.1f%%' % err_a2 if best_a2 > 0 else 'N/A'],
        ['N particles', str(N), '—', '—'],
        ['Wall time', '%.0fs' % res['wall_s'], '—', '—'],
        ['Steps', str(res['n_steps']), '—', '—'],
    ]
    col_labels = ['Metric', 'SPH', 'Experiment', 'Error']
    tbl = ax.table(cellText=table_data, colLabels=col_labels,
                   loc='center', cellLoc='center', bbox=[0, 0, 1, 1])
    tbl.auto_set_font_size(False); tbl.set_fontsize(8.5)
    for (rr, cc), cell in tbl.get_celld().items():
        if rr == 0:
            cell.set_facecolor('#1a4e8c')
            cell.set_text_props(color='w', fontweight='bold')
        elif cc == 3 and rr > 0:
            txt = table_data[rr-1][3]
            try:
                val = float(txt.replace('%', '').replace('N/A','999'))
                bg  = '#d4edda' if val < 10 else '#fff3cd' if val < 25 else '#f8d7da'
            except Exception:
                bg = '#f8f9fa'
            cell.set_facecolor(bg)
    ax.set_title('Results Summary', fontsize=9, fontweight='bold', pad=6)

    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close()
    print('Saved', fname)


plot_case(
    r1,
    'SPH HVI: Al-Al  (1cm Al sphere @ 6.18 km/s on 4mm Al plate, dx=1mm, N=585)',
    exp_crater=3.10, exp_aspect=1.39,
    crater_t=4.0, aspect_t=8.0,
    fname='/home/claude/sph_output/Fig1_AlAl.png'
)

plot_case(
    r2,
    'SPH HVI: Al-Cu  (1cm Al sphere @ 5.75 km/s on 1.5mm Cu plate, dx=1mm, N=322)',
    exp_crater=2.12, exp_aspect=1.39,
    crater_t=2.0, aspect_t=4.0,
    fname='/home/claude/sph_output/Fig2_AlCu.png'
)
print('All figures done.')
