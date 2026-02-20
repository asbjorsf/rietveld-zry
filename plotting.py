"""
Publication-quality Rietveld plot.

Layout (top → bottom)
----------------------
  Main panel  : observed (crosses), calculated (line), background (dashed)
  Tick panel  : reflection positions per phase (vertical tick marks)
  Difference  : y_obs - y_calc
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

from rietveld_engine import generate_reflections


_PHASE_COLORS = ['tab:blue', 'tab:green', 'tab:orange', 'tab:purple']


def plot_rietveld(ref, title=None, save_path=None, dpi=150):
    """
    Draw the standard Rietveld three-panel plot.

    Parameters
    ----------
    ref        : RietveldRefinement instance (already refined)
    title      : optional figure title
    save_path  : if given, save figure to this path
    dpi        : resolution for saved figure
    """
    two_theta = ref.two_theta
    y_obs     = ref.y_obs
    y_calc    = ref.calc()
    diff      = y_obs - y_calc

    # background alone
    from rietveld_engine import chebyshev_background
    y_bg = chebyshev_background(two_theta, ref.bg_coeffs,
                                two_theta.min(), two_theta.max())

    n_phases = len(ref.phases)
    height_ratios = [4, 0.35 * n_phases, 1.5]

    fig = plt.figure(figsize=(12, 7))
    gs  = gridspec.GridSpec(3, 1, height_ratios=height_ratios, hspace=0.04)

    ax_main = fig.add_subplot(gs[0])
    ax_tick = fig.add_subplot(gs[1], sharex=ax_main)
    ax_diff = fig.add_subplot(gs[2], sharex=ax_main)

    # ── Main panel ──────────────────────────────────────────────────────────
    ax_main.plot(two_theta, y_obs,  'k+', ms=2.5, lw=0.5,
                 label='Observed', zorder=3)
    ax_main.plot(two_theta, y_calc, 'r-', lw=1.2,
                 label='Calculated', zorder=4)
    ax_main.plot(two_theta, y_bg,   'b--', lw=0.8, alpha=0.6,
                 label='Background', zorder=2)

    ax_main.set_ylabel('Intensity (arb. units)', fontsize=12)
    ax_main.legend(fontsize=9, loc='upper right')
    ax_main.set_xlim(two_theta.min(), two_theta.max())
    ax_main.yaxis.set_minor_locator(plt.MultipleLocator(1000))

    if title:
        ax_main.set_title(title, fontsize=13, pad=6)

    # R-factor annotation
    from rietveld_engine import rwp as _rwp, rp as _rp
    _r_wp = _rwp(y_obs, y_calc, ref.weights)
    _r_p  = _rp(y_obs, y_calc)
    ax_main.text(0.985, 0.96,
                 f'$R_{{wp}}$ = {_r_wp:.4f}\n$R_p$     = {_r_p:.4f}',
                 transform=ax_main.transAxes,
                 ha='right', va='top', fontsize=9,
                 bbox=dict(boxstyle='round,pad=0.3',
                           facecolor='lightyellow', alpha=0.8))

    # ── Tick-mark panel ──────────────────────────────────────────────────────
    n = n_phases
    y_centres = np.linspace(0.75, 0.25, n)

    for i, ph in enumerate(ref.phases):
        refs = generate_reflections(ph, ref.wavelength,
                                    two_theta.max() + 2,
                                    two_theta.min() - 2)
        positions = [r['two_theta'] for r in refs]
        color = _PHASE_COLORS[i % len(_PHASE_COLORS)]
        yc = y_centres[i]
        ax_tick.vlines(positions, yc - 0.12, yc + 0.12,
                       colors=color, lw=0.7,
                       label=ph.name.capitalize() + ' ZrO₂')

    ax_tick.set_ylim(0, 1)
    ax_tick.set_yticks([])
    ax_tick.legend(fontsize=8, loc='upper right',
                   ncol=n, framealpha=0.7)
    ax_tick.set_ylabel('hkl', fontsize=9)

    # ── Difference panel ─────────────────────────────────────────────────────
    ax_diff.plot(two_theta, diff, color='dimgray', lw=0.7)
    ax_diff.axhline(0, color='k', lw=0.6, ls='--')
    ax_diff.set_ylabel('Difference', fontsize=10)
    ax_diff.set_xlabel(r'2$\theta$ (degrees)', fontsize=12)

    # hide x-tick labels on upper panels
    plt.setp(ax_main.get_xticklabels(), visible=False)
    plt.setp(ax_tick.get_xticklabels(), visible=False)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print(f'Figure saved to {save_path}')

    plt.show()
    return fig


def plot_all_samples(results, save_dir=None):
    """
    Quick overview: plot Rwp and phase fractions vs sample name.

    Parameters
    ----------
    results  : list of dicts with keys 'name', 'rwp', 'wt_cubic', 'wt_tet', 'a_cubic', 'a_tet', 'c_tet'
    save_dir : directory for saved figures (optional)
    """
    names    = [r['name']     for r in results]
    rwps     = [r['rwp']      for r in results]
    wt_c     = [r['wt_cubic'] for r in results]
    wt_t     = [r['wt_tet']   for r in results]
    x        = np.arange(len(names))

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    axes[0].bar(x - 0.2, wt_c, 0.4, label='Cubic',      color='tab:blue')
    axes[0].bar(x + 0.2, wt_t, 0.4, label='Tetragonal', color='tab:green')
    axes[0].set_ylabel('Weight fraction (%)')
    axes[0].legend()
    axes[0].set_title('Phase fractions')

    axes[1].plot(x, rwps, 'ko-', ms=5)
    axes[1].set_ylabel('$R_{wp}$')
    axes[1].set_title('Goodness of fit')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(names, rotation=30, ha='right')

    plt.tight_layout()
    if save_dir:
        p = Path(save_dir) / 'summary.png'
        plt.savefig(p, dpi=150, bbox_inches='tight')
        print(f'Summary figure saved to {p}')
    plt.show()
    return fig
