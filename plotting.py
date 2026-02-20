"""
Rietveld plot functions for Zr(1-x)Y(x)O(2-x/2).

Functions
---------
plot_rietveld             : standard 3-panel plot (obs/calc/diff + hkl ticks)
plot_phase_contributions  : per-phase contribution panels
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

from rietveld_engine import (generate_reflections, chebyshev_background,
                              calculate_pattern, rwp as _rwp, rp as _rp)

_PHASE_COLORS = ['tab:blue', 'tab:green', 'tab:orange', 'tab:purple']


# ─── Helper: per-phase pattern (no background) ───────────────────────────────

def _phase_pattern(ref, phase_idx):
    """
    Return the pattern contribution from one phase only (background excluded).
    """
    ph  = ref.phases[phase_idx]
    sc  = ref.scale_factors[phase_idx]
    return calculate_pattern(
        ref.two_theta, [ph], [sc],
        bg_coeffs=np.zeros_like(ref.bg_coeffs),   # no background
        U=ref.U, V=ref.V, W=ref.W,
        eta=ref.eta, zero_offset=ref.zero_offset,
        wavelength=ref.wavelength,
    )


# ─── Main Rietveld plot ───────────────────────────────────────────────────────

def plot_rietveld(ref, title=None, save_path=None, dpi=150):
    """
    Three-panel Rietveld plot.

    Panel 1 (main)
      ○  Observed data          open black circles
      ─  Calculated pattern     red line
      -- Background             blue dashed

    Panel 2  hkl tick marks for each phase (short vertical lines)

    Panel 3  Difference  (y_obs − y_calc)
    """
    tth    = ref.two_theta
    y_obs  = ref.y_obs
    y_calc = ref.calc()
    diff   = y_obs - y_calc
    y_bg   = chebyshev_background(tth, ref.bg_coeffs)

    n_ph   = len(ref.phases)

    # Layout: main | ticks (one row per phase) | diff
    hr = [5] + [0.4] * n_ph + [1.8]
    fig = plt.figure(figsize=(12, 7))
    gs  = gridspec.GridSpec(2 + n_ph, 1, height_ratios=hr, hspace=0.0)

    ax_main  = fig.add_subplot(gs[0])
    ax_ticks = [fig.add_subplot(gs[1 + i], sharex=ax_main) for i in range(n_ph)]
    ax_diff  = fig.add_subplot(gs[1 + n_ph], sharex=ax_main)

    # ── Main panel ────────────────────────────────────────────────────────────
    ax_main.plot(tth, y_obs,
                 'o', ms=2.0, markerfacecolor='none', markeredgecolor='black',
                 markeredgewidth=0.6, lw=0, label='Observed', zorder=3)
    ax_main.plot(tth, y_calc, '-', color='red', lw=1.3,
                 label='Calculated', zorder=4)
    ax_main.plot(tth, y_bg, '--', color='steelblue', lw=0.9, alpha=0.7,
                 label='Background', zorder=2)

    ax_main.set_ylabel('Intensity (arb. units)', fontsize=12)
    ax_main.legend(fontsize=9, loc='upper right', framealpha=0.85)
    ax_main.set_xlim(tth.min(), tth.max())
    if title:
        ax_main.set_title(title, fontsize=13, pad=5)

    # R-factor box
    r_wp = _rwp(y_obs, y_calc, ref.weights)
    r_p  = _rp(y_obs, y_calc)
    ax_main.text(0.985, 0.97,
                 f'$R_{{wp}}$ = {r_wp:.4f}\n$R_p$     = {r_p:.4f}',
                 transform=ax_main.transAxes,
                 ha='right', va='top', fontsize=9,
                 bbox=dict(boxstyle='round,pad=0.3',
                           facecolor='lightyellow', edgecolor='gray',
                           alpha=0.9))

    # ── hkl tick panels ───────────────────────────────────────────────────────
    for i, (ph, ax_tk) in enumerate(zip(ref.phases, ax_ticks)):
        refs = generate_reflections(ph, ref.wavelength,
                                    tth.max() + 2, tth.min() - 2)
        positions = [r['two_theta'] for r in refs]
        color = _PHASE_COLORS[i % len(_PHASE_COLORS)]

        ax_tk.vlines(positions, 0.05, 0.95, colors=color, lw=0.8)
        ax_tk.set_ylim(0, 1)
        ax_tk.set_yticks([])
        ax_tk.set_ylabel(f'{ph.name.capitalize()[:3]}.',
                         fontsize=8, rotation=0, labelpad=22, va='center')
        ax_tk.tick_params(bottom=False, labelbottom=False)
        for spine in ['top', 'bottom', 'right']:
            ax_tk.spines[spine].set_visible(False)

    # ── Difference panel ──────────────────────────────────────────────────────
    ax_diff.plot(tth, diff, '-', color='dimgray', lw=0.8)
    ax_diff.axhline(0, color='black', lw=0.6, ls='--', alpha=0.5)
    ax_diff.set_ylabel('Difference', fontsize=10)
    ax_diff.set_xlabel(r'2$\theta$ (degrees)', fontsize=12)

    # Hide x-tick labels on upper panels
    for ax in [ax_main] + ax_ticks:
        plt.setp(ax.get_xticklabels(), visible=False)
        ax.tick_params(bottom=False)

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.0)

    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print(f'  Rietveld plot saved → {Path(save_path).name}')

    plt.show()
    return fig


# ─── Phase contribution plots ─────────────────────────────────────────────────

def plot_phase_contributions(ref, title=None, save_path=None, dpi=150):
    """
    One subplot per phase showing its individual pattern contribution.

    Each subplot shows:
      ○  Observed data (open black circles)
      ─  Total calculated pattern (red, thin)
      ■  Phase contribution + background (filled, phase colour)

    This makes it easy to see which peaks are attributed to which phase.
    """
    tth   = ref.two_theta
    y_obs = ref.y_obs
    y_bg  = chebyshev_background(tth, ref.bg_coeffs)

    n_ph  = len(ref.phases)
    phase_patterns = [_phase_pattern(ref, i) for i in range(n_ph)]
    y_calc_total   = ref.calc()

    fig, axes = plt.subplots(n_ph, 1,
                             figsize=(12, 4.5 * n_ph),
                             sharex=True)
    if n_ph == 1:
        axes = [axes]

    phase_labels = ['Cubic ZrO₂ (Fm-3m)',
                    'Tetragonal ZrO₂ (P4₂/nmc)']

    for i, (ax, ph, y_ph) in enumerate(zip(axes, ref.phases, phase_patterns)):
        color = _PHASE_COLORS[i % len(_PHASE_COLORS)]

        # Observed
        ax.plot(tth, y_obs,
                'o', ms=2.0, markerfacecolor='none', markeredgecolor='black',
                markeredgewidth=0.5, lw=0, label='Observed', zorder=3)

        # Total calculated (thin red, for reference)
        ax.plot(tth, y_calc_total, '-', color='red', lw=0.8, alpha=0.5,
                label='Total calculated', zorder=4)

        # This phase's contribution + background (filled area)
        y_contrib = y_ph + y_bg
        ax.fill_between(tth, y_bg, y_contrib,
                        color=color, alpha=0.35, zorder=2)
        ax.plot(tth, y_contrib, '-', color=color, lw=1.2,
                label=f'{phase_labels[i] if i < len(phase_labels) else ph.name} + bg',
                zorder=2)

        # Background line
        ax.plot(tth, y_bg, '--', color='steelblue', lw=0.8, alpha=0.6,
                label='Background', zorder=1)

        # hkl tick marks along the bottom
        refs_hkl = generate_reflections(ph, ref.wavelength,
                                        tth.max() + 2, tth.min() - 2)
        tick_y = y_obs.min() - 0.04 * (y_obs.max() - y_obs.min())
        tick_h = 0.03 * (y_obs.max() - y_obs.min())
        positions = [r['two_theta'] for r in refs_hkl]
        ax.vlines(positions, tick_y - tick_h, tick_y + tick_h,
                  colors=color, lw=0.8, zorder=5)

        ax.set_xlim(tth.min(), tth.max())
        ax.set_ylabel('Intensity (arb. units)', fontsize=11)
        ax.legend(fontsize=9, loc='upper right', framealpha=0.85)

        label = phase_labels[i] if i < len(phase_labels) else ph.name.capitalize()
        ax.set_title(f'Phase contribution: {label}', fontsize=11, pad=4)

    axes[-1].set_xlabel(r'2$\theta$ (degrees)', fontsize=12)

    if title:
        fig.suptitle(title, fontsize=13, y=1.01)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print(f'  Phase contribution plot saved → {Path(save_path).name}')

    plt.show()
    return fig


# ─── Batch summary ────────────────────────────────────────────────────────────

def plot_all_samples(results, save_dir=None):
    """Bar chart of phase fractions and Rwp for all samples in a batch run."""
    names = [r['name']     for r in results]
    rwps  = [r['rwp']      for r in results]
    wt_c  = [r['wt_cubic'] for r in results]
    wt_t  = [r['wt_tet']   for r in results]
    x     = np.arange(len(names))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    ax1.bar(x - 0.2, wt_c, 0.4, label='Cubic',      color='tab:blue',  alpha=0.85)
    ax1.bar(x + 0.2, wt_t, 0.4, label='Tetragonal', color='tab:green', alpha=0.85)
    ax1.set_ylabel('Weight fraction (%)', fontsize=11)
    ax1.set_ylim(0, 105)
    ax1.legend(fontsize=10)
    ax1.set_title('Phase fractions', fontsize=12)

    ax2.plot(x, rwps, 'ko-', ms=5)
    ax2.set_ylabel('$R_{wp}$', fontsize=11)
    ax2.set_title('Goodness of fit', fontsize=12)
    ax2.set_xticks(x)
    ax2.set_xticklabels(names, rotation=30, ha='right', fontsize=9)

    plt.tight_layout()

    if save_dir:
        p = Path(save_dir) / 'summary.png'
        fig.savefig(p, dpi=150, bbox_inches='tight')
        print(f'  Summary figure saved → {p.name}')

    plt.show()
    return fig
