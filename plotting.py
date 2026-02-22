"""
Rietveld plot functions for Zr(1-x)Y(x)O(2-x/2).

Functions
---------
plot_rietveld             : full-range 3-panel plot
plot_phase_contributions  : per-phase contribution panels (full range)
plot_zoom_contributions   : zoomed deconvolution view around a peak group

Design principles
-----------------
- Wong (2011) colorblind-safe palette throughout.
- Minimum 11 pt font; consistent rcParams context.
- Legends placed outside the data area (below axes) to avoid overlap.
- hkl tick positions are derived from the refined unit-cell parameters
  (a, c, z_O) after the Rietveld refinement — NOT from fixed presets.
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D
from pathlib import Path

from rietveld_engine import (generate_reflections, chebyshev_background,
                              calculate_pattern, rwp as _rwp, rp as _rp)

# ─── Wong (2011) colorblind-safe palette ─────────────────────────────────────
_BLACK      = '#000000'
_ORANGE     = '#E69F00'
_SKY_BLUE   = '#56B4E9'
_GREEN      = '#009E73'
_BLUE       = '#0072B2'
_VERMILLION = '#D55E00'
_PINK       = '#CC79A7'
_YELLOW     = '#F0E442'

_PHASE_COLORS  = [_BLUE, _VERMILLION, _GREEN, _PINK]
_CALC_COLOR    = _ORANGE
_BG_COLOR      = _SKY_BLUE
_DIFF_COLOR    = '#666666'

# ─── Shared rcParams ─────────────────────────────────────────────────────────
_RC = {
    'font.family':      'sans-serif',
    'font.size':        11,
    'axes.titlesize':   12,
    'axes.labelsize':   12,
    'legend.fontsize':  10,
    'xtick.labelsize':  10,
    'ytick.labelsize':  10,
    'axes.linewidth':   0.8,
    'xtick.major.width': 0.8,
    'ytick.major.width': 0.8,
}


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _phase_pattern(ref, phase_idx):
    """Return the pattern contribution from one phase only (no background)."""
    ph = ref.phases[phase_idx]
    sc = ref.scale_factors[phase_idx]
    return calculate_pattern(
        ref.two_theta, [ph], [sc],
        bg_coeffs=np.zeros_like(ref.bg_coeffs),
        U=ref.U, V=ref.V, W=ref.W,
        eta=ref.eta, zero_offset=ref.zero_offset,
        wavelength=ref.wavelength,
    )


def _k_formatter(x, _):
    """Format y-axis values as e.g. '12k' or '500'."""
    return f'{x/1000:.0f}k' if x >= 1000 else f'{x:.0f}'


def _obs_circles(ax, tth, y_obs, **kw):
    """Observed data as open black circles."""
    defaults = dict(
        marker='o', markersize=2.5,
        markerfacecolor='none', markeredgecolor=_BLACK,
        markeredgewidth=0.5, linestyle='none',
        zorder=3, label='Observed',
    )
    defaults.update(kw)
    return ax.plot(tth, y_obs, **defaults)[0]


def _r_factor_text(ax, y_obs, y_calc, weights):
    """Add Rwp / Rp text box at the top-left of *ax*."""
    r_wp = _rwp(y_obs, y_calc, weights)
    r_p  = _rp(y_obs, y_calc)
    ax.text(
        0.01, 0.97,
        f'$R_{{wp}}$ = {r_wp:.4f}\n$R_p$   = {r_p:.4f}',
        transform=ax.transAxes,
        ha='left', va='top', fontsize=10,
        fontfamily='monospace',
        bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                  edgecolor='#aaaaaa', alpha=0.92, linewidth=0.8),
    )


def _outside_legend(ax, handles, labels, ncol=3, y_offset=-0.28):
    """
    Place legend below the axes, outside the data area.
    y_offset is in axes-fraction units (negative = below).
    """
    return ax.legend(
        handles, labels,
        loc='upper center',
        bbox_to_anchor=(0.5, y_offset),
        ncol=ncol,
        framealpha=0.92, edgecolor='#aaaaaa',
        handlelength=2.2, handletextpad=0.6,
        borderpad=0.6, columnspacing=1.0,
        fontsize=10,
    )


# ─── Main Rietveld plot ───────────────────────────────────────────────────────

def plot_rietveld(ref, title=None, save_path=None, dpi=150):
    """
    Three-panel Rietveld plot:
      ○  Observed          open black circles
      ─  Calculated        orange line
      -- Background        sky-blue dashed
      |  hkl ticks         per phase, coloured
      ─  Difference        grey line below
    """
    with plt.rc_context(_RC):
        tth    = ref.two_theta
        y_obs  = ref.y_obs
        y_calc = ref.calc()
        diff   = y_obs - y_calc
        y_bg   = chebyshev_background(tth, ref.bg_coeffs)
        n_ph   = len(ref.phases)

        hr  = [5] + [0.45] * n_ph + [1.8]
        fig = plt.figure(figsize=(13, 8))
        gs  = gridspec.GridSpec(
            2 + n_ph, 1, height_ratios=hr,
            hspace=0.0, left=0.09, right=0.97, top=0.93, bottom=0.09,
        )
        ax_main  = fig.add_subplot(gs[0])
        ax_ticks = [fig.add_subplot(gs[1 + i], sharex=ax_main)
                    for i in range(n_ph)]
        ax_diff  = fig.add_subplot(gs[1 + n_ph], sharex=ax_main)

        # ── Main panel ──────────────────────────────────────────────────────
        _obs_circles(ax_main, tth, y_obs)
        ax_main.plot(tth, y_calc, '-', color=_CALC_COLOR, lw=1.5,
                     label='Calculated', zorder=4)
        ax_main.plot(tth, y_bg, '--', color=_BG_COLOR, lw=1.0, alpha=0.85,
                     label='Background', zorder=2)

        ax_main.set_ylabel('Intensity (arb. units)')
        ax_main.set_xlim(tth.min(), tth.max())
        ax_main.yaxis.set_major_formatter(mticker.FuncFormatter(_k_formatter))

        # Legend — top-right, enlarged markers
        leg = ax_main.legend(
            loc='upper right',
            framealpha=0.92, edgecolor='#aaaaaa',
            handlelength=2.5, handletextpad=0.8,
            borderpad=0.7, labelspacing=0.5,
            markerscale=3.0, fontsize=11,
        )
        leg.get_frame().set_linewidth(0.8)

        _r_factor_text(ax_main, y_obs, y_calc, ref.weights)

        if title:
            ax_main.set_title(title, fontsize=13, pad=6, fontweight='bold')

        # ── hkl tick panels ──────────────────────────────────────────────────
        # Positions come from refined unit-cell parameters (a, c, z_O)
        for i, (ph, ax_tk) in enumerate(zip(ref.phases, ax_ticks)):
            refs = generate_reflections(ph, ref.wavelength,
                                        tth.max() + 2, tth.min() - 2)
            positions = [r['two_theta'] for r in refs]
            color = _PHASE_COLORS[i % len(_PHASE_COLORS)]
            ax_tk.vlines(positions, 0.1, 0.9, colors=color, lw=0.9)
            ax_tk.set_ylim(0, 1)
            ax_tk.set_yticks([])

            lbl = 'Cubic' if ph.crystal_system == 'cubic' else 'Tetrag.'
            ax_tk.set_ylabel(lbl, fontsize=9, rotation=0,
                             labelpad=30, va='center',
                             color=color, fontweight='bold')
            ax_tk.tick_params(bottom=False, labelbottom=False)
            for sp in ['top', 'bottom', 'right']:
                ax_tk.spines[sp].set_visible(False)

        # ── Difference panel ─────────────────────────────────────────────────
        ax_diff.plot(tth, diff, '-', color=_DIFF_COLOR, lw=0.8)
        ax_diff.axhline(0, color=_BLACK, lw=0.6, ls='--', alpha=0.4)
        ax_diff.set_ylabel('Difference')
        ax_diff.set_xlabel(r'2$\theta$ (degrees)')
        ax_diff.yaxis.set_major_formatter(mticker.FuncFormatter(_k_formatter))

        for ax in [ax_main] + ax_ticks:
            plt.setp(ax.get_xticklabels(), visible=False)
            ax.tick_params(bottom=False)

        if save_path:
            fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
            print(f'  Rietveld plot saved → {Path(save_path).name}')

        plt.show()
    return fig


# ─── Full-range phase contribution plots ─────────────────────────────────────

def plot_phase_contributions(ref, title=None, save_path=None, dpi=150):
    """
    One subplot per phase showing its absolute contribution above the background.

      ○  Observed
      ─  Total calculated (reference)
      ▓  Phase contribution + background (filled area)
    """
    with plt.rc_context(_RC):
        tth            = ref.two_theta
        y_obs          = ref.y_obs
        y_bg           = chebyshev_background(tth, ref.bg_coeffs)
        phase_patterns = [_phase_pattern(ref, i) for i in range(len(ref.phases))]
        y_calc_total   = ref.calc()

        n_ph  = len(ref.phases)
        fig, axes = plt.subplots(n_ph, 1, figsize=(13, 5.5 * n_ph),
                                 sharex=True, constrained_layout=True)
        if n_ph == 1:
            axes = [axes]

        phase_labels = ['Cubic ZrO₂  (Fm-3m)',
                        'Tetragonal ZrO₂  (P4₂/nmc)']

        for i, (ax, ph, y_ph) in enumerate(zip(axes, ref.phases, phase_patterns)):
            color = _PHASE_COLORS[i % len(_PHASE_COLORS)]
            label = phase_labels[i] if i < len(phase_labels) else ph.name

            obs_h  = _obs_circles(ax, tth, y_obs)
            calc_h = ax.plot(tth, y_calc_total, '-', color=_CALC_COLOR,
                             lw=0.9, alpha=0.6, label='Total calculated',
                             zorder=4)[0]

            y_contrib = y_ph + y_bg
            ax.fill_between(tth, y_bg, y_contrib,
                            color=color, alpha=0.30, zorder=2)
            contrib_h = ax.plot(tth, y_contrib, '-', color=color, lw=1.4,
                                label=f'{label}  + background', zorder=3)[0]
            bg_h = ax.plot(tth, y_bg, '--', color=_BG_COLOR, lw=0.9,
                           alpha=0.7, label='Background', zorder=1)[0]

            # hkl ticks from refined parameters
            refs_hkl  = generate_reflections(ph, ref.wavelength,
                                             tth.max() + 2, tth.min() - 2)
            tick_base = y_obs.min() - 0.05 * (y_obs.max() - y_obs.min())
            tick_h    = 0.025 * (y_obs.max() - y_obs.min())
            positions = [r['two_theta'] for r in refs_hkl]
            ax.vlines(positions, tick_base - tick_h, tick_base + tick_h,
                      colors=color, lw=0.9)
            tick_h_line = Line2D([0], [0], color=color, lw=1.5,
                                 label=f'{label}  hkl positions')

            ax.set_xlim(tth.min(), tth.max())
            ax.set_ylabel('Intensity (arb. units)')
            ax.yaxis.set_major_formatter(mticker.FuncFormatter(_k_formatter))
            ax.set_title(f'Phase contribution: {label}',
                         fontweight='bold', pad=5)

            # Legend below the axes
            handles = [obs_h, calc_h, contrib_h, bg_h, tick_h_line]
            labels  = [h.get_label() for h in handles]
            _outside_legend(ax, handles, labels, ncol=3, y_offset=-0.14)

        axes[-1].set_xlabel(r'2$\theta$ (degrees)')

        if title:
            fig.suptitle(title, fontsize=14, fontweight='bold')

        if save_path:
            fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
            print(f'  Phase contribution plot saved → {Path(save_path).name}')

        plt.show()
    return fig


# ─── Zoomed deconvolution plot ────────────────────────────────────────────────

def plot_zoom_contributions(ref, tth_center, half_window=2.0,
                            title=None, save_path=None, dpi=150):
    """
    Zoomed deconvolution view centred on *tth_center* ± *half_window* degrees.

    Each phase contribution is shown as a filled area above the background
    (independently, not stacked).  Where both phases have peaks at the same
    2θ the fills overlap — the overlap is deliberately visible so the viewer
    can judge how much each phase contributes.

    The solid orange line is the total calculated pattern
    (background + all phases), not associated with any individual hkl.

    hkl tick mark heights are scaled by relative |F|² × multiplicity so
    stronger reflections have taller ticks.
    """
    tth_lo = tth_center - half_window
    tth_hi = tth_center + half_window

    tth   = ref.two_theta
    mask  = (tth >= tth_lo) & (tth <= tth_hi)

    tth_z   = tth[mask]
    y_obs_z = ref.y_obs[mask]
    y_bg_z  = chebyshev_background(tth_z, ref.bg_coeffs)
    y_calc_z = ref.calc()[mask]

    phase_patterns_z = [_phase_pattern(ref, i)[mask]
                        for i in range(len(ref.phases))]

    with plt.rc_context(_RC):
        n_ph = len(ref.phases)
        # Layout: main panel + one hkl tick row per phase
        hr  = [6] + [0.55] * n_ph
        fig = plt.figure(figsize=(9, 6.5))
        gs  = gridspec.GridSpec(
            1 + n_ph, 1, height_ratios=hr,
            hspace=0.0, left=0.11, right=0.97,
            top=0.88,
            bottom=0.22,   # leave room for the legend below
        )
        ax_main  = fig.add_subplot(gs[0])
        ax_ticks = [fig.add_subplot(gs[1 + i], sharex=ax_main)
                    for i in range(n_ph)]

        # ── Background ───────────────────────────────────────────────────────
        bg_h = ax_main.plot(tth_z, y_bg_z, '--', color=_BG_COLOR,
                            lw=1.1, alpha=0.85, label='Background', zorder=1)[0]

        # ── Per-phase overlapping fills (each above background, NOT stacked) ─
        phase_labels = ['Cubic ZrO₂  (Fm-3m)',
                        'Tetragonal ZrO₂  (P4₂/nmc)']
        contrib_handles = []
        for i, (ph, y_ph_z) in enumerate(zip(ref.phases, phase_patterns_z)):
            color = _PHASE_COLORS[i % len(_PHASE_COLORS)]
            label = phase_labels[i] if i < len(phase_labels) else ph.name
            y_top = y_bg_z + y_ph_z
            ax_main.fill_between(tth_z, y_bg_z, y_top,
                                 color=color, alpha=0.38, zorder=2 + i)
            h = ax_main.plot(tth_z, y_top, '-', color=color, lw=1.7,
                             label=label, zorder=3 + i)[0]
            contrib_handles.append(h)

        # ── Total calculated — sum of all phases, not per-hkl ────────────────
        calc_h = ax_main.plot(
            tth_z, y_calc_z, '-', color=_CALC_COLOR, lw=2.0,
            label='Total calculated  (sum of phases)', zorder=6,
        )[0]

        # ── Observed data (larger circles in zoom view) ───────────────────────
        obs_h = ax_main.plot(
            tth_z, y_obs_z,
            'o', markersize=5.0,
            markerfacecolor='none', markeredgecolor=_BLACK,
            markeredgewidth=0.9, linestyle='none',
            label='Observed', zorder=7,
        )[0]

        # y-limits: zero to 10 % above max observed in window
        ax_main.set_ylim(0, y_obs_z.max() * 1.10)
        ax_main.set_xlim(tth_lo, tth_hi)
        ax_main.set_ylabel('Intensity (arb. units)')
        ax_main.yaxis.set_major_formatter(mticker.FuncFormatter(_k_formatter))

        ttl = title or f'Deconvolution  2θ ≈ {tth_center:.0f}°'
        ax_main.set_title(ttl, fontsize=12, pad=5, fontweight='bold')

        # ── Legend below the entire figure ───────────────────────────────────
        all_handles = [obs_h, calc_h] + contrib_handles + [bg_h]
        all_labels  = [h.get_label() for h in all_handles]
        fig.legend(
            all_handles, all_labels,
            loc='lower center',
            bbox_to_anchor=(0.5, 0.01),
            ncol=3,
            framealpha=0.92, edgecolor='#aaaaaa',
            handlelength=2.2, handletextpad=0.6,
            borderpad=0.6, columnspacing=1.0,
            fontsize=10,
            markerscale=1.5,
        )

        # ── hkl tick panels ──────────────────────────────────────────────────
        # Positions computed from REFINED unit-cell parameters after Rietveld.
        for i, (ph, ax_tk) in enumerate(zip(ref.phases, ax_ticks)):
            color = _PHASE_COLORS[i % len(_PHASE_COLORS)]
            refs_hkl = generate_reflections(ph, ref.wavelength,
                                            tth_hi + 0.5, tth_lo - 0.5)
            for r in refs_hkl:
                s  = 1.0 / (2.0 * r['d'])
                F2 = ph.structure_factor_sq(r['h'], r['k'], r['l'], s)
                rel = min(r['multiplicity'] * F2 / 1e6, 1.0)   # normalised height
                h_half = 0.15 + 0.65 * rel
                ax_tk.vlines(r['two_theta'],
                             0.5 - h_half, 0.5 + h_half,
                             colors=color, lw=1.3)

            ax_tk.set_ylim(0, 1)
            ax_tk.set_yticks([])
            lbl = 'Cubic' if ph.crystal_system == 'cubic' else 'Tetrag.'
            ax_tk.set_ylabel(lbl, fontsize=9, rotation=0,
                             labelpad=30, va='center',
                             color=color, fontweight='bold')
            ax_tk.tick_params(bottom=(i == n_ph - 1),
                              labelbottom=(i == n_ph - 1))
            for sp in ['top', 'right']:
                ax_tk.spines[sp].set_visible(False)
            if i < n_ph - 1:
                ax_tk.spines['bottom'].set_visible(False)

        plt.setp(ax_main.get_xticklabels(), visible=False)
        ax_main.tick_params(bottom=False)
        ax_ticks[-1].set_xlabel(r'2$\theta$ (degrees)')

        # Annotation: hkl source
        ax_main.annotate(
            'hkl positions from refined unit-cell parameters',
            xy=(0.99, 0.02), xycoords='axes fraction',
            ha='right', va='bottom', fontsize=7.5,
            color='#666666', style='italic',
        )

        if save_path:
            fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
            print(f'  Zoom plot saved → {Path(save_path).name}')

        plt.show()
    return fig


# ─── Batch summary ────────────────────────────────────────────────────────────

def plot_all_samples(results, save_dir=None):
    """Bar chart of phase fractions and Rwp for all samples."""
    with plt.rc_context(_RC):
        names = [r['name']     for r in results]
        rwps  = [r['rwp']      for r in results]
        wt_c  = [r['wt_cubic'] for r in results]
        wt_t  = [r['wt_tet']   for r in results]
        x     = np.arange(len(names))

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6),
                                        sharex=True, constrained_layout=True)
        ax1.bar(x - 0.2, wt_c, 0.4, label='Cubic',
                color=_BLUE, alpha=0.85)
        ax1.bar(x + 0.2, wt_t, 0.4, label='Tetragonal',
                color=_VERMILLION, alpha=0.85)
        ax1.set_ylabel('Weight fraction (%)')
        ax1.set_ylim(0, 105)
        ax1.legend(framealpha=0.92, edgecolor='#aaaaaa')
        ax1.set_title('Phase fractions', fontweight='bold')

        ax2.plot(x, rwps, 'o-', color=_BLACK, ms=6)
        ax2.set_ylabel('$R_{wp}$')
        ax2.set_title('Goodness of fit', fontweight='bold')
        ax2.set_xticks(x)
        ax2.set_xticklabels(names, rotation=30, ha='right')

        if save_dir:
            p = Path(save_dir) / 'summary.png'
            fig.savefig(p, dpi=150, bbox_inches='tight')
            print(f'  Summary figure saved → {p.name}')

        plt.show()
    return fig
