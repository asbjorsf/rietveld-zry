"""
Rietveld plot functions for Zr(1-x)Y(x)O(2-x/2).

Functions
---------
plot_rietveld             : full-range 3-panel plot (obs/calc/diff + hkl ticks)
plot_phase_contributions  : per-phase contribution panels (full range)
plot_zoom_contributions   : zoomed deconvolution view around a chosen peak group

Colour palette
--------------
Wong (2011) colorblind-safe palette is used throughout.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker
from pathlib import Path

from rietveld_engine import (generate_reflections, chebyshev_background,
                              calculate_pattern, rwp as _rwp, rp as _rp)

# ─── Wong (2011) colorblind-safe palette ────────────────────────────────────
# Distinguishable for deuteranopia, protanopia, and tritanopia.
_BLACK      = '#000000'
_ORANGE     = '#E69F00'
_SKY_BLUE   = '#56B4E9'
_GREEN      = '#009E73'
_YELLOW     = '#F0E442'
_BLUE       = '#0072B2'
_VERMILLION = '#D55E00'
_PINK       = '#CC79A7'

_PHASE_COLORS  = [_BLUE, _VERMILLION, _GREEN, _PINK]
_CALC_COLOR    = _ORANGE
_BG_COLOR      = _SKY_BLUE
_DIFF_COLOR    = '#666666'

# ─── Shared rcParams for readability ────────────────────────────────────────
_RC = {
    'font.size':        11,
    'axes.titlesize':   12,
    'axes.labelsize':   12,
    'legend.fontsize':  10,
    'xtick.labelsize':  10,
    'ytick.labelsize':  10,
    'axes.linewidth':   0.8,
    'xtick.major.width': 0.8,
    'ytick.major.width': 0.8,
    'lines.linewidth':  1.3,
    'figure.dpi':       100,
}


# ─── Helper: per-phase pattern (no background) ───────────────────────────────

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


def _obs_circles(ax, tth, y_obs, **kw):
    """Plot observed data as open black circles (standard Rietveld style)."""
    defaults = dict(
        marker='o', ms=2.5, markerfacecolor='none',
        markeredgecolor=_BLACK, markeredgewidth=0.5,
        linestyle='none', zorder=3, label='Observed',
    )
    defaults.update(kw)
    return ax.plot(tth, y_obs, **defaults)[0]


# ─── Main Rietveld plot ───────────────────────────────────────────────────────

def plot_rietveld(ref, title=None, save_path=None, dpi=150):
    """
    Three-panel Rietveld plot.

    Panel 1  Observed (○), Calculated (─), Background (--)
    Panel 2  hkl tick marks per phase
    Panel 3  Difference curve
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
        gs  = gridspec.GridSpec(2 + n_ph, 1, height_ratios=hr,
                                hspace=0.0, left=0.10, right=0.97,
                                top=0.93, bottom=0.08)

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
        ax_main.yaxis.set_major_formatter(
            mticker.FuncFormatter(lambda x, _: f'{x/1000:.0f}k'
                                  if x >= 1000 else f'{x:.0f}'))

        # Legend — large markers, clear box, placed top-right
        leg = ax_main.legend(
            loc='upper right',
            framealpha=0.92, edgecolor='#aaaaaa',
            handlelength=2.5, handletextpad=0.8,
            borderpad=0.7, labelspacing=0.5,
            markerscale=3.0,          # enlarge legend markers
            fontsize=11,
        )
        leg.get_frame().set_linewidth(0.8)

        if title:
            ax_main.set_title(title, fontsize=13, pad=6, fontweight='bold')

        # R-factor box — clear, top-left so it doesn't clash with legend
        r_wp = _rwp(y_obs, y_calc, ref.weights)
        r_p  = _rp(y_obs, y_calc)
        ax_main.text(
            0.01, 0.97,
            f'$R_{{wp}}$ = {r_wp:.4f}\n$R_p$   = {r_p:.4f}',
            transform=ax_main.transAxes,
            ha='left', va='top', fontsize=10,
            fontfamily='monospace',
            bbox=dict(boxstyle='round,pad=0.4',
                      facecolor='white', edgecolor='#aaaaaa',
                      alpha=0.92, linewidth=0.8),
        )

        # ── hkl tick panels ─────────────────────────────────────────────────
        for i, (ph, ax_tk) in enumerate(zip(ref.phases, ax_ticks)):
            refs = generate_reflections(ph, ref.wavelength,
                                        tth.max() + 2, tth.min() - 2)
            positions = [r['two_theta'] for r in refs]
            color = _PHASE_COLORS[i % len(_PHASE_COLORS)]
            ax_tk.vlines(positions, 0.1, 0.9, colors=color, lw=0.9)
            ax_tk.set_ylim(0, 1)
            ax_tk.set_yticks([])

            label = ('Cubic' if ph.crystal_system == 'cubic'
                     else 'Tetrag.')
            ax_tk.set_ylabel(label, fontsize=9, rotation=0,
                             labelpad=30, va='center', color=color,
                             fontweight='bold')
            ax_tk.tick_params(bottom=False, labelbottom=False)
            for spine in ['top', 'bottom', 'right']:
                ax_tk.spines[spine].set_visible(False)

        # ── Difference panel ─────────────────────────────────────────────────
        ax_diff.plot(tth, diff, '-', color=_DIFF_COLOR, lw=0.8)
        ax_diff.axhline(0, color=_BLACK, lw=0.6, ls='--', alpha=0.4)
        ax_diff.set_ylabel('Difference')
        ax_diff.set_xlabel(r'2$\theta$ (degrees)')

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
    One subplot per phase showing its contribution across the full range.

      ○  Observed
      ─  Total calculated (thin reference line)
      ▓  Phase contribution + background (filled area)
    """
    with plt.rc_context(_RC):
        tth            = ref.two_theta
        y_obs          = ref.y_obs
        y_bg           = chebyshev_background(tth, ref.bg_coeffs)
        phase_patterns = [_phase_pattern(ref, i) for i in range(len(ref.phases))]
        y_calc_total   = ref.calc()

        n_ph  = len(ref.phases)
        fig, axes = plt.subplots(n_ph, 1, figsize=(13, 5 * n_ph),
                                 sharex=True, constrained_layout=True)
        if n_ph == 1:
            axes = [axes]

        phase_labels = ['Cubic ZrO₂  (Fm-3m)',
                        'Tetragonal ZrO₂  (P4₂/nmc)']

        for i, (ax, ph, y_ph) in enumerate(zip(axes, ref.phases, phase_patterns)):
            color = _PHASE_COLORS[i % len(_PHASE_COLORS)]
            label = phase_labels[i] if i < len(phase_labels) else ph.name

            _obs_circles(ax, tth, y_obs)
            ax.plot(tth, y_calc_total, '-', color=_CALC_COLOR, lw=0.9,
                    alpha=0.6, label='Total calculated', zorder=4)

            y_contrib = y_ph + y_bg
            ax.fill_between(tth, y_bg, y_contrib,
                            color=color, alpha=0.30, zorder=2)
            ax.plot(tth, y_contrib, '-', color=color, lw=1.4,
                    label=f'{label}  + background', zorder=3)
            ax.plot(tth, y_bg, '--', color=_BG_COLOR, lw=0.9, alpha=0.7,
                    label='Background', zorder=1)

            # hkl ticks just below x-axis zero line
            refs_hkl = generate_reflections(ph, ref.wavelength,
                                            tth.max() + 2, tth.min() - 2)
            tick_base = y_obs.min() - 0.05 * (y_obs.max() - y_obs.min())
            tick_h    = 0.025 * (y_obs.max() - y_obs.min())
            positions = [r['two_theta'] for r in refs_hkl]
            ax.vlines(positions, tick_base - tick_h, tick_base + tick_h,
                      colors=color, lw=0.9)

            ax.set_xlim(tth.min(), tth.max())
            ax.set_ylabel('Intensity (arb. units)')
            ax.set_title(f'Phase contribution: {label}',
                         fontweight='bold', pad=5)
            ax.legend(loc='upper right', framealpha=0.92,
                      edgecolor='#aaaaaa', markerscale=3.0,
                      handlelength=2.5)

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
    Zoomed deconvolution plot centred on *tth_center* ± *half_window* degrees.

    Shows how the two phases add up to produce the observed peak group.
    y-axis is auto-scaled to the peak maximum so the deconvolution fills
    the panel.

    Parameters
    ----------
    tth_center  : centre of zoom region [degrees 2θ]
    half_window : half-width of zoom region [degrees]
    """
    tth_lo = tth_center - half_window
    tth_hi = tth_center + half_window

    tth   = ref.two_theta
    mask  = (tth >= tth_lo) & (tth <= tth_hi)

    tth_z = tth[mask]
    y_obs_z = ref.y_obs[mask]
    y_bg_z  = chebyshev_background(tth_z, ref.bg_coeffs)

    # Pattern contributions in zoom window
    phase_patterns_z = []
    for i in range(len(ref.phases)):
        full = _phase_pattern(ref, i)
        phase_patterns_z.append(full[mask])
    y_calc_z = ref.calc()[mask]

    with plt.rc_context(_RC):
        # Two panels: main zoom | hkl ticks
        n_ph = len(ref.phases)
        hr   = [6] + [0.55] * n_ph
        fig  = plt.figure(figsize=(9, 5.5))
        gs   = gridspec.GridSpec(1 + n_ph, 1, height_ratios=hr,
                                 hspace=0.0, left=0.12, right=0.97,
                                 top=0.90, bottom=0.11)

        ax_main  = fig.add_subplot(gs[0])
        ax_ticks = [fig.add_subplot(gs[1 + i], sharex=ax_main)
                    for i in range(n_ph)]

        # ── Stacked filled contributions ─────────────────────────────────
        # Draw from bottom up: background, then each phase stacked
        y_stack = y_bg_z.copy()
        phase_labels = ['Cubic ZrO₂  (Fm-3m)',
                        'Tetragonal ZrO₂  (P4₂/nmc)']

        # Background as thin dashed line
        ax_main.plot(tth_z, y_bg_z, '--', color=_BG_COLOR,
                     lw=1.0, alpha=0.8, label='Background', zorder=1)

        for i, (ph, y_ph_z) in enumerate(zip(ref.phases, phase_patterns_z)):
            color = _PHASE_COLORS[i % len(_PHASE_COLORS)]
            label = phase_labels[i] if i < len(phase_labels) else ph.name
            y_top = y_stack + y_ph_z
            ax_main.fill_between(tth_z, y_stack, y_top,
                                 color=color, alpha=0.40, zorder=2 + i)
            ax_main.plot(tth_z, y_top, '-', color=color,
                         lw=1.6, label=f'{label}', zorder=3 + i)
            y_stack = y_top

        # Total calculated
        ax_main.plot(tth_z, y_calc_z, '-', color=_CALC_COLOR,
                     lw=1.8, alpha=0.8, label='Total calculated', zorder=6)

        # Observed data — slightly larger markers for zoom view
        ax_main.plot(tth_z, y_obs_z,
                     'o', ms=4.5, markerfacecolor='none',
                     markeredgecolor=_BLACK, markeredgewidth=0.8,
                     linestyle='none', label='Observed', zorder=7)

        # y-limits: base at 0, top 10% above peak max
        y_top_lim = y_obs_z.max() * 1.10
        ax_main.set_ylim(0, y_top_lim)
        ax_main.set_xlim(tth_lo, tth_hi)
        ax_main.set_ylabel('Intensity (arb. units)')

        ax_main.yaxis.set_major_formatter(
            mticker.FuncFormatter(lambda x, _: f'{x/1000:.0f}k'
                                  if x >= 1000 else f'{x:.0f}'))

        # Legend — two columns to save vertical space
        leg = ax_main.legend(
            loc='upper right', ncol=2,
            framealpha=0.92, edgecolor='#aaaaaa',
            handlelength=2.0, handletextpad=0.6,
            borderpad=0.6, labelspacing=0.45,
            markerscale=1.8, fontsize=10,
        )
        leg.get_frame().set_linewidth(0.8)

        if title:
            ax_main.set_title(title, fontsize=12, pad=5, fontweight='bold')
        else:
            ax_main.set_title(
                f'Deconvolution  2θ ≈ {tth_center:.0f}°',
                fontsize=12, pad=5, fontweight='bold')

        # ── hkl tick panels ──────────────────────────────────────────────
        for i, (ph, ax_tk) in enumerate(zip(ref.phases, ax_ticks)):
            refs_hkl = generate_reflections(ph, ref.wavelength,
                                            tth_hi + 1, tth_lo - 1)
            color = _PHASE_COLORS[i % len(_PHASE_COLORS)]
            positions = [r['two_theta'] for r in refs_hkl]
            intensities = []
            for r in refs_hkl:
                s  = 1.0 / (2.0 * r['d'])
                F2 = ph.structure_factor_sq(r['h'], r['k'], r['l'], s)
                intensities.append(r['multiplicity'] * F2)

            # Scale tick heights by relative intensity for visual clarity
            if intensities:
                i_max = max(intensities)
                for pos, inten in zip(positions, intensities):
                    h_rel = 0.2 + 0.7 * (inten / i_max)
                    ax_tk.vlines(pos, 0.5 - h_rel/2, 0.5 + h_rel/2,
                                 colors=color, lw=1.2)

            ax_tk.set_ylim(0, 1)
            ax_tk.set_yticks([])
            label = 'Cubic' if ph.crystal_system == 'cubic' else 'Tetrag.'
            ax_tk.set_ylabel(label, fontsize=9, rotation=0,
                             labelpad=30, va='center', color=color,
                             fontweight='bold')
            ax_tk.tick_params(bottom=(i == n_ph - 1),
                              labelbottom=(i == n_ph - 1))
            for spine in ['top', 'right']:
                ax_tk.spines[spine].set_visible(False)
            if i < n_ph - 1:
                ax_tk.spines['bottom'].set_visible(False)

        plt.setp(ax_main.get_xticklabels(), visible=False)
        ax_main.tick_params(bottom=False)
        ax_ticks[-1].set_xlabel(r'2$\theta$ (degrees)')

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

        ax1.bar(x - 0.2, wt_c, 0.4, label='Cubic',      color=_BLUE,      alpha=0.85)
        ax1.bar(x + 0.2, wt_t, 0.4, label='Tetragonal', color=_VERMILLION, alpha=0.85)
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
