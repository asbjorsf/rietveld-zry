"""
Main Rietveld refinement script for Zr(1-x)Y(x)O(2-x/2).

Usage
-----
    # Refine a single file
    python run_refinement.py path/to/sample.xy

    # Refine all .xy files in a directory (batch mode)
    python run_refinement.py path/to/XRD/

    # With explicit Y content hint (helps initial guess)
    python run_refinement.py sample.xy --x_Y 0.08

Data format
-----------
    Two-column ASCII:  2theta [deg]   intensity [counts]
    Lines beginning with # are ignored.
    σ is estimated as sqrt(intensity) if not supplied.
"""

import sys
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')   # non-interactive backend – works without a display
import numpy as np

from crystal_structures import CubicZrO2, TetragonalZrO2
from refinement import RietveldRefinement
from plotting import plot_rietveld, plot_phase_contributions, plot_all_samples


# ─── Constants ───────────────────────────────────────────────────────────────

WAVELENGTH    = 1.540562    # Cu Kα1 [Angstrom]
TWO_THETA_MIN = 20.0        # degrees  – crop to data range
TWO_THETA_MAX = 110.0       # degrees


# ─── Initial crystal structure parameters ────────────────────────────────────
# These are literature values; the refinement will optimise them.

def make_phases(x_Y_guess=0.05):
    """
    Create initial phase objects.

    x_Y_guess : estimated total Y content.  Partitions evenly as starting
                point; let the refinement adjust.
    """
    cubic = CubicZrO2(
        a         = 5.1270 + 0.123 * x_Y_guess,   # linear approx: a increases with Y
        x_Y       = x_Y_guess,
        Biso_ZrY  = 0.5,
        Biso_O    = 0.8,
    )
    tetragonal = TetragonalZrO2(
        a         = 3.6041,
        c         = 5.1733,
        z_O       = 0.2065,
        x_Y       = x_Y_guess / 2,
        Biso_ZrY  = 0.5,
        Biso_O    = 0.8,
    )
    return cubic, tetragonal


# ─── Data loading ─────────────────────────────────────────────────────────────

def load_xy(filepath):
    """
    Load a two-column (or three-column) powder diffraction file.

    Returns
    -------
    two_theta, y_obs, sigma : 1-D numpy arrays
    """
    data = np.loadtxt(filepath, comments=['#', '!', "'"])
    two_theta = data[:, 0]
    y_obs     = data[:, 1]
    sigma     = data[:, 2] if data.shape[1] >= 3 else np.sqrt(np.maximum(y_obs, 1.0))
    return two_theta, y_obs, sigma


# ─── Single-file refinement ───────────────────────────────────────────────────

def refine_file(filepath, x_Y_guess=0.05, plot=True, save_results=True):
    """
    Run a full Rietveld refinement on one .xy file.

    Parameters
    ----------
    filepath    : path to data file
    x_Y_guess   : initial Y-content guess
    plot        : show Rietveld plot after refinement
    save_results: write JSON summary next to data file

    Returns
    -------
    dict of key refined parameters
    """
    filepath = Path(filepath)
    stem     = filepath.stem
    print(f'\n{"="*60}')
    print(f'  Sample: {stem}')
    print(f'{"="*60}')

    # ── Load data
    two_theta, y_obs, sigma = load_xy(filepath)
    mask      = (two_theta >= TWO_THETA_MIN) & (two_theta <= TWO_THETA_MAX)
    two_theta = two_theta[mask]
    y_obs     = y_obs[mask]
    sigma     = sigma[mask]
    print(f'  Data: {two_theta.min():.2f}–{two_theta.max():.2f}°'
          f'  ({len(two_theta)} points)')

    # ── Initialise
    cubic, tetragonal = make_phases(x_Y_guess)
    ref = RietveldRefinement(two_theta, y_obs, sigma,
                             phases=[cubic, tetragonal],
                             wavelength=WAVELENGTH)
    ref.auto_scale()

    # ── Run sequential refinement
    ref.run_sequential(verbose=True)

    # ── Report
    ref.report()

    # ── Plot
    if plot:
        save_fig  = filepath.parent / f'{stem}_rietveld.png'
        save_fig2 = filepath.parent / f'{stem}_contributions.png'
        plot_rietveld(ref, title=stem, save_path=str(save_fig))
        plot_phase_contributions(ref, title=stem, save_path=str(save_fig2))

    # ── Collect results
    from rietveld_engine import rwp as _rwp, rp as _rp
    yc  = ref.calc()
    rf  = ref.r_factors()

    ph_c, ph_t = ref.phases
    s_c,  s_t  = ref.scale_factors

    w_c = s_c * ph_c.Z * ph_c.volume()
    w_t = s_t * ph_t.Z * ph_t.volume()
    tot = w_c + w_t

    results = {
        'name':      stem,
        'rwp':       float(rf['Rwp']),
        'rp':        float(rf['Rp']),
        'wt_cubic':  float(100 * w_c / tot),
        'wt_tet':    float(100 * w_t / tot),
        # cubic cell
        'a_cubic':   float(ph_c.a),
        'x_Y_cubic': float(ph_c.x_Y),
        # tetragonal cell
        'a_tet':     float(ph_t.a),
        'c_tet':     float(ph_t.c),
        'z_O_tet':   float(ph_t.z_O),
        'x_Y_tet':   float(ph_t.x_Y),
        # profile
        'U': float(ref.U), 'V': float(ref.V), 'W': float(ref.W),
        'eta': float(ref.eta),
        'zero_offset': float(ref.zero_offset),
    }

    if save_results:
        out = filepath.parent / f'{stem}_results.json'
        with open(out, 'w') as f:
            json.dump(results, f, indent=2)
        print(f'  Results saved to {out.name}')

    return results


# ─── Batch mode ───────────────────────────────────────────────────────────────

def batch_refine(directory, x_Y_guess=0.05, plot_summary=True):
    """
    Refine all .xy files in *directory*.

    Returns
    -------
    List of result dicts, one per file.
    """
    directory = Path(directory)
    files = sorted(directory.glob('*.xy'))
    if not files:
        print(f'No .xy files found in {directory}')
        return []

    print(f'Found {len(files)} .xy file(s) in {directory}')

    all_results = []
    for fp in files:
        try:
            r = refine_file(fp, x_Y_guess=x_Y_guess,
                            plot=False, save_results=True)
            all_results.append(r)
        except Exception as e:
            print(f'  ERROR processing {fp.name}: {e}')

    # Summary table
    print('\n' + '='*70)
    print(f'{"Sample":<20} {"Rwp":>6} {"cubic wt%":>10} {"tet wt%":>8}'
          f' {"a_c (Å)":>9} {"a_t (Å)":>9} {"c_t (Å)":>9}')
    print('-'*70)
    for r in all_results:
        print(f'{r["name"]:<20} {r["rwp"]:6.4f} {r["wt_cubic"]:10.1f}'
              f' {r["wt_tet"]:8.1f}'
              f' {r["a_cubic"]:9.5f} {r["a_tet"]:9.5f} {r["c_tet"]:9.5f}')
    print('='*70)

    # Save combined CSV
    import csv
    csv_path = directory / 'all_results.csv'
    if all_results:
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=all_results[0].keys())
            writer.writeheader()
            writer.writerows(all_results)
        print(f'\nCombined results saved to {csv_path}')

    if plot_summary and all_results:
        plot_all_samples(all_results, save_dir=str(directory))

    return all_results


# ─── Entry point ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Rietveld refinement for Zr(1-x)Y(x)O(2-x/2)')
    parser.add_argument('path',
        help='Path to a .xy file OR a directory containing .xy files')
    parser.add_argument('--x_Y', type=float, default=0.05,
        help='Initial Y content guess  [0–0.5]  (default: 0.05)')
    parser.add_argument('--no-plot', action='store_true',
        help='Suppress individual Rietveld plots')
    args = parser.parse_args()

    p = Path(args.path)
    if p.is_dir():
        batch_refine(p, x_Y_guess=args.x_Y, plot_summary=True)
    elif p.is_file():
        refine_file(p, x_Y_guess=args.x_Y, plot=not args.no_plot)
    else:
        print(f'Error: {p} is not a valid file or directory.')
        sys.exit(1)


if __name__ == '__main__':
    main()
