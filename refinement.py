"""
Parameter management and sequential least-squares refinement.

Usage example
-------------
    from crystal_structures import CubicZrO2, TetragonalZrO2
    from refinement import RietveldRefinement

    cubic = CubicZrO2(a=5.127, x_Y=0.10)
    tet   = TetragonalZrO2(a=3.604, c=5.173, x_Y=0.04)

    ref = RietveldRefinement(two_theta, y_obs, sigma,
                             phases=[cubic, tet])
    ref.auto_scale()
    ref.run_sequential()
    ref.report()

Parameter naming convention
---------------------------
    'bg_N'           → background coefficient index N
    'scale_N'        → scale factor for phase N
    'U', 'V', 'W'    → Caglioti peak-width parameters
    'eta'            → pseudo-Voigt mixing
    'zero_offset'    → 2θ zero-point correction [degrees]
    'phase_N_attr'   → attribute 'attr' of phase N  (e.g. 'phase_0_a')
"""

import numpy as np
from scipy.optimize import least_squares

from rietveld_engine import (calculate_pattern, snip_background,
                              fit_chebyshev_to_background,
                              rwp as _rwp, rp as _rp, chi2 as _chi2)

# ─── Background initialisation ────────────────────────────────────────────────

N_BG_TERMS = 6   # Chebyshev terms — matches GSAS-II default recommendation (3–6 for lab XRD)


def _init_bg_from_snip(two_theta, y_obs, n_terms=N_BG_TERMS):
    """
    Estimate Chebyshev background coefficients using the SNIP algorithm.

    SNIP (Ryan & Jamieson 1988) progressively clips peaks by replacing
    each point with the average of its neighbours at decreasing distances.
    The result hugs the true background without overfitting noise.
    A Chebyshev polynomial is then fitted through the SNIP result so that
    the coefficients can be refined in the Rietveld optimisation.
    """
    bg_array = snip_background(y_obs)
    return fit_chebyshev_to_background(two_theta, bg_array, n_terms)


class RietveldRefinement:
    """Multi-phase Rietveld refinement for the Zr(1-x)Y(x)O(2-x/2) system."""

    def __init__(self, two_theta, y_obs, sigma, phases, wavelength=1.540562):
        self.two_theta  = np.asarray(two_theta,  dtype=float)
        self.y_obs      = np.asarray(y_obs,      dtype=float)
        self.sigma      = np.asarray(sigma,       dtype=float)
        self.weights    = 1.0 / np.maximum(self.sigma ** 2, 1.0)
        self.phases     = list(phases)
        self.wavelength = float(wavelength)

        # ── Global parameters (initial values) ──────────────────────────────
        self.scale_factors = [1.0] * len(phases)
        self.bg_coeffs     = _init_bg_from_snip(self.two_theta, self.y_obs)
        self._bg_coeffs_snip = self.bg_coeffs.copy()   # kept for bounds
        self.U           =  0.010
        self.V           = -0.005
        self.W           =  0.010
        self.eta         =  0.5
        self.zero_offset =  0.0

    # ── Parameter pack / unpack ──────────────────────────────────────────────

    def _get(self, name):
        if name.startswith('bg_'):
            return self.bg_coeffs[int(name[3:])]
        if name.startswith('scale_'):
            return self.scale_factors[int(name[6:])]
        if name.startswith('phase_'):
            _, idx, attr = name.split('_', 2)
            return getattr(self.phases[int(idx)], attr)
        return getattr(self, name)

    def _set(self, name, value):
        if name.startswith('bg_'):
            self.bg_coeffs[int(name[3:])] = value
        elif name.startswith('scale_'):
            self.scale_factors[int(name[6:])] = value
        elif name.startswith('phase_'):
            _, idx, attr = name.split('_', 2)
            setattr(self.phases[int(idx)], attr, value)
        else:
            setattr(self, name, value)

    def _pack(self, names):
        return np.array([self._get(n) for n in names], dtype=float)

    def _unpack(self, x, names):
        for v, n in zip(x, names):
            self._set(n, float(v))

    # ── Pattern calculation ──────────────────────────────────────────────────

    def calc(self):
        return calculate_pattern(
            self.two_theta, self.phases, self.scale_factors, self.bg_coeffs,
            self.U, self.V, self.W, self.eta, self.zero_offset,
            wavelength=self.wavelength,
        )

    # ── Agreement indices ────────────────────────────────────────────────────

    def r_factors(self):
        yc = self.calc()
        return {
            'Rwp':  _rwp(self.y_obs, yc, self.weights),
            'Rp':   _rp(self.y_obs, yc),
        }

    # ── Auto-scale initialisation ────────────────────────────────────────────

    def auto_scale(self):
        """
        Set scale factors so the initial calculated pattern roughly matches
        the data.  Uses a simple least-squares scale estimate.
        """
        # Temporarily equalise scales and compute pattern
        for i in range(len(self.scale_factors)):
            self.scale_factors[i] = 1.0

        yc = self.calc()
        yc_sum = max(yc.sum(), 1e-6)
        yo_sum = self.y_obs.sum()

        global_scale = yo_sum / yc_sum
        for i in range(len(self.scale_factors)):
            self.scale_factors[i] = global_scale / len(self.scale_factors)

    # ── Single refinement step ───────────────────────────────────────────────

    def refine(self, param_names, extra_bounds=None, verbose=True):
        """
        Refine the parameters listed in *param_names*.

        Parameters
        ----------
        param_names  : list[str]
        extra_bounds : dict  name → (lo, hi)   overrides defaults
        verbose      : print convergence info
        """
        x0 = self._pack(param_names)

        # Default bounds
        _bounds = {
            'eta':          (0.0,   1.0),
            'W':            (1e-6,  2.0),
            'U':            (-1.0,  2.0),
            'V':            (-2.0,  2.0),
            'zero_offset':  (-0.5,  0.5),
        }
        for i in range(len(self.phases)):
            _bounds[f'scale_{i}']              = (0.0,   np.inf)
            _bounds[f'phase_{i}_x_Y']          = (0.0,   0.5)
            _bounds[f'phase_{i}_Biso_ZrY']     = (0.001, 5.0)
            _bounds[f'phase_{i}_Biso_O']       = (0.001, 5.0)
            _bounds[f'phase_{i}_a']            = (3.0,   7.0)
            _bounds[f'phase_{i}_c']            = (3.0,   8.0)
            _bounds[f'phase_{i}_z_O']          = (0.10,  0.40)

        # Background coefficients: unconstrained — SNIP initialisation places them
        # in a physically sensible region; 6 Chebyshev terms cannot overfit a smooth bg.
        # (GSAS-II also leaves Chebyshev background coefficients unconstrained by default.)

        if extra_bounds:
            _bounds.update(extra_bounds)

        lo = np.array([_bounds.get(n, (-np.inf,  np.inf))[0] for n in param_names])
        hi = np.array([_bounds.get(n, (-np.inf,  np.inf))[1] for n in param_names])

        def residuals(x):
            self._unpack(x, param_names)
            yc = self.calc()
            return np.sqrt(self.weights) * (self.y_obs - yc)

        result = least_squares(
            residuals, x0,
            bounds=(lo, hi),
            method='trf',
            ftol=1e-10, xtol=1e-10, gtol=1e-10,
            max_nfev=5000,
            verbose=1 if verbose else 0,
        )
        self._unpack(result.x, param_names)

        rf = self.r_factors()
        if verbose:
            print(f"  → Rwp = {rf['Rwp']:.5f}   Rp = {rf['Rp']:.5f}")

        return result

    # ── Sequential refinement strategy ──────────────────────────────────────

    def run_sequential(self, verbose=True):
        """
        Run the recommended multi-step Rietveld strategy:

        Step 1  Background
        Step 2  Scale factors
        Step 3  Unit cell parameters
        Step 4  Peak profile (Caglioti + eta)
        Step 5  Zero-point correction
        Step 6  Oxygen z coordinate in tetragonal phase
        Step 7  Displacement parameters
        Step 8  Combined final refinement
        """
        n = len(self.phases)

        # collect phase-specific parameter names
        cell_params   = []
        disp_params   = []
        for i, ph in enumerate(self.phases):
            cell_params.append(f'phase_{i}_a')
            if ph.crystal_system == 'tetragonal':
                cell_params.append(f'phase_{i}_c')
            disp_params += [f'phase_{i}_Biso_ZrY', f'phase_{i}_Biso_O']

        z_O_params = [f'phase_{i}_z_O'
                      for i, ph in enumerate(self.phases)
                      if ph.crystal_system == 'tetragonal']

        scale_params   = [f'scale_{i}' for i in range(n)]
        bg_params      = [f'bg_{i}' for i in range(N_BG_TERMS)]
        profile_params = ['U', 'V', 'W', 'eta']

        steps = [
            ("1  Background",                    bg_params),
            ("2  Scale factors",                 scale_params),
            ("3  Unit cell parameters",          cell_params),
            ("4  Peak profile",                  profile_params),
            ("5  Zero-point correction",         ['zero_offset']),
            ("6  Oxygen z in tetragonal phase",  z_O_params),
            ("7  Displacement parameters",       disp_params),
            ("8  Combined final refinement",
             bg_params + scale_params + cell_params +
             profile_params + ['zero_offset'] +
             z_O_params + disp_params),
        ]

        for label, params in steps:
            if not params:
                continue
            if verbose:
                print(f"\n── Step {label} {'─'*(45-len(label))}")
            self.refine(params, verbose=verbose)

    # ── Report ───────────────────────────────────────────────────────────────

    def report(self):
        """Print a summary of refined parameters and R-factors."""
        yc = self.calc()
        rf = self.r_factors()

        sep = '=' * 60
        print(f'\n{sep}')
        print('  RIETVELD REFINEMENT RESULTS')
        print(sep)

        print(f'\n  Rwp = {rf["Rwp"]:.5f}     Rp = {rf["Rp"]:.5f}')

        print('\n  Global parameters')
        print(f'  {"zero offset":20s}: {self.zero_offset:+.4f} deg')
        print(f'  {"U, V, W":20s}: {self.U:.5f}  {self.V:.5f}  {self.W:.5f}')
        print(f'  {"eta":20s}: {self.eta:.4f}')
        print(f'  {"background":20s}: ' +
              '  '.join(f'{b:.2f}' for b in self.bg_coeffs))

        for i, (ph, sc) in enumerate(zip(self.phases, self.scale_factors)):
            print(f'\n  Phase {i+1}: {ph.name.capitalize()} ZrO2')
            print(f'  {"scale factor":20s}: {sc:.4e}')
            for attr, val in ph.get_params().items():
                print(f'  {attr:20s}: {val:.6f}')

        # Phase weight fractions (Hill & Howard formula)
        if len(self.phases) == 2:
            ph0, ph1 = self.phases
            s0, s1   = self.scale_factors

            # W_i ∝ S_i * Z_i * M_i * V_i  (use relative volumes; M absorbed into S)
            w0 = s0 * ph0.Z * ph0.volume()
            w1 = s1 * ph1.Z * ph1.volume()
            wt0 = 100 * w0 / (w0 + w1)
            wt1 = 100 * w1 / (w0 + w1)
            print(f'\n  Phase weight fractions (approx.)')
            print(f'  {ph0.name.capitalize():20s}: {wt0:.1f} wt%')
            print(f'  {ph1.name.capitalize():20s}: {wt1:.1f} wt%')

        print(f'\n{sep}')
