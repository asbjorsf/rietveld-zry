"""
Core Rietveld pattern calculation for Cu Kα powder X-ray diffraction.

Key components
--------------
snip_background       : SNIP peak-stripping background estimate (signal processing)
chebyshev_background  : Chebyshev polynomial background (refineable)
generate_reflections  : enumerate observable (hkl) reflections for a phase
lorentz_polarization  : LP correction (optionally with graphite monochromator)
caglioti_fwhm         : Caglioti peak-width formula
pseudo_voigt          : normalised profile function
calculate_pattern     : assemble the full calculated pattern
Rwp, Rp, chi2         : agreement indices
"""

import numpy as np
from scipy.ndimage import uniform_filter1d

# ─── Wavelengths ─────────────────────────────────────────────────────────────

LAMBDA_Ka1 = 1.540562   # Angstrom
LAMBDA_Ka2 = 1.544390   # Angstrom
Ka2_Ka1_RATIO = 0.5     # relative intensity  I(Kα2)/I(Kα1)


# ─── SNIP background estimation ──────────────────────────────────────────────

def snip_background(y_obs, max_window=None, smooth_window=None):
    """
    Statistics-sensitive Non-linear Iterative Peak-clipping (SNIP).

    Reference:  Ryan & Jamieson, Spectrochim. Acta Part B 43 (1988) 1553.

    The algorithm progressively erodes peaks by replacing each point with
    the average of its two neighbours at distance *w*, but only if that
    average is lower than the current value.  Iterating from large *w*
    down to 1 strips peaks while leaving the slowly-varying background intact.

    Parameters
    ----------
    y_obs        : 1-D intensity array
    max_window   : starting window half-width in data points.
                   Default: n // 30  (≈ 3° for a 90° pattern at 0.013°/step)
    smooth_window: Gaussian smoothing after stripping (points).
                   Default: max_window // 8

    Returns
    -------
    bg : 1-D background array, same shape as y_obs
    """
    n = len(y_obs)
    if max_window is None:
        max_window = max(n // 30, 5)
    if smooth_window is None:
        smooth_window = max(max_window // 8, 3)

    bg = y_obs.astype(float).copy()
    idx = np.arange(n)

    for w in range(max_window, 0, -1):
        i_lo = np.clip(idx - w, 0, n - 1)
        i_hi = np.clip(idx + w, 0, n - 1)
        avg  = 0.5 * (bg[i_lo] + bg[i_hi])
        bg   = np.minimum(bg, avg)

    # Light smoothing to remove residual noise from the stripped background
    if smooth_window > 1:
        bg = uniform_filter1d(bg, size=smooth_window)

    return bg


def fit_chebyshev_to_background(two_theta, bg_array, n_terms):
    """
    Fit a Chebyshev polynomial to a pre-estimated background array.

    Uses the same [-1, 1] x-mapping as chebyshev_background() so that
    the returned coefficients are directly compatible.

    Returns
    -------
    coeffs : 1-D array of length n_terms
    """
    x = 2.0 * (two_theta - two_theta.min()) / (two_theta.max() - two_theta.min()) - 1.0
    c = np.polynomial.chebyshev.chebfit(x, bg_array, n_terms - 1)
    if len(c) < n_terms:
        c = np.pad(c, (0, n_terms - len(c)))
    return c[:n_terms]


# ─── Reflection generation ───────────────────────────────────────────────────

def generate_reflections(phase, wavelength, two_theta_max, two_theta_min=5.0):
    """
    Enumerate all powder-observable (hkl) reflections for *phase*.

    Uses brute-force enumeration over all (h,k,l) in [-hmax, hmax].
    Reflections at the same d-spacing (within 1e-4 Å tolerance) are grouped;
    their count gives the powder multiplicity.

    Returns
    -------
    list of dicts, each with keys:
        h, k, l          : representative Miller indices
        d                : d-spacing [Angstrom]
        two_theta        : peak position for Kα1 [degrees]
        multiplicity     : powder multiplicity
    Sorted by increasing two_theta.
    """
    d_min = wavelength / (2.0 * np.sin(np.radians(two_theta_max / 2.0)))
    d_max = wavelength / (2.0 * np.sin(np.radians(max(two_theta_min, 0.5) / 2.0)))

    cs = phase.crystal_system
    if cs == 'cubic':
        hmax = int(phase.a / d_min) + 1
    elif cs == 'tetragonal':
        hmax = int(max(phase.a, phase.c) / d_min) + 1
    else:
        raise ValueError(f'Unknown crystal system: {cs}')

    seen = {}

    for h in range(-hmax, hmax + 1):
        for k in range(-hmax, hmax + 1):
            for l in range(-hmax, hmax + 1):
                if h == 0 and k == 0 and l == 0:
                    continue
                if cs == 'cubic':
                    q = h*h + k*k + l*l
                    d = phase.a / np.sqrt(q)
                else:  # tetragonal
                    q = (h*h + k*k) / phase.a**2 + l*l / phase.c**2
                    if q == 0:
                        continue
                    d = 1.0 / np.sqrt(q)

                if d < d_min or d > d_max:
                    continue

                key = round(d, 4)
                if key not in seen:
                    # Store representative with smallest non-negative indices
                    seen[key] = {'d': d, 'rep': (h, k, l), 'count': 0}
                seen[key]['count'] += 1

    reflections = []
    for entry in seen.values():
        d     = entry['d']
        h, k, l = entry['rep']
        tth   = np.degrees(2.0 * np.arcsin(wavelength / (2.0 * d)))
        reflections.append({
            'h': h, 'k': k, 'l': l,
            'd': d,
            'two_theta': tth,
            'multiplicity': entry['count'],
        })

    return sorted(reflections, key=lambda r: r['two_theta'])


# ─── Lorentz-Polarization correction ─────────────────────────────────────────

def lorentz_polarization(two_theta_deg, monochromator=True,
                         cos2_2theta_mono=0.7998):
    """
    LP correction for X-ray powder diffraction.

    Parameters
    ----------
    two_theta_deg       : float or array, 2θ in degrees
    monochromator       : if True use graphite-monochromator form
    cos2_2theta_mono    : cos²(2θ_m) for graphite 002 with Cu Kα (≈ 0.800)

    Returns
    -------
    LP : same shape as two_theta_deg
    """
    tth = np.radians(two_theta_deg)
    th  = tth / 2.0
    cos2_2th = np.cos(tth) ** 2
    sin2_th  = np.sin(th)  ** 2
    cos_th   = np.cos(th)

    if monochromator:
        lp = (1.0 + cos2_2theta_mono * cos2_2th) / (2.0 * sin2_th * cos_th)
    else:
        lp = (1.0 + cos2_2th) / (2.0 * sin2_th * cos_th)

    return lp


# ─── Peak profile ─────────────────────────────────────────────────────────────

def caglioti_fwhm(two_theta_deg, U, V, W):
    """
    FWHM from the Caglioti equation:  FWHM² = U·tan²θ + V·tanθ + W

    Returns FWHM in degrees.  Clipped at 0.001° to avoid numerical issues.
    """
    tan_th = np.tan(np.radians(two_theta_deg / 2.0))
    fwhm2  = U * tan_th**2 + V * tan_th + W
    return np.sqrt(np.clip(fwhm2, 1e-6, None))


def pseudo_voigt(x, fwhm, eta):
    """
    Normalised pseudo-Voigt profile (integrates to 1).

    Parameters
    ----------
    x    : deviation from peak centre [degrees]
    fwhm : full width at half maximum [degrees]
    eta  : Lorentzian fraction (0 = pure Gaussian, 1 = pure Lorentzian)

    Returns
    -------
    profile : same shape as x
    """
    hwhm = fwhm / 2.0
    gauss   = (np.sqrt(np.log(2.0) / np.pi) / hwhm) * np.exp(-np.log(2.0) * (x / hwhm)**2)
    lorentz = (1.0 / (np.pi * hwhm)) / (1.0 + (x / hwhm)**2)
    return eta * lorentz + (1.0 - eta) * gauss


# ─── Background ───────────────────────────────────────────────────────────────

def chebyshev_background(two_theta, coeffs, tth_min=None, tth_max=None):
    """
    Background modelled as a sum of Chebyshev polynomials of the first kind.
    Mapping 2θ → [-1, 1] improves numerical stability.

    Parameters
    ----------
    two_theta : array of 2θ values
    coeffs    : 1-D array of Chebyshev coefficients  [b0, b1, b2, ...]
    """
    if tth_min is None: tth_min = two_theta.min()
    if tth_max is None: tth_max = two_theta.max()

    x = 2.0 * (two_theta - tth_min) / (tth_max - tth_min) - 1.0
    n = len(coeffs)
    if n == 0:
        return np.zeros_like(two_theta)

    T = np.zeros((n, len(two_theta)))
    if n >= 1: T[0] = 1.0
    if n >= 2: T[1] = x
    for i in range(2, n):
        T[i] = 2.0 * x * T[i-1] - T[i-2]

    return coeffs @ T


# ─── Full pattern calculation ─────────────────────────────────────────────────

def calculate_pattern(two_theta, phases, scale_factors, bg_coeffs,
                      U, V, W, eta, zero_offset,
                      wavelength=LAMBDA_Ka1,
                      include_Ka2=True,
                      monochromator=True,
                      cutoff_fwhm=12):
    """
    Calculate the full Rietveld pattern.

    Parameters
    ----------
    two_theta    : 1-D array of data 2θ [degrees]
    phases       : list of phase objects
    scale_factors: list of floats, one per phase
    bg_coeffs    : array of Chebyshev background coefficients
    U, V, W      : Caglioti FWHM parameters
    eta          : pseudo-Voigt Lorentzian fraction
    zero_offset  : 2θ zero-point correction [degrees]
    wavelength   : Kα1 wavelength [Angstrom]
    include_Ka2  : add Kα2 satellite peaks
    monochromator: use graphite-monochromator LP correction
    cutoff_fwhm  : profile cut-off in units of FWHM

    Returns
    -------
    y_calc : 1-D array of calculated intensities
    """
    tth_corr = two_theta - zero_offset
    tth_min  = two_theta.min()
    tth_max  = two_theta.max()

    y_calc = chebyshev_background(two_theta, bg_coeffs, tth_min, tth_max)

    for phase, scale in zip(phases, scale_factors):
        if scale <= 0.0:
            continue

        refs = generate_reflections(phase, wavelength, tth_max + 2, tth_min - 2)

        for ref in refs:
            h, k, l  = ref['h'], ref['k'], ref['l']
            d        = ref['d']
            tth_k    = ref['two_theta']
            mult     = ref['multiplicity']

            s   = 1.0 / (2.0 * d)                  # sin(theta)/lambda
            F2  = phase.structure_factor_sq(h, k, l, s)
            if F2 < 1e-9:
                continue                            # systematically absent

            LP   = lorentz_polarization(tth_k, monochromator)
            fwhm = caglioti_fwhm(tth_k, U, V, W)

            I_k = scale * mult * LP * F2

            # ── Kα1 contribution
            mask = np.abs(tth_corr - tth_k) < cutoff_fwhm * fwhm
            if np.any(mask):
                y_calc[mask] += I_k * pseudo_voigt(tth_corr[mask] - tth_k, fwhm, eta)

            # ── Kα2 contribution (shifted peak, half intensity)
            if include_Ka2:
                lam2 = LAMBDA_Ka2
                if lam2 / (2.0 * d) <= 1.0:
                    tth_k2  = np.degrees(2.0 * np.arcsin(lam2 / (2.0 * d)))
                    fwhm2   = caglioti_fwhm(tth_k2, U, V, W)
                    mask2   = np.abs(tth_corr - tth_k2) < cutoff_fwhm * fwhm2
                    if np.any(mask2):
                        y_calc[mask2] += (I_k * Ka2_Ka1_RATIO *
                                          pseudo_voigt(tth_corr[mask2] - tth_k2, fwhm2, eta))

    return y_calc


# ─── Agreement indices ────────────────────────────────────────────────────────

def rwp(y_obs, y_calc, weights):
    """Weighted profile R-factor."""
    num = np.sum(weights * (y_obs - y_calc) ** 2)
    den = np.sum(weights * y_obs ** 2)
    return np.sqrt(num / den)


def rp(y_obs, y_calc):
    """Unweighted profile R-factor."""
    return np.sum(np.abs(y_obs - y_calc)) / np.sum(y_obs)


def chi2(y_obs, y_calc, weights, n_params):
    """Reduced chi-squared (goodness-of-fit)."""
    return np.sum(weights * (y_obs - y_calc) ** 2) / max(len(y_obs) - n_params, 1)
