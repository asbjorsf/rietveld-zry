"""
Crystal structure definitions for the Zr(1-x)Y(x)O(2-x/2) system.

Phases implemented:
  - CubicZrO2      Fm-3m  (No. 225)  fluorite structure
  - TetragonalZrO2 P4_2/nmc (No. 137)

Atomic scattering factors:  Cromer-Mann 4-Gaussian parameterisation
  f(s) = sum_i a_i * exp(-b_i * s^2) + c    where s = sin(theta)/lambda [A^-1]

Y substitution model:  Zr(1-x)Y(x)O(2-x/2)
  - Zr/Y site:  f_eff = (1-x)*f_Zr + x*f_Y
  - O site:     occupancy = 1 - x/4  (one vacancy per two Y substitutions)
"""

import numpy as np

# ─── Cromer-Mann coefficients (International Tables Vol. C, Table 6.1.1.4) ───

_CROMER_MANN = {
    #          a1       a2       a3       a4       b1        b2       b3        b4       c
    'O':  ([3.0485,  2.2868,  1.5463,  0.8670], [13.2771,  5.7011,  0.3239, 32.9089],  0.2508),
    'Zr': ([17.8765, 10.9480,  5.4173,  3.6572], [ 1.2762, 11.9160,  0.1176, 87.6627],  2.0693),
    'Y':  ([17.1760,  4.8445, 15.5616,  3.6611], [ 1.1265, 27.5493,  0.0957, 99.1890],  1.0252),
}


def atomic_scattering_factor(element: str, s: np.ndarray) -> np.ndarray:
    """
    Atomic scattering factor f(s).

    Parameters
    ----------
    element : 'O', 'Zr', or 'Y'
    s       : sin(theta)/lambda  [A^-1], scalar or array

    Returns
    -------
    f : same shape as s
    """
    a, b, c = _CROMER_MANN[element]
    a = np.array(a)[:, None]
    b = np.array(b)[:, None]
    s2 = np.atleast_1d(s).astype(float) ** 2
    return float(c) + np.sum(a * np.exp(-b * s2), axis=0)


# ─── Cubic ZrO2  (Fm-3m, No. 225) ───────────────────────────────────────────

class CubicZrO2:
    """
    Cubic ZrO2 – fluorite structure, space group Fm-3m (No. 225).
    Z = 4 formula units per unit cell.

    Wyckoff positions
    -----------------
    Zr/Y  4a  (0,0,0) + FCC translations          → 4 atoms
    O     8c  (1/4,1/4,1/4) + FCC + inversion     → 8 atoms

    Parameters
    ----------
    a        : unit-cell parameter [Angstrom]
    x_Y      : Y fraction on Zr site  (0 ≤ x_Y ≤ 1)
    Biso_ZrY : isotropic ADP for Zr/Y site [Angstrom^2]
    Biso_O   : isotropic ADP for O site    [Angstrom^2]
    """

    name           = 'cubic'
    crystal_system = 'cubic'
    Z              = 4  # formula units per cell

    # All symmetry-equivalent fractional coordinates (hardcoded for speed)
    _ZrY_frac = np.array([
        [0.00, 0.00, 0.00],
        [0.00, 0.50, 0.50],
        [0.50, 0.00, 0.50],
        [0.50, 0.50, 0.00],
    ])

    _O_frac = np.array([
        [0.25, 0.25, 0.25],
        [0.75, 0.75, 0.25],
        [0.75, 0.25, 0.75],
        [0.25, 0.75, 0.75],
        [0.75, 0.75, 0.75],
        [0.25, 0.25, 0.75],
        [0.25, 0.75, 0.25],
        [0.75, 0.25, 0.25],
    ])

    def __init__(self, a=5.1270, x_Y=0.0, Biso_ZrY=0.5, Biso_O=0.8):
        self.a       = float(a)
        self.x_Y     = float(x_Y)
        self.Biso_ZrY = float(Biso_ZrY)
        self.Biso_O   = float(Biso_O)

    def d_spacing(self, h, k, l):
        return self.a / np.sqrt(h*h + k*k + l*l)

    def structure_factor_sq(self, h, k, l, s):
        """
        |F(hkl)|^2 including Debye-Waller and Y-substitution / O-vacancy model.

        s = sin(theta)/lambda for this reflection [A^-1].
        """
        x   = self.x_Y
        s_  = float(s)

        DW_ZrY = np.exp(-self.Biso_ZrY * s_ * s_)
        DW_O   = np.exp(-self.Biso_O   * s_ * s_)

        f_ZrY = ((1.0 - x) * atomic_scattering_factor('Zr', np.array([s_]))[0]
                 + x       * atomic_scattering_factor('Y',  np.array([s_]))[0]) * DW_ZrY

        occ_O = 1.0 - x / 4.0           # oxygen site occupancy
        f_O   = occ_O * atomic_scattering_factor('O', np.array([s_]))[0] * DW_O

        phase_ZrY = np.sum(np.exp(2j * np.pi * (
            h * self._ZrY_frac[:, 0] +
            k * self._ZrY_frac[:, 1] +
            l * self._ZrY_frac[:, 2])))

        phase_O = np.sum(np.exp(2j * np.pi * (
            h * self._O_frac[:, 0] +
            k * self._O_frac[:, 1] +
            l * self._O_frac[:, 2])))

        F = f_ZrY * phase_ZrY + f_O * phase_O
        return abs(F) ** 2

    def volume(self):
        return self.a ** 3

    def get_params(self):
        return dict(a=self.a, x_Y=self.x_Y,
                    Biso_ZrY=self.Biso_ZrY, Biso_O=self.Biso_O)

    def set_params(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, float(v))


# ─── Tetragonal ZrO2  (P4_2/nmc, No. 137, origin choice 2) ──────────────────

class TetragonalZrO2:
    """
    Tetragonal ZrO2, space group P4_2/nmc (No. 137, origin choice 2).
    Z = 2 formula units per unit cell.

    Wyckoff positions
    -----------------
    Zr   2a  (0,0,0), (1/2,1/2,1/2)
    O    4d  (0,1/2,z_O), (1/2,0,z_O+1/2), (0,1/2,1/2-z_O), (1/2,0,1-z_O)

    Reference:  Howard, Hill & Reichert, Acta Cryst. B44 (1988) 116.
    z_O ≈ 0.2065 displaces O from ideal fluorite (1/4) along c,
    giving 4 short + 4 long Zr-O bonds.

    Parameters
    ----------
    a, c     : unit-cell parameters [Angstrom]
    z_O      : O fractional coordinate along c (free parameter)
    x_Y      : Y fraction on Zr site
    Biso_ZrY : isotropic ADP for Zr/Y [Angstrom^2]
    Biso_O   : isotropic ADP for O     [Angstrom^2]
    """

    name           = 'tetragonal'
    crystal_system = 'tetragonal'
    Z              = 2

    _Zr_frac_fixed = np.array([
        [0.0, 0.0, 0.0],
        [0.5, 0.5, 0.5],
    ])

    def __init__(self, a=3.6041, c=5.1733, z_O=0.2065,
                 x_Y=0.0, Biso_ZrY=0.5, Biso_O=0.8):
        self.a       = float(a)
        self.c       = float(c)
        self.z_O     = float(z_O)
        self.x_Y     = float(x_Y)
        self.Biso_ZrY = float(Biso_ZrY)
        self.Biso_O   = float(Biso_O)

    @property
    def _O_frac(self):
        z = self.z_O
        return np.array([
            [0.0, 0.5,       z        ],
            [0.5, 0.0,       z + 0.5  ],
            [0.0, 0.5,  0.5 - z      ],
            [0.5, 0.0,  1.0 - z      ],
        ])

    def d_spacing(self, h, k, l):
        return 1.0 / np.sqrt((h*h + k*k) / self.a**2 + l*l / self.c**2)

    def structure_factor_sq(self, h, k, l, s):
        x   = self.x_Y
        s_  = float(s)

        DW_ZrY = np.exp(-self.Biso_ZrY * s_ * s_)
        DW_O   = np.exp(-self.Biso_O   * s_ * s_)

        f_ZrY = ((1.0 - x) * atomic_scattering_factor('Zr', np.array([s_]))[0]
                 + x       * atomic_scattering_factor('Y',  np.array([s_]))[0]) * DW_ZrY

        occ_O = 1.0 - x / 4.0
        f_O   = occ_O * atomic_scattering_factor('O', np.array([s_]))[0] * DW_O

        zf = self._Zr_frac_fixed
        of = self._O_frac

        phase_Zr = np.sum(np.exp(2j * np.pi * (
            h * zf[:, 0] + k * zf[:, 1] + l * zf[:, 2])))
        phase_O  = np.sum(np.exp(2j * np.pi * (
            h * of[:, 0] + k * of[:, 1] + l * of[:, 2])))

        F = f_ZrY * phase_Zr + f_O * phase_O
        return abs(F) ** 2

    def volume(self):
        return self.a**2 * self.c

    def get_params(self):
        return dict(a=self.a, c=self.c, z_O=self.z_O, x_Y=self.x_Y,
                    Biso_ZrY=self.Biso_ZrY, Biso_O=self.Biso_O)

    def set_params(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, float(v))
