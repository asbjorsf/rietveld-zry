# Rietveld Refinement – Zr(1-x)Y(x)O(2-x/2)

Custom Python Rietveld refinement scripts for the yttria–zirconia system,
covering the cubic (Fm-3m) and tetragonal (P4₂/nmc) phases from lab Cu Kα data.

## Requirements

```bash
pip install numpy scipy matplotlib
```

## Usage

### Single file
```bash
python run_refinement.py path/to/sample.xy
python run_refinement.py path/to/sample.xy --x_Y 0.08   # with Y-content hint
```

### Batch (all .xy files in a folder)
```bash
python run_refinement.py path/to/XRD/
```

Outputs per sample: `<sample>_rietveld.png` and `<sample>_results.json`.
Batch mode additionally writes `all_results.csv` and a summary figure.

## File structure

| File | Description |
|---|---|
| `crystal_structures.py` | Phase definitions (CubicZrO2, TetragonalZrO2), Cromer-Mann scattering factors |
| `rietveld_engine.py` | Pattern calculation: LP correction, pseudo-Voigt profile, Chebyshev background, hkl generation |
| `refinement.py` | Parameter management, scipy `least_squares` wrapper, sequential refinement strategy |
| `plotting.py` | Three-panel Rietveld plot (obs/calc/diff + tick marks) |
| `run_refinement.py` | CLI entry point for single-file and batch refinement |

## Physics / crystallography notes

- **Scattering factors**: Cromer-Mann 4-Gaussian parameterisation (Int. Tables Vol. C).
- **Peak profile**: pseudo-Voigt with Caglioti FWHM (U, V, W parameters).
- **LP correction**: graphite-monochromator form by default.
- **Kα doublet**: Kα1 + Kα2 (I₂/I₁ = 0.5) included automatically.
- **Background**: Chebyshev polynomial (stable over wide 2θ range).
- **Y-substitution model**: Zr(1-x)Y(x)O(2-x/2)
  → O site occupancy = 1 − x/4 (one vacancy per 2 Y ions).

## Refinement strategy (8 sequential steps)

1. Background coefficients
2. Scale factors
3. Unit cell parameters (a for cubic; a, c for tetragonal)
4. Peak profile (U, V, W, η)
5. Zero-point correction
6. O z-coordinate in tetragonal phase
7. Displacement parameters (Biso)
8. Combined final refinement (all parameters simultaneously)

## References

- Evans & Evans, *J. Chem. Ed.* **98** (2021) 495 – Rietveld method explained via Excel
- Howard, Hill & Reichert, *Acta Cryst.* B44 (1988) 116 – tetragonal ZrO₂ structure
- GSAS-II tutorials: https://advancedphotonsource.github.io/GSAS-II-tutorials/
