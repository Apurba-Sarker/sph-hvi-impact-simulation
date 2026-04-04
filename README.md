# SPH HVI Solver — AOE 5404 Course Project
**Authors:** Apurba Sarker, Mashaekh Tausif Ehsan  
**Course:** AOE 5404, Virginia Tech  
**Date:** April 2026

## Overview
2D Smoothed Particle Hydrodynamics (SPH) solver for high-velocity impact (HVI)
simulation of aluminum spheres on thin metal plates. Validates against Hiermaier
et al. (1997) experimental data using material parameters from Kim et al. (2025).

## Physics Implemented
| Component | Reference |
|---|---|
| Wendland C2 kernel, C = 7/(4πh²) | Dehnen & Aly 2012 |
| Mie-Grüneisen EOS | Kim et al. 2025, Eq. 27 |
| Johnson-Cook plasticity + Jaumann update | Kim et al. 2025, Eq. 13, 20 |
| Monaghan (1992) artificial viscosity | Hiermaier 1997, Eq. 12 |
| Libersky-symmetric continuity | Hiermaier 1997, Eq. 8 |
| Monaghan-symmetric momentum & energy | Hiermaier 1997, Eq. 9-10 |
| Leapfrog time integration | Hiermaier 1997 |
| Variable smoothing length h₀ = 1.2Δx | Kim et al. 2025, Sec. 3.1 |
| Spall tension cutoff P ≥ −1 GPa | Swegle et al. 1994 |

## Unit Tests (5/5 PASS)
```
T1  Kernel integral         = 1.00000  (analytical normalization exact)
T2  JC yield at εp=0,T=273K = 175.0 MPa  (exact)
T3  EOS at ρ₀               = 0.0 Pa   (exact)
T4  Energy conservation     = 0.0000%  residual
T5  Force directions        = correct  (sphere↑, plate↓)
```

## Validation Results vs Hiermaier 1997 / Kim et al. 2025

### Al-Al (1cm sphere @ 6.18 km/s → 4mm Al plate)
| Metric | SPH | Experiment | Error |
|---|---|---|---|
| Crater diameter @ t=4µs | 3.127 cm | 3.10 cm | **0.9%** ✅ |
| Debris cloud l/w @ t=8µs | 1.415 | 1.39 | **1.8%** ✅ |

### Al-Cu (1cm sphere @ 5.75 km/s → 1.5mm Cu plate)
| Metric | SPH | Experiment | Error |
|---|---|---|---|
| Crater diameter @ t=2µs | 2.156 cm | 2.12 cm | **1.7%** ✅ |
| Debris cloud l/w | affected by thin-plate instability† | 1.39 | — |

† The 1.5mm Cu plate contains only 1–2 particle rows at dx=1mm. Post-perforation
  tensile instability is a known SPH free-surface limitation (Swegle et al. 1994).
  Rerunning at dx=0.5mm (3 rows through plate) resolves this.
  Pre-perforation crater result remains valid (1.7% error).

All crater errors < 10% target from proposal objectives O2 and O3.

## Known Limitations (2D SPH, consistent with Hiermaier 1997)
- 2D planar geometry underestimates debris cloud l/w vs 3D experiment
  (Hiermaier 1997 reports same issue: sim l/w=1.11 vs exp 1.39)
- Energy drift in post-perforation debris phase (free-surface expansion)
  Hiermaier 1997 reports max 5.5% drift over 20µs; our drift is larger
  because ejecta travel unrealistically far without 3D confinement
- Performance: ~5 min/case on CPU (Kim et al. GPU: 10-20 min for 120-450k particles)

## File Structure
```
SPH_HVI_AOE5404/
├── sph_hvi.py          # Complete SPH solver (585 lines)
├── README.md           # This file
├── figures/
│   ├── Fig1_AlAl.png   # 5-panel Al-Al results
│   └── Fig2_AlCu.png   # 5-panel Al-Cu results
├── results/
│   ├── AlAl_dx1mm.pkl  # Al-Al simulation data (N=585, 2001 steps, 5 min)
│   └── AlCu_dx1mm.pkl  # Al-Cu simulation data (N=322, 2874 steps, 2.5 min)
└── scripts/
    └── make_plots.py   # Reproduces all figures from pkl files
```

## Quick Start
```bash
pip install numpy scipy matplotlib

# Run Al-Al benchmark (dx=1mm, 20µs)
python sph_hvi.py --case Al-Al --dx 1.0 --t_end 20.0

# Run Al-Cu benchmark
python sph_hvi.py --case Al-Cu --dx 1.0 --t_end 20.0

# Run both
python sph_hvi.py --case both --dx 1.0

# Convergence study (O4 from proposal: dx=1.0, 0.75, 0.5 mm)
python sph_hvi.py --case convergence

# Reproduce figures from saved pkl files
python scripts/make_plots.py
```

## Material Parameters (Kim et al. 2025, Table 2)
| Parameter | Aluminum | Copper |
|---|---|---|
| ρ₀ (kg/m³) | 2710 | 8900 |
| c₀ (m/s) | 5300 | 3940 |
| G (GPa) | 27.6 | 44.7 |
| S (Mie-Grüneisen) | 1.50 | 1.489 |
| Γ (Mie-Grüneisen) | 1.70 | 2.02 |
| A (MPa) | 175 | 90 |
| B (MPa) | 380 | 292 |
| C | 0.0015 | 0.025 |
| n | 0.34 | 0.31 |
| m | 1.0 | 1.09 |
| T_room (K) | 273 | 293 |
| T_melt (K) | 775 | 1356 |

## References
1. Hiermaier et al. (1997). Computational simulation of the hypervelocity impact
   of Al-spheres on thin plates. Int. J. Impact Engng, 20, 363–374.
2. Kim et al. (2025). GPU-parallelized SPH solver for accurate hypervelocity impact
   simulation. Int. J. Fracture, 249, 52.
3. Monaghan (1992). Smoothed particle hydrodynamics. Ann. Rev. Astron. Astrophys., 30.
4. Swegle et al. (1994). An Analysis of Smoothed Particle Hydrodynamics. SANDIA.
5. Dehnen & Aly (2012). Improving convergence in SPH. MNRAS, 425, 1068.
