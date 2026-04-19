# Resistivity of andesitic arc volcanoes

Post-processing code for computing electrical resistivity from coupled magmatic-hydrothermal simulations (CSMP++). Accompanies the manuscript:

> Scott, S.W. and Gresse, M. (in prep). The resistivity structure of andesitic arc volcanoes. *Journal of Volcanology and Geothermal Research*.

## Overview

This code converts CSMP++ VTU output fields (temperature, pressure, fluid saturations, salinity, melt fraction) into bulk electrical conductivity at mesh nodes. It implements a dual-domain framework:

- **Magmatic domain**: three-phase generalized Archie's law mixing crystals, silicate melt, and exsolved volatiles with the conservation-of-connectedness constraint (Glover, 2009, 2010; Samrock et al., 2021).
- **Hydrothermal domain**: generalized Archie's law for pore fluid combined with a surface-conduction term. Five surface-conduction parameterizations are available (Waxman & Smits, 1968; Revil et al., 2002, 2019; Lévy et al., 2018; and two hybrid dispatches).

## Modules

| File | Description |
|------|-------------|
| `conductivity.py` | Core conductivity library. All physical models, mixing laws, and the nodal pipeline. |
| `viscosity.py` | Dynamic viscosity of H2O-NaCl fluids (Klyukin et al., 2017; Huber et al., 2009). |
| `vtu_io.py` | VTU file I/O and pipeline orchestrator. Loads CSMP++ output and calls `conductivity.py`. |
| `Driesners_eqs.py` | NaCl-H2O equation of state (Driesner & Heinrich, 2007). Density, phase boundaries. |
| `run_conductivity_test.py` | Example script: loads a VTU timestep and plots conductivity with temperature contours. |

## Physical models

### Fluid conductivity

Per-phase conductivity is routed by fluid density so a single pipeline covers subcritical liquid, supercritical fluid, and dilute steam:

- ρ ≥ 450 kg/m³ — Watanabe et al. (2021) viscosity-based empirical formula (calibrated 20–525 °C)
- 200 ≤ ρ < 450 kg/m³ — density-model parameterization following Watanabe et al. (2022, §3.2.2)
- ρ < 200 kg/m³ — Sinmyo & Keppler (2017)

To avoid step discontinuities in σ where (T, P, X) trajectories cross the routing-boundary densities (the three models disagree by ~30–40 % at ρ = 450 and somewhat more at ρ = 200), the routing applies a linear-in-ρ blend across narrow transition windows (default ±30 kg/m³ at ρ = 450 and ±20 kg/m³ at ρ = 200). Outside the windows each model is used at its published form. See `_sigma_fluid_by_density` for the implementation; the half-widths are keyword arguments and can be set to 0 to recover the original hard-step routing.

### Melt conductivity

- Dacite: Laumonier et al. (2019)
- Andesite: Guo et al. (2017)
- Rhyolite: Guo et al. (2016)
- Composition interpolation via the Samrock et al. (2021) Lagrange scheme

### Rock matrix

Dry silicate rock uses Watanabe et al. (2022) Eq. 2 (quadratic fit to Olhoeft, 1981) valid 25–800 °C. Halite (where it precipitates) uses the intrinsic-regime Arrhenius formula of Mapother, Crooks & Maurer (1950).

### Surface conduction

Selectable via `config['surface_conduction_model']`:

| Option | Equation | Notes |
|--------|----------|-------|
| `'waxman_smits'` | WS (1968) Eq. 19 | Classical σ_w-dependent counterion mobility `B(σ_w, T)` with β_25 recalibrated to Revil (2002)'s surface-bound value (0.53×10⁻⁸ m²/(V·s)) and α_T = 0.040 /°C. Keeps the WS 1968 α=0.6, γ=1.3 S/m structure. |
| `'revil'` | Revil et al. (2002, 2019) Bussian DEM | Full differential effective medium closed-form bulk conductivity (Revil 2019 Eq. 10), with apparent Na⁺ mobility `B₂₅ = 3.1×10⁻⁹ m²/(V·s)` and anion contribution `λ₂₅ = 3.0×10⁻¹⁰ m²/(V·s)` fit on 205 volcanic cores from Kilauea, White Island, Krafla, Yellowstone, etc.; Arrhenius-T with `E_a = 16 kJ/mol`. Calibrated specifically for clay-cap imaging in geothermal fields. Captures the Dukhin-number rollover near the isoconductivity point that linear additive surface-conduction forms miss. |
| `'levy'` | Lévy et al. (2018) Eq. 16 | Three-pathway empirical form `σ_S = B'(T)·(CEC/CEC_0)·(1−φ)/φ^(1−m)`, calibrated on Krafla smectite-bearing clay caps. Captures the interfoliar conduction pathway through connected smectite interlayers. |
| `'waxman_smits_revil'` | WS (non-clay) + Revil 2019 DEM (clay caps) | Dispatches the Revil 2019 Bussian DEM in regions listed in `config['clay_cap_regions']`, WS everywhere else. |
| `'waxman_smits_levy'` | WS (non-clay) + Lévy 2018 (clay caps) | Production default for smectite-bearing clay caps. |

Surface conduction is added to the mixing-law bulk term, with all formulations using `φ_eff = φ·(1 − S_halite)` so precipitated halite volume is excluded from the fluid-accessible pore space.

### CEC unit convention

CEC values are specified per region in `meq/100g`. Internally converted to C/kg via `MEQ_TO_CKG = 964.85` (= F·10⁻³/0.1, with F the Faraday constant). Earlier versions of this library used 9.6485 (100× too small); this has been fixed. All Waxman-Smits and Revil surface conduction results from versions prior to commit `7dc33c2` were suppressed by a factor of 100 by this unit bug — Lévy was unaffected because it only uses `CEC/CEC_0` as a ratio.

### Mixing laws

Per-region cementation exponents with literature-derived defaults:
- Intrusive/crystalline basement: m = 1.7
- Volcanic edifice: m = 2.1
- Clay cap (altered volcanics): m = 2.2
- Fractured/outflow zones: m = 1.5–1.6

Two bulk-mixing options:
- `'glover'` (default): Generalized Archie's law with the conservation-of-connectedness constraint `Σᵢ φᵢ^mᵢ = 1` (Glover, 2009, 2010). m_fluid is the regional value; m_solid is derived from the unity constraint.
- `'hashin_shtrikman'`: Two-step HS upper bound for the two-phase fluid, then Archie for rock–fluid mixing.

## Quick start

```python
from vtu_io import run_conductivity

config = {
    # Archie
    'porosity_exponent_m': 1.8,
    'saturation_exponent_n': 2.0,

    # Bulk mixing and surface conduction
    'mixing_law': 'glover',
    'surface_conduction_model': 'waxman_smits_levy',

    # Per-region lithology (cementation exponent, grain density,
    # smectite-effective CEC in meq/100g)
    'regions': {
        # Cooled intrusive body / crystalline basement
        1:  {'porosity_exponent_m': 1.7, 'grain_density': 2800.0, 'CEC_meq_per_100g': 0.3},
        2:  {'porosity_exponent_m': 1.7, 'grain_density': 2800.0, 'CEC_meq_per_100g': 0.2},
        5:  {'porosity_exponent_m': 1.7, 'grain_density': 2750.0, 'CEC_meq_per_100g': 0.2},
        6:  {'porosity_exponent_m': 1.7, 'grain_density': 2750.0, 'CEC_meq_per_100g': 0.2},

        # Volcanic edifice (andesite, fresh to mildly altered)
        7:  {'porosity_exponent_m': 2.1, 'grain_density': 2700.0, 'CEC_meq_per_100g': 1.5},
        8:  {'porosity_exponent_m': 2.1, 'grain_density': 2700.0, 'CEC_meq_per_100g': 1.5},

        # Fractured outflow zone
        10: {'porosity_exponent_m': 1.6, 'grain_density': 2700.0, 'CEC_meq_per_100g': 3.0},

        # Smectite clay caps (outer + smectite-rich central)
        11: {'porosity_exponent_m': 2.2, 'grain_density': 2600.0, 'CEC_meq_per_100g': 10.0},
        12: {'porosity_exponent_m': 2.2, 'grain_density': 2600.0, 'CEC_meq_per_100g': 20.0},
    },
    'default_region': {'grain_density': 2800.0, 'CEC_meq_per_100g': 0.2},
    'clay_cap_regions': [11, 12],

    # Magmatic domain
    'intrusion_id': 1,
    'magma_composition': {'type': 'dacite', 'SiO2_wt_percent': 67.8},
}

results = run_conductivity(
    'yuz_basecase/Variables_100000.vtu',
    'yuz_basecase/Initial.vtu',
    config=config,
)

sigma_bulk = results['sigma_bulk']   # [S/m] at each node
x, y = results['coordinates']        # [km]
```

See `run_conductivity_test.py` for a complete example with plotting.

## Test data

The `yuz_basecase/` directory contains VTU output from a 2D CSMP++ simulation of the Yuzawa geothermal system (NE Japan). The simulation includes a cooling dacitic intrusion with magmatic volatile exsolution, hydrothermal circulation, and smectite clay-cap development.

## Dependencies

- Python 3.8+
- NumPy, SciPy, Matplotlib
- [iapws](https://pypi.org/project/iapws/) (IAPWS-95/97 water properties)
- [numba](https://numba.pydata.org/) (JIT compilation for rock conductivity)
- [VTK](https://vtk.org/) with Python bindings (VTU file reading)

## Key references

- Archie, G.E. (1942). *Trans. AIME* 146, 54.
- Driesner, T. & Heinrich, C.A. (2007). *Geochim. Cosmochim. Acta* 71, 4880.
- Glover, P.W.J. (2009, 2010). *The Leading Edge* 28, 82; *Geophysics* 75(6), E247.
- Guo, X. et al. (2016, 2017). *EPSL* 433, 54; *EPSL* 468, 113.
- Klyukin, Y.I. et al. (2017). *Chem. Geol.* 471, 78.
- Kristinsdóttir, L.H. et al. (2010). *Geothermics* 39, 94.
- Laumonier, M. et al. (2019). *EPSL* 521, 79.
- Lévy, L. et al. (2018). *GJI* 215, 1558.
- Mapother, D. et al. (1950). *J. Chem. Phys.* 18, 1231.
- Olhoeft, G.R. (1981). *USGS Open-File Report* 81-1379.
- Revil, A. et al. (1998, 2002, 2019). *JGR* 103, 23925; *JGR* 107(B8), 2168; *JGR Solid Earth* 124, 4367.
- Samrock, F. et al. (2021). *Geophys. Res. Lett.* 48, e2020GL092370.
- Sinmyo, R. & Keppler, H. (2017). *Contrib. Mineral. Petrol.* 172, 4.
- Watanabe, N. et al. (2021, 2022). *Geofluids* 2021, 5514593; *Geothermics* 105, 102543.
- Waxman, M.H. & Smits, L.J.M. (1968). *Soc. Pet. Eng. J.* 8, 107.

## License

To be determined.
