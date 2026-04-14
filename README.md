# Resistivity of andesitic arc volcanoes

Post-processing code for computing electrical resistivity from coupled magmatic-hydrothermal simulations (CSMP++). Accompanies the manuscript:

> Scott, S.W. and Gresse, M. (in prep). The resistivity structure of andesitic arc volcanoes. *Journal of Volcanology and Geothermal Research*.

## Overview

This code converts CSMP++ VTU output fields (temperature, pressure, fluid saturations, salinity, melt fraction) into bulk electrical conductivity at mesh nodes. It implements a dual-domain framework:

- **Magmatic domain**: three-phase Archie's law mixing crystals, silicate melt, and exsolved volatiles (Glover, 2010; Samrock et al., 2021)
- **Hydrothermal domain**: modified Archie's law with surface conduction for clay-bearing rock (Revil et al., 2017)

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
- **Liquid**: Watanabe et al. (2021) viscosity-dependent empirical model, calibrated 20-525 C
- **Vapor**: Sinmyo & Keppler (2017) for density < 400 kg/m3; Watanabe for denser vapor

### Melt conductivity
- Andesite: Guo et al. (2017), dacite: Laumonier et al. (2019), rhyolite: Guo et al. (2016)
- Composition interpolation via Samrock et al. (2021) Lagrange scheme

### Rock matrix
- Olhoeft (1981) Arrhenius approximation for dry granite

### Surface conduction
- Waxman-Smits / Revil et al. (2017) framework with temperature-dependent counterion mobility

### Mixing laws
- Per-region cementation exponents (Glover, 2009; Revil et al., 2024; Zhang & Revil, 2023)
- Hashin-Shtrikman upper bound or per-phase Archie for two-phase fluid mixing

## Quick start

```python
from vtu_io import run_conductivity

config = {
    'porosity_exponent_m': 1.8,
    'two_phase_mixing': 'hashin_shtrikman',
    'magma_composition': {'type': 'dacite'},
    'clay_cap_regions': [11, 12],
    'regions': {
        11: {'CEC_meq_per_100g': 80.0, 'porosity_exponent_m': 1.8},
        12: {'CEC_meq_per_100g': 150.0, 'porosity_exponent_m': 1.8},
    },
}

results = run_conductivity(
    'yuz_basecase/Variables_100000.vtu',
    'yuz_basecase/Initial.vtu',
    config=config,
)

sigma_bulk = results['sigma_bulk']        # [S/m] at each node
x, y = results['coordinates']            # [km]
```

See `run_conductivity_test.py` for a complete example with plotting.

## Test data

The `yuz_basecase/` directory contains VTU output from a 2D axisymmetric CSMP++ simulation of the Yuzawa geothermal system (NE Japan). The simulation includes a cooling dacitic intrusion with magmatic volatile exsolution, hydrothermal circulation, and clay cap development.

## Dependencies

- Python 3.8+
- NumPy, SciPy, Matplotlib
- [iapws](https://pypi.org/project/iapws/) (IAPWS-95/97 water properties)
- [numba](https://numba.pydata.org/) (JIT compilation for rock conductivity)
- [VTK](https://vtk.org/) with Python bindings (VTU file reading)

## License

To be determined.
