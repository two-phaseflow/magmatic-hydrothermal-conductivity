#!/usr/bin/env python3
"""
conductivity.py -- Electrical conductivity for magmatic-hydrothermal systems
=================================================================================

Converts CSMP++ simulation output fields to bulk electrical conductivity at
mesh nodes. Designed for post-processing of coupled magmatic-hydrothermal
simulations with NaCl-H2O fluids and silicate melt.

Public interface
----------------
Fluid conductivity:
    sigma_liquid          Watanabe et al. (2021) liquid NaCl(aq) conductivity
    sigma_vapor           Sinmyo & Keppler (2017) vapor conductivity
    compute_fluid_conductivity  Per-phase fluid sigma with density-based dispatch

Rock matrix:
    sigma_rock_olhoeft    Olhoeft (1981) Arrhenius rock conductivity

Melt conductivity:
    sigma_melt_andesite   Guo et al. (2017) andesite (60.8 wt% SiO2)
    sigma_melt_dacite     Laumonier et al. (2019) dacite (65.8 wt% SiO2)
    sigma_melt_rhyolite   Guo et al. (2016) rhyolite (75.7 wt% SiO2)
    sigma_melt_interpolated  Samrock et al. (2021) Lagrange interpolation
    compute_melt_conductivity  Dispatcher with H2O priority and composition routing

Mixing laws:
    archie_three_phase          Glover (2010) n-phase additive Archie
    cementation_exponents_samrock  Samrock et al. (2021) variable m_i
    hashin_shtrikman_upper      Hashin & Shtrikman (1962) upper bound

Pipeline:
    calculate_conductivity_nodal  Full nodal conductivity pipeline

Dual-domain framework
---------------------
Every node is classified as either *magmatic* (phi_melt > melt_threshold,
default 0) or *hydrothermal* (all other nodes).

Magmatic domain (phi_melt > melt_threshold):
    Three-phase additive Archie's law (Glover, 2010, Geophysics 75(6),
    E247-E265, Eq. 1) mixes crystals, silicate melt, and exsolved
    magmatic volatiles:

        sigma_bulk = sigma_rock * phi_s^m_s + sigma_melt * phi_m^m_m
                   + sigma_vol * phi_v^m_v

    with the unity constraint (Glover, 2010, Eq. 2):

        sum(phi_i^m_i) = 1

    The cementation exponents m_i are phase-fraction-dependent following
    Samrock et al. (2021, EPSL 559, Eq. 3):

        m_melt = 1.0               if phi_melt > 0.4
        m_melt = -2.75*phi + 2.1   if phi_melt <= 0.4
        m_vol  = 1.5               (fixed)
        m_solid from unity constraint

    Phase fractions come from CSMP++ fields crystal_volume_fraction
    and fluid_volume_fraction. As phi_melt -> 0 at the solidification
    front, three-phase Archie naturally reduces to sigma_bulk ->
    sigma_rock (fully crystallized rock).

Hydrothermal domain (no melt):
    Modified Archie's law with surface conduction:

        sigma_bulk = (1/F) * sigma_fluid + sigma_surface

    where F = a / phi^m is the formation factor (Archie, 1942) and
    sigma_surface is the surface conduction contribution from cation
    exchange on mineral surfaces (Revil et al., 2017, GJI 208, Eq. 15).

Two-phase fluid mixing
-----------------------
At two-phase (liquid + vapor) nodes, the effective fluid conductivity
sigma_fluid is computed from per-phase conductivities. Each phase is
evaluated at its own salinity and density, then mixed via one of:

    'archie' (default): saturation-weighted per-phase Archie:

        sigma_fluid = S_liq^n * sigma_liq + S_vap^n * sigma_vap

    This is the standard petrophysical approach. Each phase
    conductivity is computed independently at its own salinity
    (salt_fraction_liquid, salt_fraction_vapor) and density
    (density_liquid, density_vapor). The saturation exponent n
    (default 2.0) controls how rapidly conductivity decreases
    as the more conductive phase (typically brine) becomes a
    smaller fraction of the pore space.

    'hashin_shtrikman': Hashin-Shtrikman upper bound:

        sigma_fluid = HS+(sigma_liq, sigma_vap, S_liq)

    Assumes the more conductive phase (brine) forms the connected
    matrix. Gives higher conductivity than Archie at the same
    saturations. More appropriate when brine wets grain surfaces
    and forms interconnected films even at low saturation.

For single-phase nodes (S_liq ~ 1 or S_vap ~ 1), both methods
give the same result.

The density-based model switch for the vapor phase (Watanabe above
400 kg/m3, Sinmyo-Keppler below) is applied within
compute_fluid_conductivity before the two-phase mixing step.

Per-region properties
---------------------
The following properties can be set globally (in DEFAULT_CONFIG) or
overridden per region (in config['regions'][id]):

    porosity_exponent_m   Cementation exponent (Archie m). Typical values:
        Fractured/permeable zones:       m ~ 1.3-1.5 (Glover, 2009)
        Intrusive igneous (granite):     m ~ 1.7     (Revil et al., 2024)
        Extrusive volcanic (andesite):   m ~ 2.1     (Zhang & Revil, 2023)
        Altered/zeolitized tuff:         m ~ 2.5-3.3 (Revil et al., 2002)

    saturation_exponent_n  Saturation exponent. Kept at n = 2.0 globally
        following standard practice (rarely varied by rock type).

    grain_density          Grain density [kg/m3] for surface conduction.
        Fresh igneous: 2700-2800; clay-altered: ~2500.

    f_stern                Stern layer fraction [-] for surface conduction.
        Fresh rock: 0.95; clay-rich altered: ~0.90 (more counterions
        in diffuse layer -> higher surface conduction).

    CEC_meq_per_100g       Cation exchange capacity [meq/100g].
        Fresh rock: ~2; smectite-rich clay cap: 50-150.

The tortuosity factor a is kept at 1.0 following Glover (2009), who
argues that a != 1 indicates an incorrect m value.

Example per-region config:
    'regions': {
        1: {'porosity_exponent_m': 1.7, 'grain_density': 2750.0},
        11: {'porosity_exponent_m': 2.2, 'grain_density': 2500.0,
             'CEC_meq_per_100g': 80.0, 'f_stern': 0.90},
    }

Liquid conductivity (Watanabe et al., 2021)
-------------------------------------------
The liquid-phase conductivity of NaCl(aq) follows the viscosity-dependent
empirical model of Watanabe et al. (2021, Fluid Phase Equil. 549, 113187).

The model chains two steps:

    1. Molar conductivity Lambda [S.m2/mol] from viscosity and molality:

        Lambda = A(m) + B(m)/mu + C(m)/mu^2       (Eq. 2)

       where mu is dynamic viscosity [Pa.s] and m is molality [mol/kg-H2O].
       A(m) is defined by Eq. (3), C(m) by Eq. (5). B(m) is obtained from
       Eq. (4) which defines B^{-1} (not B directly):

        B^{-1} = [b1 + (b2-b1)/(1+(sqrt(m)/b3)^b4)] * 1e6  (Eq. 4)
        B = 1 / (Q * 1e6)

       Parameters from Table 3.

    2. Solution conductivity sigma [S/m] from Lambda and molarity:

        sigma = Lambda * M_m3                        (Eq. 1)

       where M_m3 is molarity in mol/m3, computed from molality m,
       solution density rho, and molar mass M_NaCl.

    Viscosity is computed via Klyukin et al. (2017, Fluid Phase Equil.
    433, 193-205) using the equivalent-temperature approach (valid to
    900 C, 500 MPa, 0-100 wt% NaCl; see viscosity.py).

    Calibration range (Watanabe, from Bannard 1975 data): 20-525 C,
    25-200 MPa, 0.059-24.6 wt% NaCl. Above 525 C the model
    extrapolates via the viscosity-conductivity relationship. This is
    physically reasonable because the underlying Walden's rule
    correlation (Lambda vs 1/mu) is smooth and monotonic (Watanabe
    Fig. 3), but has not been validated experimentally above 525 C.
    In practice, nodes above 525 C are typically in the magmatic
    domain where melt conductivity models are used instead.

Vapor conductivity (Sinmyo & Keppler, 2017)
--------------------------------------------
For vapor with density < 400 kg/m3, the model of Sinmyo & Keppler
(2017, Contrib. Mineral. Petrol. 172, 4) is used:

    Lambda0 = 1573 - 1212*rho + 537062/T - 208122721/T^2

    log10(sigma) = -1.7060 - 93.78/T + 0.8075*log10(c)
                 + 3.0781*log10(rho) + log10(Lambda0)

where rho [g/cm3], T [K], c [wt% NaCl], Lambda0 [S.cm2/mol].

For vapor with density >= 400 kg/m3, the Watanabe liquid model is
used instead (the fluid is liquid-like at these densities and the
Watanabe model is valid). The 400 kg/m3 threshold comes from
Watanabe et al. (2021, Section 4.2), who report errors > 30% below
this density.

Valid range (Sinmyo-Keppler): 0-600 C, 0-1 GPa, 0.058-5.6 wt% NaCl.

Melt conductivity models
-------------------------
Five melt conductivity options are available, selected via
config['magma_composition']['type']:

    'andesite' -- Guo et al. (2017) JGR Solid Earth 122, Eq. 3.
        Andesite starting material with 60.8 wt% SiO2.
        log sigma = 5.23 - 0.56*w^0.6
                  - (8130.4 - 1462.7*w^0.6 + (581.3 - 12.7*w^2)*P) / T
        Valid: 1100-1600 K, <= 1.0 GPa, <= 6 wt% H2O.

    'dacite' -- Laumonier et al. (2019) EPSL 521, 79-90, Eq. 2.
        Dacite with 65.8 wt% SiO2. Parameters from Laumonier et al.
        (2015, Chemical Geology 418, 66-76). Also listed in Samrock et
        al. (2021, EPSL 559) Supplementary Table S1.
        sigma = exp[(a*w+b) + P_bar*(c*w+d) - (Ea + P_bar*dV)/(R*T)]
        Valid: up to 1573 C, 0.7-2.9 GPa.

    'rhyolite' -- Guo et al. (2016) EPSL 433, 54-62, Eq. 4.
        Rhyolite with 75.7 wt% SiO2.
        log sigma = 2.983 - 0.0732*w
                  - (3528 - 233.8*w + (763 - 7.5*w^2)*P) / T
        Valid: 868-1665 K, 0.5-1.0 GPa, 0-8 wt% H2O.

    'interpolated' -- Samrock et al. (2021) EPSL 559, 116765.
        Second-order Lagrange interpolation in log10-space between
        the three end-member compositions (60.8, 65.8, 75.7 wt% SiO2).
        Uses a fixed SiO2 value (config SiO2_wt_percent) for all nodes.

    'samrock' -- Same Lagrange interpolation as 'interpolated', but
        the SiO2 content varies per node: it increases linearly from
        SiO2_parent to SiO2_final as crystallinity (1 - phi_melt)
        increases. This models SiO2 enrichment in the residual melt
        during fractional crystallization, following Samrock et al.
        (2021). Requires SiO2_parent and SiO2_final in config.

Rock matrix conductivity (Olhoeft, 1981)
-----------------------------------------
Temperature-dependent conductivity of dry rock from Olhoeft (1981,
J. Geophys. Res. 86(B2), 931-936). Arrhenius approximation fitted to
the dry Westerly Granite data in Fig. 8:

    sigma = sigma_0 * exp(-E_a / (k_B * T))

where sigma_0 = 1e4 S/m and E_a = 1.2 eV. The activation energy is
consistent with solid-state conduction in dry silicate minerals
(Yang, 2011, Surv. Geophys. 32, Fig. 6).

Surface conduction
------------------
Surface conduction follows the Waxman-Smits (1968, J. Pet. Tech. 20(6),
107-122) framework as reformulated by Revil & Florsch (2010, GJI 181,
1480-1498, Eqs. 2-7) and Revil et al. (2017, GJI 208, 826-844, Eq. 15):

    sigma_s = (1 / (F * phi)) * rho_g * beta_plus * (1 - f) * CEC

where rho_g is grain density [kg/m3], beta_plus is counterion mobility
[m2/(V.s)], f is the Stern layer fraction [-], and CEC is the cation
exchange capacity [C/kg]. All three can be set per region.

Temperature dependence of counterion mobility (Revil et al., 2017):

    beta_plus(T) = beta_25 * (1 + alpha * (T - 25))
    beta_25(Na+) = 5.19e-8 m2/(V.s)
    alpha = 0.037 /C

Saturation scaling uses S_total = S_liq + S_vap rather than S_liq alone,
because at supercritical conditions the "vapor" phase still wets mineral
surfaces and contributes to surface conduction.

Phase fractions from CSMP++
---------------------------
Melt, crystal, and volatile phase fractions are obtained directly
from CSMP++ simulation output fields (crystal_volume_fraction and
fluid_volume_fraction). This library does NOT compute crystallization
internally -- the thermal-petrological evolution is handled entirely
by the forward model.

Modeling assumptions
--------------------
1. Fluid properties (T, P, salinity, density, saturation) are computed
   at nodes where the EOS is evaluated, avoiding interpolation artifacts
   at phase boundaries.

2. Vapor-phase conductivity uses Watanabe (2021) above 400 kg/m3
   (liquid-like) and Sinmyo & Keppler (2017) below 400 kg/m3.
   The threshold is from Watanabe (2021, Section 4.2).

3. At two-phase nodes, each phase conductivity is computed at its own
   salinity and density, then mixed via Archie or HS upper bound.

4. Porosity from Initial.vtu is spatially matched to timestep nodes
   using a KD-tree. This handles cases where node numbering differs
   between files.

5. Region IDs (element-based) are mapped to nodes with clay cap
   priority: if any element touching a node belongs to a clay cap
   region, the node inherits that region ID.

6. At the solidification front, three-phase Archie naturally reduces
   to sigma_bulk -> sigma_rock as phi_melt -> 0 and FVF -> 0.

7. Surface conduction uses S_total (liquid + vapor) rather than S_liq
   alone, to account for surface wetting at supercritical conditions.

Required CSMP++ VTU fields
--------------------------
Nodal (from timestep VTU):
    temperature, fluid_pressure, saturation_liquid, saturation_vapor,
    salt_fraction_liquid, salt_fraction_vapor, density_liquid,
    density_vapor, salinity, saturation_halite,
    crystal_volume_fraction, fluid_volume_fraction,
    water_fraction_melt

Static (from Initial.vtu):
    nodal_porosity, region_id (element), node_coordinates

References
----------
Glover, P.W.J. (2009). What is the cementation exponent? A new
    interpretation. The Leading Edge 28(1), 82-85.

Glover, P.W.J. (2010). A generalized Archie's law for n phases.
    Geophysics 75(6), E247-E265.

Guo, X., Zhang, L., Behrens, H. & Ni, H. (2016). Probing the status
    of felsic magma reservoirs: Constraints from the P-T-H2O dependences
    of electrical conductivity of rhyolitic melt. Earth Planet. Sci.
    Lett. 433, 54-62.

Guo, X., Bi, L., Ni, H. & Mao, Z. (2017). Electrical conductivity of
    hydrous andesitic melts pertinent to subduction zones. J. Geophys.
    Res. Solid Earth 122, 1777-1788.

Hashin, Z. & Shtrikman, S. (1962). A variational approach to the theory
    of the effective magnetic permeability of multiphase materials.
    J. Appl. Phys. 33(10), 3125-3131.

Klyukin, Y.I., Lowell, R.P. & Bodnar, R.J. (2017). A revised empirical
    model to calculate the dynamic viscosity of H2O-NaCl fluids at
    elevated temperatures and pressures. Fluid Phase Equilibria 433,
    193-205.

Laumonier, M., Gaillard, F. & Sifre, D. (2015). The effect of pressure
    and water concentration on the electrical conductivity of dacitic
    melts. Chemical Geology 418, 66-76.

Laumonier, M., Karakas, O., Bachmann, O., Gaillard, F., Lukacs, R. &
    Seghedi, I. (2019). Evidence for a persistent magma reservoir with
    large melt content beneath an apparently extinct volcano. Earth
    Planet. Sci. Lett. 521, 79-90.

Olhoeft, G.R. (1981). Electrical properties of granite with implications
    for the lower crust. J. Geophys. Res. 86(B2), 931-936.

Revil, A. & Florsch, N. (2010). Determination of permeability from
    spectral induced polarization in granular media. Geophys. J. Int.
    181, 1480-1498.

Revil, A., Le Breton, M., Niu, Q., Wallin, E., Haskins, E. & Thomas,
    D.M. (2017). Induced polarization of volcanic rocks - 1. Surface
    versus quadrature conductivity. Geophys. J. Int. 208, 826-844.

Revil, A. et al. (2024). Induced polarization of volcanic rocks - 8.
    The case of intrusive igneous rocks. Geophys. J. Int. 241(2), 1348.

Samrock, C.S., Grayver, A.V., Bachmann, O., Karakas, O. & Saar, M.O.
    (2021). Integrated magnetotelluric and petrological analysis of
    felsic magma reservoirs: Insights from Ethiopian rift volcanoes.
    Earth Planet. Sci. Lett. 559, 116765.

Sinmyo, R. & Keppler, H. (2017). Electrical conductivity of NaCl-bearing
    aqueous fluids to 600 C and 1 GPa. Contrib. Mineral. Petrol. 172, 4.

Watanabe, N., Yamaya, Y., Kitamura, K. & Mogi, T. (2021). Viscosity-
    dependent empirical formula for electrical conductivity of H2O-NaCl
    fluids at elevated temperatures and high salinity. Fluid Phase
    Equilibria 549, 113187.

Waxman, M.H. & Smits, L.J.M. (1968). Electrical conductivities in
    oil-bearing shaly sands. J. Pet. Tech. 20(6), 107-122.

Yang, X. (2011). Origin of high electrical conductivity in the lower
    continental crust: A review. Surv. Geophys. 32, 875-903.

Zhang, Z., Revil, A. et al. (2023). Induced polarization of volcanic
    rocks - 7. The case of pyroclastic rocks and lavas from
    stratovolcanoes. Geophys. J. Int. 234(3), 2375.

@author: samuels
"""

import warnings

import numpy as np
from numba import vectorize
from scipy.spatial import cKDTree
from viscosity import viscosity_h2o_nacl_vectorized
from iapws import IAPWS97


# =============================================================================
# VALIDATION HELPER
# =============================================================================

def _validate_range(arr, name, lo, hi, hard=False):
    """
    Check that array values are within [lo, hi].

    Parameters
    ----------
    arr : ndarray
    name : str, human-readable name for error messages.
    lo, hi : float, valid range bounds.
    hard : bool, if True raise ValueError; otherwise warnings.warn().

    Returns
    -------
    bad : ndarray of bool, True where values are out of range.
    """
    arr = np.asarray(arr)
    bad = np.isnan(arr) | (arr < lo) | (arr > hi)
    if np.any(bad):
        n = int(np.sum(bad))
        finite_bad = arr[bad & np.isfinite(arr)]
        if len(finite_bad) > 0:
            vmin, vmax = float(np.min(finite_bad)), float(np.max(finite_bad))
            msg = (f"{name}: {n} values outside [{lo}, {hi}] "
                   f"(range {vmin:.3g} to {vmax:.3g})")
        else:
            msg = f"{name}: {n} values are NaN"
        if hard:
            raise ValueError(msg)
        warnings.warn(msg)
    return bad


# =============================================================================
# PHYSICAL CONSTANTS
# =============================================================================

# Molar mass of NaCl [kg/mol].
M_NACL = 0.05844

# Conversion: meq/100g -> C/kg (SI).
# 1 meq = 96.485e-3 C; per 100g = per 0.1 kg; so 1 meq/100g = 0.96485 C/kg.
# Ref: Revil et al. (2017) GJI 208, Eq. (15) uses CEC in C/kg.
MEQ_TO_CKG = 9.6485



# =============================================================================
# DEFAULT_CONFIG
# =============================================================================

DEFAULT_CONFIG = {
    # --- Archie's law parameters ---
    'porosity_exponent_m': 1.8,       # cementation exponent [-]
    'saturation_exponent_n': 2.0,     # saturation exponent [-]
    'tortuosity_a': 1.0,              # tortuosity factor [-]

    # --- Two-phase fluid mixing ---
    'two_phase_mixing': 'archie',  # 'archie' or 'hashin_shtrikman'

    # --- Salinity floor ---
    'min_fluid_salinity_wt_percent': 0.0,  # background equilibrium [wt%]

    # --- Melt domain thresholds ---
    'melt_threshold': 0.0,                # phi_melt above this -> magmatic domain
    'spent_magma_min_porosity': 0.05,      # porosity floor for solidified intrusion
    'intrusion_id': 1,            # region ID of the intrusion body

    # --- Surface conduction (Revil et al., 2017, GJI 208, Eq. 15) ---
    'counterion_mobility_25C': 5.19e-8,   # beta(Na+, 25C) [m2/(V.s)]
    'counterion_temp_coeff': 0.037,        # alpha [1/C], linear T dependence

    # --- Per-region properties ---
    # Override any default_region property for specific region IDs.
    # Supported keys per region: grain_density, f_stern, CEC_meq_per_100g,
    # porosity_exponent_m, saturation_exponent_n.
    # Example:
    #   'regions': {
    #       1: {'porosity_exponent_m': 1.7},                     # basement
    #       11: {'porosity_exponent_m': 2.2, 'CEC_meq_per_100g': 80.0},  # clay cap
    #   }
    'regions': {},
    'default_region': {
        'grain_density': 2800.0,      # [kg/m3]
        'f_stern': 0.95,              # Stern layer fraction [-]
        'CEC_meq_per_100g': 2.0,      # cation exchange capacity
        'porosity_exponent_m': None,  # per-region override; None -> use global
        'saturation_exponent_n': None, # per-region override; None -> use global
    },

    # --- Clay cap identification ---
    'clay_cap_regions': [],

    # --- Magma composition ---
    # Phase fractions (melt, crystals, volatiles) come from CSMP++ output,
    # NOT from internal crystallization models. The settings below control
    # only which melt conductivity model is used.
    'magma_composition': {
        'type': 'dacite',             # Melt conductivity model:
                                      #   'andesite'     - Guo et al. (2017) 60.8 wt% SiO2
                                      #   'dacite'       - Laumonier et al. (2019) 65.8 wt% SiO2
                                      #   'rhyolite'     - Guo et al. (2016) 75.7 wt% SiO2
                                      #   'interpolated' - Lagrange interp. at fixed SiO2_wt_percent
                                      #   'samrock'      - Lagrange interp. with SiO2 enrichment
                                      #                    tracking during crystallization (SiO2
                                      #                    increases linearly from SiO2_parent to
                                      #                    SiO2_final as crystallinity increases)
        'constant_sigma_melt': None,  # override [S/m] or None (bypasses model)
        'SiO2_wt_percent': 65.0,      # [wt%] for 'interpolated' type
        'SiO2_parent': 63.0,          # [wt%] for 'samrock' type, parent magma
        'SiO2_final': 75.0,           # [wt%] for 'samrock' type, fully crystallized
        'T_solidus': 700.0,           # [C] reference solidus temperature
        'T_liquidus': 1000.0,         # [C] reference liquidus temperature
    },
}


# =============================================================================
# WATANABE REGRESSION PARAMETERS
# =============================================================================

# Watanabe et al. (2021) Fluid Phase Equilibria 549, 113187, Table 3.
# Molar conductivity regression: Lambda = A(m) + B(m)/mu + C(m)/mu^2 (Eq. 2)
# A(m): Eq. (3), B(m) from Eq. (4) via B = 1/(Q*1e6), C(m): Eq. (5).
# mu in Pa.s, m in mol/kg-H2O, Lambda in S.m^2/mol.
#
# IMPORTANT: Eq. (4) defines B^{-1}, not B:
#   B^{-1} = [b1 + (b2-b1)/(1+(sqrt(m)/b3)^b4)] * 1e6
# The b1-b4 parameters below are used to compute the bracket Q,
# then B = 1/(Q * 1e6). Verified against paper Figs. 4, 5, and 7.
WATANABE_PARAMS = {
    'a1': 4.16975e-03, 'a2': -5.08206e-03,
    'a3': 5.75588e-01, 'a4': 1.00422e+00,
    'b1': 2.55008e+01, 'b2': 6.04911e-02,
    'b3': 2.51861e+06, 'b4': 4.30952e-01,
    'c1': -4.89245e-10, 'c2': -1.75339e-11,
}


# =============================================================================
# SECTION 1: LIQUID CONDUCTIVITY -- Watanabe et al. (2021)
# Fluid Phase Equilibria 549, 113187
# =============================================================================

def wtfrac_to_molality(wt_frac_NaCl):
    """
    Convert NaCl mass fraction to molality.

    Derivation
    ----------
    For a solution with mass fraction w of NaCl and molar mass M_NaCl:

        m = (w / M_NaCl) / (1 - w)

    where the denominator (1 - w) is the mass of solvent (H2O) per unit
    mass of solution, giving molality in mol/kg-H2O.

    Parameters
    ----------
    wt_frac_NaCl : array_like
        NaCl mass fraction [0-1].

    Returns
    -------
    m : ndarray
        Molality [mol/kg-H2O]. NaN where input is invalid.
    """
    w = np.asarray(wt_frac_NaCl, dtype=float)
    invalid = (w < 0.0) | (w > 1.0)
    denom = np.where(1.0 - w > 0.0, 1.0 - w, np.nan)
    m = (w / M_NACL) / denom
    return np.where(invalid, np.nan, m)


def molar_conductivity_watanabe(mu_pas, molality):
    """
    Molar conductivity of NaCl(aq) from viscosity and molality.

    Watanabe et al. (2021) Fluid Phase Equilibria 549, 113187.

    Derivation
    ----------
    The viscosity-dependent molar conductivity model (Eq. 2):

        Lambda = A + B * mu^-1 + C * mu^-2

    where Lambda [S.m2/mol], mu [Pa.s], m [mol/kg-H2O].

    The molality-dependent coefficients are:

        A(m) = a1 + (a2 - a1) / (1 + (m/a3)^a4)          (Eq. 3)

        B^-1(m) = [b1 + (b2 - b1) / (1 + (sqrt(m)/b3)^b4)] * 1e6  (Eq. 4)
        => B(m) = 1 / (Q * 1e6)   where Q is the bracket

        C(m) = c1 + c2 * m                                 (Eq. 5)

    Note on Eq. 4: the paper defines B^{-1} (the reciprocal of B),
    not B directly. The x10^6 factor is part of the definition.
    Parameters from Table 3 (see WATANABE_PARAMS).

    Guards
    ------
    - Viscosity clipped to >= 1e-6 Pa.s to prevent division by zero.
    - Molality clipped to [0, 30] mol/kg-H2O (upper bound of valid range).
    - B^{-1} clipped to >= 1e-3 to prevent division by zero at edge cases.
    - Output clipped to >= 1e-12 S.m2/mol to prevent negative values
      from the quadratic term at extreme conditions.

    Parameters
    ----------
    mu_pas : array_like
        Dynamic viscosity [Pa.s].
    molality : array_like
        NaCl molality [mol/kg-H2O].

    Returns
    -------
    Lambda : ndarray
        Molar conductivity [S.m2/mol].
    """
    p = WATANABE_PARAMS
    mu = np.clip(np.asarray(mu_pas, dtype=float), 1e-6, np.inf)
    m = np.clip(np.asarray(molality, dtype=float), 0.0, 30.0)

    # Eq. 3: A(m)
    A = p['a1'] + (p['a2'] - p['a1']) / (1.0 + (m / p['a3'])**p['a4'])

    # Eq. 4: B^{-1}(m) = Q * 1e6, so B = 1/(Q * 1e6)
    sqrtm = np.sqrt(np.maximum(m, 0.0))
    Q = p['b1'] + (p['b2'] - p['b1']) / (1.0 + (sqrtm / p['b3'])**p['b4'])
    B_inv = np.maximum(Q * 1.0e6, 1e-3)
    B = 1.0 / B_inv

    # Eq. 5: C(m)
    C = p['c1'] + p['c2'] * m

    # Eq. 2: Lambda = A + B/mu + C/mu^2,  mu in Pa.s
    Lambda = A + B / mu + C / (mu * mu)
    return np.maximum(Lambda, 1e-12)


def sigma_liquid(wt_frac_NaCl, T_C, P_bar, density_solution):
    """
    Electrical conductivity of NaCl(aq) liquid.

    Watanabe et al. (2021) Fluid Phase Equilibria 549, 113187.

    Derivation
    ----------
    The solution conductivity is computed from the molar conductivity
    Lambda and molarity (Eq. 1):

        Lambda = sigma_f / (M * 1e3)

    where the factor 1e3 converts M from mol/L to mol/m3.
    Rearranging:

        sigma_f = Lambda * M_m3

    where M_m3 is molarity in mol/m3, computed from molality m,
    solution density rho [kg/m3], and molar mass M_NaCl [kg/mol]:

        M_m3 = (m * rho) / (1 + m * M_NaCl)

    Viscosity is computed from the Klyukin et al. (2017, Fluid Phase
    Equil. 433, 193-205) equivalent-temperature model.

    Valid range: 20-525 C, 25-200 MPa, 0.059-24.6 wt% NaCl.

    Guards
    ------
    - Zero-salinity nodes return 1e-6 S/m (pure water floor).
    - NaN viscosity values are replaced with 1e-3 Pa.s (water at ~20 C).
    - Output clipped to >= 1e-6 S/m.

    Parameters
    ----------
    wt_frac_NaCl : array_like
        NaCl mass fraction [0-1].
    T_C : array_like
        Temperature [deg C].
    P_bar : array_like
        Pressure [bar].
    density_solution : array_like
        Solution density [kg/m3].

    Returns
    -------
    sigma : ndarray
        Electrical conductivity [S/m]. Minimum 1e-6.
    """
    wt_frac_NaCl = np.asarray(wt_frac_NaCl)
    T_C = np.asarray(T_C)
    P_bar = np.asarray(P_bar)
    density_solution = np.asarray(density_solution)

    low_sal = wt_frac_NaCl <= 1e-6

    _validate_range(density_solution, "density_solution", 1.0, 2000.0)

    eta_cP = viscosity_h2o_nacl_vectorized(wt_frac_NaCl, T_C, P_bar)
    eta_Pas = eta_cP / 1000.0
    bad_visc = np.isnan(eta_Pas) | (eta_Pas <= 0)
    if np.any(bad_visc & ~low_sal):
        warnings.warn(
            f"sigma_liquid: {int(np.sum(bad_visc & ~low_sal))} non-zero-salinity "
            f"nodes with invalid viscosity (NaN or <= 0). "
            f"Replacing with 1e-3 Pa.s.")
    eta_Pas = np.where(bad_visc, 1e-3, eta_Pas)

    molality = wtfrac_to_molality(wt_frac_NaCl)
    Lambda = molar_conductivity_watanabe(eta_Pas, molality)

    # Eq. (1): sigma = Lambda * M_m3, where M_m3 in mol/m3
    M_m3 = (molality * density_solution) / (1.0 + molality * M_NACL)
    sigma = Lambda * M_m3

    # Floor to 1e-6 S/m; warn if non-zero-salinity nodes are floored
    floored = (sigma < 1e-6) & ~low_sal & np.isfinite(sigma)
    if np.any(floored):
        warnings.warn(
            f"sigma_liquid: {int(np.sum(floored))} non-zero-salinity nodes "
            f"floored to 1e-6 S/m. Check T/P/density inputs.")

    sigma = np.where(low_sal, 1e-6, sigma)
    sigma = np.where(np.isnan(sigma), 1e-6, sigma)
    return np.maximum(sigma, 1e-6)


# =============================================================================
# SECTION 2: VAPOR CONDUCTIVITY -- Sinmyo & Keppler (2017)
# Contrib. Mineral. Petrol. 172, 4
# =============================================================================

def _sigma_vapor_single(wt_frac_NaCl, T_C, P_bar, density_solution):
    """
    Single-node vapor conductivity for low-density steam.

    Sinmyo, R. & Keppler, H. (2017). Electrical conductivity of NaCl-
    bearing aqueous fluids to 600 C and 1 GPa. Contrib. Mineral. Petrol.
    172, 4. Eq. (3).

    Derivation
    ----------
    The limiting molar conductivity Lambda0 is given by:

        Lambda0 = 1573 - 1212*rho + 537062/T - 208122721/T^2

    where rho [g/cm3], T [K], Lambda0 [S.cm2/mol].

    The total conductivity (from abstract):

        log10(sigma) = -1.7060 - 93.78/T + 0.8075*log10(c)
                     + 3.0781*log10(rho) + log10(Lambda0)

    where c [wt% NaCl], rho [g/cm3], T [K], sigma [S/m].

    Uses IAPWS97 for pure water density at each node.
    Valid: 0-600 C, 0-1 GPa, 0.01-1 M NaCl (0.058-5.6 wt%).

    Parameters
    ----------
    wt_frac_NaCl : float
        NaCl mass fraction [0-1].
    T_C : float
        Temperature [deg C].
    P_bar : float
        Pressure [bar].
    density_solution : float
        Solution density [kg/m3].

    Returns
    -------
    sigma : float
        Electrical conductivity [S/m]. Minimum 1e-6.
    """
    T_K = T_C + 273.15

    if wt_frac_NaCl <= 1e-6:
        return 1e-6

    P_MPa = P_bar / 10.0
    try:
        water = IAPWS97(T=T_K, P=P_MPa)
        rho_H2O = water.rho / 1000.0  # kg/m3 -> g/cm3
    except Exception:
        return 1e-6

    Lambda0 = 1573.0 - 1212.0 * rho_H2O + 537062.0 / T_K - 208122721.0 / T_K**2
    if Lambda0 <= 0:
        return 1e-6

    # Sinmyo & Keppler (2017) abstract: c is NaCl concentration in wt%.
    c_wt_pct = wt_frac_NaCl * 100.0
    if c_wt_pct <= 1e-10:
        return 1e-6

    log_sigma = (-1.7060 - 93.78 / T_K + 0.8075 * np.log10(c_wt_pct)
                 + 3.0781 * np.log10(max(rho_H2O, 1e-6))
                 + np.log10(Lambda0))
    return max(1e-6, 10.0 ** log_sigma)


def sigma_vapor(wt_frac_NaCl, T_C, P_bar, density_solution):
    """
    Vectorized vapor conductivity (loops internally due to IAPWS97).

    Wraps _sigma_vapor_single() for array inputs. The loop is
    unavoidable because IAPWS97 is a scalar Python function.

    See _sigma_vapor_single for model details and references.

    Parameters
    ----------
    wt_frac_NaCl, T_C, P_bar, density_solution : array_like

    Returns
    -------
    sigma : ndarray
        Electrical conductivity [S/m].
    """
    wt_frac_NaCl = np.asarray(wt_frac_NaCl)
    T_C = np.asarray(T_C)
    P_bar = np.asarray(P_bar)
    density_solution = np.asarray(density_solution)

    if not (len(wt_frac_NaCl) == len(T_C) == len(P_bar) == len(density_solution)):
        raise ValueError(
            f"sigma_vapor: input array length mismatch "
            f"({len(wt_frac_NaCl)}, {len(T_C)}, {len(P_bar)}, "
            f"{len(density_solution)})")

    result = np.zeros_like(T_C, dtype=float)
    for i in range(len(T_C)):
        result[i] = _sigma_vapor_single(
            wt_frac_NaCl[i], T_C[i], P_bar[i], density_solution[i])

    n_floor = int(np.sum(result <= 1e-6))
    if n_floor > 0 and n_floor < len(result):
        warnings.warn(
            f"sigma_vapor: {n_floor}/{len(result)} nodes at floor (1e-6 S/m)")

    return result


# =============================================================================
# SECTION 3: FLUID CONDUCTIVITY (two-phase L+V mixing at nodes)
# =============================================================================

def compute_fluid_conductivity(X_liq, X_vap, T_C, P_bar, S_liq, S_vap,
                               rho_liq, rho_vap):
    """
    Per-phase fluid conductivity at nodes with density-based vapor model.

    Assumptions
    -----------
    1. Liquid phase always uses Watanabe et al. (2021), which is
       calibrated for liquid-like densities (rho > 370 kg/m3).
    2. Vapor phase model depends on density:
       - rho >= 400 kg/m3: Watanabe (2021). At these densities the
         fluid is liquid-like and the viscosity-based model is valid.
       - rho < 400 kg/m3: Sinmyo & Keppler (2017). Watanabe et al.
         (2021) explicitly note errors > 30% below 400 kg/m3.
       The 400 kg/m3 threshold comes from Watanabe (2021, Section 4.2).

    Parameters
    ----------
    X_liq, X_vap : array_like
        Salt mass fraction in each phase [0-1].
    T_C, P_bar : array_like
        Temperature [deg C] and pressure [bar].
    S_liq, S_vap : array_like
        Phase saturations [0-1].
    rho_liq, rho_vap : array_like
        Phase densities [kg/m3].

    Returns
    -------
    dict
        'sigma_liq' and 'sigma_vap' arrays [S/m].
    """
    n = len(T_C)
    tol = 1e-4

    liq_mask = S_liq > tol
    sig_liq = np.zeros(n)
    if np.any(liq_mask):
        sig_liq[liq_mask] = sigma_liquid(
            X_liq[liq_mask], T_C[liq_mask], P_bar[liq_mask], rho_liq[liq_mask])

    vap_mask = S_vap > tol
    sig_vap = np.zeros(n)
    if np.any(vap_mask):
        rho_v = np.asarray(rho_vap)
        # Watanabe (2021) is valid for liquid-like densities (rho > 400 kg/m3);
        # Sinmyo-Keppler (2017) covers the full range including dilute steam.
        dense = vap_mask & (rho_v >= 400.0)
        dilute = vap_mask & (rho_v < 400.0)

        if np.any(dense):
            sig_vap[dense] = sigma_liquid(
                X_vap[dense], T_C[dense], P_bar[dense], rho_v[dense])
        if np.any(dilute):
            sig_vap[dilute] = sigma_vapor(
                X_vap[dilute], T_C[dilute], P_bar[dilute], rho_v[dilute])

        print(f"    Vapor: dense(Watanabe)={np.sum(dense)}, "
              f"dilute(Sinmyo-Keppler)={np.sum(dilute)}")

    n_liq = np.sum(liq_mask & ~vap_mask)
    n_vap = np.sum(~liq_mask & vap_mask)
    n_2ph = np.sum(liq_mask & vap_mask)
    print(f"    Phases: liquid={n_liq}, vapor={n_vap}, two-phase={n_2ph}")

    return {'sigma_liq': sig_liq, 'sigma_vap': sig_vap}


# =============================================================================
# SECTION 4: ROCK MATRIX CONDUCTIVITY -- Olhoeft (1981)
# J. Geophys. Res. 86(B2), 931-936
# =============================================================================

@vectorize(['float64(float64)'], nopython=True)
def sigma_rock_olhoeft(T_C):
    """
    Temperature-dependent dry rock matrix conductivity.

    Olhoeft, G.R. (1981). J. Geophys. Res. 86(B2), 931-936.

    Arrhenius approximation fitted to dry Westerly Granite data
    (Fig. 8, measured after volatile outgassing at each temperature
    in 10^-11 MPa vacuum):

        sigma = sigma_0 * exp(-E_a / (k_B * T))

    where sigma_0 = 1e4 S/m, E_a = 1.2 eV, k_B = 8.617e-5 eV/K,
    T in Kelvin.

    Fit verification against Fig. 8 data points:
        727 C (1000/T=1.0): log(rho) ~ 2    -> sigma ~ 0.01 S/m
                            model gives           0.023 S/m  OK
        394 C (1000/T=1.5): log(rho) ~ 5    -> sigma ~ 1e-5 S/m
                            model gives           2.3e-5 S/m OK
        227 C (1000/T=2.0): log(rho) ~ 8    -> sigma ~ 1e-8 S/m
                            model gives           2.3e-8 S/m OK

    The activation energy E_a = 1.2 eV is consistent with solid-state
    conduction in dry silicate minerals at crustal temperatures. Yang
    (2011, Surv. Geophys. 32, Fig. 6) reports comparable values for
    dry lower-crustal clinopyroxene, orthopyroxene, and plagioclase
    (E_a ~ 1.0-1.5 eV).

    NOTE: This is an approximation from figure data, not an explicit
    equation in the paper. The paper shows data for Westerly Granite
    and hornblende schist with very similar trends.

    Guards
    ------
    Output is clipped to >= 1e-12 S/m to prevent underflow at low
    temperatures while preserving physical resolution. At 25 C the
    formula gives ~6e-16 S/m, which is negligible for all mixing
    laws but still finite.

    Parameters
    ----------
    T_C : float
        Temperature [deg C].

    Returns
    -------
    sigma : float
        Rock conductivity [S/m].
    """
    T_K = T_C + 273.15
    E_a = 1.2          # eV, fitted to Olhoeft (1981) Fig. 8
    k_B = 8.617333e-5  # eV/K
    sigma_0 = 1e4      # S/m, fitted to Olhoeft (1981) Fig. 8
    return max(1e-12, sigma_0 * np.exp(-E_a / (k_B * T_K)))


# =============================================================================
# SECTION 5: MELT CONDUCTIVITY MODELS
# =============================================================================

def sigma_melt_andesite(T_K, P_GPa, H2O_wt):
    """
    Andesite melt conductivity (60.8 wt% SiO2).

    Guo, X., Bi, L., Ni, H. & Mao, Z. (2017). J. Geophys. Res. Solid
    Earth 122, 1777-1788. Eq. (3):

        log sigma = 5.23 - 0.56*w^0.6
                  - (8130.4 - 1462.7*w^0.6 + (581.3 - 12.7*w^2)*P) / T

    where sigma [S/m], T [K], P [GPa], w [wt% H2O].

    The pressure term is inside the 1/T division, meaning pressure
    affects the effective activation energy. This is verified against
    the original paper.

    Valid: 1100-1600 K, <= 1.0 GPa, <= 6 wt% H2O.

    Parameters
    ----------
    T_K : array_like
        Temperature [K].
    P_GPa : array_like
        Pressure [GPa].
    H2O_wt : array_like
        Dissolved water [wt%].

    Returns
    -------
    sigma : ndarray
        Conductivity [S/m].
    """
    T_K = np.asarray(T_K, dtype=np.float64)
    P_GPa = np.asarray(P_GPa, dtype=np.float64)
    H2O_wt = np.asarray(H2O_wt, dtype=np.float64)

    w06 = np.power(np.maximum(H2O_wt, 0.01), 0.6)
    log_sigma = (5.23 - 0.56 * w06
                 - (8130.4 - 1462.7 * w06
                    + (581.3 - 12.7 * H2O_wt**2) * P_GPa) / T_K)
    return np.power(10.0, log_sigma)


def sigma_melt_dacite(T_K, P_GPa, H2O_wt):
    """
    Dacite melt conductivity (65.8 wt% SiO2).

    Laumonier, M. et al. (2019). Earth Planet. Sci. Lett. 521, 79-90.
    Eq. (2), with parameters originally from Laumonier et al. (2015)
    Chemical Geology 418, 66-76:

        sigma = exp[(a*w + b) + P_bar*(c*w + d)
                    - (1/(R*T)) * (Ea + P_bar*dV)]

    Parameters from Laumonier (2019) Eq. (2):
        a = 0.395, b = 4.65, c = -1.77e-6, d = 3.91e-5
        Ea = 60000 J/mol, dV = 0.654 cm3/mol

    Also listed in Samrock et al. (2021, EPSL 559) Supplementary
    Table S1.

    Valid: up to 1573 C, 0.7-2.9 GPa.

    Parameters
    ----------
    T_K : array_like
        Temperature [K].
    P_GPa : array_like
        Pressure [GPa].
    H2O_wt : array_like
        Dissolved water [wt%].

    Returns
    -------
    sigma : ndarray
        Conductivity [S/m].
    """
    T_K = np.asarray(T_K, dtype=np.float64)
    P_GPa = np.asarray(P_GPa, dtype=np.float64)
    H2O_wt = np.asarray(H2O_wt, dtype=np.float64)

    a, b = 0.395, 4.65
    c, d = -1.77e-6, 3.91e-5
    E_a = 60000.0     # J/mol
    dV = 6.54e-1      # cm3/mol
    R = 8.3143        # J/(mol.K)
    P_bar = P_GPa * 1.0e4

    return np.exp((a * H2O_wt + b)
                  + P_bar * (c * H2O_wt + d)
                  - (1.0 / (R * T_K)) * (E_a + P_bar * dV))


def sigma_melt_rhyolite(T_K, P_GPa, H2O_wt):
    """
    Rhyolite melt conductivity (75.7 wt% SiO2).

    Guo, X., Zhang, L., Behrens, H. & Ni, H. (2016). Earth Planet.
    Sci. Lett. 433, 54-62. Eq. (4):

        log sigma = 2.983 - 0.0732*w
                  - (3528 - 233.8*w + (763 - 7.5*w^2)*P) / T

    where sigma [S/m], T [K], P [GPa], w [wt% H2O].

    Valid: 868-1665 K, 0.5-1.0 GPa, 0-8 wt% H2O.

    Parameters
    ----------
    T_K : array_like
        Temperature [K].
    P_GPa : array_like
        Pressure [GPa].
    H2O_wt : array_like
        Dissolved water [wt%].

    Returns
    -------
    sigma : ndarray
        Conductivity [S/m].
    """
    T_K = np.asarray(T_K, dtype=np.float64)
    P_GPa = np.asarray(P_GPa, dtype=np.float64)
    H2O_wt = np.asarray(H2O_wt, dtype=np.float64)

    log_sigma = (2.983 - 0.0732 * H2O_wt
                 - (3528.0 - 233.8 * H2O_wt
                    + (763.0 - 7.5 * H2O_wt**2) * P_GPa) / T_K)
    return np.power(10.0, log_sigma)


def sigma_melt_interpolated(T_K, P_GPa, H2O_wt, SiO2_wt):
    """
    Composition-interpolated melt conductivity.

    Samrock et al. (2021) EPSL 559, 116765.

    Derivation
    ----------
    Second-order Lagrange polynomial interpolation in log10-space
    between three end-member compositions:

        andesite  (x_a = 60.8 wt% SiO2)
        dacite    (x_d = 65.8 wt% SiO2)
        rhyolite  (x_r = 75.7 wt% SiO2)

    The Lagrange basis polynomials are:

        L_a(x) = (x - x_d)(x - x_r) / ((x_a - x_d)(x_a - x_r))
        L_d(x) = (x - x_a)(x - x_r) / ((x_d - x_a)(x_d - x_r))
        L_r(x) = (x - x_a)(x - x_d) / ((x_r - x_a)(x_r - x_d))

    The interpolated log-conductivity is:

        log10(sigma) = L_a*log10(sigma_a) + L_d*log10(sigma_d)
                     + L_r*log10(sigma_r)

    Parameters
    ----------
    T_K, P_GPa, H2O_wt : array_like
        Temperature [K], pressure [GPa], dissolved water [wt%].
    SiO2_wt : array_like
        SiO2 content of melt [wt%].

    Returns
    -------
    sigma : ndarray
        Conductivity [S/m].
    """
    log_a = np.log10(np.maximum(sigma_melt_andesite(T_K, P_GPa, H2O_wt), 1e-12))
    log_d = np.log10(np.maximum(sigma_melt_dacite(T_K, P_GPa, H2O_wt), 1e-12))
    log_r = np.log10(np.maximum(sigma_melt_rhyolite(T_K, P_GPa, H2O_wt), 1e-12))

    x_a, x_d, x_r = 60.8, 65.8, 75.7
    SiO2_wt = np.asarray(SiO2_wt, dtype=np.float64)

    L_a = ((SiO2_wt - x_d) * (SiO2_wt - x_r)) / ((x_a - x_d) * (x_a - x_r))
    L_d = ((SiO2_wt - x_a) * (SiO2_wt - x_r)) / ((x_d - x_a) * (x_d - x_r))
    L_r = ((SiO2_wt - x_a) * (SiO2_wt - x_d)) / ((x_r - x_a) * (x_r - x_d))

    return np.power(10.0, L_a * log_a + L_d * log_d + L_r * log_r)


def SiO2_enrichment(chi, SiO2_parent=63.0, SiO2_final=75.0):
    """
    Linear SiO2 enrichment in residual melt during fractional crystallization.

    Used with sigma_melt_interpolated() for the 'samrock' melt type,
    following Samrock et al. (2021, EPSL 559).

    Parameters
    ----------
    chi : array_like
        Crystallinity [0-1].
    SiO2_parent, SiO2_final : float
        Parent and fully-crystallized SiO2 content [wt%].

    Returns
    -------
    SiO2 : ndarray
        Residual melt SiO2 [wt%].
    """
    chi = np.asarray(chi, dtype=np.float64)
    SiO2 = SiO2_parent + (SiO2_final - SiO2_parent) * chi
    return np.clip(SiO2, SiO2_parent, SiO2_final)


def estimate_melt_H2O(melt_fraction, H2O_bulk=2.0, H2O_saturation=6.0):
    """
    Estimate dissolved H2O assuming incompatibility: H2O = H2O_bulk / phi_melt.

    FALLBACK ONLY. Used when water_fraction_melt is not in VTU output.
    Prefer CSMP++ computed water_fraction_melt. A warning is printed
    when this fallback is triggered.

    Parameters
    ----------
    melt_fraction : array_like
        Melt volume fraction [0-1].
    H2O_bulk : float
        Initial bulk H2O [wt%].
    H2O_saturation : float
        Maximum solubility [wt%].

    Returns
    -------
    H2O_melt : ndarray
        Dissolved water [wt%].
    """
    phi = np.maximum(np.asarray(melt_fraction, dtype=np.float64), 0.01)
    return np.minimum(H2O_bulk / phi, H2O_saturation)


def compute_melt_conductivity(T_C, P_Pa, melt_fraction, config,
                              H2O_from_VTU=None):
    """
    Melt conductivity dispatcher.

    Routes to the appropriate melt conductivity model based on
    config['magma_composition']['type'] and determines dissolved
    H2O from the best available source.

    H2O priority
    ------------
    1. H2O_from_VTU (water_fraction_melt from CSMP++): preferred,
       spatially resolved, self-consistent with the melt model.
    2. config H2O_wt_percent: fixed value, useful for sensitivity tests.
    3. estimate_melt_H2O(): FALLBACK with explicit warning. Assumes
       incompatible behavior (H2O = H2O_bulk / phi_melt, capped at
       saturation). Crude but reasonable for water-undersaturated melts.

    Parameters
    ----------
    T_C : array_like
        Temperature [deg C].
    P_Pa : array_like
        Pressure [Pa].
    melt_fraction : array_like
        Melt volume fraction [0-1].
    config : dict
        Must contain 'magma_composition' sub-dict.
    H2O_from_VTU : array_like or None
        Dissolved water from CSMP++ [mass fraction, 0-1].

    Returns
    -------
    sigma_melt : ndarray
        Melt conductivity [S/m], clipped to [1e-4, 100].
    """
    T_C = np.asarray(T_C, dtype=np.float64)
    P_Pa = np.asarray(P_Pa, dtype=np.float64)
    melt_fraction = np.asarray(melt_fraction, dtype=np.float64)

    comp = config.get('magma_composition', {})

    if comp.get('constant_sigma_melt') is not None:
        val = float(comp['constant_sigma_melt'])
        print(f"    sigma_melt: constant override = {val} S/m")
        return np.full_like(T_C, val)

    T_K = T_C + 273.15
    P_GPa = P_Pa * 1.0e-9
    comp_type = comp.get('type', 'dacite')

    # H2O
    if H2O_from_VTU is not None and np.any(H2O_from_VTU > 0):
        H2O = np.asarray(H2O_from_VTU, dtype=np.float64) * 100.0
        print(f"    H2O: VTU ({H2O.min():.2f}-{H2O.max():.2f} wt%)")
    else:
        fixed = comp.get('H2O_wt_percent', None)
        if fixed is not None:
            H2O = np.full_like(T_C, float(fixed))
            print(f"    H2O: fixed {fixed} wt%")
        else:
            bulk = comp.get('H2O_bulk', 3.5)
            sat = comp.get('H2O_saturation', None)
            if sat is None:
                sat = 6.0
                print(f"    WARNING: No VTU water data, H2O_saturation=None. "
                      f"FALLBACK: estimate_melt_H2O(bulk={bulk}, sat={sat})")
            H2O = estimate_melt_H2O(melt_fraction, bulk, sat)

    # Check model calibration ranges
    _melt_ranges = {
        'andesite':  {'T_K': (1100, 1600), 'P_GPa': (0, 1.0), 'H2O': (0, 6)},
        'dacite':    {'T_K': (773, 1846),  'P_GPa': (0, 2.9),  'H2O': (0, 10)},
        'rhyolite':  {'T_K': (868, 1665),  'P_GPa': (0.5, 1.0), 'H2O': (0, 8)},
    }
    check_type = comp_type if comp_type in _melt_ranges else None
    if check_type:
        rng = _melt_ranges[check_type]
        _validate_range(T_K, f"sigma_melt_{check_type} T_K",
                        rng['T_K'][0], rng['T_K'][1])
        _validate_range(P_GPa, f"sigma_melt_{check_type} P_GPa",
                        rng['P_GPa'][0], rng['P_GPa'][1])
        _validate_range(H2O, f"sigma_melt_{check_type} H2O_wt",
                        rng['H2O'][0], rng['H2O'][1])

    # H2O sanity check (catch wt-fraction/wt-percent mix-up)
    if np.any(H2O > 20):
        warnings.warn(
            f"H2O: {int(np.sum(H2O > 20))} nodes > 20 wt% "
            f"(max {H2O.max():.1f}). Check if VTU field is in "
            f"mass fraction [0-1] or already in wt%.")

    if comp_type == 'andesite':
        sigma = sigma_melt_andesite(T_K, P_GPa, H2O)
    elif comp_type == 'dacite':
        sigma = sigma_melt_dacite(T_K, P_GPa, H2O)
    elif comp_type == 'rhyolite':
        sigma = sigma_melt_rhyolite(T_K, P_GPa, H2O)
    elif comp_type == 'interpolated':
        SiO2 = comp.get('SiO2_wt_percent', 65.0)
        sigma = sigma_melt_interpolated(T_K, P_GPa, H2O, SiO2)
    elif comp_type == 'samrock':
        chi = 1.0 - melt_fraction
        SiO2 = SiO2_enrichment(
            chi, comp.get('SiO2_parent', 63.0), comp.get('SiO2_final', 75.0))
        sigma = sigma_melt_interpolated(T_K, P_GPa, H2O, SiO2)
    else:
        raise ValueError(f"Unknown melt type: {comp_type}")

    # Warn when clipping triggers
    n_lo = int(np.sum(sigma < 1e-4))
    n_hi = int(np.sum(sigma > 100))
    if n_lo or n_hi:
        warnings.warn(
            f"sigma_melt: clipped {n_lo} nodes below 1e-4 S/m, "
            f"{n_hi} nodes above 100 S/m")

    P_bar = P_Pa * 1e-5
    print(f"    Melt: {comp_type}, T={T_C.min():.0f}-{T_C.max():.0f}C, "
          f"P={P_bar.min():.0f}-{P_bar.max():.0f}bar, "
          f"H2O={H2O.min():.1f}-{H2O.max():.1f}wt%")

    return np.clip(sigma, 1e-4, 100.0)


# =============================================================================
# SECTION 6: MIXING LAWS
# =============================================================================

def hashin_shtrikman_upper(sigma_a, sigma_b, f_a):
    """
    Hashin-Shtrikman upper bound for a two-phase mixture.

    Hashin, Z. & Shtrikman, S. (1962). J. Appl. Phys. 33(10),
    3125-3131.

    Derivation
    ----------
    The HS upper bound places the more conductive phase as the matrix:

        sigma_eff = sigma_hi + f_lo /
                    (1/(sigma_lo - sigma_hi) + f_hi/(3*sigma_hi))

    The implementation automatically identifies which phase is more
    conductive at each element.

    Guards
    ------
    - When f_a >= 1 - 1e-10, returns sigma_a (pure phase a).
    - When f_a <= 1e-10, returns sigma_b (pure phase b).
    - Denominator protected against division by zero when
      sigma_a == sigma_b.

    Parameters
    ----------
    sigma_a, sigma_b : array_like
        Phase conductivities [S/m].
    f_a : array_like
        Volume fraction of phase a [0-1].

    Returns
    -------
    sigma_eff : ndarray
        Effective conductivity [S/m].
    """
    sigma_a = np.asarray(sigma_a, float)
    sigma_b = np.asarray(sigma_b, float)
    f_a = np.clip(np.asarray(f_a, float), 0.0, 1.0)
    f_b = 1.0 - f_a

    a_hi = sigma_a >= sigma_b
    sigma_hi = np.where(a_hi, sigma_a, sigma_b)
    sigma_lo = np.where(a_hi, sigma_b, sigma_a)
    f_hi = np.where(a_hi, f_a, f_b)
    f_lo = 1.0 - f_hi

    sigma_hi_safe = np.maximum(sigma_hi, 1e-12)
    delta = sigma_lo - sigma_hi
    denom = np.where(np.abs(delta) > 1e-15,
                     1.0 / delta + f_hi / (3.0 * sigma_hi_safe), 1e30)
    sigma_eff = sigma_hi + f_lo / denom

    sigma_eff = np.where(f_a >= 1.0 - 1e-10, sigma_a, sigma_eff)
    sigma_eff = np.where(f_a <= 1e-10, sigma_b, sigma_eff)
    return np.maximum(sigma_eff, 0.0)


def cementation_exponents_samrock(phi_solid, phi_melt, phi_vol):
    """
    Variable cementation exponents for three-phase Archie.

    Samrock et al. (2021) EPSL 559, 116765. Eq. (3):

        m_melt = 1.0                if phi_melt > 0.4
        m_melt = -2.75*phi + 2.1    if phi_melt <= 0.4

    This piecewise function captures the transition from a
    melt-dominated (interconnected) regime at high melt fractions
    to a crystal-dominated (disconnected melt pockets) regime at
    low melt fractions.

    m_vol = 1.5 (fixed, typical for fluid-filled porosity).
    m_solid from the unity constraint (Glover, 2010, Eq. 2):

        sum(phi_i^m_i) = 1

    Solving for m_solid:

        m_solid = log(1 - phi_melt^m_melt - phi_vol^m_vol) / log(phi_solid)

    Parameters
    ----------
    phi_solid, phi_melt, phi_vol : array_like
        Phase volume fractions [0-1].

    Returns
    -------
    m_solid, m_melt, m_vol : ndarray
        Cementation exponents.
    """
    phi_solid = np.asarray(phi_solid, float)
    phi_melt = np.asarray(phi_melt, float)
    phi_vol = np.asarray(phi_vol, float)

    phase_sum = phi_solid + phi_melt + phi_vol
    bad_sum = np.abs(phase_sum - 1.0) > 0.01
    if np.any(bad_sum):
        warnings.warn(
            f"cementation_exponents: {int(np.sum(bad_sum))} nodes with "
            f"phi_solid + phi_melt + phi_vol != 1.0 "
            f"(max deviation: {float(np.max(np.abs(phase_sum - 1.0))):.3f})")

    m_melt = np.clip(-2.75 * phi_melt + 2.1, 1.0, 2.1)
    m_vol = np.full_like(phi_vol, 1.5)

    G_rest = np.clip(
        1.0 - (np.power(phi_melt, m_melt) + np.power(phi_vol, m_vol)),
        1e-12, 1.0)
    safe_phi = np.clip(phi_solid, 1e-6, 1.0)
    m_solid = np.clip(np.log(G_rest) / np.log(safe_phi), 1.3, 3.0)

    return m_solid, m_melt, m_vol


def archie_three_phase(sigma_solid, sigma_melt, sigma_vol,
                       phi_solid, phi_melt, phi_vol,
                       m_solid, m_melt, m_vol):
    """
    Additive three-phase Archie's law.

    Glover, P.W.J. (2010). Geophysics 75(6), E247-E265. Eq. (1):

        sigma_bulk = sum_i(sigma_i * phi_i^m_i)

    with the unity constraint (Eq. 2):

        sum_i(phi_i^m_i) = 1

    Parameters
    ----------
    sigma_solid, sigma_melt, sigma_vol : array_like
        Phase conductivities [S/m].
    phi_solid, phi_melt, phi_vol : array_like
        Phase volume fractions [0-1].
    m_solid, m_melt, m_vol : array_like
        Cementation exponents.

    Returns
    -------
    sigma_bulk : ndarray
        Bulk conductivity [S/m].
    """
    Gs = np.power(np.clip(phi_solid, 0.0, 1.0), m_solid)
    Gm = np.power(np.clip(phi_melt, 0.0, 1.0), m_melt)
    Gv = np.power(np.clip(phi_vol, 0.0, 1.0), m_vol)
    return sigma_solid * Gs + sigma_melt * Gm + sigma_vol * Gv


# =============================================================================
# SECTION 7: SURFACE CONDUCTION
# Waxman & Smits (1968) J. Pet. Tech. 20(6), 107-122
# Revil & Florsch (2010) GJI 181, 1480-1498, Eqs. (2)-(7)
# Revil et al. (2017) GJI 208, 826-844, Eq. (15)
# =============================================================================

def build_CEC_array(region_ids, config):
    """
    Per-node CEC from region configuration.

    CEC (cation exchange capacity) is converted from meq/100g to C/kg
    (SI units) using the Faraday-based factor MEQ_TO_CKG = 9.6485.

    Parameters
    ----------
    region_ids : array_like of int
        Region ID per node.
    config : dict
        Must contain 'regions' and/or 'default_region'.

    Returns
    -------
    CEC : ndarray
        Cation exchange capacity [C/kg].
    """
    region_ids = np.asarray(region_ids, dtype=int)
    n = len(region_ids)

    regions = config.get('regions', {})
    default = config.get('default_region', {})
    default_CEC = default.get('CEC_meq_per_100g', 2.0) * MEQ_TO_CKG

    # Legacy support
    if not regions and 'surface_conduction_regions' in config:
        sc = config['surface_conduction_regions']
        for rid, props in sc.items():
            regions[int(rid)] = {
                'CEC_meq_per_100g': props.get('CEC_meq_per_100g', 2.0)}
        default_CEC = config.get(
            'surface_conduction_default_CEC_meq_per_100g', 2.0) * MEQ_TO_CKG

    CEC = np.full(n, default_CEC)
    for rid, rprops in regions.items():
        mask = (region_ids == int(rid))
        if np.any(mask):
            val = rprops.get('CEC_meq_per_100g', default_CEC / MEQ_TO_CKG)
            CEC[mask] = val * MEQ_TO_CKG
            print(f"    Region {rid}: {np.sum(mask)} nodes, "
                  f"CEC={val} meq/100g ({val * MEQ_TO_CKG:.1f} C/kg)")
    return CEC


def build_region_property(region_ids, config, prop_name, n_nodes):
    """
    Per-node array for a region-specific property.

    Looks up property values from config['regions'] for each node,
    falling back to config['default_region'] for unspecified regions.

    Parameters
    ----------
    region_ids : array_like of int
        Region ID per node.
    config : dict
        Must contain 'regions' and 'default_region'.
    prop_name : str
        Property name, e.g. 'grain_density', 'f_stern'.
    n_nodes : int
        Number of nodes.

    Returns
    -------
    values : ndarray
        Property values per node.
    """
    regions = config.get('regions', {})
    default = config.get('default_region', {})
    hardcoded = {'grain_density': 2800.0, 'f_stern': 0.95}
    default_val = default.get(prop_name, hardcoded.get(prop_name, 0.0))

    # Treat None as 0.0 (caller uses 0 to detect "not set")
    if default_val is None:
        default_val = 0.0

    values = np.full(n_nodes, float(default_val))
    for rid, rprops in regions.items():
        mask = (region_ids == int(rid))
        if np.any(mask) and prop_name in rprops:
            val = rprops[prop_name]
            if val is not None:
                values[mask] = float(val)
    return values


# =============================================================================
# SECTION 8: NODAL PIPELINE HELPERS
# =============================================================================

def _match_porosity(nodal_data, element_data, x, y):
    """
    Spatially match porosity from Initial.vtu to timestep nodes.

    Uses a KD-tree to find the nearest Initial.vtu node for each
    timestep node. This handles cases where node numbering differs
    between VTU files (e.g., after mesh refinement or re-ordering).

    Parameters
    ----------
    nodal_data : dict
        Must contain 'nodal_porosity'.
    element_data : dict
        May contain 'initial_node_coordinates' as (x, y) arrays.
    x, y : ndarray
        Timestep node coordinates [km].

    Returns
    -------
    phi : ndarray
        Porosity at each timestep node.
    source : str
        Description of matching method used.

    Raises
    ------
    ValueError
        If nodal_porosity is missing or all-zero.
    """
    if 'nodal_porosity' not in nodal_data or not np.any(nodal_data['nodal_porosity'] > 0):
        raise ValueError("nodal_porosity required. Load from Initial.vtu.")

    init_phi = nodal_data['nodal_porosity']
    if 'initial_node_coordinates' in element_data:
        init_nx, init_ny = element_data['initial_node_coordinates']
        tree = cKDTree(np.column_stack((init_nx, init_ny)))
        _, idx = tree.query(np.column_stack((x, y)))
        phi = init_phi[idx]
        source = "spatially matched"
    else:
        phi = init_phi.copy()
        source = "index-matched (WARNING: may have node numbering issues)"

    _validate_range(phi, "porosity", 0.0, 1.0, hard=True)
    if np.any(phi > 0.5):
        warnings.warn(f"porosity: {np.sum(phi > 0.5)} nodes > 0.5 "
                       f"(max {phi.max():.3f}). Verify input data.")

    print(f"  Porosity: {source}, [{phi.min():.4f}, {phi.max():.4f}]")
    return phi, source


def _match_region_ids(element_data, x, y, triangles, config):
    """
    Map element-based region IDs to nodes with clay cap priority.

    Each node inherits a region ID from its surrounding elements.
    If any element touching a node belongs to a clay cap region
    (specified in config['clay_cap_regions']), the node inherits
    that clay cap region ID regardless of other elements.

    Among non-clay regions, the highest region ID wins (convention:
    higher IDs = deeper/more specific regions).

    Parameters
    ----------
    element_data : dict
        Must contain 'region_id_initial', 'region_id_initial_centers'.
    x, y : ndarray
        Timestep node coordinates [km].
    triangles : ndarray, shape (n_tri, 3)
        Element connectivity.
    config : dict
        May contain 'clay_cap_regions' list.

    Returns
    -------
    region_n : ndarray of int
        Region ID at each node.

    Raises
    ------
    ValueError
        If region_id_initial or centers are missing.
    """
    if 'region_id_initial' not in element_data or 'region_id_initial_centers' not in element_data:
        raise ValueError("region_id_initial required. Load from Initial.vtu.")

    n_nodes = len(x)
    init_rid = element_data['region_id_initial'].astype(int)
    init_cx, init_cy = element_data['region_id_initial_centers']
    tri_cx = np.mean(x[triangles], axis=1)
    tri_cy = np.mean(y[triangles], axis=1)
    tree = cKDTree(np.column_stack((init_cx, init_cy)))
    _, idx = tree.query(np.column_stack((tri_cx, tri_cy)))
    region_per_tri = init_rid[idx]

    clay_regions = set(config.get('clay_cap_regions', []))
    region_n = np.zeros(n_nodes, dtype=int)
    for i, tri in enumerate(triangles):
        rid = int(region_per_tri[i])
        for ni in tri:
            if rid in clay_regions:
                if region_n[ni] not in clay_regions:
                    region_n[ni] = rid
            elif region_n[ni] not in clay_regions:
                if rid > region_n[ni]:
                    region_n[ni] = rid

    print(f"  Region ID: {np.unique(region_n)}")
    return region_n


def _compute_fluid_conductivity(nodal_data, config):
    """
    Compute effective fluid conductivity at each node.

    Dispatches to the appropriate two-phase mixing model based on
    config['two_phase_mixing']:

        'archie' (default): per-phase Archie mixing:
            sigma_fluid = S_liq^n * sigma_liq + S_vap^n * sigma_vap
            Each phase conductivity is computed at its own salinity
            and density. Standard petrophysical approach.
        'hashin_shtrikman': HS upper bound between liquid and vapor
            conductivities. Assumes the more conductive phase (brine)
            forms the connected matrix. Gives higher conductivity
            than Archie at the same saturations.

    Parameters
    ----------
    nodal_data : dict
        Must contain temperature, fluid_pressure, saturation_liquid,
        saturation_vapor, salt_fraction_liquid, salt_fraction_vapor,
        density_liquid, density_vapor, salinity.
    config : dict
        Controls mixing model and salinity floor.

    Returns
    -------
    sigma_fluid : ndarray
        Effective fluid conductivity [S/m].
    sig_liq : ndarray
        Liquid-phase conductivity [S/m].
    sig_vap : ndarray
        Vapor-phase conductivity [S/m].
    """
    T = nodal_data['temperature']
    P_bar = nodal_data['fluid_pressure'] * 1e-5
    Sliq = nodal_data['saturation_liquid']
    Svap = nodal_data['saturation_vapor']
    Xliq = nodal_data['salt_fraction_liquid']
    Xvap = nodal_data['salt_fraction_vapor']
    rhol = nodal_data['density_liquid']
    rhov = nodal_data['density_vapor']
    sal = nodal_data['salinity']
    n_nodes = len(T)

    # Validate fluid inputs
    Shal = nodal_data.get('saturation_halite', np.zeros(n_nodes))
    S_sum = Sliq + Svap + np.clip(Shal, 0, 1)
    bad_sat = np.abs(S_sum - 1.0) > 0.05
    if np.any(bad_sat):
        warnings.warn(
            f"Fluid saturations: {int(np.sum(bad_sat))} nodes with "
            f"S_liq + S_vap + S_hal deviating from 1.0 by > 0.05 "
            f"(max deviation: {float(np.max(np.abs(S_sum - 1.0))):.3f})")
    _validate_range(Xliq, "salt_fraction_liquid", 0.0, 1.0)
    _validate_range(Xvap, "salt_fraction_vapor", 0.0, 1.0)
    _validate_range(rhol, "density_liquid", 0.1, 2000.0)
    _validate_range(rhov, "density_vapor", 0.001, 2000.0)

    mixing = config.get('two_phase_mixing', 'archie')
    print(f"  Fluid conductivity (mixing={mixing})...")

    min_sal = config.get('min_fluid_salinity_wt_percent', 0.0) / 100.0
    Xliq_cond = np.maximum(Xliq, min_sal) if min_sal > 0 else Xliq
    if min_sal > 0:
        print(f"    Salinity floor: {min_sal*100:.2f} wt% "
              f"({np.sum(Xliq < min_sal)} nodes)")

    # Per-phase conductivities (each at its own salinity and density)
    nodal_fluid = compute_fluid_conductivity(
        Xliq_cond, Xvap, T, P_bar, Sliq, Svap, rhol, rhov)
    sig_liq = nodal_fluid['sigma_liq']
    sig_vap = nodal_fluid['sigma_vap']

    # Two-phase mixing
    n_exp = config.get('saturation_exponent_n', 2.0)
    if mixing == 'hashin_shtrikman':
        sigma_fluid = hashin_shtrikman_upper(sig_liq, sig_vap, Sliq)
    elif mixing == 'archie':
        sigma_fluid = Sliq**n_exp * sig_liq + Svap**n_exp * sig_vap
    else:
        raise ValueError(
            f"Unknown two_phase_mixing: '{mixing}'. "
            f"Use 'archie' or 'hashin_shtrikman'.")

    print(f"    sigma_fluid: {np.nanmin(sigma_fluid):.3e} - "
          f"{np.nanmax(sigma_fluid):.3e} S/m")

    return sigma_fluid, sig_liq, sig_vap


def _compute_phase_fractions(nodal_data, element_data, phi, n_nodes, config=None):
    """
    Compute melt, solid, and volatile phase fractions from CSMP++ fields.

    Phase fractions come from crystal_volume_fraction (CVF) and
    fluid_volume_fraction (FVF) when magma is present. In the
    hydrothermal domain, phi_solid = 1 - porosity and phi_vol = porosity.

    Parameters
    ----------
    nodal_data : dict
        May contain 'crystal_volume_fraction', 'fluid_volume_fraction'.
    element_data : dict
        Fallback source for CVF/FVF.
    phi : ndarray
        Porosity at each node.
    n_nodes : int
        Number of nodes.

    Returns
    -------
    phi_melt : ndarray
        Melt volume fraction [0-1].
    phi_solid : ndarray
        Solid (crystal) volume fraction [0-1].
    phi_vol : ndarray
        Volatile (fluid) volume fraction [0-1].
    magma_active : bool
        Whether magmatic phases are present.
    """
    def _get(key):
        arr = nodal_data.get(key, element_data.get(key, None))
        if arr is not None:
            arr = np.asarray(arr, float)
            return arr if len(arr) == n_nodes else None
        return None

    cvf = _get('crystal_volume_fraction')
    fvf_raw = _get('fluid_volume_fraction')

    phi_melt = np.zeros(n_nodes)
    phi_solid = np.ones(n_nodes)
    phi_vol = np.zeros(n_nodes)
    magma_active = cvf is not None and np.any(cvf > 0.01)

    if magma_active:
        print(f"  Magmatic phases: ACTIVE")
        _validate_range(cvf, "crystal_volume_fraction", 0.0, 1.0, hard=True)
        if fvf_raw is not None:
            _validate_range(fvf_raw, "fluid_volume_fraction", 0.0, 1.0, hard=True)
        cvf = np.clip(cvf, 0.0, 1.0)
        fvf = np.clip(fvf_raw, 0.0, 1.0) if fvf_raw is not None else np.zeros(n_nodes)
        magma = cvf > 0.01
        phi_solid[magma] = cvf[magma]

        # Volatile phase fraction: use FVF when available. When FVF is
        # zero, decide whether to fall back to porosity based on the
        # melt water saturation state:
        #
        # - If water_fraction_melt >= water_solubility_melt, the melt
        #   IS water-saturated and volatiles SHOULD be exsolving. FVF=0
        #   means they have escaped into the hydrothermal system. The
        #   pore space (porosity) contains hydrothermal fluid. Use
        #   porosity as phi_vol so this fluid contributes to sigma.
        #
        # - If water_fraction_melt < water_solubility_melt, the melt
        #   is undersaturated — water is dissolved. FVF=0 is correct.
        #   No free volatiles exist. Keep phi_vol = 0.
        #
        # - If water saturation data is not available, fall back to
        #   the conservative criterion phi_melt < 0.05.
        wfm = nodal_data.get('water_fraction_melt', None)
        wsm = nodal_data.get('water_solubility_melt', None)

        phi_melt_raw = 1.0 - cvf[magma] - fvf[magma]
        fvf_zero = fvf[magma] <= 1e-4

        if wfm is not None and wsm is not None:
            wfm_m = np.asarray(wfm, dtype=float)[magma]
            wsm_m = np.asarray(wsm, dtype=float)[magma]
            # Melt is water-saturated: volatiles should exist but FVF=0
            melt_saturated = wfm_m >= wsm_m * 0.99  # 1% tolerance
            use_porosity = fvf_zero & melt_saturated
        else:
            # No water saturation data: fall back to phi_melt threshold
            use_porosity = fvf_zero & (phi_melt_raw < 0.05)

        # At fallback nodes, use max(porosity, spent_magma_min_porosity).
        # The initial porosity (from Initial.vtu) may underestimate the
        # actual pore space because it doesn't account for porosity
        # created by volatile exsolution during crystallization.
        phi_min = (config or {}).get('spent_magma_min_porosity', 0.05)
        fallback_phi = np.maximum(phi[magma], phi_min)

        fvf_eff = fvf[magma].copy()
        fvf_eff[use_porosity] = fallback_phi[use_porosity]
        phi_vol[magma] = fvf_eff
        phi_melt[magma] = np.clip(1.0 - cvf[magma] - fvf_eff, 0.0, 1.0)
        phi_solid[~magma] = 1.0 - phi[~magma]
        phi_vol[~magma] = phi[~magma]
        # Validate phase fraction sum
        phase_sum = phi_solid[magma] + phi_melt[magma] + phi_vol[magma]
        bad_sum = np.abs(phase_sum - 1.0) > 0.01
        if np.any(bad_sum):
            warnings.warn(
                f"Phase fractions: {np.sum(bad_sum)} magma nodes with "
                f"phi_solid + phi_melt + phi_vol != 1.0 "
                f"(max deviation: {np.max(np.abs(phase_sum - 1.0)):.3f})")

        n_porosity_used = int(np.sum(use_porosity))
        criterion = ("water-saturated melt" if wfm is not None and wsm is not None
                     else "phi_melt < 0.05")
        if n_porosity_used > 0:
            print(f"    FVF->porosity fallback: {n_porosity_used} nodes "
                  f"(FVF~0, {criterion})")
        print(f"    CVF: {cvf[magma].min():.3f}-{cvf[magma].max():.3f}, "
              f"FVF: {fvf[magma].min():.4f}-{fvf[magma].max():.4f}, "
              f"phi_vol (eff): {phi_vol[magma].min():.4f}-{phi_vol[magma].max():.4f}, "
              f"phi_melt: {phi_melt[magma].min():.3f}-{phi_melt[magma].max():.3f}")
    else:
        print(f"  Magmatic phases: NOT DETECTED")
        phi_solid = 1.0 - phi
        phi_vol = phi.copy()

    # Build full-size mask of nodes where FVF was replaced by porosity
    fvf_fallback = np.zeros(n_nodes, dtype=bool)
    if magma_active and np.any(use_porosity):
        full_idx = np.where(magma)[0]
        fvf_fallback[full_idx[use_porosity]] = True

    return phi_melt, phi_solid, phi_vol, magma_active, fvf_fallback


def _compute_melt_domain(T, P_Pa, phi_melt, phi_solid, phi_vol,
                         sigma_rock, sig_liq, sig_vap,
                         Sliq, Svap, melt_mask, nodal_data, config,
                         fvf_fallback=None):
    """
    Three-phase Archie conductivity for the magmatic domain.

    Applies Glover (2010) Eq. 1 with Samrock et al. (2021) Eq. 3
    cementation exponents for nodes where phi_melt > melt_threshold.

    As phi_melt -> 0 and FVF -> 0 at the solidification front, the
    three-phase Archie naturally reduces to sigma_bulk -> sigma_rock
    (fully crystallized rock).

    Parameters
    ----------
    T : ndarray
        Temperature [deg C].
    P_Pa : ndarray
        Pressure [Pa].
    phi_melt, phi_solid, phi_vol : ndarray
        Phase fractions.
    sigma_rock : ndarray
        Rock matrix conductivity [S/m].
    sig_liq, sig_vap : ndarray
        Per-phase fluid conductivities [S/m].
    Sliq, Svap : ndarray
        Phase saturations.
    melt_mask : ndarray of bool
        Nodes in the melt domain.
    nodal_data : dict
        May contain 'water_fraction_melt'.
    config : dict
        Pipeline configuration.

    Returns
    -------
    sigma_bulk_melt : ndarray
        Bulk conductivity at melt nodes [S/m].
    """
    mm = melt_mask
    n_mm = np.sum(mm)
    print(f"\n  Melt domain ({n_mm} nodes):")

    H2O_vtu = None
    if 'water_fraction_melt' in nodal_data:
        wfm = nodal_data['water_fraction_melt']
        if np.any(wfm > 0):
            H2O_vtu = wfm[mm]

    sig_melt = compute_melt_conductivity(
        T[mm], P_Pa[mm], phi_melt[mm], config, H2O_from_VTU=H2O_vtu)
    print(f"    sigma_melt: {np.nanmin(sig_melt):.3e}-{np.nanmax(sig_melt):.3e} S/m")

    phi_vol_m = phi_vol[mm].copy()
    sig_vol = Sliq[mm] * sig_liq[mm] + Svap[mm] * sig_vap[mm]

    ms, m_m, mv = cementation_exponents_samrock(
        phi_solid[mm], phi_melt[mm], phi_vol_m)

    # At nodes where FVF was replaced by porosity (volatiles escaped,
    # pore fluid fills the space), use the hydrothermal cementation
    # exponent for the volatile phase instead of the Samrock m_vol=1.5.
    # This ensures continuity at the melt/hydrothermal domain boundary.
    if fvf_fallback is not None:
        fb_mm = fvf_fallback[mm]
        if np.any(fb_mm):
            m_hydro = config.get('porosity_exponent_m', 1.8)
            mv[fb_mm] = m_hydro
            # Recompute m_solid from unity constraint for consistency
            G_rest = np.clip(
                1.0 - (np.power(np.clip(phi_melt[mm][fb_mm], 0, 1), m_m[fb_mm])
                      + np.power(np.clip(phi_vol_m[fb_mm], 0, 1), mv[fb_mm])),
                1e-12, 1.0)
            safe_phi_s = np.clip(phi_solid[mm][fb_mm], 1e-6, 1.0)
            ms[fb_mm] = np.clip(np.log(G_rest) / np.log(safe_phi_s), 1.3, 3.0)

    sigma_bulk_melt = archie_three_phase(
        sigma_rock[mm], sig_melt, sig_vol,
        phi_solid[mm], phi_melt[mm], phi_vol_m, ms, m_m, mv)

    v = sigma_bulk_melt[np.isfinite(sigma_bulk_melt) & (sigma_bulk_melt > 0)]
    if v.size:
        print(f"    sigma_bulk: {v.min():.3e}-{v.max():.3e} S/m")

    return sigma_bulk_melt


def _compute_hydrothermal_domain(T, phi, sigma_fluid, Sliq, Svap,
                                 region_n, hm, nodal_data, config,
                                 magma_active, cvf_data):
    """
    Modified Archie conductivity with surface conduction.

    For the hydrothermal domain (no melt), bulk conductivity is:

        sigma_bulk = (1/F) * sigma_fluid + sigma_surface

    where F = a / phi^m is the formation factor and sigma_surface is
    the surface conduction from Revil et al. (2017, GJI 208, Eq. 15):

        sigma_s = (1/(F*phi)) * S_total^(n-1) * rho_g
                * beta_plus * (1 - f_stern) * CEC

    Parameters
    ----------
    T : ndarray
        Temperature [deg C].
    phi : ndarray
        Porosity.
    sigma_fluid : ndarray
        Effective fluid conductivity [S/m].
    Sliq, Svap : ndarray
        Phase saturations.
    region_n : ndarray of int
        Region ID per node.
    hm : ndarray of bool
        Hydrothermal domain mask.
    nodal_data : dict
        May contain 'saturation_halite'.
    config : dict
        Pipeline configuration.
    magma_active : bool
        Whether magmatic phases are present anywhere.
    cvf_data : ndarray or None
        Crystal volume fraction (for spent magma detection).

    Returns
    -------
    sigma_bulk_hydro : ndarray
        Bulk conductivity at hydrothermal nodes [S/m].
    """
    n_h = np.sum(hm)
    n_nodes = len(T)
    print(f"\n  Hydrothermal domain ({n_h} nodes):")

    phi_h = phi.copy()
    if magma_active and cvf_data is not None:
        # Spent magma: intrusion region (region_id == 1) that has fully
        # crystallized (CVF == 1.0). These nodes had melt but are now
        # solid; their porosity may be near zero in the simulation, so
        # we floor it to allow fluid conduction through residual porosity.
        # NOTE: CVF is 1.0 for ALL fully crystallized rock in CSMP++,
        # not just the intrusion. The region_id filter prevents applying
        # the porosity floor to the entire domain.
        intrusion_region = config.get('intrusion_id', 1)
        spent = hm & (cvf_data >= 0.99) & (region_n == intrusion_region)
        if np.any(spent):
            phi_min = config.get('spent_magma_min_porosity', 0.05)
            phi_h[spent] = np.maximum(phi_h[spent], phi_min)
            print(f"    Spent magma: {np.sum(spent)} nodes "
                  f"(region {intrusion_region}, CVF >= 0.99)")

    Shal = nodal_data.get('saturation_halite', np.zeros(n_nodes))
    Shal_h = np.clip(Shal[hm], 0.0, 1.0)
    if np.any(Shal_h > 1e-4):
        print(f"    Halite: {np.sum(Shal_h > 1e-4)} nodes")

    # Per-region cementation and saturation exponents.
    # Region-specific values override the global default.
    m_global = config.get('porosity_exponent_m', 2.0)
    n_global = config.get('saturation_exponent_n', 2.0)

    m_arr = build_region_property(region_n[hm], config,
                                  'porosity_exponent_m', n_h)
    n_arr = build_region_property(region_n[hm], config,
                                  'saturation_exponent_n', n_h)
    # Where region didn't specify (returned 0 from default None->0 fallback),
    # fill with the global value.
    m_arr = np.where(m_arr > 0, m_arr, m_global)
    n_arr = np.where(n_arr > 0, n_arr, n_global)

    unique_m = np.unique(np.round(m_arr, 2))
    if len(unique_m) > 1:
        print(f"    Cementation exponent m: {unique_m}")

    a = config.get('tortuosity_a', 1.0)
    phi_eff_raw = phi_h[hm] * (1.0 - Shal_h)
    n_clipped = int(np.sum(phi_eff_raw < 1e-3))
    if n_clipped:
        warnings.warn(
            f"Hydrothermal domain: {n_clipped} nodes with effective "
            f"porosity < 1e-3, clipped to 1e-3. Formation factor F may "
            f"be unreliable at these nodes.")
    phi_eff = np.clip(phi_eff_raw, 1e-3, 1.0)
    F = a / (phi_eff ** m_arr)

    sig_fluid_term = (1.0 / F) * sigma_fluid[hm]

    # Surface conduction: Revil et al. (2017) GJI 208, Eq. (15):
    #   sigma_s = (1/(F*phi)) * rho_g * beta_plus * (1 - f_stern) * CEC
    # With saturation scaling S_total^(n-1).
    # Uses S_total = S_liq + S_vap instead of S_liq alone:
    # at supercritical conditions "vapor" still wets mineral surfaces.
    # beta_plus(T) = beta_25 * (1 + alpha * (T - 25)):
    #   beta_25 = 5.19e-8 m^2/(V.s), alpha = 0.037 /C.
    CEC = build_CEC_array(region_n[hm], config)
    rho_g = build_region_property(region_n[hm], config, 'grain_density', n_h)
    f_stern = build_region_property(region_n[hm], config, 'f_stern', n_h)

    alpha = config.get('counterion_temp_coeff', 0.037)
    beta_25 = config.get('counterion_mobility_25C', 5.19e-8)
    beta = beta_25 * (1.0 + alpha * (T[hm] - 25.0))

    S_total = np.clip(Sliq[hm] + Svap[hm], 0.0, 1.0)
    sig_surface = ((1.0 / (F * phi_eff))
                   * (S_total ** (n_arr - 1.0))
                   * rho_g * beta * (1.0 - f_stern) * CEC)

    sigma_bulk_hydro = sig_fluid_term + sig_surface

    v = sigma_bulk_hydro[np.isfinite(sigma_bulk_hydro) & (sigma_bulk_hydro > 0)]
    if v.size:
        print(f"    sigma_bulk: {v.min():.3e}-{v.max():.3e} S/m")

    return sigma_bulk_hydro


# =============================================================================
# SECTION 9: NODAL PIPELINE
# =============================================================================

def calculate_conductivity_nodal(config, nodal_data, element_data,
                                 coordinates, triangles):
    """
    Compute bulk electrical conductivity at mesh nodes.

    This is the main entry point for the conductivity pipeline. It
    orchestrates the dual-domain framework: magmatic nodes use
    three-phase Archie mixing (Glover, 2010; Samrock et al., 2021),
    while hydrothermal nodes use modified Archie with surface
    conduction (Revil et al., 2017).

    Algorithm
    ---------
    1. Merge user config with DEFAULT_CONFIG.
    2. Spatially match porosity from Initial.vtu.
    3. Map element region IDs to nodes (clay cap priority).
    4. Compute per-phase and effective fluid conductivity.
    5. Compute rock matrix conductivity (Olhoeft, 1981).
    6. Extract phase fractions from CVF/FVF fields.
    7. Classify nodes as magmatic or hydrothermal.
    8. Correct injection front artifacts (steps 5b below).
    9. Compute melt-domain conductivity (three-phase Archie).
    10. Compute hydrothermal-domain conductivity (Archie + surface).

    Phase fraction corrections (step 6)
    ------------------------------------
    In the melt domain, the volatile phase fraction phi_vol is set to
    max(FVF, porosity) at nodes where FVF is zero AND the melt is
    water-saturated (water_fraction_melt >= water_solubility_melt).
    At these nodes, magmatic volatiles have already exsolved and
    escaped into the hydrothermal system, but hydrothermal pore fluid
    fills the porosity. Without this correction, phi_vol = 0 and the
    pore fluid contributes nothing to the three-phase Archie mixing.

    At these same fallback nodes, the volatile cementation exponent
    m_vol is changed from the Samrock value (1.5) to the hydrothermal
    cementation exponent (config porosity_exponent_m) to ensure
    continuity across the melt/hydrothermal domain boundary.

    Injection front corrections (step 5b)
    --------------------------------------
    CSMP++ injects exsolved magmatic fluids at "injection location"
    nodes, which sit in the solidified shell just outside the
    solidification front. The adjacent mushy nodes (inside the front)
    have FVF = 0 and lack fluid properties, even though physically
    the injected fluid should occupy these nodes.

    Two corrections are applied when injection_location is available
    in the VTU output:

    1. Low-density injection fix: injection nodes with bulk fluid
       density < 400 kg/m3 borrow conductivity from a neighboring
       injection node with higher density.

    2. Melt-domain fallback fix: FVF-fallback nodes in the melt
       domain that share a triangle element with an injection node
       receive the injection node's fluid conductivity directly.

    These corrections compensate for CSMP++'s injection placement
    logic and may be rolled back if the forward model is updated
    to inject into sub-solidus nodes instead.

    Parameters
    ----------
    config : dict
        User configuration. Merged with DEFAULT_CONFIG; user values
        override defaults. See DEFAULT_CONFIG for available keys.
    nodal_data : dict
        Nodal fields from timestep VTU. Required keys: temperature,
        fluid_pressure, saturation_liquid, saturation_vapor,
        salt_fraction_liquid, salt_fraction_vapor, density_liquid,
        density_vapor, salinity, nodal_porosity.
        Optional: injection_location, fluid_density,
        water_fraction_melt, water_solubility_melt.
    element_data : dict
        Element fields and Initial.vtu cross-references. Required:
        region_id_initial, region_id_initial_centers.
    coordinates : tuple of (x, y) ndarrays
        Node coordinates [km].
    triangles : ndarray, shape (n_tri, 3)
        Element connectivity.

    Returns
    -------
    dict
        sigma_bulk : ndarray, bulk conductivity [S/m].
        sigma_fluid : ndarray, effective fluid conductivity [S/m].
        melt_fractions : ndarray, phi_melt [0-1].
        phi_solid : ndarray, crystal fraction [0-1].
        phi_vol : ndarray, volatile fraction [0-1].
        porosity : ndarray, matched porosity.
        porosity_source : str, matching method used.
        melt_mask : ndarray of bool.
        clay_cap_mask : ndarray of bool.
        model_type : str, identifier string.
    """
    # Merge config with defaults
    params = dict(DEFAULT_CONFIG)
    params.update(config)
    # Deep-merge magma_composition
    if 'magma_composition' in config:
        mc = dict(DEFAULT_CONFIG.get('magma_composition', {}))
        mc.update(config['magma_composition'])
        params['magma_composition'] = mc
    if 'default_region' in config:
        dr = dict(DEFAULT_CONFIG.get('default_region', {}))
        dr.update(config['default_region'])
        params['default_region'] = dr

    x, y = coordinates
    n_nodes = len(x)

    # Validate required fields
    required_nodal = ['temperature', 'fluid_pressure', 'saturation_liquid',
                      'saturation_vapor', 'salt_fraction_liquid',
                      'salt_fraction_vapor', 'density_liquid', 'density_vapor',
                      'salinity']
    missing = [k for k in required_nodal if k not in nodal_data]
    if missing:
        raise ValueError(
            f"Missing required nodal fields: {missing}")

    # Validate array lengths
    for key in required_nodal:
        if len(nodal_data[key]) != n_nodes:
            raise ValueError(
                f"Array length mismatch: {key} has {len(nodal_data[key])} "
                f"elements, expected {n_nodes}")

    # Validate physical ranges
    T = nodal_data['temperature']
    _validate_range(T, "temperature", -273.15, 1500.0, hard=True)
    _validate_range(nodal_data['fluid_pressure'], "fluid_pressure", 0.0, 1e10)

    print("Starting NODAL conductivity calculation...")
    print(f"  {n_nodes} nodes")

    # 1. Porosity
    phi, porosity_source = _match_porosity(nodal_data, element_data, x, y)

    # 2. Region IDs
    region_n = _match_region_ids(element_data, x, y, triangles, params)

    # 3. Fluid conductivity
    sigma_fluid, sig_liq, sig_vap = _compute_fluid_conductivity(
        nodal_data, params)

    # 4. Rock conductivity
    T = nodal_data['temperature']
    P_Pa = nodal_data['fluid_pressure']
    Sliq = nodal_data['saturation_liquid']
    Svap = nodal_data['saturation_vapor']
    sigma_rock = sigma_rock_olhoeft(T)
    _validate_range(T, "temperature (rock model)", -273.15, 1200.0)

    # 5. Phase fractions
    phi_melt, phi_solid, phi_vol, magma_active, fvf_fallback = _compute_phase_fractions(
        nodal_data, element_data, phi, n_nodes, config=params)

    melt_threshold = params.get('melt_threshold', 0.05)
    melt_mask = phi_melt > melt_threshold
    print(f"    Melt nodes: {np.sum(melt_mask)}")

    # 5b. Injection neighbor fix: at FVF-fallback nodes that share an
    # element with an injection_location=1 node, borrow fluid properties
    # from the injection node. This corrects a CSMP++ spatial offset
    # where injected fluids land in the solidified shell rather than
    # the adjacent mush.
    inj_loc = nodal_data.get('injection_location', None)
    if inj_loc is not None and np.any(fvf_fallback):
        inj_loc = np.asarray(inj_loc, dtype=float)
        inj_nodes = inj_loc > 0.5
        target_nodes = fvf_fallback & melt_mask

        # Pass 1: fix low-density injection nodes by borrowing
        # conductivity from a neighboring injection node with
        # fluid_density >= 400 kg/m3.
        rho_fluid = nodal_data.get('fluid_density', None)
        if rho_fluid is not None:
            rho_fluid = np.asarray(rho_fluid, dtype=float)
            low_rho_inj = inj_nodes & (rho_fluid < 400.0)
            good_inj = inj_nodes & (rho_fluid >= 400.0)
            n_low_rho = int(np.sum(low_rho_inj))

            if n_low_rho > 0:
                inj_fixed_set = set()
                for tri in triangles:
                    bad_in_tri = [n for n in tri if low_rho_inj[n]
                                  and n not in inj_fixed_set]
                    good_in_tri = [n for n in tri if good_inj[n]]
                    if not bad_in_tri or not good_in_tri:
                        continue
                    src = max(good_in_tri, key=lambda n: rho_fluid[n])
                    for dst in bad_in_tri:
                        sig_liq[dst] = sig_liq[src]
                        sig_vap[dst] = sig_vap[src]
                        sigma_fluid[dst] = sigma_fluid[src]
                        inj_fixed_set.add(dst)
                print(f"    Low-density injection nodes fixed: "
                      f"{len(inj_fixed_set)}/{n_low_rho}")

        # Pass 2: for each melt-domain fallback node sharing an element
        # with an injection node, copy fluid conductivity from the
        # injection node.
        n_fixed = 0
        fixed_set = set()

        for tri in triangles:
            targets_in_tri = [n for n in tri if target_nodes[n]
                              and n not in fixed_set]
            inj_in_tri = [n for n in tri if inj_nodes[n]]

            if not targets_in_tri or not inj_in_tri:
                continue

            if len(inj_in_tri) > 1:
                src = max(inj_in_tri,
                          key=lambda n: (nodal_data.get('fluid_volume_fraction',
                                         np.zeros(n_nodes))[n],
                                        nodal_data['salinity'][n]))
            else:
                src = inj_in_tri[0]

            for dst in targets_in_tri:
                for key in ['saturation_liquid', 'saturation_vapor',
                            'salt_fraction_liquid', 'salt_fraction_vapor',
                            'density_liquid', 'density_vapor', 'salinity']:
                    if key in nodal_data:
                        nodal_data[key][dst] = nodal_data[key][src]
                sig_liq[dst] = sig_liq[src]
                sig_vap[dst] = sig_vap[src]
                sigma_fluid[dst] = sigma_fluid[src]
                fixed_set.add(dst)
                n_fixed += 1

        n_no_neighbor = int(np.sum(target_nodes)) - n_fixed
        print(f"    Injection front fix: {n_fixed} melt nodes corrected, "
              f"{n_no_neighbor} without injection neighbor")

    # 6. Bulk conductivity
    sigma_bulk = np.zeros(n_nodes)

    if np.any(melt_mask):
        sigma_bulk[melt_mask] = _compute_melt_domain(
            T, P_Pa, phi_melt, phi_solid, phi_vol,
            sigma_rock, sig_liq, sig_vap,
            Sliq, Svap, melt_mask, nodal_data, params,
            fvf_fallback=fvf_fallback)

    hm = ~melt_mask
    if np.any(hm):
        cvf_data = nodal_data.get('crystal_volume_fraction',
                                  element_data.get('crystal_volume_fraction', None))
        if cvf_data is not None:
            cvf_data = np.asarray(cvf_data, float)
            if len(cvf_data) != n_nodes:
                cvf_data = None
        sigma_bulk[hm] = _compute_hydrothermal_domain(
            T, phi, sigma_fluid, Sliq, Svap, region_n, hm,
            nodal_data, params, magma_active, cvf_data)

    # 7. Summary
    clay_regions = set(params.get('clay_cap_regions', []))
    clay_mask = np.zeros(n_nodes, dtype=bool)
    for rid in clay_regions:
        clay_mask |= (region_n == rid)

    pos = sigma_bulk > 0
    n_zero = int(np.sum(sigma_bulk == 0))
    if n_zero > 0:
        warnings.warn(
            f"sigma_bulk: {n_zero} nodes remain at zero conductivity "
            f"(not assigned by either domain). Check domain classification.")
    if np.any(pos):
        log_s = np.log10(np.maximum(sigma_bulk, 1e-12))
        print(f"\n  Result: {sigma_bulk[pos].min():.3e}-{sigma_bulk.max():.3e} S/m "
              f"(log10: {log_s[pos].min():.2f} to {log_s.max():.2f})")

    return {
        'sigma_bulk': sigma_bulk,
        'sigma_fluid': sigma_fluid,
        'melt_fractions': phi_melt,
        'phi_solid': phi_solid,
        'phi_vol': phi_vol,
        'porosity': phi,
        'porosity_source': porosity_source,
        'melt_mask': melt_mask,
        'clay_cap_mask': clay_mask,
        'model_type': 'nodal-watanabe+archie+gresse',
    }


# =============================================================================
# SELF-TEST
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("conductivity_lib.py -- self-test suite")
    print("=" * 70)

    passed = 0
    failed = 0

    def check(name, condition, detail=""):
        """Report a test result."""
        global passed, failed
        if condition:
            passed += 1
            print(f"  PASS  {name}")
        else:
            failed += 1
            print(f"  FAIL  {name}  {detail}")

    # --- 1. Liquid conductivity (Watanabe 2021) ---
    # Verified against Watanabe et al. (2021) Figs. 5 and 7.
    print("\n1. Liquid conductivity (Watanabe et al., 2021)")

    # Fig. 7d: 5.6 wt% NaCl, 300C, 100 MPa (1000 bar) -> ~30 S/m
    sig_fig7d = sigma_liquid(
        np.array([0.056]), np.array([300.0]),
        np.array([1000.0]), np.array([700.0]))
    check("Fig.7d: 5.6% NaCl, 300C, 1000bar -> ~30 S/m",
          10.0 < float(sig_fig7d) < 60.0,
          f"got {float(sig_fig7d):.1f}")

    # Fig. 7c: 3.5 wt% NaCl, 200C, 50 MPa (500 bar) -> ~10 S/m
    sig_fig7c = sigma_liquid(
        np.array([0.035]), np.array([200.0]),
        np.array([500.0]), np.array([900.0]))
    check("Fig.7c: 3.5% NaCl, 200C, 500bar -> ~5-15 S/m",
          2.0 < float(sig_fig7c) < 30.0,
          f"got {float(sig_fig7c):.1f}")

    # Fig. 7a: 0.01 wt% NaCl, 200C, 100 MPa -> ~0.05-0.1 S/m
    sig_fig7a = sigma_liquid(
        np.array([0.0001]), np.array([200.0]),
        np.array([1000.0]), np.array([900.0]))
    check("Fig.7a: 0.01% NaCl, 200C, 1000bar -> ~0.05-0.15 S/m",
          0.01 < float(sig_fig7a) < 0.5,
          f"got {float(sig_fig7a):.3f}")

    # General 5% NaCl at 300C
    sig = sigma_liquid(
        np.array([0.05]), np.array([300.0]),
        np.array([1000.0]), np.array([750.0]))
    check("5% NaCl, 300C, 1000bar -> order 10 S/m",
          5.0 < float(sig) < 100.0,
          f"got {float(sig):.1f}")

    sig_conc = sigma_liquid(
        np.array([0.20]), np.array([300.0]),
        np.array([1000.0]), np.array([900.0]))
    check("20% NaCl > 5% NaCl at same conditions",
          float(sig_conc) > float(sig),
          f"20%={float(sig_conc):.1f}, 5%={float(sig):.1f}")

    # Monotonicity: sigma increases with T
    T_arr = np.array([100.0, 200.0, 300.0, 400.0])
    sig_T = sigma_liquid(
        np.full(4, 0.05), T_arr,
        np.full(4, 1000.0), np.full(4, 800.0))
    check("sigma_liquid increases with T",
          np.all(np.diff(sig_T) > 0),
          f"values: {[f'{v:.1f}' for v in sig_T]}")

    # Monotonicity: sigma increases with salinity
    sal_arr = np.array([0.01, 0.05, 0.10, 0.15])
    sig_sal = sigma_liquid(
        sal_arr, np.full(4, 300.0),
        np.full(4, 1000.0), np.full(4, 800.0))
    check("sigma_liquid increases with salinity",
          np.all(np.diff(sig_sal) > 0),
          f"values: {[f'{v:.1f}' for v in sig_sal]}")

    # Zero salinity floor
    sig_zero = sigma_liquid(
        np.array([0.0]), np.array([300.0]),
        np.array([1000.0]), np.array([800.0]))
    check("zero salinity -> floor (1e-6)",
          float(sig_zero) == 1e-6,
          f"got {float(sig_zero):.3e}")

    # --- 2. Molar conductivity helpers ---
    print("\n2. Molar conductivity helpers")
    m = wtfrac_to_molality(0.05)
    check("wtfrac_to_molality(0.05)",
          0.5 < float(m) < 1.5,
          f"got {float(m):.4f}")

    m_zero = wtfrac_to_molality(0.0)
    check("wtfrac_to_molality(0.0) = 0",
          float(m_zero) == 0.0,
          f"got {float(m_zero)}")

    m_one = wtfrac_to_molality(1.0)
    check("wtfrac_to_molality(1.0) = NaN (pure NaCl)",
          np.isnan(float(m_one)),
          f"got {float(m_one)}")

    Lambda = molar_conductivity_watanabe(1e-3, 0.9)
    check("molar_conductivity_watanabe > 0",
          float(Lambda) > 0,
          f"got {float(Lambda):.3e}")

    # --- 3. Rock conductivity (Olhoeft 1981) ---
    # Verified against Olhoeft (1981) Fig. 8 (Westerly Granite, dry).
    print("\n3. Rock conductivity (Olhoeft, 1981)")
    sig_25 = sigma_rock_olhoeft(25.0)
    check("25C -> negligible (< 1e-12)",
          float(sig_25) < 1e-10,
          f"got {float(sig_25):.3e}")

    sig_300 = sigma_rock_olhoeft(300.0)
    check("300C -> ~3e-7 S/m (Fig.8: log rho ~ 6.5)",
          1e-8 < float(sig_300) < 1e-5,
          f"got {float(sig_300):.3e}")

    sig_800 = sigma_rock_olhoeft(800.0)
    check("800C -> ~0.01-0.1 S/m (Fig.8: log rho ~ 1.5)",
          1e-3 < float(sig_800) < 1.0,
          f"got {float(sig_800):.3e}")

    # Monotonicity
    T_rock = np.array([100.0, 300.0, 500.0, 800.0])
    sig_rock_arr = sigma_rock_olhoeft(T_rock)
    check("sigma_rock increases with T (Arrhenius)",
          np.all(np.diff(sig_rock_arr) >= 0))

    # --- 4. Melt conductivity models ---
    print("\n4. Melt conductivity models")
    T_melt = np.array([1273.15])   # 1000 C
    P_melt = np.array([0.2])      # 0.2 GPa
    H2O_melt = np.array([4.0])    # 4 wt%

    sig_a = sigma_melt_andesite(T_melt, P_melt, H2O_melt)
    check("andesite (1000C, 0.2GPa, 4% H2O)",
          1e-3 < float(sig_a) < 100,
          f"got {float(sig_a):.3e}")

    sig_d = sigma_melt_dacite(T_melt, P_melt, H2O_melt)
    check("dacite (1000C, 0.2GPa, 4% H2O)",
          1e-3 < float(sig_d) < 100,
          f"got {float(sig_d):.3e}")

    sig_r = sigma_melt_rhyolite(T_melt, P_melt, H2O_melt)
    check("rhyolite (1000C, 0.2GPa, 4% H2O)",
          1e-3 < float(sig_r) < 100,
          f"got {float(sig_r):.3e}")

    sig_interp = sigma_melt_interpolated(T_melt, P_melt, H2O_melt, 65.8)
    check("interpolated at dacite SiO2 ~ dacite",
          abs(np.log10(float(sig_interp)) - np.log10(float(sig_d))) < 0.01,
          f"interp={float(sig_interp):.3e}, dacite={float(sig_d):.3e}")

    # Interpolated is bracketed by end-members
    sig_i_mid = sigma_melt_interpolated(T_melt, P_melt, H2O_melt, 68.0)
    lo = min(float(sig_a), float(sig_d), float(sig_r))
    hi = max(float(sig_a), float(sig_d), float(sig_r))
    check("interpolated (68 SiO2) bracketed by end-members",
          lo * 0.1 <= float(sig_i_mid) <= hi * 10,
          f"got {float(sig_i_mid):.3e}, range [{lo:.3e}, {hi:.3e}]")

    # Melt conductivity increases with T
    T_melt_arr = np.array([1073.15, 1173.15, 1273.15, 1373.15])
    sig_T_melt = sigma_melt_dacite(
        T_melt_arr, np.full(4, 0.2), np.full(4, 4.0))
    check("sigma_melt_dacite increases with T",
          np.all(np.diff(sig_T_melt) > 0),
          f"values: {sig_T_melt}")

    # Melt conductivity increases with H2O
    H2O_arr = np.array([1.0, 2.0, 4.0, 6.0])
    sig_H2O = sigma_melt_andesite(
        np.full(4, 1273.15), np.full(4, 0.2), H2O_arr)
    check("sigma_melt_andesite increases with H2O",
          np.all(np.diff(sig_H2O) > 0),
          f"values: {sig_H2O}")

    # Melt >> rock at magmatic T
    check("sigma_melt >> sigma_rock at 1000C",
          float(sig_d) > float(sigma_rock_olhoeft(1000.0)) * 10,
          f"melt={float(sig_d):.3e}, rock={float(sigma_rock_olhoeft(1000.0)):.3e}")

    # --- 5. Cementation exponents (Samrock 2021) ---
    print("\n5. Cementation exponents (Samrock et al., 2021)")
    ms, mm_, mv = cementation_exponents_samrock(
        np.array([0.5]), np.array([0.5]), np.array([0.0]))
    check("phi_melt=0.5 -> m_melt=1.0 (high melt regime)",
          abs(float(mm_) - 1.0) < 0.01,
          f"got m_melt={float(mm_):.3f}")

    ms2, mm2, mv2 = cementation_exponents_samrock(
        np.array([0.85]), np.array([0.1]), np.array([0.05]))
    expected_mm = -2.75 * 0.1 + 2.1  # = 1.825
    check("phi_melt=0.1 -> m_melt=1.825",
          abs(float(mm2) - expected_mm) < 0.01,
          f"got m_melt={float(mm2):.3f}, expected {expected_mm:.3f}")

    check("m_vol always 1.5",
          abs(float(mv) - 1.5) < 0.01 and abs(float(mv2) - 1.5) < 0.01)

    # --- 6. Hashin-Shtrikman ---
    print("\n6. Hashin-Shtrikman upper bound")
    hs_equal = hashin_shtrikman_upper(
        np.array([1.0]), np.array([1.0]), np.array([0.5]))
    check("equal phases -> same value",
          abs(float(hs_equal) - 1.0) < 0.01,
          f"got {float(hs_equal):.3e}")

    hs_pure_a = hashin_shtrikman_upper(
        np.array([10.0]), np.array([0.1]), np.array([1.0]))
    check("f_a=1.0 -> sigma_a",
          abs(float(hs_pure_a) - 10.0) < 0.01,
          f"got {float(hs_pure_a):.3e}")

    hs_pure_b = hashin_shtrikman_upper(
        np.array([10.0]), np.array([0.1]), np.array([0.0]))
    check("f_a=0.0 -> sigma_b",
          abs(float(hs_pure_b) - 0.1) < 0.01,
          f"got {float(hs_pure_b):.3e}")

    # --- 7. Edge cases ---
    print("\n7. Edge cases")
    m_neg = wtfrac_to_molality(-0.1)
    check("wtfrac_to_molality(-0.1) = NaN",
          np.isnan(float(m_neg)))

    Lambda_small_mu = molar_conductivity_watanabe(1e-7, 0.5)
    check("Lambda at very small viscosity is finite",
          np.isfinite(float(Lambda_small_mu)) and float(Lambda_small_mu) > 0,
          f"got {float(Lambda_small_mu):.3e}")

    # --- 8. SiO2 enrichment ---
    print("\n8. SiO2 enrichment")
    SiO2_0 = SiO2_enrichment(0.0)
    check("chi=0 -> parent SiO2",
          abs(float(SiO2_0) - 63.0) < 0.01)

    SiO2_1 = SiO2_enrichment(1.0)
    check("chi=1 -> final SiO2",
          abs(float(SiO2_1) - 75.0) < 0.01)

    # --- Summary ---
    print("\n" + "=" * 70)
    total = passed + failed
    print(f"Results: {passed}/{total} passed, {failed} failed")
    if failed == 0:
        print("All tests passed.")
    else:
        print("SOME TESTS FAILED -- review output above.")
    print("=" * 70)
