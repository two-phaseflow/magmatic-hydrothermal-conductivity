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
    sigma_density_model   Density model for intermediate densities (200-450 kg/m3)
    sigma_vapor           Sinmyo & Keppler (2017) low-density vapor conductivity
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
    exchange on mineral surfaces. Five surface-conduction options are
    available via config['surface_conduction_model'] (see the Surface
    conduction section of the module docstring for full details):

        'waxman_smits'       Waxman & Smits (1968) Eq. 19 structural
                             form for B(sigma_w) (alpha = 0.6,
                             gamma = 1.3 S/m) with beta_25 recalibrated
                             to Revil 2002's surface-bound value
                             (0.53e-8 m^2/(V.s), alpha_T = 0.040 /C).
                             Preserves WS's sigma_w dependence while
                             using realistic surface-bound mobility.
        'revil'              Revil et al. (2002) DC-calibrated intrinsic
                             surface conductivity sigma_S from their
                             Eq. 7, (2/3)*rho_g*beta_s*CEC (reproduces
                             their Fig. 7b). For the full bulk
                             Eqs. 11-13 with Dukhin-number rollover,
                             see revil2002_bulk_conductivity().
        'levy'               Lévy et al. (2018, Eq. 16) empirical
                             smectite-inclusive parameterization.
        'waxman_smits_revil' WS in non-clay regions, Revil 2002 in
                             config['clay_cap_regions'].
        'waxman_smits_levy'  WS in non-clay regions, Lévy in
                             config['clay_cap_regions'].

Mixing laws
-----------
The hydrothermal domain supports two mixing law options
(config['mixing_law']):

    'glover' (default): Generalized Archie's law (Glover, 2010,
        Geophysics 75(6), E247-E265), applied to three phases — rock,
        liquid, and vapor — with the "conservation of connectedness"
        unity constraint (Glover 2010 Eq. 2; Glover 2009, TLE 28(1),
        82-85):

            sum_i(phi_i^m_i) = 1

        The bulk conductivity is the sum of per-phase contributions:

            sigma_bulk = sigma_rock * phi_s^m_solid
                       + sigma_liq * phi_liq^m_liq
                       + sigma_vap * phi_vap^m_vap
                       + sigma_surface

        where phi_s = 1 - phi, phi_liq = phi * S_liq,
        phi_vap = phi * S_vap.

        Per-phase cementation exponents follow the Archie convention
        in which m describes the FLUID pathway through the pore
        network. The regional m values from the literature (e.g.
        Revil et al. 2024 m=1.7 for granite; Zhang & Revil 2023 m=2.1
        for andesite; m=2.2 for clay caps) are therefore assigned to
        m_liq and m_vap. The rock cementation exponent m_solid is
        DERIVED from the unity constraint, so the mineral framework's
        connectedness is physically consistent with the fluid
        pathways rather than being set equal to m_fluid by convention.
        This matches the magmatic-domain treatment (Samrock et al.
        2021 for m_melt, m_vol fixed; m_solid derived from unity),
        applying the same Glover (2010) framework uniformly across
        the magmatic-hydrothermal system. See
        cementation_exponents_hydrothermal.

    'hashin_shtrikman': Two-step mixing. First combine liquid and
        vapor via the HS upper bound (assumes brine is the connected
        phase within the pore space), then mix with rock via the
        formation factor F:

            sigma_fluid = HS+(sigma_liq, sigma_vap, S_liq)
            sigma_bulk = (1/F) * sigma_fluid + sigma_surface

        This gives higher bulk conductivity than Glover at low
        brine saturations, because it assumes brine connectivity
        within the pore space independent of pore geometry.

Surface conduction (sigma_surface) is added in both cases using the
configured surface_conduction_model (see Surface conduction section
below). The recommended production setting for smectite-bearing clay
caps is 'waxman_smits_levy'.

The density-based model switch for the vapor phase (Watanabe above
450 kg/m3, density model at 200-450 kg/m3, Sinmyo-Keppler below
200 kg/m3) is applied within compute_fluid_conductivity before
either mixing law.

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

    CEC_meq_per_100g       Cation exchange capacity [meq/100g].
        Fresh rock: ~2; smectite-rich clay cap: 50-150.

The tortuosity factor a is kept at 1.0 following Glover (2009), who
argues that a != 1 indicates an incorrect m value.

Example per-region config:
    'regions': {
        1: {'porosity_exponent_m': 1.7, 'grain_density': 2750.0},
        11: {'porosity_exponent_m': 2.2, 'grain_density': 2500.0,
             'CEC_meq_per_100g': 80.0},
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
Vapor conductivity is computed using three density regimes:

    rho >= 450 kg/m3: Watanabe (2021) liquid model. The fluid is
        liquid-like at these densities and the viscosity-based model
        is validated.

    200 <= rho < 450 kg/m3: Density model, following Watanabe et al.
        (2022, Geothermics 101, Section 3.2.2). An empirical fit
        derived from Watanabe (2021) conductivity values computed in
        the single-phase region (rho 450-700) and extrapolated to
        lower densities:

            log10(sigma) = a*rho^2 + b*rho + c*log10(m) + d

        Fitted to 8151 data points at T = 375-800 C, sal = 0.1-10 wt%
        (R2 = 0.994). This bypasses the viscosity calculation, which
        fails at intermediate densities where IAPWS97/Klyukin valid
        ranges are exceeded.

    rho < 200 kg/m3: Sinmyo & Keppler (2017, Contrib. Mineral.
        Petrol. 172, 4). For very dilute low-density steam:

            log10(sigma) = -1.7060 - 93.78/T + 0.8075*log10(c)
                         + 3.0781*log10(rho) + log10(Lambda0)

        where rho [g/cm3], T [K], c [wt% NaCl].

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
J. Geophys. Res. 86(B2), 931-936), using the quadratic fit of
Watanabe et al. (2022, Geothermics 101, 102361, Eq. 2):

    log10(sigma_s^{-1}) = a*T^2 + b*T + c

where a = 7.34406e-6, b = -1.95002e-2, c = 13.5479, T in deg C.
This gives sigma_s = 10^{-14} to 10^{-5} S/m at 25-600 C.

Surface conduction
------------------
Three surface-conduction parameterizations and two mixed-dispatch
options are available, selected via config['surface_conduction_model']:

  'waxman_smits' (default): classical Waxman & Smits (1968, J. Pet. Tech.
      20(6), 107-122) in the volume-averaged form of Revil et al. (1998,
      J. Geophys. Res. 103, 23925-23936, Eq. 8):

          sigma_s = (1/(F*phi_eff)) * (1-phi_eff) * rho_g * B(sigma_w, T)
                    * CEC * S_total^(n-1)

          B(sigma_w, T) = beta(T) * (1 - alpha * exp(-sigma_w / gamma))
          beta(T)       = beta_25 * (1 + alpha_T * (T - 25))

      B is the Waxman-Smits "apparent counterion mobility" with the
      salinity dependence of their Eq. 19 (Fig. 6). Default
      parameters are a hybrid:

        alpha = 0.6, gamma = 1.3 S/m   (WS 1968 Eq. 19 original)
        beta_25 = 0.53e-8 m^2/(V.s)    (Revil 2002 Fig. 7b recalibration)
        alpha_T = 0.040 /C              (Revil 2002 Eq. 19, nu_S)

      We keep the WS 1968 sigma_w-dependent structure but recalibrate
      beta_25 to Revil 2002's DC-measured surface-bound mobility.
      WS 1968's original beta_max = 4.77e-8 m^2/(V.s) is the free-
      solution Na+ mobility (their high-sigma_w saturation limit),
      which over-predicts real surface-bound counterion contribution
      by ~10x at geothermal conditions. Revil 2002's 0.53e-8 is the
      measured value on altered volcanic rocks at comparable regimes.

  'revil': Revil et al. (2002, J. Geophys. Res. 107, 2168) DC-calibrated
      intrinsic surface conductivity from their Eq. 7:

          sigma_s = (2/3) * rho_g * beta_s(T) * CEC * S_total^(n-1)

          beta_s(T) = beta_s_25 * (1 + alpha_T * (T - 25))

      The (2/3) factor is the DEM geometric factor for spherical
      grains (Revil & Glover 1998; Bruggeman 1935). sigma_s here is
      the INTRINSIC surface conductivity of the clay fraction, the
      quantity plotted on the y-axis of Revil 2002 Fig. 7b; it is a
      property of the rock itself, independent of pore fluid or of
      F. No salinity dependence; beta_s is an empirical surface
      mobility that absorbs the Stern/diffuse-layer partition into a
      single constant.

      Default: beta_s_25(Na+) = 0.53e-8 m^2/(V.s) (Revil 2002
      Fig. 7b, Cobalt-CEC linear fit); alpha_T = 0.040 /C (Revil
      2002 Eq. 19, nu_S).

      When combined with the library's additive-form bulk mixing law
      (sigma_bulk = sigma_f/F + sigma_surface, as used for WS and
      Levy), this is equivalent to Revil 2002's high-salinity limit
      (Eq. 10, with the small -sigma_f*t_+/F correction omitted).
      For faithful treatment at clay-cap conditions where xi =
      sigma_S/sigma_f approaches the isoconductivity point xi_+ =
      1 - t_+ ~ 0.61 for NaCl, call revil2002_bulk_conductivity()
      directly -- it implements the full Eqs. 11-13 with the Dukhin
      rollover.

      NOTE: earlier versions of this library used Revil 1998 Eq. 8
      geometry (1-phi)/(F*phi) or the (F-1)/F prefactor from Revil
      1998's bulk formula. Both were incorrect -- Revil 2002 Eq. 7
      has no F factor at all in the intrinsic sigma_S.

  'levy': Levy et al. (2018, GJI 215, 1558-1582) three-pathway
      framework (their Eqs. 13 and 16), capturing both edge-EDL and
      smectite-interfoliar conduction:

          sigma_s = B'(T) * (CEC/CEC_0) * (1-phi_eff) / phi_eff^(1-m)

          B'(T) = B'_ref * (1 + alpha_T * (T - T_ref))

      Default: B'_ref = 0.77 S/m (Levy 2018 Fig. 10), CEC_0 = 91
      meq/100g (Levy 2018 Fig. 4), alpha_T = 0.04 /C, T_ref = 25 C.
      Levy Eq. 16 is used here as the additive sigma_surface term,
      while the pore-fluid term sigma_w/F is still supplied by the
      mixing law.

  'waxman_smits_revil': classical Waxman-Smits in all hydrothermal
      regions, with the Revil 2002 parameterization substituted
      within config['clay_cap_regions']. Useful when a clay-cap-
      specific DC-calibrated surface mobility is preferred without
      invoking the Levy empirical interlayer model.

  'waxman_smits_levy': classical Waxman-Smits in all hydrothermal
      regions, with the Levy 2018 parameterization substituted within
      config['clay_cap_regions']. This is the recommended production
      setting for smectite-bearing clay caps, because Levy's B'
      empirically includes the interfoliar (intra-solid) conduction
      pathway that Waxman-Smits cannot represent.

All four formulations use phi_eff = porosity * (1 - saturation_halite)
so volume occupied by precipitated halite is excluded from the fluid-
accessible pore space. Saturation scaling uses S_total = S_liq + S_vap
rather than S_liq alone, because at supercritical conditions the
"vapor" phase still wets mineral surfaces.

The previously-used Revil et al. (2017, GJI 208, Eq. 16) IP-calibrated
form with an explicit Stern partition (1-f_stern) is not a good DC
surface-conduction predictor: at f_stern = 0.95 (typical) it under-
estimates measured bulk conductivities of altered volcaniclastics
(Revil et al. 2002 Fig. 7b) by ~100x. It has therefore been removed.

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

2. Vapor-phase conductivity uses three density regimes: Watanabe
   (2021) above 450 kg/m3, a density model (Watanabe et al. 2022
   approach) at 200-450 kg/m3, and Sinmyo & Keppler (2017) below
   200 kg/m3.

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
Allnatt, A.R. & Pantelis, P. (1968). Anomalous high temperature
    electrical conductance of NaCl. Solid State Commun. 6, 309-312.

Archie, G.E. (1942). The electrical resistivity log as an aid in
    determining some reservoir characteristics. Trans. AIME 146, 54-62.

Driesner, T. & Heinrich, C.A. (2007). The system H2O-NaCl. Part I:
    Correlation formulae for phase relations in temperature-pressure-
    composition space from 0 to 1000 C, 0 to 5000 bar, and 0 to 1
    X_NaCl. Geochim. Cosmochim. Acta 71, 4880-4901.

Glover, P.W.J., Hole, M.J. & Pous, J. (2000). A modified Archie's law
    for two conducting phases. Earth Planet. Sci. Lett. 180, 369-383.

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

Kristinsdottir, L.H., Flovenz, O.G., Arnason, K., Bruhn, D., Milsch, H.,
    Spangenberg, E. & Kulenkampff, J. (2010). Electrical conductivity and
    P-wave velocity in rock samples from high-temperature Icelandic
    geothermal fields. Geothermics 39, 94-105.

Laumonier, M., Gaillard, F. & Sifre, D. (2015). The effect of pressure
    and water concentration on the electrical conductivity of dacitic
    melts. Chemical Geology 418, 66-76.

Laumonier, M., Karakas, O., Bachmann, O., Gaillard, F., Lukacs, R. &
    Seghedi, I. (2019). Evidence for a persistent magma reservoir with
    large melt content beneath an apparently extinct volcano. Earth
    Planet. Sci. Lett. 521, 79-90.

Levy, L., Gibert, B., Sigmundsson, F., Flovenz, O.G., Hersir, G.P.,
    Briole, P., Pezard, P.A. & Doin, M.P. (2018). The role of smectites
    in the electrical conductivity of active hydrothermal systems:
    Electrical properties of core samples from Krafla volcano, Iceland.
    Geophys. J. Int. 215, 1558-1582.

Mapother, D., Crooks, H.N. & Maurer, R. (1950). Self-diffusion of sodium
    in sodium chloride and sodium bromide. J. Chem. Phys. 18, 1231-1236.

Olhoeft, G.R. (1981). Electrical properties of granite with implications
    for the lower crust. J. Geophys. Res. 86(B2), 931-936.

Revil, A., Cathles, L.M., Losh, S. & Nunn, J.A. (1998). Electrical
    conductivity in shaly sands with geophysical applications. J.
    Geophys. Res. 103(B10), 23925-23936.

Revil, A., Hermitte, D., Spangenberg, E. & Cocheme, J.J. (2002).
    Electrical properties of zeolitized volcaniclastic materials. J.
    Geophys. Res. 107(B8), 2168.

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

Watanabe, N., Mogi, T., Yamaya, Y., Kitamura, K., Asanuma, H. &
    Tsuchiya, N. (2022). Electrical conductivity of H2O-NaCl fluids
    under supercritical geothermal conditions and implications for deep
    conductors observed by the magnetotelluric method. Geothermics 101,
    102361.

Waxman, M.H. & Smits, L.J.M. (1968). Electrical conductivities in
    oil-bearing shaly sands. J. Pet. Tech. 20(6), 107-122.

Yang, X. (2011). Origin of high electrical conductivity in the lower
    continental crust: A review. Surv. Geophys. 32, 875-903.

@author: samuels
"""

import warnings

import numpy as np
from numba import vectorize
from scipy.spatial import cKDTree
from viscosity import viscosity_h2o_nacl_vectorized
from iapws import IAPWS97, IAPWS95


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
# Unit-conversion factor: cation exchange capacity from meq/100g to C/kg.
#
#   1 equivalent = 1 mole of charge x F (Faraday) = 96,485 C
#   1 meq        = 10^-3 equivalent                = 96.485 C
#   1 meq/100g   = 96.485 C / 0.1 kg              = 964.85 C/kg
#
# So CEC_Ckg = CEC_meq_per_100g * 964.85. Used for Waxman-Smits / Revil
# CEC in SI units (where sigma_s = ... * rho_g * beta * CEC in kg/m^3 *
# m^2/(V.s) * C/kg = S/m) and for the denominator of Levy's CEC/CEC_0
# ratio (where both numerator and denominator get the same conversion,
# so its value is irrelevant to the Levy branch specifically).
#
# Earlier versions of this library had MEQ_TO_CKG = 9.6485, which is
# 100x too small. That bug silently suppressed Waxman-Smits and Revil
# surface-conduction contributions by 100x in every calculation.
# Lévy was not affected because its CEC appears only as a ratio.
# Fixed 2026-04 against Revil (2002) Fig. 7b.
MEQ_TO_CKG = 964.85



# =============================================================================
# DEFAULT_CONFIG
# =============================================================================

DEFAULT_CONFIG = {
    # --- Archie's law parameters ---
    'porosity_exponent_m': 1.8,       # cementation exponent [-]
    'saturation_exponent_n': 2.0,     # saturation exponent [-]
    'tortuosity_a': 1.0,              # tortuosity factor [-]

    # --- Mixing law for hydrothermal domain ---
    'mixing_law': 'glover',  # 'glover' or 'hashin_shtrikman'

    # --- Salinity floor ---
    'min_fluid_salinity_wt_percent': 0.0,  # background equilibrium [wt%]

    # --- Melt domain thresholds ---
    'melt_threshold': 0.0,                # phi_melt above this -> magmatic domain
    'spent_magma_min_porosity': 0.05,      # porosity floor for solidified intrusion
    'intrusion_id': 1,            # region ID of the intrusion body

    # --- Surface conduction model ---
    # One of:
    #   'waxman_smits'       -- classical Waxman & Smits (1968) everywhere,
    #                           with sigma_w-dependent B(sigma_w) from their
    #                           Eq. 7 / Levy 2018 Eq. 7.
    #   'revil'              -- Revil et al. (2002) DC-calibrated surface
    #                           mobility beta_s everywhere.
    #   'levy'               -- Levy et al. (2018) Eq. 16 everywhere.
    #   'waxman_smits_revil' -- WS in non-clay regions, Revil 2002 in
    #                           clay_cap_regions.
    #   'waxman_smits_levy'  -- WS in non-clay regions, Levy 2018 in
    #                           clay_cap_regions. Recommended production
    #                           setting for smectite-bearing clay caps.
    'surface_conduction_model': 'waxman_smits_levy',

    # Classical Waxman-Smits (1968) parameters, used when
    # surface_conduction_model is 'waxman_smits', 'waxman_smits_revil',
    # or 'waxman_smits_levy'. We use the WS 1968 Eq. 19 structural form
    # (alpha = 0.6, gamma = 1.3 S/m from their Fig. 6, in SI units)
    # but with beta_25 recalibrated to Revil et al. (2002)'s
    # surface-bound counterion mobility rather than the WS 1968
    # saturation limit:
    #
    #   B(sigma_w, 25C) = [1 - 0.6 * exp(-sigma_w / 1.3 S/m)] * 0.53e-8
    #
    # WS 1968 Eq. 19 in their original units gives beta_max =
    # 0.046 mho.cm^2/meq = 4.77e-8 m^2/(V.s), but this is the FREE-
    # SOLUTION Na+ mobility (WS's high-sigma_w asymptote where
    # counterions are assumed to behave as in bulk electrolyte). DC
    # measurements on altered volcanic rocks (Revil et al. 2002
    # Fig. 7b) give beta_s = 0.53e-8 for surface-bound counterions --
    # ~9x lower than free-solution, reflecting that bound counterions
    # are actually less mobile than WS 1968 assumed. Applied at
    # geothermal temperatures (T > 100 C, alpha_T ~ 0.04 /C gives
    # x6-8 boosts), the free-solution beta_25 = 4.77e-8 produces
    # wildly over-predicted surface conduction. We therefore use the
    # Revil-calibrated beta_25 = 0.53e-8 while keeping WS's alpha and
    # gamma parameterization for the sigma_w dependence of B.
    #
    # The temperature coefficient is also switched to Revil 2002
    # Eq. 19 nu_S = 0.040 /C (surface), not nu_f = 0.023 /C (free
    # fluid). Physically: since beta_25 now represents bound
    # counterions, it inherits the surface temperature coefficient.
    'ws_B_25': 0.53e-8,   # B(Na+, 25C, sigma_w->inf) [m2/(V.s)], Revil 2002 calibration
    'ws_alpha_T': 0.040,  # temperature coefficient of B [1/C], Revil 2002 Eq. 19 (nu_S)
    'ws_alpha_sw': 0.6,   # alpha in B(sigma_w) = beta*(1 - alpha*exp(-sigma_w/gamma)), WS 1968 Eq. 19
    'ws_gamma_sw': 1.3,   # gamma [S/m] = 0.013 mho/cm, WS 1968 Eq. 19

    # Revil et al. (2002) parameters (used when surface_conduction_model
    # is 'revil' or 'waxman_smits_revil'). beta_s_25 = 0.53e-8 m^2/(V.s)
    # is from Revil 2002 Fig. 7b (Cobalt-CEC linear regression), fit to
    # their Eq. 7: sigma_S = (2/3)*rho_g*beta_s*(CEC - CEC_r) where
    # sigma_S is the INTRINSIC clay surface conductivity, independent
    # of pore fluid or F. The _revil_surface_conductivity helper
    # implements this Eq. 7 literally (no F factor). For the full bulk
    # Eqs. 11-13 including the Dukhin-number rollover across the
    # isoconductivity point, call revil2002_bulk_conductivity() --
    # required at xi = sigma_S/sigma_f below xi_+ = 1 - t_+ = 0.61.
    'revil_beta_s_25': 0.53e-8,  # beta_s(Na+, 25C) [m2/(V.s)], Revil 2002 Eq. 7 + Fig. 7b
    'revil_alpha_T': 0.040,      # temperature coefficient [1/C], Revil 2002 Eq. 19 (nu_S)

    # Levy et al. (2018) parameters (used when surface_conduction_model
    # is 'levy' or 'waxman_smits_levy'):
    'levy_CEC_0_meq_per_100g': 91.0,  # pure smectite reference CEC [meq/100g]
    'levy_B_prime': 0.77,             # proportionality constant B' [S/m]
    'levy_alpha_T': 0.040,            # temperature coefficient [1/C]
    'levy_T_ref': 25.0,               # reference temperature [deg C]

    # --- Per-region properties ---
    # Override any default_region property for specific region IDs.
    # Supported keys per region: grain_density, CEC_meq_per_100g,
    # porosity_exponent_m, saturation_exponent_n.
    # Example:
    #   'regions': {
    #       1:  {'porosity_exponent_m': 1.7},                     # basement
    #       11: {'porosity_exponent_m': 2.2, 'CEC_meq_per_100g': 80.0},  # clay cap
    #   }
    'regions': {},
    'default_region': {
        'grain_density': 2800.0,      # [kg/m3]
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
    - Molality floored at 0 (non-negative) but NOT capped on the high
      side: magmatic-hydrothermal brines in the V+H (halite-saturating)
      region can reach salt mass fractions > 0.9, corresponding to
      m > 150 mol/kg-H2O. The Watanabe (2021) fit was calibrated up to
      moderate molality, so values beyond ~30 mol/kg are extrapolated.
      The sigmoidal A(m) and B^{-1}(m) saturate at their asymptotes;
      the linear C(m) grows unboundedly but its contribution is damped
      by 1/mu^2. Extrapolation is smooth and preferable to clipping,
      which would produce step artifacts at the molality boundary.
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
    m = np.maximum(np.asarray(molality, dtype=float), 0.0)

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

    # Floor to 1e-6 S/m; warn if non-zero-salinity nodes are floored and
    # report the conditions at floored nodes so phase-transition
    # extrapolation artifacts can be diagnosed (e.g. Watanabe breaking
    # down at supercritical-like "liquid" densities near the critical
    # point).
    floored = (sigma < 1e-6) & ~low_sal & np.isfinite(sigma)
    if np.any(floored):
        m = floored
        warnings.warn(
            f"sigma_liquid: {int(np.sum(m))} non-zero-salinity nodes "
            f"floored to 1e-6 S/m. Floored nodes: "
            f"T={T_C[m].min():.0f}-{T_C[m].max():.0f} C, "
            f"P={P_bar[m].min():.0f}-{P_bar[m].max():.0f} bar, "
            f"rho={density_solution[m].min():.1f}-"
            f"{density_solution[m].max():.1f} kg/m3, "
            f"wt%NaCl={wt_frac_NaCl[m].min()*100:.3g}-"
            f"{wt_frac_NaCl[m].max()*100:.3g}.")

    sigma = np.where(low_sal, 1e-6, sigma)
    sigma = np.where(np.isnan(sigma), 1e-6, sigma)
    return np.maximum(sigma, 1e-6)


# =============================================================================
# SECTION 2: VAPOR / INTERMEDIATE-DENSITY CONDUCTIVITY
# =============================================================================

def sigma_density_model(rho, sal_wt_frac):
    """
    Fluid conductivity from the density model (200-450 kg/m3).

    Empirical fit derived from Watanabe et al. (2021) conductivity
    values computed in the single-phase region (rho = 450-700 kg/m3)
    and extrapolated to lower densities, following the approach of
    Watanabe et al. (2022, Geothermics 101, Section 3.2.2).

    The fit relates log10(sigma) to fluid density and molality:

        log10(sigma) = a*rho^2 + b*rho + c*log10(m) + d

    where rho [kg/m3], m = molality [mol/kg-H2O], sigma [S/m].

    Coefficients fitted to 8151 Watanabe (2021) single-phase data
    points at T = 375-800 C, sal = 0.1-10 wt% NaCl, P up to 500 MPa
    (R2 = 0.994, RMSE = 0.044 log10 units).

    Parameters
    ----------
    rho : array_like
        Fluid density [kg/m3].
    sal_wt_frac : array_like
        NaCl weight fraction [0-1].

    Returns
    -------
    sigma : ndarray
        Electrical conductivity [S/m].
    """
    rho = np.asarray(rho, dtype=float)
    sal_wt_frac = np.asarray(sal_wt_frac, dtype=float)

    # Molality from weight fraction
    m = (sal_wt_frac / M_NACL) / np.maximum(1.0 - sal_wt_frac, 1e-10)
    log_m = np.log10(np.maximum(m, 1e-8))

    # Fitted coefficients
    a = -2.8582514881e-06
    b = 4.55810616e-03
    c = 0.7947
    d = -0.3554

    log_sigma = a * rho**2 + b * rho + c * log_m + d
    return np.maximum(10.0**log_sigma, 1e-6)


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
        try:
            water = IAPWS95(T=T_K, P=P_MPa)
            rho_H2O = water.rho / 1000.0
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

    # Report conditions at floor nodes so ultra-dilute physical cases can
    # be distinguished from Sinmyo-Keppler extrapolation failures
    # (Lambda0 <= 0 at the edge of the fit, or IAPWS failures).
    m = result <= 1e-6
    n_floor = int(np.sum(m))
    if n_floor > 0 and n_floor < len(result):
        warnings.warn(
            f"sigma_vapor: {n_floor}/{len(result)} nodes at floor "
            f"(1e-6 S/m). Floored nodes: "
            f"T={T_C[m].min():.0f}-{T_C[m].max():.0f} C, "
            f"P={P_bar[m].min():.0f}-{P_bar[m].max():.0f} bar, "
            f"rho={density_solution[m].min():.2f}-"
            f"{density_solution[m].max():.2f} kg/m3, "
            f"wt%NaCl={wt_frac_NaCl[m].min()*100:.3g}-"
            f"{wt_frac_NaCl[m].max()*100:.3g}.")

    return result


# =============================================================================
# SECTION 3: FLUID CONDUCTIVITY (two-phase L+V mixing at nodes)
# =============================================================================

def _sigma_fluid_by_density(wt_frac_NaCl, T_C, P_bar, rho):
    """
    Fluid conductivity routed by density.

    Routing:
        rho >= 450 kg/m3 -> Watanabe (2021) viscosity-based model
        200 <= rho < 450 -> density model (Watanabe 2022 Section 3.2.2)
        rho < 200        -> Sinmyo & Keppler (2017) dilute-steam model

    The 450 kg/m3 threshold is the lower bound of the density-model's
    fit range (Watanabe 2022 fit the density model on Watanabe 2021
    single-phase data at 450-700 kg/m3). Below 200 kg/m3 the density
    model is extrapolating and Sinmyo-Keppler is calibrated for
    dilute steam, so we switch there.

    Applied identically to the liquid and vapor phases because CSMP++
    may label supercritical fluid as either (what matters is density,
    not the phase label).

    Parameters
    ----------
    wt_frac_NaCl, T_C, P_bar, rho : array_like
        Per-node inputs, same length.

    Returns
    -------
    sigma : ndarray
        Fluid conductivity [S/m].
    counts : dict
        {'watanabe': n_dense, 'density_model': n_intermediate,
         'sinmyo': n_dilute} for diagnostic printout.
    """
    wt_frac_NaCl = np.asarray(wt_frac_NaCl, dtype=float)
    T_C = np.asarray(T_C, dtype=float)
    P_bar = np.asarray(P_bar, dtype=float)
    rho = np.asarray(rho, dtype=float)

    dense = rho >= 450.0
    intermediate = (rho >= 200.0) & (rho < 450.0)
    dilute = rho < 200.0

    sigma = np.zeros_like(rho)
    if np.any(dense):
        sigma[dense] = sigma_liquid(
            wt_frac_NaCl[dense], T_C[dense], P_bar[dense], rho[dense])
    if np.any(intermediate):
        sigma[intermediate] = sigma_density_model(
            rho[intermediate], wt_frac_NaCl[intermediate])
    if np.any(dilute):
        sigma[dilute] = sigma_vapor(
            wt_frac_NaCl[dilute], T_C[dilute], P_bar[dilute], rho[dilute])

    counts = {'watanabe': int(np.sum(dense)),
              'density_model': int(np.sum(intermediate)),
              'sinmyo': int(np.sum(dilute))}
    return sigma, counts


def _fmt_range(values, mask):
    """Format min-max range of masked positive finite values."""
    if not np.any(mask):
        return "n/a (0 nodes)"
    v = values[mask]
    v = v[np.isfinite(v) & (v > 0)]
    if v.size == 0:
        return "n/a (all zero/non-finite)"
    return f"{v.min():.3e} - {v.max():.3e} S/m"


def compute_fluid_conductivity(X_liq, X_vap, T_C, P_bar, S_liq, S_vap,
                               rho_liq, rho_vap):
    """
    Per-phase fluid conductivity at nodes with density-based routing.

    Both phases use the same routing because CSMP++ can label
    supercritical fluid as either "liquid" (expanded low-density liquid
    above the critical point) or "vapor"; what matters for conductivity
    is the fluid state, not the phase label.

    Routing (per node, applied to both phases):
        rho >= 450 kg/m3 -> Watanabe (2021) viscosity-based model
        200 <= rho < 450 -> density model (Watanabe 2022, Section 3.2.2)
        rho < 200        -> Sinmyo & Keppler (2017) dilute-steam model

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

    liq_mask = S_liq > 0.0
    sig_liq = np.zeros(n)
    if np.any(liq_mask):
        sig_liq[liq_mask], c_liq = _sigma_fluid_by_density(
            X_liq[liq_mask], T_C[liq_mask], P_bar[liq_mask],
            rho_liq[liq_mask])
        print(f"    Liquid: Watanabe(rho>=450)={c_liq['watanabe']}, "
              f"density_model(200-450)={c_liq['density_model']}, "
              f"Sinmyo(<200)={c_liq['sinmyo']}")
        print(f"            sigma_liq range: {_fmt_range(sig_liq, liq_mask)}")

    vap_mask = S_vap > 0.0
    sig_vap = np.zeros(n)
    if np.any(vap_mask):
        sig_vap[vap_mask], c_vap = _sigma_fluid_by_density(
            X_vap[vap_mask], T_C[vap_mask], P_bar[vap_mask],
            rho_vap[vap_mask])
        print(f"    Vapor:  Watanabe(rho>=450)={c_vap['watanabe']}, "
              f"density_model(200-450)={c_vap['density_model']}, "
              f"Sinmyo(<200)={c_vap['sinmyo']}")
        print(f"            sigma_vap range: {_fmt_range(sig_vap, vap_mask)}")

    n_liq = int(np.sum(liq_mask & ~vap_mask))
    n_vap = int(np.sum(~liq_mask & vap_mask))
    n_2ph = int(np.sum(liq_mask & vap_mask))
    print(f"    Phases: liquid-only={n_liq}, vapor-only={n_vap}, "
          f"two-phase={n_2ph}")

    return {'sigma_liq': sig_liq, 'sigma_vap': sig_vap}


# =============================================================================
# SECTION 4: ROCK MATRIX CONDUCTIVITY -- Olhoeft (1981)
# J. Geophys. Res. 86(B2), 931-936
# =============================================================================

@vectorize(['float64(float64)'], nopython=True)
def sigma_rock_olhoeft(T_C):
    """
    Temperature-dependent dry rock matrix conductivity.

    Quadratic fit to the dry Westerly Granite data of Olhoeft (1981,
    J. Geophys. Res. 86(B2), 931-936), as parameterized by Watanabe
    et al. (2022, Geothermics 101, 102361, Eq. 2):

        log10(sigma_s^{-1}) = a*T^2 + b*T + c

    where T is temperature in degrees Celsius and sigma_s is
    conductivity in S/m. Coefficients:

        a = 7.34406e-6
        b = -1.95002e-2
        c = 13.5479

    This gives sigma_s = 10^{-14} to 10^{-5} S/m at 25-600 C,
    consistent with the Olhoeft (1981) Fig. 8 data for dry granite.

    Guards
    ------
    Output is clipped to >= 1e-14 S/m to prevent underflow.

    Parameters
    ----------
    T_C : float
        Temperature [deg C].

    Returns
    -------
    sigma : float
        Rock conductivity [S/m].
    """
    # Watanabe et al. (2022) Eq. 2 coefficients
    a = 7.34406e-6
    b = -1.95002e-2
    c = 13.5479
    log10_rho = a * T_C * T_C + b * T_C + c
    sigma = 10.0 ** (-log10_rho)
    return max(1e-14, sigma)


def sigma_halite(T_C):
    """
    Electrical conductivity of pure halite (NaCl) in the intrinsic regime.

    Mapother, D., Crooks, H.N. & Maurer, R. (1950). Self-diffusion of
    sodium in sodium chloride and sodium bromide. J. Chem. Phys. 18,
    1231-1236. Eq. (2) with parameters from Table I, Row B (derived
    from direct conductivity measurements on pure NaCl):

        sigma(T) = (sigma_0 / T_K) * exp(-E_a / (k_B * T_K))

    where
        E_a = epsilon + epsilon'/2 = 1.89 eV
            is the activation energy for intrinsic, Schottky-defect-
            mediated cation migration, and
        sigma_0 = D_0 * N * e^2 / k_B ~ 5.8e10 S.K/m
            is obtained from D_0 = 14 cm^2/s (Mapother Table I Row B)
            via the Einstein relation (sigma/D = N*e^2/(k_B*T)) with
            N = rho_NaCl * N_A / M_NaCl = 2.23e28 Na+ ions / m^3.

    Consistent with the independent Schottky+Frenkel analysis of
    Allnatt & Pantelis (1968, Solid State Commun. 6, 309-312), whose
    defect parameters h = 2.167 eV, Delta_h_1 = 0.658 eV imply
    E_a = h/2 + Delta_h_1 = 1.74 eV (~8% below Mapother's 1.89 eV;
    both are within the scatter of different analyses and sample
    purities).

    Validity
    --------
    Intrinsic regime, T > ~350 C. Below this temperature extrinsic
    divalent-impurity conduction dominates (Mapother Fig. 5, Eq. 5),
    but this is irrelevant for hydrothermal modeling: halite does
    not exist as a solid phase in H2O-NaCl systems below the fluid's
    halite-saturation curve (T > ~300 C at relevant geothermal
    pressures; Driesner & Heinrich, 2007).

    Magnitude at V+H-relevant temperatures:
        T = 500 C: sigma ~ 3.5e-5 S/m   (~15x dry silicate rock)
        T = 600 C: sigma ~ 1.1e-3 S/m   (~35x dry silicate rock)
        T = 700 C: sigma ~ 9.6e-3 S/m   (~30x dry silicate rock)
        T = 800 C: sigma ~ 5.2e-2 S/m   (~24x dry silicate rock)

    Parameters
    ----------
    T_C : array_like
        Temperature [deg C].

    Returns
    -------
    sigma : ndarray
        Halite electrical conductivity [S/m].
    """
    T_K = np.asarray(T_C, dtype=float) + 273.15
    sigma_0 = 5.8e10        # S.K/m  (= D_0 * N * e^2 / k_B)
    E_a_eV = 1.89           # Mapother Table I Row B
    k_B_eV = 8.617333e-5    # eV/K
    return (sigma_0 / T_K) * np.exp(-E_a_eV / (k_B_eV * T_K))


def sigma_solid_phase(T_C, phi, S_halite=None, sigma_rock=None):
    """
    Effective conductivity of the solid phase (silicate rock + halite).

    In most of the hydrothermal domain the solid phase is just
    silicate rock matrix (sigma = sigma_rock from sigma_rock_olhoeft,
    Watanabe 2022 Eq. 2). In the vapor-halite coexistence region
    where halite precipitates (saturation_halite > 0, from the
    coupled H2O-NaCl equation of state), the solid phase is a
    mixture of rock and halite precipitate. This helper returns the
    arithmetic volume-weighted average:

        sigma_solid = f_rock * sigma_rock + f_halite * sigma_halite

    where the fractions are taken within the total solid phase:

        phi_solid = (1 - phi) + phi * S_halite
        f_rock    = (1 - phi)       / phi_solid
        f_halite  = (phi * S_halite) / phi_solid

    Arithmetic (parallel / upper Wiener bound) weighting is used
    because the Glover unity-constraint already captures the
    geometric connectedness of the solid phase as a whole through
    phi_solid^m_solid; what we need here is the material-weighted
    "internal" conductivity of that phase.

    Halite conductivity exceeds dry silicate by ~15-35x in the
    500-800 C range (see sigma_halite), and halite volume fraction
    can approach unity as the vapor phase dries out in the V+H
    region, so the correction is non-negligible there. Below
    ~400 C the two are within a factor of 2 and the correction
    is small.

    Efficiency
    ----------
    sigma_halite() is evaluated only at nodes where S_halite > 0.
    In typical simulations this is a small subset (a few percent of
    hydrothermal nodes). Nodes with S_halite = 0 pass through as
    sigma_rock with no additional computation.

    Parameters
    ----------
    T_C : array_like
        Temperature per node [deg C].
    phi : array_like
        Porosity per node (original, NOT halite-reduced phi_eff).
    S_halite : array_like or None, optional
        Halite saturation per node (fraction of pore volume occupied
        by halite). If None or all zero, no halite contribution.
    sigma_rock : array_like or None, optional
        Pre-computed silicate rock conductivity per node [S/m]. If
        None, sigma_rock_olhoeft(T_C) is called internally.

    Returns
    -------
    sigma_solid : ndarray
        Effective solid-phase conductivity per node [S/m]. Same
        shape as T_C.
    """
    T_C_arr = np.asarray(T_C, dtype=float)
    phi_arr = np.asarray(phi, dtype=float)

    if sigma_rock is None:
        sigma_rock_arr = sigma_rock_olhoeft(T_C_arr)
    else:
        sigma_rock_arr = np.asarray(sigma_rock, dtype=float)

    # No halite anywhere -> pure silicate
    if S_halite is None:
        return sigma_rock_arr.copy()

    S_halite_arr = np.asarray(S_halite, dtype=float)
    has_halite = S_halite_arr > 0.0
    if not np.any(has_halite):
        return sigma_rock_arr.copy()

    # Compute halite conductivity ONLY where needed
    sigma_solid = sigma_rock_arr.copy()
    T_hal = T_C_arr[has_halite]
    phi_hal = phi_arr[has_halite]
    S_hal = S_halite_arr[has_halite]

    # Volume fractions within the solid phase
    phi_solid = (1.0 - phi_hal) + phi_hal * S_hal
    phi_solid_safe = np.clip(phi_solid, 1e-6, 1.0)
    f_halite = (phi_hal * S_hal) / phi_solid_safe
    f_halite = np.clip(f_halite, 0.0, 1.0)
    f_rock = 1.0 - f_halite

    sigma_solid[has_halite] = (
        f_rock * sigma_rock_arr[has_halite]
        + f_halite * sigma_halite(T_hal))

    return sigma_solid


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


def cementation_exponents_hydrothermal(phi_solid, phi_liq, phi_vap,
                                       m_fluid):
    """
    Three-phase cementation exponents for the hydrothermal domain.

    Glover (2009, The Leading Edge 28(1), 82-85) interprets the
    cementation exponent m as a measure of the connectedness G = phi^m
    of each phase: low m means high connectedness, high m means low
    connectedness. In a multi-phase conducting system (Glover et al.,
    2000, EPSL 180, 369-383; Glover, 2010, Geophysics 75(6), E247-E265),
    the cementation exponents of the phases are not independent — they
    are linked by the "conservation of connectedness" constraint
    (Glover, 2010, Eq. 2):

        sum_i(phi_i^m_i) = 1

    which expresses the physical requirement that the phases together
    fill the sample and share its electrical pathways.

    In the Archie convention, the cementation exponent of a porous
    rock describes the pathway through the FLUID phase (the brine
    occupying the pore network). Published regional values
    (e.g. Revil et al. 2024: m = 1.7 for granite; Zhang & Revil 2023:
    m = 2.1 for andesite; m = 2.2 for smectite-rich clay) are all
    fluid cementation exponents. The rock matrix, being a continuous
    mineral framework, has its own degree of connectedness that must
    be derived so the unity constraint is satisfied.

    This helper therefore sets:

        m_liq   = m_fluid           (regional, from config)
        m_vap   = m_fluid           (same pore network)
        m_solid = log(1 - phi_liq^m_liq - phi_vap^m_vap) / log(phi_solid)

    which is the hydrothermal analogue of the magmatic routine
    `cementation_exponents_samrock`, and applies the same Glover (2010)
    conservation-of-connectedness principle uniformly across the
    magmatic-hydrothermal system.

    Parameters
    ----------
    phi_solid : array_like
        Rock volume fraction [0-1], = 1 - phi.
    phi_liq : array_like
        Liquid volume fraction [0-1], = phi * S_liq.
    phi_vap : array_like
        Vapor volume fraction [0-1], = phi * S_vap.
    m_fluid : array_like
        Regional fluid cementation exponent [-], typically 1.5-2.3.

    Returns
    -------
    m_solid : ndarray
        Solid (rock) cementation exponent, derived from unity.
        Clipped to [0.01, 3.0] for numerical safety.
    m_liq : ndarray
        Liquid cementation exponent, = m_fluid.
    m_vap : ndarray
        Vapor cementation exponent, = m_fluid.

    Notes
    -----
    - For a fully saturated rock (phi_vap = 0), the constraint reduces
      to the two-phase form (Glover 2009, Eqs. 3-4):
          phi^m_liq + (1-phi)^m_solid = 1
    - Typical m_solid values are near zero (well-connected mineral
      framework), reflecting the physical fact that the rock matrix
      forms a continuous solid.
    - G_rest (= 1 - G_liq - G_vap) is floored to 1e-12 to avoid log(0)
      at fully fluid-saturated, infinitesimally-porous edge cases.
    - phi_solid is floored to 1e-6 before the logarithm to avoid
      division by log(0) at the fluid-only limit. Such nodes are not
      physically meaningful in the hydrothermal domain.
    """
    phi_solid = np.asarray(phi_solid, float)
    phi_liq = np.asarray(phi_liq, float)
    phi_vap = np.asarray(phi_vap, float)
    m_fluid = np.asarray(m_fluid, float)

    phase_sum = phi_solid + phi_liq + phi_vap
    bad_sum = np.abs(phase_sum - 1.0) > 0.01
    if np.any(bad_sum):
        warnings.warn(
            f"cementation_exponents_hydrothermal: "
            f"{int(np.sum(bad_sum))} nodes with "
            f"phi_solid + phi_liq + phi_vap != 1.0 "
            f"(max deviation: "
            f"{float(np.max(np.abs(phase_sum - 1.0))):.3f})")

    m_liq = m_fluid
    m_vap = m_fluid

    G_liq = np.power(np.clip(phi_liq, 0.0, 1.0), m_liq)
    G_vap = np.power(np.clip(phi_vap, 0.0, 1.0), m_vap)
    G_rest = np.clip(1.0 - G_liq - G_vap, 1e-12, 1.0)
    safe_phi = np.clip(phi_solid, 1e-6, 1.0)
    m_solid = np.clip(np.log(G_rest) / np.log(safe_phi), 0.01, 3.0)

    return m_solid, m_liq, m_vap


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
# Revil et al. (1998) J. Geophys. Res. 103, 23925-23936, Eq. (8)
# Revil et al. (2002) J. Geophys. Res. 107(B8), 2168, Eq. (7), Fig. 7b
# Levy et al. (2018) GJI 215, 1558-1582, Eqs. (13), (16), (17)
# =============================================================================

def build_CEC_array(region_ids, config):
    """
    Per-node CEC from region configuration.

    CEC (cation exchange capacity) is converted from meq/100g to C/kg
    (SI units) using the Faraday-based factor MEQ_TO_CKG = 964.85.

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
    # Default CEC fallback lowered from 2.0 to 0.2 meq/100g: a value of
    # 2.0 is reasonable only for mildly altered volcanics, and using it
    # as a silent default for unassigned region IDs produced large
    # surface conduction in every node of the domain once the
    # MEQ_TO_CKG unit-conversion bug was fixed. 0.2 meq/100g
    # corresponds to fresh crystalline basement (Revil 2002 Table 3),
    # a conservative choice that doesn't accidentally add significant
    # surface conduction to unmapped regions.
    default_CEC = default.get('CEC_meq_per_100g', 0.2) * MEQ_TO_CKG

    # Legacy support
    if not regions and 'surface_conduction_regions' in config:
        sc = config['surface_conduction_regions']
        for rid, props in sc.items():
            regions[int(rid)] = {
                'CEC_meq_per_100g': props.get('CEC_meq_per_100g', 0.2)}
        default_CEC = config.get(
            'surface_conduction_default_CEC_meq_per_100g', 0.2) * MEQ_TO_CKG

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
        Property name, e.g. 'grain_density'.
    n_nodes : int
        Number of nodes.

    Returns
    -------
    values : ndarray
        Property values per node.
    """
    regions = config.get('regions', {})
    default = config.get('default_region', {})
    hardcoded = {'grain_density': 2800.0}
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


def _levy_surface_conductivity(phi_eff, CEC, T, m_arr, config):
    """
    Levy et al. (2018) smectite surface conductivity model.

    This helper implements Levy et al. (2018, GJI 215) Eq. (16) with the
    linear temperature correction from Eq. (17):

        sigma_surface = B'(T) * (CEC / CEC_0) * (1 - phi) / phi^(1 - m)
        B'(T) = B'_ref * (1 + alpha_T * (T - T_ref))

    In the original paper, Eq. (16) is a low-fluid-conductivity empirical
    fit to the combined edge double-layer and connected smectite interlayer
    response of altered Krafla basalts. Here it is used as the additive
    sigma_surface term within Levy's three-pathway bulk framework
    (Eq. 13):

        sigma_bulk = sigma_w / F + sigma_EDL + sigma_intra_solid

    with the pore-fluid term sigma_w / F still supplied by the selected
    bulk mixing law. This preserves the intended low-conductivity behavior
    while remaining well behaved when fluid conductivity is high.

    Parameters
    ----------
    phi_eff : ndarray
        Effective porosity [-]. Re-clipped internally to [1e-3, 1] as a
        safety guard against singular behavior at very low porosity.
    CEC : ndarray
        Cation exchange capacity [C/kg]. Negative values are clipped to 0.
    T : ndarray
        Temperature [deg C].
    m_arr : ndarray
        Cementation exponent [-]. Typically ~1.3-3.3 in this workflow.
    config : dict
        Must provide or default the following keys:
        levy_CEC_0_meq_per_100g : float
            Pure smectite reference CEC [meq/100g]. Default 91.
        levy_B_prime : float
            Empirical proportionality constant B' [S/m]. Default 0.77.
        levy_alpha_T : float
            Temperature coefficient [1/C]. Default 0.040.
        levy_T_ref : float
            Reference temperature [deg C]. Default 25.

    Returns
    -------
    sigma_surface : ndarray
        Surface conductivity contribution [S/m].

    Notes
    -----
    Assumptions:
    - Linear temperature correction is used exactly as in Levy Eq. (17).
    - The sample-specific intra-solid fitting term from Levy Eq. (15) is
      not modeled explicitly; its average effect is absorbed into B'.
    - B' was calibrated on Krafla smectite-bearing volcanic samples, so
      the model is most defensible for smectite-rich hydrothermal rocks.

    Guards:
    - phi is clipped to avoid division by zero.
    - CEC is clipped to remain non-negative.
    - If the temperature correction would make B'(T) negative, it is
      clipped to zero and a warning is emitted.
    """
    phi_safe = np.clip(np.asarray(phi_eff, dtype=float), 1e-3, 1.0)
    CEC_safe = np.clip(np.asarray(CEC, dtype=float), 0.0, None)
    T_arr = np.asarray(T, dtype=float)
    m_safe = np.asarray(m_arr, dtype=float)

    cec0_meq = max(float(config.get('levy_CEC_0_meq_per_100g', 91.0)), 1e-12)
    cec0 = cec0_meq * MEQ_TO_CKG
    B_prime_ref = max(float(config.get('levy_B_prime', 0.77)), 0.0)
    alpha_T = float(config.get('levy_alpha_T', 0.040))
    T_ref = float(config.get('levy_T_ref', 25.0))

    temp_factor = 1.0 + alpha_T * (T_arr - T_ref)
    n_negative = int(np.sum(temp_factor < 0.0))
    if n_negative:
        warnings.warn(
            f"Levy surface conductivity: {n_negative} nodes gave a negative "
            f"temperature factor; clipped B'(T) to zero.")
    temp_factor = np.clip(temp_factor, 0.0, None)

    B_prime_T = B_prime_ref * temp_factor
    cec_ratio = CEC_safe / cec0

    sigma_surface = (B_prime_T
                     * cec_ratio
                     * (1.0 - phi_safe)
                     / np.power(phi_safe, 1.0 - m_safe))
    return sigma_surface


def _waxman_smits_surface_conductivity(phi_eff, F, S_total, n_arr, rho_g,
                                       CEC, T, sigma_w, config):
    """
    Waxman & Smits (1968) surface conductivity with a Revil 2002
    recalibration of beta_25 for surface-bound counterions.

    Implements Waxman & Smits (1968, J. Pet. Tech. 20(6), 107-122) in
    the volume-averaged bulk form of Revil et al. (1998, J. Geophys.
    Res. 103, 23925-23936, Eq. 8):

        sigma_s = (1/(F*phi_eff)) * (1-phi_eff) * rho_g * B(sigma_w, T)
                  * CEC * S_total^(n-1)

    The salinity dependence of the apparent counterion mobility follows
    Waxman & Smits (1968) Eq. 19:

        B(sigma_w, T) = beta(T) * (1 - alpha * exp(-sigma_w / gamma))
        beta(T)       = beta_25 * (1 + alpha_T * (T - 25))

    Default parameters are a hybrid: we keep WS 1968's original Eq. 19
    structural form and empirical alpha / gamma, but recalibrate
    beta_25 to Revil et al. (2002)'s DC-measured surface counterion
    mobility rather than the WS 1968 free-solution saturation limit:

        alpha    = 0.6                          (WS 1968 Eq. 19)
        gamma    = 1.3 S/m   (= 0.013 mho/cm)   (WS 1968 Eq. 19)
        beta_25  = 0.53e-8 m^2/(V.s)             (Revil 2002 Fig. 7b)
        alpha_T  = 0.040 /C                      (Revil 2002 Eq. 19,
                                                  nu_S surface)

    Why the hybrid: WS 1968's Eq. 19 implies beta_max = 4.77e-8 m^2/(V.s)
    at saturation, which is essentially the FREE-SOLUTION Na+ mobility.
    Applied at geothermal temperatures with alpha_T = 0.023 /C, this
    over-predicts sigma_surface by ~10x because real surface-bound
    counterions have beta ~ 0.5e-8 (Revil 2002 DC measurements on
    altered volcanic rocks). Using Revil's calibration here keeps the
    WS sigma_w-dependence (important for unaltered low-salinity rocks
    where the diffuse layer expands and mobility is reduced) while
    correcting the magnitude.

    At high sigma_w the mobility asymptotes to beta(T). At low sigma_w
    it is reduced by (1-alpha) because the expanded electrical double
    layer impedes ion mobility (Waxman & Smits 1968 Sec V; Levy et
    al. 2018 Sec 2.2).

    In the high-sigma_w saturation limit, this branch reduces to
    approximately:
        sigma_s_WS ~ (1-phi)/(F phi) * rho_g * 0.53e-8 * (1+0.04(T-25)) * CEC
    which is the WS Eq. 8 volume-averaging geometry with Revil 2002's
    chemistry. Compare to the Revil 2002 'revil' branch which uses the
    same chemistry but in the DEM geometry:
        sigma_s_Revil = (2/3) * rho_g * 0.53e-8 * (1+0.04(T-25)) * CEC.
    The geometric prefactor (1-phi)/(F phi) typically suppresses WS by
    a factor of ~5-10 vs Revil at clay-cap conditions (phi~0.1, F~100).

    Parameters
    ----------
    phi_eff : ndarray
        Effective porosity (halite-corrected) [-].
    F : ndarray
        Formation factor F = a / phi_eff^m [-].
    S_total : ndarray
        Total wetting saturation S_liq + S_vap [-].
    n_arr : ndarray
        Saturation exponent [-].
    rho_g : ndarray
        Grain density [kg/m3].
    CEC : ndarray
        Cation exchange capacity [C/kg].
    T : ndarray
        Temperature [deg C].
    sigma_w : ndarray
        Pore-fluid (brine) conductivity [S/m]. Used for the non-linear
        B(sigma_w) dependence.
    config : dict
        Must provide or default: ws_B_25, ws_alpha_T, ws_alpha_sw,
        ws_gamma_sw.

    Returns
    -------
    sigma_surface : ndarray
        Surface conductivity contribution [S/m], additive to the
        mixing-law bulk term.
    """
    # Defaults: WS 1968 Eq. 19 structural form (alpha, gamma) with
    # beta_25 and alpha_T recalibrated to Revil 2002 for surface-bound
    # counterion mobility. See module-level DEFAULT_CONFIG comments
    # for the rationale.
    B_25 = float(config.get('ws_B_25', 0.53e-8))
    alpha_T = float(config.get('ws_alpha_T', 0.040))
    alpha_sw = float(config.get('ws_alpha_sw', 0.6))
    gamma_sw = float(config.get('ws_gamma_sw', 1.3))

    T_arr = np.asarray(T, dtype=float)
    sw = np.maximum(np.asarray(sigma_w, dtype=float), 0.0)

    beta_T = B_25 * (1.0 + alpha_T * (T_arr - 25.0))
    B = beta_T * (1.0 - alpha_sw * np.exp(-sw / gamma_sw))
    B = np.clip(B, 0.0, None)

    return ((1.0 / (F * phi_eff))
            * (1.0 - phi_eff)
            * (S_total ** (n_arr - 1.0))
            * rho_g * B * CEC)


def _revil_surface_conductivity(phi_eff, F, S_total, n_arr, rho_g,
                                CEC, T, config):
    """
    Revil et al. (2002) intrinsic clay-surface conductivity, Eq. 7.

    Revil, A., Hermitte, D., Spangenberg, E. & Cochemé, J.J. (2002).
    J. Geophys. Res. 107(B8), 2168.

        sigma_S = (2/3) * rho_g * beta_S(T) * CEC * S_total^(n-1)
        beta_S(T) = beta_s_25 * (1 + alpha_T * (T - 25))

    sigma_S in Revil 2002 is the INTRINSIC surface conductivity of
    the clay fraction -- a property of the rock itself, independent
    of pore fluid and of the formation factor F. It is the quantity
    plotted on the y-axis of Revil 2002 Fig. 7b. The (2/3) factor is
    the DEM geometric averaging factor for spherical grains (Revil &
    Glover 1998; Bruggeman 1935).

    IMPORTANT: in Revil 2002, the dependence of the BULK rock
    conductivity on F enters separately via their Eqs. 11-13, which
    build sigma_bulk = sigma_f/F * H(xi) from sigma_S and the
    pore-fluid conductivity. See revil2002_bulk_conductivity() for a
    full Eq. 11/12/13 implementation that includes the Dukhin-number
    rollover below the isoconductivity point.

    When this helper's output is used in the library's additive-form
    bulk mixing law (sigma_bulk = sigma_f/F + sigma_surface, as used
    for WS and Levy), it is equivalent to applying Revil 2002's
    high-salinity limit (their Eq. 10, with the small correction
    -sigma_f*t_+/F omitted). This is accurate for xi = sigma_S/sigma_f
    well above the isoconductivity point xi_+ = 1 - t_+ ~ 0.61 for
    NaCl; at lower salinity, use revil2002_bulk_conductivity().

    Note: Revil 2002 Eq. 7 also includes a residual-CEC subtraction
    (CEC - CEC_r) to remove the zeolite fraction that is measured by
    Cobalt titration but does not conduct. For the Yuzawa smectite
    clay cap, CEC_r is taken to be 0 and the per-region CEC is the
    smectite-effective value. The verify_revil2002_fig7b.py script
    applies CEC_r = 3 meq/100g externally for the Fig. 7b sanity
    check.

    Earlier versions of this helper (i) used Revil 1998 Eq. 8
    geometry (1-phi)/(F*phi), which under-predicted Fig. 7b by
    ~7-12x, and later (ii) included a (F-1)/F factor that does not
    appear in Revil 2002 Eq. 7 (that was from the Revil 1998 bulk
    formula). Both have been removed; this helper now matches Eq. 7
    literally.

    The saturation scaling S_total^(n-1) is retained for consistency
    with the WS and Levy branches (partial-saturation correction).

    Parameters
    ----------
    phi_eff : ndarray
        Effective porosity (halite-corrected) [-]. Not used in Eq. 7
        itself but retained in the signature for interface
        consistency with the WS and Levy helpers.
    F : ndarray
        Formation factor [-]. Not used in Eq. 7 itself but retained
        in the signature for interface consistency.
    S_total : ndarray
        Total wetting saturation [-].
    n_arr : ndarray
        Saturation exponent [-].
    rho_g : ndarray
        Grain density [kg/m3].
    CEC : ndarray
        Cation exchange capacity [C/kg]. In Yuzawa production,
        CEC is already the smectite-effective value (no CEC_r
        subtraction required).
    T : ndarray
        Temperature [deg C].
    config : dict
        Must provide or default: revil_beta_s_25, revil_alpha_T.

    Returns
    -------
    sigma_surface : ndarray
        Intrinsic surface conductivity sigma_S [S/m].
    """
    beta_s_25 = float(config.get('revil_beta_s_25', 0.53e-8))
    alpha_T = float(config.get('revil_alpha_T', 0.040))

    T_arr = np.asarray(T, dtype=float)
    beta_s = beta_s_25 * (1.0 + alpha_T * (T_arr - 25.0))
    beta_s = np.clip(beta_s, 0.0, None)

    return ((2.0 / 3.0)
            * rho_g
            * beta_s
            * CEC
            * (S_total ** (n_arr - 1.0)))


def revil2002_bulk_conductivity(sigma_fluid, F, sigma_S, t_plus=0.39):
    """
    Full Revil et al. (2002) bulk conductivity via Eqs. 11-13 with
    Dukhin-number rollover across the isoconductivity point.

    Reference: Revil, A., Hermitte, D., Spangenberg, E. & Cochemé,
    J.J. (2002). Electrical properties of zeolitized volcaniclastic
    materials. J. Geophys. Res. 107(B8), 2168. Equations 11, 12, 13
    (page 8 of the published paper).

    Their framework decomposes the bulk rock conductivity into cation
    and anion contributions:

        sigma = sigma_(+) + sigma_(-)     (Eq. 3)
        sigma_(-) = (sigma_f / F) * (1 - t_+)  (only anions in bulk
                                                fluid; cations are
                                                mostly surface-bound)

    and combines them into a single relation in terms of the Dukhin
    number xi = sigma_S / sigma_f:

        sigma / sigma_f = (1/F) * H(xi)                        (Eq. 11)

    with H(xi) piecewise in xi:

        xi_+ = 1 - t_+     (isoconductivity point; continuity of H)

        H(xi) = 1 - t_+ + F*xi                                 (Eq. 13)
            for xi >= xi_+  (high-salinity regime)
            This is equivalent to the simple additive form
            sigma = sigma_S + (sigma_f / F) * (1 - t_+).

        H(xi) = 1 - t_+ + F*xi
              + (1/2) * (1 - t_+ - xi)
                * [1 - sqrt((1 - xi/t_+)^2 + 4*F*xi/t_+)]      (Eq. 12)
            for xi < xi_+  (low-salinity / close-to-isoconductivity)
            The correction term (1/2)(1-t_+-xi)*[1 - sqrt(...)]
            subtracts from the high-salinity asymptote, reflecting
            the reduced surface-conduction contribution when
            xi < xi_+.

    Both branches evaluate to the same value at xi = xi_+ = 1 - t_+
    (the correction term's (1-t_+-xi) factor vanishes), so H is
    continuous there.

    Parameters
    ----------
    sigma_fluid : ndarray
        Pore-fluid (brine) conductivity [S/m].
    F : ndarray
        Formation factor [-]. Usually F = 1 / phi^m in Archie.
    sigma_S : ndarray
        Intrinsic surface conductivity of the clay fraction [S/m],
        as returned by _revil_surface_conductivity (Eq. 7).
    t_plus : float, default 0.39
        Cation transport number in the pore fluid. Default 0.39 is
        the Na+ transport number in aqueous NaCl at 25 C (Robinson
        & Stokes). Mildly temperature-dependent; the default is
        adequate for geothermal applications within its ~10%
        uncertainty.

    Returns
    -------
    sigma_bulk : ndarray
        Bulk rock electrical conductivity [S/m].

    Notes
    -----
    Unlike the additive form used for WS and Levy, this function
    returns the BULK conductivity directly -- there is no "surface
    contribution" to add to sigma_fluid/F. The pore-fluid contribution
    (1-t_+) * sigma_f/F is already baked into H(xi).

    For Yuzawa production (which uses the additive framework), the
    _revil_surface_conductivity helper's output is used as sigma_s
    in sigma_bulk = sigma_f/F + sigma_s, which approximates this
    full result at xi >> xi_+.
    """
    sigma_f = np.asarray(sigma_fluid, dtype=float)
    F_arr = np.asarray(F, dtype=float)
    sigma_S_arr = np.asarray(sigma_S, dtype=float)
    t_p = float(t_plus)

    # Dukhin number
    sigma_f_safe = np.where(sigma_f > 0.0, sigma_f, 1e-30)
    xi = sigma_S_arr / sigma_f_safe
    xi_plus = 1.0 - t_p  # isoconductivity point

    # High-salinity branch (Eq. 13): H = (1 - t_+) + F * xi
    H_high = (1.0 - t_p) + F_arr * xi

    # Low-salinity branch (Eq. 12): H = H_high + correction
    # correction = (1/2) * (1 - t_+ - xi) * [1 - sqrt((1 - xi/t_+)^2
    #                                                + 4 F xi / t_+)]
    # Guard against xi/t_+ making the sqrt argument negative (it
    # can't mathematically, but guard numerically).
    xi_over_tp = xi / max(t_p, 1e-12)
    sqrt_arg = (1.0 - xi_over_tp) ** 2 + 4.0 * F_arr * xi / max(t_p, 1e-12)
    sqrt_arg = np.maximum(sqrt_arg, 0.0)
    correction = 0.5 * (1.0 - t_p - xi) * (1.0 - np.sqrt(sqrt_arg))
    H_low = H_high + correction

    # Piecewise assembly
    H = np.where(xi >= xi_plus, H_high, H_low)

    # Eq. 11: sigma/sigma_f = H / F
    return sigma_f * H / F_arr


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
    Compute per-phase fluid conductivity at each node.

    Computes sigma_liq and sigma_vap independently at their own
    salinity and density, using the three-regime vapor model
    (Watanabe > 450 kg/m3, density model 200-450, Sinmyo-Keppler
    < 200). Also returns a pre-mixed sigma_fluid using per-phase
    Archie (S_liq^n * sigma_liq + S_vap^n * sigma_vap), which is
    used by the injection front fix. The actual mixing law (Glover
    or HS) is applied in _compute_hydrothermal_domain.
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

    # Density is only physical where the phase is present. CSMP++ writes
    # rho = 0 for the absent phase in single-phase nodes, which would
    # otherwise fail the range check.
    liq_present = Sliq > 0.0
    vap_present = Svap > 0.0
    if np.any(liq_present):
        _validate_range(rhol[liq_present],
                        "density_liquid (where S_liq>0)", 0.1, 2000.0)
    if np.any(vap_present):
        _validate_range(rhov[vap_present],
                        "density_vapor (where S_vap>0)", 0.001, 2000.0)

    mixing_law = config.get('mixing_law', 'glover')
    print(f"  Fluid conductivity (mixing_law={mixing_law})...")

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

    # Pre-mixed sigma_fluid using per-phase Archie. This is used by
    # the injection front fix (step 5b) and as a fallback. The actual
    # mixing law (Glover or HS) is applied in the hydrothermal domain.
    n_exp = config.get('saturation_exponent_n', 2.0)
    sigma_fluid = Sliq**n_exp * sig_liq + Svap**n_exp * sig_vap

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


def _compute_hydrothermal_domain(T, phi, sigma_fluid, sig_liq, sig_vap,
                                 sigma_rock, Sliq, Svap,
                                 region_n, hm, nodal_data, config,
                                 magma_active, cvf_data):
    """
    Bulk conductivity for the hydrothermal domain (no melt).

    Two mixing law options (config['mixing_law']):

    'glover' (default): Generalized Archie's law (Glover, 2010,
        Geophysics 75(6), E247-E265), applied to three phases
        (rock, liquid, vapor) with the conservation-of-connectedness
        unity constraint (Glover 2010, Eq. 2; Glover 2009, TLE 28(1),
        82-85):

            sum_i(phi_i^m_i) = 1

        The bulk conductivity is:

            sigma_bulk = sigma_rock * phi_s^m_solid
                       + sigma_liq * phi_liq^m_liq
                       + sigma_vap * phi_vap^m_vap
                       + sigma_surface

        where phi_s = 1 - phi_eff, phi_liq = phi_eff * S_liq,
        phi_vap = phi_eff * S_vap. Regional m values (m_arr, from
        config) are Archie fluid cementation exponents and are used
        as m_liq and m_vap. The rock cementation exponent m_solid is
        derived from the unity constraint via
        cementation_exponents_hydrothermal. This matches the
        magmatic-domain treatment in cementation_exponents_samrock,
        applying the same Glover framework across both domains.

    'hashin_shtrikman': Two-step mixing. First combine liquid and
        vapor via the HS upper bound (assumes brine is the connected
        phase), then mix with rock via the formation factor:

            sigma_fluid = HS_upper(sigma_liq, sigma_vap, S_liq)
            sigma_bulk = (1/F) * sigma_fluid + sigma_surface

    Surface conduction (sigma_surface) is added in both cases using the
    configured surface_conduction_model. Available options:
    'waxman_smits' (classical WS with sigma_w-dependent B),
    'revil' (Revil 2002 DC-calibrated beta_s),
    'levy' (Levy 2018 Eq. 16),
    'waxman_smits_revil' (WS outside, Revil 2002 in clay_cap_regions),
    'waxman_smits_levy' (WS outside, Levy in clay_cap_regions).

    Parameters
    ----------
    T : ndarray
        Temperature [deg C].
    phi : ndarray
        Porosity.
    sigma_fluid : ndarray
        Pre-computed effective fluid conductivity [S/m] (used by HS path).
    sig_liq, sig_vap : ndarray
        Per-phase fluid conductivities [S/m] (used by Glover path).
    sigma_rock : ndarray
        Rock matrix conductivity [S/m] (used by Glover path).
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

    # Surface conduction: five options (see module docstring for details).
    # All formulations are additive terms that enter alongside the mixing
    # law, and all use phi_eff (halite-corrected porosity).
    CEC = build_CEC_array(region_n[hm], config)
    rho_g = build_region_property(region_n[hm], config, 'grain_density', n_h)
    T_h = T[hm]
    S_total = np.clip(Sliq[hm] + Svap[hm], 0.0, 1.0)
    sigma_w_h = sig_liq[hm]   # pore-fluid conductivity, used by classical WS

    surface_model = str(
        config.get('surface_conduction_model', 'waxman_smits_levy')).lower()

    def _ws(mask):
        return _waxman_smits_surface_conductivity(
            phi_eff[mask], F[mask], S_total[mask], n_arr[mask],
            rho_g[mask], CEC[mask], T_h[mask], sigma_w_h[mask], config)

    def _revil(mask):
        return _revil_surface_conductivity(
            phi_eff[mask], F[mask], S_total[mask], n_arr[mask],
            rho_g[mask], CEC[mask], T_h[mask], config)

    def _levy(mask):
        return _levy_surface_conductivity(
            phi_eff[mask], CEC[mask], T_h[mask], m_arr[mask], config)

    all_mask = np.ones(n_h, dtype=bool)

    if surface_model == 'waxman_smits':
        sig_surface = _ws(all_mask)
        print(f"    Surface model: Waxman-Smits (classical, with "
              f"B(sigma_w)) everywhere ({n_h} nodes)")
    elif surface_model == 'revil':
        sig_surface = _revil(all_mask)
        print(f"    Surface model: Revil (2002) everywhere "
              f"({n_h} nodes)")
    elif surface_model == 'levy':
        sig_surface = _levy(all_mask)
        print(f"    Surface model: Levy (2018) everywhere "
              f"({n_h} nodes)")
    elif surface_model in ('waxman_smits_revil', 'waxman_smits_levy'):
        clay_cap_regions = [int(rid)
                            for rid in config.get('clay_cap_regions', [])]
        clay_mask = np.isin(region_n[hm], clay_cap_regions)
        sig_surface = np.empty(n_h, dtype=float)

        if len(clay_cap_regions) == 0:
            warnings.warn(
                f"surface_conduction_model='{surface_model}' but "
                f"clay_cap_regions is empty; Waxman-Smits is applied "
                f"everywhere (no clay-cap substitution).")

        if np.any(~clay_mask):
            sig_surface[~clay_mask] = _ws(~clay_mask)
        if np.any(clay_mask):
            if surface_model == 'waxman_smits_revil':
                sig_surface[clay_mask] = _revil(clay_mask)
            else:  # waxman_smits_levy
                sig_surface[clay_mask] = _levy(clay_mask)

        clay_model = ('Revil (2002)' if surface_model == 'waxman_smits_revil'
                      else 'Levy (2018)')
        print(f"    Surface model: Waxman-Smits on "
              f"{int(np.sum(~clay_mask))} non-clay nodes, "
              f"{clay_model} on {int(np.sum(clay_mask))} clay-cap nodes")
    else:
        raise ValueError(
            f"Unknown surface_conduction_model: '{surface_model}'. Use "
            "'waxman_smits', 'revil', 'levy', 'waxman_smits_revil', or "
            "'waxman_smits_levy'.")

    # Mixing law: Glover (2010) three-phase or Hashin-Shtrikman
    mixing_law = config.get('mixing_law', 'glover')

    if mixing_law == 'glover':
        # Generalized Archie (Glover, 2010, Geophysics 75(6), E247-E265):
        # three phases (rock, liquid, vapor) mixed additively, with the
        # per-phase cementation exponents linked by the "conservation
        # of connectedness" constraint sum_i(phi_i^m_i) = 1
        # (Glover 2010 Eq. 2; Glover et al. 2000, EPSL 180, 369-383;
        # Glover 2009, TLE 28(1), 82-85).
        #
        # Regional m values (m_arr) are Archie cementation exponents
        # describing the FLUID pathway through the pore network, so
        # they are used as m_liq and m_vap. The rock cementation
        # exponent m_solid is then derived from the unity constraint,
        # matching the magmatic-domain treatment in
        # cementation_exponents_samrock.
        phi_solid_h = 1.0 - phi_eff
        phi_liq_h = phi_eff * Sliq[hm]
        phi_vap_h = phi_eff * Svap[hm]

        m_solid_h, m_liq_h, m_vap_h = cementation_exponents_hydrothermal(
            phi_solid_h, phi_liq_h, phi_vap_h, m_arr)

        # Compute phase contributions (connectednesses G_i = phi_i^m_i)
        Gs = np.power(np.clip(phi_solid_h, 0.0, 1.0), m_solid_h)
        Gl = np.power(np.clip(phi_liq_h, 1e-10, 1.0), m_liq_h)
        Gv = np.power(np.clip(phi_vap_h, 1e-10, 1.0), m_vap_h)

        # Effective solid-phase conductivity: silicate rock + halite
        # precipitate where S_halite > 0 (Mapother et al. 1950). Most
        # nodes have S_halite = 0 and pass through as sigma_rock; see
        # sigma_solid_phase for details.
        sigma_solid_h = sigma_solid_phase(
            T[hm], phi_h[hm], S_halite=Shal_h, sigma_rock=sigma_rock[hm])

        sigma_bulk_hydro = (sigma_solid_h * Gs
                          + sig_liq[hm] * Gl
                          + sig_vap[hm] * Gv
                          + sig_surface)

    elif mixing_law == 'hashin_shtrikman':
        # Two-step: HS upper bound for fluid mixing, then formation
        # factor for fluid-rock mixing.
        sig_fluid_hs = hashin_shtrikman_upper(
            sig_liq[hm], sig_vap[hm], Sliq[hm])
        sig_fluid_term = (1.0 / F) * sig_fluid_hs
        sigma_bulk_hydro = sig_fluid_term + sig_surface

    else:
        raise ValueError(
            f"Unknown mixing_law: '{mixing_law}'. "
            f"Use 'glover' or 'hashin_shtrikman'.")

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
            T, phi, sigma_fluid, sig_liq, sig_vap, sigma_rock,
            Sliq, Svap, region_n, hm,
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
    print("conductivity.py -- self-test suite")
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

    # --- 3. Rock conductivity (Olhoeft 1981, Watanabe 2022 Eq. 2) ---
    print("\n3. Rock conductivity (Olhoeft 1981 / Watanabe 2022 Eq. 2)")
    sig_25 = sigma_rock_olhoeft(25.0)
    check("25C -> ~1e-13 S/m",
          1e-14 < float(sig_25) < 1e-12,
          f"got {float(sig_25):.3e}")

    sig_300 = sigma_rock_olhoeft(300.0)
    check("300C -> ~1e-9 to 1e-8 S/m",
          1e-10 < float(sig_300) < 1e-7,
          f"got {float(sig_300):.3e}")

    sig_800 = sigma_rock_olhoeft(800.0)
    check("800C -> ~1e-3 to 1e-2 S/m",
          1e-4 < float(sig_800) < 1e-1,
          f"got {float(sig_800):.3e}")

    # Monotonicity
    T_rock = np.array([100.0, 300.0, 500.0, 800.0])
    sig_rock_arr = sigma_rock_olhoeft(T_rock)
    check("sigma_rock increases with T",
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
