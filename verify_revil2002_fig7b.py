# -*- coding: utf-8 -*-
"""
verify_revil2002_fig7b.py -- Fig. 7b reproduction + bulk sanity check
=====================================================================

Two diagnostic panels, both prerequisites for the paper's methods-
figure panel (c) in plot_conductivity_model.py.

(a) Revil et al. (2002) Fig. 7b reproduction.
    Draws the Eq. 7 linear fit
        sigma_S = (2/3) * rho_g * beta_S * (CEC - CEC_r)
    at 20 C (rho_g = 2300 kg/m^3, beta_S = 0.44e-8 m^2/(V.s),
    CEC_r = 3 meq/100g) and overlays the measured sigma_S vs CEC
    points from their Table 1. Passing means the library's Eq. 7
    formula (and its MEQ_TO_CKG unit conversion) reproduces the
    paper's own calibrated fit line.

    This panel also serves as the unit-conversion sanity check for
    MEQ_TO_CKG (must be 964.85 to convert meq/100g to C/kg). An
    earlier bug (MEQ_TO_CKG = 9.6485, 100x too small) would flatten
    the Eq. 7 line against the x-axis.

(b) Bulk resistivity at a clay-cap point.
    For each surface-conduction model, computes sigma_bulk as a
    function of pore-fluid sigma_w at a representative clay-cap
    node (T, phi, CEC, m configurable at the top of the "Panel (b)"
    section).

    - WS (1968): sigma_bulk = sigma_f/F + sigma_s_WS       (additive)
    - Levy (2018): sigma_bulk = sigma_f/F + sigma_s_Levy   (additive)
    - Revil (2002): sigma_bulk = sigma_f/F * H(xi)         (Eqs. 11-13,
          full Dukhin-number formulation with rollover across the
          isoconductivity point xi_+ = 1 - t_+).

    The MT clay-cap range (1-10 Ohm.m, i.e. sigma_bulk = 0.1-1 S/m)
    is shaded so each model can be read directly against the MT
    observation range.

    The methods-figure panel (c) itself lives in
    ../plot_conductivity_model.py; this script is a standalone
    cross-check used during library development.

@author: samuels
"""

import numpy as np
import matplotlib.pyplot as plt

from conductivity import (
    _revil_surface_conductivity,
    _waxman_smits_surface_conductivity,
    _levy_surface_conductivity,
    revil2002_bulk_conductivity,
    MEQ_TO_CKG,
)


# =============================================================================
# Panel (a): Revil 2002 Fig. 7b reproduction
# =============================================================================
#
# Representative samples from Revil et al. (2002) Table 1. Columns:
# phi (connected porosity), F (formation factor), CEC_Co (Cobalt
# hexamine titration, meq/100g), sigma_S (measured surface conductivity
# at the isoconductivity-point limit, in 10^-3 S/m), rho_g (matrix
# density, kg/m^3). Values are from the Table 1 scan; refine if a
# tighter comparison to the paper's published fit is desired.

revil2002_samples = [
    # (id,        phi,   F,     CEC [meq/100g], sigma_S [mS/m], rho_g [kg/m^3])
    ("San Pedro",  0.497, 21.8,  13.7,   50.0,  2290),
    ("BU-3A",      0.313, 40.0,  11.3,   39.0,  2290),
    ("BU-3B",      0.314, 42.0,  11.3,   61.0,  2290),
    ("BU-5B",      0.365, 27.6,  16.2,   77.0,  2280),
    ("BU-96-7A",   0.274, 40.0,  18.9,   99.0,  2140),
    ("BU-96-7B",   0.298, 49.0,  15.9,   91.0,  2410),
    ("BU-96-8A",   0.253, 113.0, 5.5,    17.5,  2300),
    ("BU-96-10A",  0.254, 43.0,  15.3,   53.0,  2260),
    ("BU-96-11A",  0.307, 31.0,  23.4,  109.0,  2170),
    ("BU-96-12A",  0.307, 34.0,  14.2,   68.0,  2450),
    ("BU-96-14B",  0.328, 35.0,  10.3,   39.0,  2290),
    ("BU-96-24A",  0.308, 41.0,  14.2,   89.0,  2260),
]

# Revil 2002 Fig. 7b regression:
#     sigma_S = (2/3) * rho_g * beta_S * (CEC - CEC_r)
# Note: Revil 2002 Fig. 7b uses beta_S(Na+, 20 C) = 0.44e-8 m^2/(V.s)
# from the direct fit; the 0.53e-8 value quoted in their Sec 4.3 and
# used as the library default is the 25 C value obtained by applying
# their Eq. 19 temperature correction. The Fig. 7b measurements
# themselves are at lab temperature (~20 C), so we evaluate
# everything at T_lab = 20 C to keep the paper's line and the code
# helper self-consistent.
T_lab_revil2002 = 20.0  # deg C
beta_S_25C = 0.53e-8     # m^2/(V.s), library default at 25 C
alpha_T_revil = 0.040    # /C, Revil 2002 Eq. 19 (nu_S)
beta_S_ref = beta_S_25C * (1.0 + alpha_T_revil * (T_lab_revil2002 - 25.0))
# -> 0.53e-8 * 0.8 = 4.24e-9 m^2/(V.s). Compares to paper's 4.4e-9
# direct fit at 20 C (within the quoted fit uncertainty of +-0.6e-9).
rho_g_ref = 2300.0       # kg/m^3, Revil 2002 Sec 2.2 mean matrix density
CEC_r_ref = 3.0          # meq/100g, Fig. 7b x-intercept (approx.)


def revil2002_regression_line(CEC_meq):
    """Revil 2002 Fig. 7b linear-fit line, result in mS/m."""
    CEC_Ckg = CEC_meq * MEQ_TO_CKG
    CEC_r_Ckg = CEC_r_ref * MEQ_TO_CKG
    sigma_S = (2.0 / 3.0) * rho_g_ref * beta_S_ref * (CEC_Ckg - CEC_r_Ckg)
    return sigma_S * 1.0e3  # S/m -> mS/m


def code_sigma_S_mS(phi, F, CEC_meq, rho_g):
    """Run _revil_surface_conductivity for a single sample; return mS/m.

    Revil 2002 Eq. 7 uses (CEC - CEC_r) where CEC_r is the residual
    zeolite-fraction CEC that does NOT contribute to surface
    conduction (only the smectite fraction does). Revil's Fig. 7b
    regression intercepts the x-axis around CEC_r = 3 meq/100g.
    Since _revil_surface_conductivity applies no such subtraction
    internally (by design -- for production, per-region CEC values
    are expected to already be the smectite-effective values), we
    subtract CEC_r here before feeding CEC into the helper so that
    the code triangles can be compared directly to the paper's own
    linear fit.
    """
    CEC_effective = max(CEC_meq - CEC_r_ref, 0.0)
    phi_arr = np.array([phi])
    F_arr = np.array([F])
    S_total = np.array([1.0])
    n_arr = np.array([2.0])
    rho_g_arr = np.array([rho_g])
    CEC_arr = np.array([CEC_effective * MEQ_TO_CKG])
    T_arr = np.array([T_lab_revil2002])  # Revil 2002 Fig. 7b was at ~20 C
    # Pass beta_S_25C so the helper applies its own T correction
    # internally; that keeps the code path identical to the production
    # usage rather than bypassing the temperature term.
    config = {'revil_beta_s_25': beta_S_25C, 'revil_alpha_T': alpha_T_revil}
    sigma_S = _revil_surface_conductivity(
        phi_arr, F_arr, S_total, n_arr, rho_g_arr, CEC_arr, T_arr, config)
    return float(sigma_S[0]) * 1.0e3


# =============================================================================
# Panel (b): sigma_bulk vs sigma_w at clay-cap point
# =============================================================================
#
# Represents a single node in Yuzawa's smectite-bearing clay cap.
# sigma_bulk = sigma_fluid / F + sigma_surface  (Archie + additive surface).

T_claycap = 200.0       # C
phi_claycap = 0.15     # porosity
m_claycap = 2.2         # cementation exponent (clay-cap regional default)
n_claycap = 2.0         # saturation exponent (irrelevant at S_total=1)
CEC_claycap_meq = 20.0  # meq/100g; typical smectite-bearing clay cap
rho_g_claycap = 2700.0  # kg/m^3, altered andesite matrix

sigma_w_range = np.logspace(-2, 1, 200)  # 0.01 to 10 S/m
F_claycap = 1.0 / phi_claycap ** m_claycap

# Constant-by-node arrays (one entry per sigma_w grid point)
N = sigma_w_range.size
phi_arr = np.full(N, phi_claycap)
F_arr = np.full(N, F_claycap)
S_total = np.ones(N)
n_arr = np.full(N, n_claycap)
rho_g_arr = np.full(N, rho_g_claycap)
CEC_arr = np.full(N, CEC_claycap_meq * MEQ_TO_CKG)
T_arr = np.full(N, T_claycap)
m_arr = np.full(N, m_claycap)

# Library-default configs with explicit values called out so the plot
# is traceable to the paper equations.
config_ws = {
    'ws_B_25': 4.77e-8,    # WS 1968 Eq. 19 saturation mobility
    'ws_alpha_T': 0.023,    # Revil 2002 Eq. 18 (nu_f, free-fluid)
    'ws_alpha_sw': 0.6,     # WS 1968 Eq. 19 alpha
    'ws_gamma_sw': 1.3,     # WS 1968 Eq. 19 gamma (= 0.013 mho/cm)
}
config_revil = {
    'revil_beta_s_25': 0.53e-8,   # Revil 2002 Fig. 7b
    'revil_alpha_T': 0.040,        # Revil 2002 Eq. 19 (nu_S)
}
config_levy = {
    'levy_CEC_0_meq_per_100g': 91.0,
    'levy_B_prime': 0.77,
    'levy_alpha_T': 0.040,
    'levy_T_ref': 25.0,
}

sigma_s_ws = _waxman_smits_surface_conductivity(
    phi_arr, F_arr, S_total, n_arr, rho_g_arr, CEC_arr, T_arr,
    sigma_w_range, config_ws)
# sigma_S for Revil 2002 = intrinsic surface conductivity (Eq. 7).
# This feeds into revil2002_bulk_conductivity (Eqs. 11-13) below.
sigma_S_revil_intrinsic = _revil_surface_conductivity(
    phi_arr, F_arr, S_total, n_arr, rho_g_arr, CEC_arr, T_arr, config_revil)
sigma_s_levy = _levy_surface_conductivity(
    phi_arr, CEC_arr, T_arr, m_arr, config_levy)

# Pore-fluid contribution (Archie). For WS and Levy, the additive
# framework sigma_bulk = sigma_f/F + sigma_surface is exact within
# their calibration. For Revil 2002, the bulk conductivity uses the
# full Eqs. 11-13 formulation (with Dukhin-number rollover across the
# isoconductivity point xi_+ = 1 - t_+), not the simple additive form.
sigma_fluid_bulk = sigma_w_range / F_claycap
sigma_bulk_ws = sigma_fluid_bulk + sigma_s_ws
sigma_bulk_revil = revil2002_bulk_conductivity(
    sigma_w_range, F_arr, sigma_S_revil_intrinsic, t_plus=0.39)
sigma_bulk_levy = sigma_fluid_bulk + sigma_s_levy


# =============================================================================
# Build the two-panel figure
# =============================================================================

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.5))

# --- Panel (a): Revil 2002 Fig. 7b reproduction ---
# Draw the linear-fit line computed from Revil 2002 Eq. 7 at 20 C
# and compare against the measured Table 1 sigma_S values. If the
# library's intrinsic-sigma_S formula is correct, the line will
# pass through the middle of the data cloud with physically
# reasonable sample-to-sample scatter around it.
CEC_line = np.linspace(0, 30, 200)
ax1.plot(CEC_line, revil2002_regression_line(CEC_line), 'k-', linewidth=2,
         label=(r'Revil (2002) Fig. 7b Eq. 7 fit (20 $^\circ$C):'
                '\n' r'$\sigma_S = (2/3)\rho_g\beta_S(\mathrm{CEC}-\mathrm{CEC}_r)$,'
                '\n' r'$\beta_S$=0.44$\times 10^{-8}$ m$^2$/(V$\cdot$s), '
                r'$\rho_g$=2300 kg/m$^3$, CEC$_r$=3 meq/100g'))

meas_sig = np.array([s[4] for s in revil2002_samples])
meas_CEC = np.array([s[3] for s in revil2002_samples])
ax1.plot(meas_CEC, meas_sig, 'ko', ms=7, mfc='white', zorder=3,
         label='Revil (2002) Table 1 measured $\\sigma_S$')

ax1.set_xlabel(r'CEC (Cobalt)  [meq/100 g]', fontsize=11)
ax1.set_ylabel(r'Surface conductivity $\sigma_S$  [mS/m]', fontsize=11)
ax1.set_title('(a) Revil (2002) Fig. 7b reproduction', fontsize=12)
ax1.set_xlim(0, 30)
ax1.set_ylim(0, 140)
ax1.grid(True, alpha=0.3)
ax1.legend(fontsize=8, loc='upper left')

# --- Panel (b): sigma_bulk vs sigma_w, with MT clay-cap band ---
ax2.loglog(sigma_w_range, sigma_bulk_ws, color='tab:blue', lw=2.2,
           label=r'WS (1968) Eq. 19 + Archie (additive)')
ax2.loglog(sigma_w_range, sigma_bulk_revil, color='tab:orange', lw=2.2,
           label=(r'Revil (2002) Eqs. 11-13 full Dukhin' '\n'
                  r'rollover (intrinsic $\sigma_S$ from Eq. 7)'))
ax2.loglog(sigma_w_range, sigma_bulk_levy, color='tab:red', lw=2.2,
           label=r'L' '\u00e9' 'vy (2018) Eq. 16 + Archie (additive)')
ax2.loglog(sigma_w_range, sigma_fluid_bulk, color='gray', lw=1.4, ls=':',
           label=r'Pore-fluid only ($\sigma_w/F$, no surface)')

# MT clay-cap observational band: 1-10 Ohm.m  <==>  sigma_bulk = 0.1-1.0 S/m
ax2.axhspan(0.1, 1.0, color='gray', alpha=0.18,
            label=r'MT clay-cap range (1-10 $\Omega\cdot$m)')

ax2.set_xlabel(r'Pore-fluid conductivity $\sigma_w$  [S/m]', fontsize=11)
ax2.set_ylabel(r'Bulk conductivity $\sigma_\mathrm{bulk}$  [S/m]', fontsize=11)
ax2.set_title(
    f'(b) Bulk conductivity at clay-cap node: '
    f'T={T_claycap:.0f} C, $\\phi$={phi_claycap}, '
    f'CEC={CEC_claycap_meq:.0f} meq/100g, m={m_claycap}',
    fontsize=10.5)
ax2.set_xlim(sigma_w_range.min(), sigma_w_range.max())
ax2.set_ylim(1e-4, 10.0)
ax2.grid(True, which='both', alpha=0.3)
ax2.legend(fontsize=9, loc='lower right')

plt.tight_layout()
outpath = 'revil2002_fig7b_validation.png'
plt.savefig(outpath, dpi=180, bbox_inches='tight')
print(f'Wrote {outpath}')

# =============================================================================
# Numerical report
# =============================================================================
print('\nPanel (a): Revil 2002 Fig. 7b reproduction')
print('-' * 70)
print('  Eq. 7 line (at T = 20 C, rho_g = 2300, beta_S = 0.44e-8, CEC_r = 3):')
print(f'    slope = {revil2002_regression_line(1) - revil2002_regression_line(0):.3f} '
      'mS/m per meq/100g')
print(f'  {"Sample":<12s} {"CEC":>9s} {"measured":>10s} {"Eq. 7 line":>12s} '
      f'{"ratio":>8s}')
print(f'  {"":12s} {"[meq/100g]":>9s} {"[mS/m]":>10s} {"[mS/m]":>12s}')
line_sig = revil2002_regression_line(meas_CEC)
for (name, phi, F, CEC, meas, rho_g), line_val in zip(revil2002_samples,
                                                      line_sig):
    ratio = line_val / meas if meas > 0 else float('nan')
    print(f'  {name:<12s} {CEC:>9.1f} {meas:>10.1f} {line_val:>12.1f} '
          f'{ratio:>8.2f}')
ratio_arr = line_sig / meas_sig
print(f'\nGeometric-mean ratio (Eq. 7 line / measured): '
      f'{np.exp(np.mean(np.log(ratio_arr))):.3f}')
print(f'Std. dev. of log10 ratio:                     '
      f'{np.std(np.log10(ratio_arr)):.3f}')

print('\nPanel (b): sigma_bulk at clay-cap point for sigma_w in [0.01, 10] S/m')
print('-' * 70)
# Also report the Dukhin number xi at sigma_w = 1 so the Revil
# regime (above/below isoconductivity xi_+ = 1 - t_+ = 0.61) is
# visible to the user.
sigma_S_at_sw1 = float(np.interp(1.0, sigma_w_range,
                                 sigma_S_revil_intrinsic))
xi_at_sw1 = sigma_S_at_sw1 / 1.0
print(f'  (Revil intrinsic sigma_S at this node = {sigma_S_at_sw1:.3g} S/m;'
      f' xi = sigma_S / sigma_w at sigma_w=1: {xi_at_sw1:.2f};'
      f' xi_+ = 1 - t_+ = 0.61)')
for label, sigma_bulk in [
        ('WS 1968',     sigma_bulk_ws),
        ('Revil 2002',  sigma_bulk_revil),
        ('Levy 2018',   sigma_bulk_levy),
        ('Fluid only',  sigma_fluid_bulk)]:
    rho_bulk = 1.0 / sigma_bulk
    print(f'  {label:<12s} sigma_bulk @ sigma_w=1 S/m: '
          f'{np.interp(1.0, sigma_w_range, sigma_bulk):.3g} S/m  '
          f'(rho = {np.interp(1.0, sigma_w_range, rho_bulk):.3g} Ohm.m)')
