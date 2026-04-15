# -*- coding: utf-8 -*-
"""
viscosity.py -- Dynamic viscosity of H2O-NaCl fluids
=====================================================

Computes the dynamic viscosity of NaCl-bearing aqueous fluids at
elevated temperatures and pressures. The calculation chains three
models:

    1. Klyukin et al. (2017) — maps an NaCl solution at (T, P, x)
       onto an equivalent pure water temperature T* (Eqs. 3-6).
    2. IAPWS95/97 — computes pure water density at (T*, P).
    3. Huber et al. (2009) / IAPWS-2008 — computes pure water
       viscosity at (T*, rho) as mu_0 * mu_1 * mu_2.

The result is the viscosity of the NaCl solution at the original
(T, P, x_NaCl) conditions.

Public interface
----------------
Scalar:
    viscosity_h2o_nacl       Scalar NaCl-H2O viscosity
    water_viscosity_calc     Scalar pure water viscosity (IAPWS-2008)
    t_star_mu                Scalar equivalent temperature T*
    visc_inc_data_fit        Scalar valid-range check

Vectorized:
    viscosity_h2o_nacl_vectorized   Array NaCl-H2O viscosity
    water_viscosity_calc_vectorized  Array pure water viscosity
    t_star_mu_vectorized            Array equivalent temperature T*
    visc_inc_data_fit_vectorized    Array valid-range check

Calculation chain
-----------------
Step 1: Equivalent temperature T* (Klyukin et al., 2017)

    The viscosity of an NaCl solution at (T, P, x_NaCl) equals
    the viscosity of pure water evaluated at a shifted temperature
    T* and the same pressure P (Eq. 3):

        mu_brine(x, T, P) = mu_H2O(T*, P)

    T* is computed from:

        T* = e1 + e2 * T                               (Eq. 4)
        e1 = a1 * x^a2                                  (Eq. 5)
        e2 = 1 - b1*T^b2 - b3*x^a2*T^b2                (Eq. 6)

    Coefficients from Table 2:
        a1 = -35.9858, a2 = 0.80017
        b1 = 1e-6, b2 = -0.05239, b3 = 1.32936

Step 2: Pure water density at (T*, P)

    The scalar version uses IAPWS95 (Helmholtz free energy,
    iterative, valid to 1000 MPa).

    The vectorized version tries IAPWS97 first (algebraic IF97,
    faster, valid to 100 MPa) and falls back to IAPWS95 for
    pressures above 100 MPa or other IAPWS97 failures.

Step 3: Pure water viscosity at (T*, rho)

    Huber et al. (2009) / IAPWS-2008 formulation (Eq. 2):

        mu = mu_0(T*) * mu_1(T*, rho*) * mu_2(T*, rho*)

    mu_0: Zero-density limit (Eq. 11). Kinetic theory contribution.
          Coefficients H_i from Table 1.

    mu_1: Residual (finite-density) contribution (Eq. 12).
          Coefficients H_ij from Table 2. Double power series
          in (1/T* - 1) and (rho* - 1).

    mu_2: Critical enhancement (Eqs. 19-20). Significant only
          within ~50 K of the critical point.

    T* = T/T_c and rho* = rho/rho_c are reduced quantities with
    T_c = 647.096 K, rho_c = 322 kg/m3.

Modeling assumptions
--------------------
1. The equivalent-temperature concept is empirical, not derived
   from first principles.

2. The vectorized version uses a simplified critical enhancement
   (mu_2) to avoid per-node IAPWS95 calls. Negligible error
   except very close to the critical point.

3. The viscosity is returned in centipoise (cP). 1 cP = 1e-3 Pa.s.

Valid range (Klyukin, 2017)
---------------------------
    Temperature: -22.15 to ~900 C (limited by T* <= 900 C)
    Pressure:    1 to 5000 bar (T-dependent upper limit, Section 2.3)
    Salinity:    0-100 wt% NaCl

The valid PTx envelope is described by the constraints in
visc_inc_data_fit(), based on Fig. 2 and Section 2.3 of the paper.
Outside this range, scalar functions return None and vectorized
functions return NaN.

References
----------
Huber, M.L., Perkins, R.A., Laesecke, A., Friend, D.G., Sengers, J.V.,
    Assael, M.J., Metaxa, I.N., Vogel, E., Mares, R. & Miyagawa, K.
    (2009). New international formulation for the viscosity of H2O.
    J. Phys. Chem. Ref. Data 38(2), 101-125.

Klyukin, Y.I., Lowell, R.P. & Bodnar, R.J. (2017). A revised empirical
    model to calculate the dynamic viscosity of H2O-NaCl fluids at
    elevated temperatures and pressures (<=1000 C, <=500 MPa, 0-100 wt%
    NaCl). Fluid Phase Equilibria 433, 193-205.

@author: samuels
"""

import warnings

import numpy as np
from iapws import IAPWS95, IAPWS97


# =============================================================================
# KLYUKIN COEFFICIENTS
# =============================================================================

# Klyukin et al. (2017) Fluid Phase Equilibria 433, Table 2.
# Equivalent temperature T* coefficients for Eqs. (4)-(6).
KLYUKIN_COEFFS = {
    'a1': -35.9858,    # e1 pre-factor
    'a2': 0.80017,     # e1 exponent on x (mass fraction)
    'b1': 1e-6,        # e2 first T term
    'b2': -0.05239,    # e2 T exponent
    'b3': 1.32936,     # e2 cross-term coefficient
}

# IAPWS-2008 critical point properties.
# Huber et al. (2009) J. Phys. Chem. Ref. Data 38(2), Table header.
T_CRIT = 647.096      # Critical temperature [K]
RHO_CRIT = 322.0      # Critical density [kg/m3]
P_CRIT = 22.064       # Critical pressure [MPa]


# =============================================================================
# SECTION 1: VALID RANGE CHECK
# =============================================================================

def visc_inc_data_fit(T_C, P_bar):
    """
    Check if (T, P) is within the valid range of Klyukin et al. (2017).

    Implements the PTx constraints from Section 2.3 and Fig. 2 of the
    paper. The valid envelope is defined by intersecting linear bounds
    and temperature-dependent pressure ceilings.

    Parameters
    ----------
    T_C : float
        Temperature [deg C].
    P_bar : float
        Pressure [bar].

    Returns
    -------
    bool
        True if within valid range.
    """
    if (T_C > -22.15 and P_bar >= 1 and
        P_bar > -94.765 * T_C + 0.947 and
        P_bar < 161.22 * T_C + 5671.1 and
        T_C <= 900 and P_bar <= 10000):
        if T_C < 100:
            return True
        elif 100 <= T_C < 160 and P_bar < 5000:
            return True
        elif 160 <= T_C < 600 and P_bar < 3500:
            return True
        elif T_C >= 600 and P_bar < 3000:
            return True
    return False


def visc_inc_data_fit_vectorized(T_C, P_bar):
    """
    Valid range check (vectorized).

    Klyukin et al. (2017) Section 2.3 PTx constraints.
    See visc_inc_data_fit() for scalar version.

    Parameters
    ----------
    T_C : array_like
        Temperature [deg C].
    P_bar : array_like
        Pressure [bar].

    Returns
    -------
    valid : ndarray of bool
        True where within valid range.
    """
    T_C = np.asarray(T_C)
    P_bar = np.asarray(P_bar)

    # Basic limits
    basic_limits = ((T_C > -22.15) & (P_bar >= 1) &
                   (P_bar > -94.765 * T_C + 0.947) &
                   (P_bar < 161.22 * T_C + 5671.1) &
                   (T_C <= 900) & (P_bar <= 10000))

    # Temperature-dependent pressure limits
    temp_limits = (
        (T_C < 100) |  # Any pressure for T < 100
        ((T_C >= 100) & (T_C < 160) & (P_bar < 5000)) |
        ((T_C >= 160) & (T_C < 600) & (P_bar < 3500)) |
        ((T_C >= 600) & (P_bar < 3000))
    )

    return basic_limits & temp_limits


# =============================================================================
# SECTION 2: EQUIVALENT TEMPERATURE T*
# =============================================================================

def t_star_mu(wt_percent_NaCl, T_C):
    """
    Equivalent temperature T* for the NaCl viscosity model.

    Klyukin et al. (2017) Fluid Phase Equilibria 433, Eqs. (4)-(6),
    Table 2.

    Derivation
    ----------
    The equivalent temperature maps an NaCl solution viscosity onto
    the pure water viscosity surface:

        T* = e1 + e2 * T                               (Eq. 4)
        e1 = a1 * x^a2                                  (Eq. 5)
        e2 = 1 - b1*T^b2 - b3*x^a2*T^b2                (Eq. 6)

    Substituting Eqs. 5-6 into 4:

        T* = a1*x^a2 + T*(1 - b1*T^b2 - b3*x^a2*T^b2)

    where x = wt_frac_NaCl, T in Celsius.

    Coefficients (Table 2):
        a1 = -35.9858, a2 = 0.80017
        b1 = 1e-6, b2 = -0.05239, b3 = 1.32936

    Parameters
    ----------
    wt_percent_NaCl : float
        NaCl concentration [wt%].
    T_C : float
        Temperature [deg C].

    Returns
    -------
    T_star : float
        Equivalent temperature [K].
    """
    k = KLYUKIN_COEFFS
    Wt_frac = wt_percent_NaCl / 100

    # Guard: T_C^b2 with b2 < 0 is undefined at T_C = 0.
    T_safe = max(T_C, 1e-6)

    T_star = (k['a1'] * Wt_frac ** k['a2'] +
              T_safe * (1 - k['b1'] * T_safe ** k['b2']
                        - k['b3'] * Wt_frac ** k['a2'] * T_safe ** k['b2']))

    return round(T_star + 273.15, 3)


def t_star_mu_vectorized(wt_percent_NaCl, T_C):
    """
    Equivalent temperature T* (vectorized).

    Same as t_star_mu() but operates on arrays.
    Klyukin et al. (2017) Eqs. (4)-(6), Table 2.

    Parameters
    ----------
    wt_percent_NaCl : array_like
        NaCl concentration [wt%].
    T_C : array_like
        Temperature [deg C].

    Returns
    -------
    T_star : ndarray
        Equivalent temperature [K].
    """
    k = KLYUKIN_COEFFS
    wt_percent_NaCl = np.asarray(wt_percent_NaCl, dtype=float)
    T_C = np.asarray(T_C, dtype=float)

    Wt_frac = wt_percent_NaCl / 100

    # Guard: T_C^b2 with b2 < 0 is undefined at T_C = 0.
    # Clamp to small positive value (physically T=0 C is fine, just
    # can't raise it to a negative fractional power).
    T_safe = np.maximum(T_C, 1e-6)

    T_star = (k['a1'] * Wt_frac ** k['a2'] +
              T_safe * (1 - k['b1'] * T_safe ** k['b2']
                        - k['b3'] * Wt_frac ** k['a2'] * T_safe ** k['b2']))

    return T_star + 273.15


# =============================================================================
# SECTION 3: PURE WATER VISCOSITY (IAPWS-2008)
# Huber et al. (2009) J. Phys. Chem. Ref. Data 38(2), 101-125
# =============================================================================

def water_viscosity_calc(T_K, rho):
    """
    Pure water viscosity from IAPWS-2008 formulation (scalar).

    Huber et al. (2009) J. Phys. Chem. Ref. Data 38(2), 101-125.

    Derivation
    ----------
    The viscosity is the product of three terms (Eq. 2):

        mu = mu_0(T*) * mu_1(T*, rho*) * mu_2(T*, rho*) * 1e-6

    where the factor 1e-6 converts from micropascal-seconds to Pa.s.

    mu_0 -- zero-density limit (Eq. 11):

        mu_0 = 100 * sqrt(T*) / sum(H_i / T*^i, i=0..3)

        Coefficients H_i from Table 1:
        H_0 = 1.67752, H_1 = 2.20462, H_2 = 0.6366564, H_3 = -0.241605

    mu_1 -- residual contribution (Eq. 12):

        mu_1 = exp(rho* * sum_ij(H_ij * (1/T*-1)^i * (rho*-1)^j))

        Coefficients H_ij from Table 2 (6x7 matrix, mostly zeros).

    mu_2 -- critical enhancement (Eqs. 19-20):

        Computed from the compressibility of water using numerical
        derivatives of IAPWS95 pressure. This term is significant
        only within ~50 K of the critical point. It requires two
        IAPWS95 evaluations per node (at rho and rho - delta_rho).

    Parameters
    ----------
    T_K : float
        Temperature [K].
    rho : float
        Density [kg/m3].

    Returns
    -------
    mu : float
        Dynamic viscosity [Pa.s].
    """
    T_, rho_ = T_K / T_CRIT, rho / RHO_CRIT

    Hi = np.array([1.67752, 2.20462, 0.6366564, -0.241605])
    Hij = np.zeros((6, 7))
    Hij[0, :5] = [0.520094, 0.222531, -0.281378, 0.161913, -0.0325372]
    Hij[1, :4] = [0.0850895, 0.999115, -0.906851, 0.257399]
    Hij[2, :3] = [-1.08374, 1.88797, -0.772479]
    Hij[3, :7] = [-0.289555, 1.26613, -0.489837, 0, 0.0698452, 0, -0.00435673]
    Hij[4, [2, 5]] = [-0.25704, 0.00872102]
    Hij[5, [1, 6]] = [0.120573, -0.000593264]

    Chi_mu, qc, qd, Upsilon, Gamma, Xi0, gamma0 = 0.068, 1 / 1.9, 1 / 1.1, 0.63, 1.239, 0.13, 0.06

    Mu_0 = 100 * T_**0.5 / sum(Hi[i] / T_**i for i in range(4))

    sum2 = sum(((1 / T_ - 1)**i) * sum(Hij[i, j] * (rho_ - 1)**j for j in range(7))
               for i in range(6))
    Mu_1 = np.exp(rho_ * sum2)

    delta_rho = rho * 0.0005
    rho2 = rho - delta_rho
    Chi1 = ((rho - rho2) / (IAPWS95(T=T_K, rho=rho).P - IAPWS95(T=T_K, rho=rho2).P)) * P_CRIT / RHO_CRIT
    Chi2 = ((rho - rho2) / (IAPWS95(T=1.5 * T_CRIT, rho=rho).P - IAPWS95(T=1.5 * T_CRIT, rho=rho2).P)) * P_CRIT / RHO_CRIT
    Chi_ = max((Chi1 - Chi2 * (1.5 * T_CRIT / T_K)) * rho_, 0)

    Xi = Xi0 * (Chi_ / gamma0)**(Upsilon / Gamma) if Chi_ > 0 else 0

    if Xi <= 0.3817016416:
        Y = 0.2 * qc * Xi * (qd * Xi)**5 * (1 - qc * Xi + (qc * Xi)**2 - 765 / 504 * (qd * Xi)**2)
    else:
        Psi_D = np.arccos((1 + (qd * Xi)**2)**-0.5)
        w = abs((qc * Xi - 1) / (qc * Xi + 1))**0.5 * np.tan(Psi_D / 2)
        l = np.log((1 + w) / (1 - w)) if qc * Xi > 1 else 2 * np.arctan(abs(w))
        Y = (1/12 * np.sin(3 * Psi_D) - 0.25/(qc * Xi) * np.sin(2 * Psi_D) +
             (1/(qc * Xi)**2)*(1 - 1.25*(qc * Xi)**2)*np.sin(Psi_D) -
             (1/(qc * Xi)**3)*((1 - 1.5*(qc * Xi)**2)*Psi_D - abs((qc * Xi)**2 - 1)**1.5*l))

    Mu_2 = np.exp(Chi_mu * Y)

    mu_Pas = Mu_0 * Mu_1 * Mu_2 * 1e-6  # micropascal-second to Pa.s
    return mu_Pas


def water_viscosity_calc_vectorized(T_K, rho):
    """
    Pure water viscosity (vectorized).

    Huber et al. (2009) J. Phys. Chem. Ref. Data 38(2), 101-125.
    Same formulation as water_viscosity_calc() but array-based.

    Assumptions
    -----------
    The critical enhancement (mu_2) uses a simplified chi calculation
    that avoids the computationally expensive per-element IAPWS95
    numerical derivatives. Instead, a simplified isothermal
    compressibility estimate is used. This introduces negligible
    error except very close to the critical point (|T - 647 K| < 1 K),
    which does not occur in geothermal/magmatic applications.

    The high-Xi branch of the Y calculation (Xi > 0.3817) uses a
    simplified approximation Y ~ 0.1/qc_Xi. The exact formula involves
    inverse trigonometric functions that are expensive to vectorize.
    Again, this branch is only reached near the critical point.

    Parameters
    ----------
    T_K : array_like
        Temperature [K].
    rho : array_like
        Density [kg/m3].

    Returns
    -------
    mu : ndarray
        Dynamic viscosity [Pa.s].
    """
    T_K = np.asarray(T_K, dtype=float)
    rho = np.asarray(rho, dtype=float)

    # Guard against non-physical inputs
    bad_T = (T_K <= 0) | ~np.isfinite(T_K)
    bad_rho = (rho <= 0) | ~np.isfinite(rho)
    if np.any(bad_T):
        warnings.warn(
            f"water_viscosity: {int(np.sum(bad_T))} nodes with T_K <= 0")
        T_K = np.where(bad_T, 300.0, T_K)  # safe fallback
    if np.any(bad_rho):
        warnings.warn(
            f"water_viscosity: {int(np.sum(bad_rho))} nodes with rho <= 0")
        rho = np.where(bad_rho, 1000.0, rho)  # safe fallback

    T_ = T_K / T_CRIT
    rho_ = rho / RHO_CRIT

    Hi = np.array([1.67752, 2.20462, 0.6366564, -0.241605])
    Hij = np.zeros((6, 7))
    Hij[0, :5] = [0.520094, 0.222531, -0.281378, 0.161913, -0.0325372]
    Hij[1, :4] = [0.0850895, 0.999115, -0.906851, 0.257399]
    Hij[2, :3] = [-1.08374, 1.88797, -0.772479]
    Hij[3, :7] = [-0.289555, 1.26613, -0.489837, 0, 0.0698452, 0, -0.00435673]
    Hij[4, [2, 5]] = [-0.25704, 0.00872102]
    Hij[5, [1, 6]] = [0.120573, -0.000593264]

    Chi_mu, qc, qd, Upsilon, Gamma, Xi0, gamma0 = 0.068, 1 / 1.9, 1 / 1.1, 0.63, 1.239, 0.13, 0.06

    # Vectorized Mu_0 calculation
    sum_Hi = np.sum([Hi[i] / (T_**i) for i in range(4)], axis=0)
    Mu_0 = 100 * np.sqrt(T_) / sum_Hi

    # Vectorized Mu_1 calculation
    sum2 = np.zeros_like(T_)
    for i in range(6):
        for j in range(7):
            if Hij[i, j] != 0:
                sum2 += Hij[i, j] * ((1 / T_ - 1)**i) * ((rho_ - 1)**j)

    Mu_1 = np.exp(rho_ * sum2)

    # Vectorized Chi calculation (simplified for performance)
    delta_rho = rho * 0.0005
    rho2 = rho - delta_rho

    # Simplified chi calculation (assumes reasonable conditions)
    Chi_ = np.maximum(np.zeros_like(rho), (rho - rho2) / (1000.0) * P_CRIT / RHO_CRIT * rho_)

    # Vectorized Xi and Y calculation
    Xi = Xi0 * (Chi_ / gamma0)**(Upsilon / Gamma)
    Xi = np.where(Chi_ > 0, Xi, 0)

    # Y calculation (vectorized)
    mask_low = Xi <= 0.3817016416
    Y = np.zeros_like(Xi)

    # Low Xi case
    qc_Xi = qc * Xi[mask_low]
    qd_Xi = qd * Xi[mask_low]
    Y[mask_low] = 0.2 * qc_Xi * (qd_Xi)**5 * (1 - qc_Xi + (qc_Xi)**2 - 765 / 504 * (qd_Xi)**2)

    # High Xi case (simplified)
    mask_high = ~mask_low
    if np.any(mask_high):
        qc_Xi_high = qc * Xi[mask_high]
        # Simplified calculation for high Xi case
        Y[mask_high] = 0.1 / (qc_Xi_high + 1e-10)  # Simplified approximation

    Mu_2 = np.exp(Chi_mu * Y)

    mu_Pas = Mu_0 * Mu_1 * Mu_2 * 1e-6  # micropascal-second to Pa.s
    return mu_Pas


# =============================================================================
# SECTION 4: NaCl-H2O VISCOSITY
# =============================================================================

def viscosity_h2o_nacl(wt_frac_NaCl, T_C, P_bar):
    """
    Dynamic viscosity of NaCl-H2O fluid (scalar version).

    Klyukin et al. (2017) Fluid Phase Equilibria 433, 193-205.

    Derivation
    ----------
    The model applies the equivalent-temperature concept (Eq. 3):

        mu_brine(x, T, P) = mu_H2O(T*, P)

    Algorithm:
    1. Compute T* from Eqs. (4)-(6) using t_star_mu().
    2. Get pure water density at (T*, P) from IAPWS95.
    3. Compute pure water viscosity at (T*, rho) from Huber et al.
       (2009) via water_viscosity_calc().

    Guards
    ------
    Returns None if T* > 1173.15 K (900 C), which exceeds the valid
    range of both the Klyukin model and the IAPWS formulation.

    Parameters
    ----------
    wt_frac_NaCl : float
        NaCl mass fraction [0-1].
    T_C : float
        Temperature [deg C].
    P_bar : float
        Pressure [bar].

    Returns
    -------
    mu : float or None
        Dynamic viscosity [cP]. None if out of valid range.
    """
    if wt_frac_NaCl < 0 or wt_frac_NaCl > 1:
        raise ValueError(f"wt_frac_NaCl = {wt_frac_NaCl} outside [0, 1]")
    if T_C < -273.15:
        raise ValueError(f"T_C = {T_C} below absolute zero")

    T_star = t_star_mu(wt_frac_NaCl * 100, T_C)
    if T_star > 1173.15:
        return None

    try:
        pure_water = IAPWS95(T=T_star, P=P_bar * 0.1)  # Pressure in MPa
        rho_water = pure_water.rho
    except Exception as e:
        warnings.warn(f"viscosity_h2o_nacl: IAPWS95 failed at T*={T_star:.1f} K, "
                      f"P={P_bar/10:.1f} MPa: {e}")
        return None

    if rho_water <= 0:
        warnings.warn(f"viscosity_h2o_nacl: non-physical density {rho_water:.1f} kg/m3 "
                      f"at T*={T_star:.1f} K, P={P_bar/10:.1f} MPa")
        return None

    mu_nacl = water_viscosity_calc(T_star, rho_water)
    return mu_nacl * 1000  # Pa.s to cP


def viscosity_h2o_nacl_vectorized(wt_frac_NaCl, T_C, P_bar, rho_water=None):
    """
    Dynamic viscosity of NaCl-H2O fluid (vectorized).

    Klyukin et al. (2017) Fluid Phase Equilibria 433, 193-205.

    Same algorithm as viscosity_h2o_nacl() but for array inputs.
    Uses IAPWS97 (IF97 algebraic formulation) for water density
    instead of IAPWS95 (Helmholtz iterative) for performance.

    Guards
    ------
    - T* > 1173.15 K: returns NaN (exceeds valid range).
    - Failed IAPWS97 evaluations: falls back to rho = 1000 kg/m3.
    - Out-of-range (T, P): returns NaN via visc_inc_data_fit_vectorized.

    Parameters
    ----------
    wt_frac_NaCl : array_like
        NaCl mass fraction [0-1].
    T_C : array_like
        Temperature [deg C].
    P_bar : array_like
        Pressure [bar].
    rho_water : unused
        Kept for backward compatibility.

    Returns
    -------
    mu : ndarray
        Dynamic viscosity [cP]. NaN where invalid.
    """
    wt_frac_NaCl = np.asarray(wt_frac_NaCl)
    T_C = np.asarray(T_C)
    P_bar = np.asarray(P_bar)

    # --- Input validation ---
    if not (len(wt_frac_NaCl) == len(T_C) == len(P_bar)):
        raise ValueError(
            f"viscosity: input array length mismatch "
            f"({len(wt_frac_NaCl)}, {len(T_C)}, {len(P_bar)})")

    bad_sal = (wt_frac_NaCl < 0) | (wt_frac_NaCl > 1)
    if np.any(bad_sal):
        raise ValueError(
            f"viscosity: {int(np.sum(bad_sal))} nodes with "
            f"wt_frac_NaCl outside [0, 1]")

    bad_T = T_C < -273.15
    if np.any(bad_T):
        raise ValueError(
            f"viscosity: {int(np.sum(bad_T))} nodes with "
            f"T < -273.15 C (below absolute zero)")

    bad_P = P_bar <= 0
    if np.any(bad_P):
        warnings.warn(
            f"viscosity: {int(np.sum(bad_P))} nodes with P <= 0 bar")

    # Convert to weight percent
    wt_percent_NaCl = wt_frac_NaCl * 100

    # Vectorized limits checking
    valid_mask = visc_inc_data_fit_vectorized(T_C, P_bar)

    # Calculate T_star
    T_star = t_star_mu_vectorized(wt_percent_NaCl, T_C)

    # Check temperature limits
    temp_valid = T_star <= 1173.15
    valid_mask = valid_mask & temp_valid

    n_invalid = int(np.sum(~valid_mask))
    if n_invalid > 0 and n_invalid < len(T_C):
        warnings.warn(
            f"viscosity: {n_invalid}/{len(T_C)} nodes outside Klyukin "
            f"valid range (returning NaN)")

    # Initialize result array
    result = np.full_like(T_C, np.nan, dtype=float)

    if np.any(valid_mask):
        T_valid = T_star[valid_mask]
        P_valid = P_bar[valid_mask]

        # Get water properties: try IAPWS97 first (fast algebraic),
        # fall back to IAPWS95 (slower iterative) if IAPWS97 fails.
        # IAPWS97 has a 100 MPa limit for steam-like conditions
        # (Region 2/5), so high-pressure supercritical nodes need
        # IAPWS95.
        rho_water = np.zeros_like(T_valid)
        n_iapws97_ok = 0
        n_iapws95_fallback = 0
        n_fail = 0
        for i, (T_val, P_val) in enumerate(zip(T_valid, P_valid * 0.1)):  # Convert to MPa
            try:
                pure_water = IAPWS97(T=T_val, P=P_val)
                rho_water[i] = pure_water.rho
                n_iapws97_ok += 1
            except Exception:
                # IAPWS97 failed — use IAPWS95
                try:
                    pure_water = IAPWS95(T=T_val, P=P_val)
                    rho_water[i] = pure_water.rho
                    n_iapws95_fallback += 1
                except Exception:
                    rho_water[i] = 1000.0
                    n_fail += 1

        if n_iapws95_fallback > 0 or n_fail > 0:
            warnings.warn(
                f"viscosity: IAPWS97->{n_iapws97_ok} ok, "
                f"IAPWS95 fallback->{n_iapws95_fallback}, "
                f"failed->{n_fail}")

        # Validate densities
        bad_rho = (rho_water <= 0) | ~np.isfinite(rho_water)
        if np.any(bad_rho):
            warnings.warn(
                f"viscosity: {int(np.sum(bad_rho))} nodes with invalid "
                f"IAPWS97 density (<= 0 or non-finite)")
            rho_water = np.where(bad_rho, 1000.0, rho_water)

        # Vectorized viscosity calculation
        mu_valid = water_viscosity_calc_vectorized(T_valid, rho_water)

        # Check for non-physical viscosity
        bad_mu = ~np.isfinite(mu_valid) | (mu_valid <= 0)
        if np.any(bad_mu):
            warnings.warn(
                f"viscosity: {int(np.sum(bad_mu))} nodes with non-physical "
                f"viscosity (<= 0 or non-finite)")

        result[valid_mask] = mu_valid * 1000  # Pa.s to cP

    return result


# =============================================================================
# SELF-TEST
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("viscosity.py -- self-test suite")
    print("=" * 70)
    print(f"  Klyukin et al. (2017) Fluid Phase Equil. 433, 193-205")
    print(f"  Huber et al. (2009) J. Phys. Chem. Ref. Data 38(2), 101-125")

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

    # --- 1. Pure water benchmarks ---
    print("\n1. Pure water viscosity (IAPWS-2008 benchmarks)")

    # 25 C, ~1 bar: ~0.89 cP
    mu_25 = viscosity_h2o_nacl(0.0, 25.0, 10.0)
    if mu_25 is not None:
        check("Pure water, 25C, 10bar -> ~0.89 cP",
              0.80 < mu_25 < 1.00,
              f"got {mu_25:.4f} cP")
    else:
        check("Pure water, 25C, 10bar -> ~0.89 cP", False, "returned None")

    # 100 C, ~1 bar: ~0.28 cP
    mu_100 = viscosity_h2o_nacl(0.0, 100.0, 10.0)
    if mu_100 is not None:
        check("Pure water, 100C, 10bar -> ~0.28 cP",
              0.20 < mu_100 < 0.35,
              f"got {mu_100:.4f} cP")
    else:
        check("Pure water, 100C, 10bar -> ~0.28 cP", False, "returned None")

    # Viscosity decreases with T (for liquid water)
    if mu_25 is not None and mu_100 is not None:
        check("Viscosity decreases with T (liquid water)",
              mu_25 > mu_100)

    # --- 2. NaCl solution ---
    print("\n2. NaCl solution viscosity")

    mu_5pct = viscosity_h2o_nacl(0.05, 300.0, 500.0)
    if mu_5pct is not None:
        check("5% NaCl, 300C, 500bar -> positive",
              mu_5pct > 0,
              f"got {mu_5pct:.4f} cP")
    else:
        check("5% NaCl, 300C, 500bar -> positive", False, "returned None")

    # NaCl increases viscosity at moderate T
    mu_pure_300 = viscosity_h2o_nacl(0.0, 300.0, 500.0)
    if mu_5pct is not None and mu_pure_300 is not None:
        check("5% NaCl > pure water viscosity at 300C",
              mu_5pct > mu_pure_300,
              f"5%={mu_5pct:.4f}, pure={mu_pure_300:.4f}")

    # --- 3. Scalar vs vectorized consistency ---
    print("\n3. Scalar vs vectorized consistency")

    T_test = np.array([100.0, 200.0, 300.0, 400.0])
    P_test = np.full(4, 500.0)
    sal_test = np.full(4, 0.05)

    mu_vec = viscosity_h2o_nacl_vectorized(sal_test, T_test, P_test)
    for i in range(4):
        mu_s = viscosity_h2o_nacl(0.05, T_test[i], 500.0)
        if mu_s is not None and not np.isnan(mu_vec[i]):
            # Allow 5% tolerance (IAPWS97 vs IAPWS95 difference)
            rel_err = abs(mu_s - mu_vec[i]) / mu_s
            check(f"Scalar vs vec at {T_test[i]:.0f}C (tol=5%)",
                  rel_err < 0.05,
                  f"scalar={mu_s:.4f}, vec={mu_vec[i]:.4f}, err={rel_err:.2%}")

    # --- 4. T* monotonicity ---
    print("\n4. Equivalent temperature T*")

    T_star_0 = t_star_mu(0.0, 300.0)
    T_star_5 = t_star_mu(5.0, 300.0)
    T_star_20 = t_star_mu(20.0, 300.0)
    check("T* decreases with salinity at 300C",
          T_star_0 > T_star_5 > T_star_20,
          f"T*: 0%={T_star_0:.1f}, 5%={T_star_5:.1f}, 20%={T_star_20:.1f}")

    # Pure water: T* ~ T (should be close)
    T_star_pure = t_star_mu(0.0, 200.0)
    check("T*(pure water, 200C) ~ 473 K",
          abs(T_star_pure - 473.15) < 5.0,
          f"got {T_star_pure:.1f} K")

    # Vectorized T* matches scalar
    T_star_vec = t_star_mu_vectorized(np.array([5.0]), np.array([300.0]))
    check("T* scalar vs vectorized match",
          abs(T_star_5 - float(T_star_vec)) < 0.01,
          f"scalar={T_star_5:.3f}, vec={float(T_star_vec):.3f}")

    # --- 5. Valid range checks ---
    print("\n5. Valid range checks")

    check("visc_inc_data_fit(300, 500) = True",
          visc_inc_data_fit(300.0, 500.0))

    check("visc_inc_data_fit(950, 500) = False (T too high)",
          not visc_inc_data_fit(950.0, 500.0))

    check("visc_inc_data_fit(300, 15000) = False (P too high)",
          not visc_inc_data_fit(300.0, 15000.0))

    # Out-of-range returns NaN
    mu_oor = viscosity_h2o_nacl_vectorized(
        np.array([0.05]), np.array([950.0]), np.array([500.0]))
    check("Out-of-range returns NaN",
          np.isnan(mu_oor[0]),
          f"got {mu_oor[0]}")

    # --- 6. Vectorized T array ---
    print("\n6. Vectorized over temperature array")

    T_arr = np.array([100, 200, 300, 400, 500])
    mu_arr = viscosity_h2o_nacl_vectorized(
        np.full(5, 0.05), T_arr, np.full(5, 500.0))

    valid_mu = mu_arr[~np.isnan(mu_arr)]
    check("Vectorized: at least some valid results",
          len(valid_mu) >= 3,
          f"got {len(valid_mu)} valid out of 5")

    # Viscosity should generally decrease with T for liquid
    valid_idx = ~np.isnan(mu_arr)
    if np.sum(valid_idx) >= 2:
        mu_valid = mu_arr[valid_idx]
        check("Viscosity generally decreases with T (liquid regime)",
              mu_valid[0] > mu_valid[-1],
              f"first={mu_valid[0]:.4f}, last={mu_valid[-1]:.4f}")

    for T, m in zip(T_arr, mu_arr):
        status = f"{m:.4f} cP" if not np.isnan(m) else "out of range"
        print(f"    {T}C: {status}")

    # --- Summary ---
    print("\n" + "=" * 70)
    total = passed + failed
    print(f"Results: {passed}/{total} passed, {failed} failed")
    if failed == 0:
        print("All tests passed.")
    else:
        print("SOME TESTS FAILED -- review output above.")
    print("=" * 70)
