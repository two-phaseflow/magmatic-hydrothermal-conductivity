# -*- coding: utf-8 -*-
"""
Created on Tue Jun  3 10:03:54 2025

@author: samuels
"""

import numpy as np
from math import exp, log10
from iapws import IAPWS95  # for pure-water properties

def T_Star_V(xNaCl, T, P):
    """
    Effective temperature for density calculations (Driesner & Heinrich, 2007).
    Parameters:
        xNaCl: NaCl mole fraction
        T: Temperature (°C)
        P: Pressure (bar)
    Returns:
        T_star_v (°C)
    """
    xH2O = 1 - xNaCl

    n11 = -54.2958 - 45.7623 * exp(-0.000944785 * P)
    n21 = -2.6142 - 0.000239092 * P
    n22 = 0.0356828 + 4.37235e-6 * P + 2.0566e-9 * P ** 2

    n300 = 7606640 / ((P + 472.051) ** 2)
    n301 = -50 - 86.1446 * exp(-0.000621128 * P)
    n302 = 294.318 * exp(-0.00566735 * P)
    n310 = -0.0732761 * exp(-0.0023772 * P) - 5.2948e-5 * P
    n311 = -47.2747 + 24.3653 * exp(-0.00125533 * P)
    n312 = -0.278529 - 0.00081381 * P
    n30 = n300 * (exp(n301 * xNaCl) - 1) + n302 * xNaCl
    n31 = n310 * exp(n311 * xNaCl) + n312 * xNaCl

    n_oneNaCl = 330.47 + 0.942876 * P ** 0.5 + 0.0817193 * P - 2.47556e-8 * P ** 2 + 3.45052e-10 * P ** 3
    n10 = n_oneNaCl
    n12 = -n11 - n10
    n20 = 1 - n21 * n22 ** 0.5
    n_twoNaCl = -0.0370751 + 0.00237723 * P ** 0.5 + 5.42049e-5 * P + 5.84709e-9 * P ** 2 - 5.99373e-13 * P ** 3
    n23 = n_twoNaCl - n20 - n21 * (1 + n22) ** 0.5
    n1 = n10 + n11 * xH2O + n12 * xH2O ** 2
    n2 = n20 + n21 * (xNaCl + n22) ** 0.5 + n23 * xNaCl
    d = n30 * exp(n31 * T)

    T_star_v = n1 + n2 * T + d
    return T_star_v

def T_star_H(xNaCl, T, P):
    """
    Calculate effective temperature (T_star_H) for enthalpy calculations.
    Equation from Driesner & Heinrich (2007).

    Parameters:
        xNaCl : float
            NaCl mole fraction
        T : float
            Temperature in °C
        P : float
            Pressure in bar

    Returns:
        tuple: (T_star_H in K, q2h coefficient)
    """

    xH2O = 1 - xNaCl

    # Coefficients calculation
    q11 = -32.1724 + 0.0621255 * P
    q21 = -1.69513 - 0.000452781 * P - 6.04279e-8 * P**2
    q22 = 0.0612567 + 1.88082e-5 * P
    q_oneNaCl = 47.9048 - 0.00936994 * P + 6.51059e-6 * P**2
    q_twoNaCl = 0.241022 + 3.45087e-5 * P - 4.28356e-9 * P**2

    q10 = q_oneNaCl
    q12 = -q11 - q10

    q20 = 1 - q21 * np.sqrt(q22)
    q23 = q_twoNaCl - q20 - q21 * np.sqrt(1 + q22)

    # Final polynomial terms
    q1 = q10 + q11 * xH2O + q12 * xH2O**2
    q2 = q20 + q21 * np.sqrt(xNaCl + q22) + q23 * xNaCl

    T_star_H = q1 + q2 * T + 273.15  # Convert to Kelvin

    return T_star_H, q2  # q2 (q2h) returned explicitly if needed elsewhere


def T_hm(P):
    """
    Halite melting curve (Eq. 1, Driesner & Heinrich, 2007).

    Parameters:
        P : float
            Pressure in bar

    Returns:
        float: Halite melting temperature (°C)
    """

    a = 0.024726
    T_tr_NaCl = 800.7  # triple point temperature (°C)
    P_tr_NaCl = 0.0005  # triple point pressure (bar)

    T_hm = T_tr_NaCl + a * (P - P_tr_NaCl)

    return T_hm

def P_Subl(T):
    """
    Halite sublimation curve pressure.
    Equation (2) from Driesner and Heinrich (2007).

    Parameters:
        T : float
            Temperature in °C

    Returns:
        float: Sublimation pressure of halite (bar)
    """

    B_subl = 11806.1
    T_Triple_NaCl = 800.7  # triple point temperature (°C)
    P_Triple_NaCl = 0.0005  # triple point pressure (bar)

    P_subl = 10 ** (
        log10(P_Triple_NaCl)
        + B_subl * (1 / (T_Triple_NaCl + 273.15) - 1 / (T + 273.15))
    )

    return P_subl


def P_Boil(T):
    """
    Halite boiling curve pressure.
    Equation (3) from Driesner and Heinrich (2007).

    Parameters:
        T : float
            Temperature in °C

    Returns:
        float: Boiling pressure of halite (bar)
    """

    B_boil = 9418.12
    T_Triple_NaCl = 800.7  # triple point temperature (°C)
    P_Triple_NaCl = 0.0005  # triple point pressure (bar)

    P_boil = 10 ** (
        log10(P_Triple_NaCl)
        + B_boil * (1 / (T_Triple_NaCl + 273.15) - 1 / (T + 273.15))
    )

    return P_boil


def X_and_P_crit(T_in):
    """
    Critical pressure (Pcrit) and critical composition (Xcrit) calculations.
    Equations (5 and 7) from Driesner and Heinrich (2007).

    Parameters:
        T_in : float
            Temperature in °C

    Returns:
        tuple: (P_crit in bar, X_crit NaCl mole fraction)
    """
    PH2O_Crit = 220.64  # IAPWS-95 critical pressure of pure water (bar)
    TH2O_Crit = 373.946  # IAPWS-95 critical temperature of pure water (°C)

    # Coefficients as per VBA code
    C = np.zeros(15)  # C[0] unused for 1-based indexing
    CA = np.zeros(12)  # CA[0] unused

    # Assign coefficients from VBA (1-based indexing)
    C[1:15] = [
        -2.36, 0.128534, -0.023707, 0.00320089,
        -0.000138917, 0.000000102789, -4.8376e-11, 2.36,
        -0.0131417, 0.00298491, -0.000130114, 0, 0, -0.000488336
    ]

    CA[1:12] = [1, 1.5, 2, 2.5, 3, 4, 5, 1, 2, 2.5, 3]

    # Calculate intermediate coefficients
    Sum1 = 0.0
    for i in range(8, 12):
        Sum1 += C[i] * (500 - TH2O_Crit) ** CA[i]
        C[13] += C[i] * CA[i] * (500 - TH2O_Crit) ** (CA[i] - 1)
    C[12] = PH2O_Crit + Sum1

    # Coefficients for X_crit (d-array from VBA)
    d = np.zeros(12)  # d[0] unused
    d[1:12] = [
        8.0e-5, 1.0e-5, -1.37125e-7, 9.46822e-10,
        -3.50549e-12, 6.57369e-15, -4.89423e-18, 0.0777761,
        0.00027042, -4.244821e-7, 2.580872e-10
    ]

    # Critical pressure calculation (P_Crit)
    T = T_in
    Sum1 = 0.0
    if T < TH2O_Crit:
        # eq. 5a
        for j in range(1, 8):
            Sum1 += C[j] * (TH2O_Crit - T) ** CA[j]
        P_Crit = PH2O_Crit + Sum1
    elif TH2O_Crit <= T <= 500:
        # eq. 5b
        for j in range(8, 12):
            Sum1 += C[j] * (T - TH2O_Crit) ** CA[j]
        P_Crit = PH2O_Crit + Sum1
    else:
        # eq. 5c
        Sum1 = 0.0
        for j in range(12, 15):
            Sum1 += C[j] * (T - 500) ** (j - 12)
        P_Crit = Sum1

    # Critical mole fraction calculation (X_Crit)
    Sum1 = 0.0
    if TH2O_Crit <= T <= 600:
        # eq. 7a
        for j in range(1, 8):
            Sum1 += d[j] * (T - TH2O_Crit) ** j
        X_crit = Sum1
    elif T > 600:
        # eq. 7b
        for j in range(8, 12):
            Sum1 += d[j] * (T - 600) ** (j - 8)
        X_crit = Sum1
    else:
        X_crit = None  # Undefined below TH2O_Crit

    return P_Crit, X_crit



def X_L_Sat(T, P):
    """
    Halite Liquidus (Eq 8, Driesner & Heinrich, 2007)
    T: °C, P: bar
    """
    e = np.zeros(6)
    e[0] = 0.0989944 + 3.30796e-6 * P - 4.71759e-10 * P ** 2
    e[1] = 0.00947257 - 8.6646e-6 * P + 1.69417e-9 * P ** 2
    e[2] = 0.610863 - 1.51716e-5 * P + 1.1929e-8 * P ** 2
    e[3] = -1.64994 + 2.03441e-4 * P - 6.46015e-8 * P ** 2
    e[4] = 3.36474 - 1.54023e-4 * P + 8.17048e-8 * P ** 2
    e[5] = 1 - np.sum(e[:5])

    T_ratio = T / T_hm(P)
    X_sat = np.sum([e[i] * T_ratio ** i for i in range(6)])
    return min(X_sat, 1.0)

def X_V_Sat(T, P):
    """
    Vapor saturation NaCl mole fraction (X_V_Sat).
    Equation (9) from Driesner & Heinrich (2007).

    Parameters:
        T : float
            Temperature in °C
        P : float
            Pressure in bar

    Returns:
        float: Vapor saturation NaCl mole fraction
    """
    k = np.array([
        -0.235694, -0.188838, 0.004, 0.0552466, 0.66918,
        396.848, 45, -3.2719e-7, 141.699, -0.292631,
        -0.00139991, 1.95965e-6, -7.3653e-10, 0.904411,
        0.000769766, -1.18658e-6
    ])

    j = np.zeros(4)
    j[0] = k[0] + k[1] * exp(-k[2] * T)
    j[1] = k[4] + (k[3] - k[4]) / (1 + exp((T - k[5]) / k[6])) + k[7] * (T + k[8]) ** 2

    j[2] = sum(k[ii + 9] * T ** ii for ii in range(4))
    j[3] = sum(k[ii + 13] * T ** ii for ii in range(3))

    P_NaCl = P_Boil(T) if T > 800.7 else P_Subl(T)

    x_l = X_L_Sat(T, P)
    P_Crit, _ = X_and_P_crit(T)

    P_Line = (P - P_NaCl) / (P_Crit - P_NaCl)
    P_Line = 1 - P_Line

    Tmp = log10(X_L_Sat(T, P_NaCl))
    Log_K_Line = (1 + j[0] * P_Line ** j[1] + j[2] * P_Line +
                  j[3] * P_Line ** 2 - (1 + j[0] + j[2] + j[3]) * P_Line ** 3)

    Log_K_supScr = Log_K_Line * (log10(P_NaCl / P_Crit) - Tmp) + Tmp
    Tmp2 = Log_K_supScr - log10(P_NaCl / P)

    X_V_Sat = x_l / 10 ** Tmp2

    return X_V_Sat


def P_VLH(T):
    """
    Vapor-Liquid-Halite coexistence pressure (P_VLH).
    Equation (10) from Driesner & Heinrich (2007).

    Parameters:
        T : float
            Temperature in °C

    Returns:
        float: Vapor-Liquid-Halite coexistence pressure in bar
    """
    T_tr_NaCl = 800.7  # °C
    P_tr_NaCl = 0.0005  # bar

    f = np.array([
        0.00464, 5e-7, 16.9078, -269.148,
        7632.04, -49563.6, 233119, -513556,
        549708, -284628, P_tr_NaCl
    ])

    f[10] -= np.sum(f[:10])

    P_VLH = sum(f[i] * (T / T_tr_NaCl) ** i for i in range(11))

    return P_VLH

def P_H2O_Boiling_Curve(T_C):
    """
    Calculates the pure water boiling curve (saturation pressure) at a given temperature.
    
    Parameters:
        T_C : float
            Temperature in °C
    
    Returns:
        float: Saturation pressure in bar
    """
    water = IAPWS95(T=T_C + 273.15, x=0)  # saturated liquid
    return water.P * 10  # Convert from MPa to bar

def X_VL_Liq(T, P):
    """
    Vapor-liquid field boundary composition from the liquid side (X_VL_Liq).
    Equation (11) from Driesner and Heinrich (2007).

    Parameters:
        T : float
            Temperature in °C
        P : float
            Pressure in bar

    Returns:
        float: NaCl mole fraction on vapor-liquid boundary (liquid side)
    """
    h = np.array([
        0,  # placeholder for 1-based indexing
        0.00168486, 0.000219379, 438.58, 18.4508,
        -5.6765e-10, 6.73704e-6, 1.44951e-7, 384.904,
        7.07477, 6.06896e-5, 0.00762859
    ])

    G1 = h[2] + (h[1] - h[2]) / (1 + exp((T - h[3]) / h[4])) + h[5] * T ** 2
    G2 = h[7] + (h[6] - h[7]) / (1 + exp((T - h[8]) / h[9])) + h[10] * exp(-h[11] * T)

    P_Crit, XN_Crit = X_and_P_crit(T)

    if T < 800.7:
        TmpUnit = P_VLH(T)
        TmpUnit2 = X_L_Sat(T, TmpUnit)
    else:
        TmpUnit = P_Boil(T)
        TmpUnit2 = 1

    if T < 373.946:
        TmpUnit3 = P_H2O_Boiling_Curve(T)
        G0 = (
            (TmpUnit2 + G1 * (TmpUnit - TmpUnit3) +
             G2 * ((P_Crit - TmpUnit3) ** 2 - (P_Crit - TmpUnit) ** 2))
            / ((P_Crit - TmpUnit) ** 0.5 - (P_Crit - TmpUnit3) ** 0.5)
        )
        X_VL_Liq = (G0 * (P_Crit - P) ** 0.5 - G0 * (P_Crit - TmpUnit3) ** 0.5
                    - G1 * (P_Crit - TmpUnit3) - G2 * (P_Crit - TmpUnit3) ** 2
                    + G1 * (P_Crit - P) + G2 * (P_Crit - P) ** 2)
    else:
        G0 = (
            (TmpUnit2 - XN_Crit - G1 * (P_Crit - TmpUnit)
             - G2 * (P_Crit - TmpUnit) ** 2)
            / (P_Crit - TmpUnit) ** 0.5
        )
        X_VL_Liq = XN_Crit + G0 * (P_Crit - P) ** 0.5 + G1 * (P_Crit - P) + G2 * (P_Crit - P) ** 2

    return X_VL_Liq


def X_VL_Vap(T, P):
    """
    Vapor-liquid field boundary composition from the vapor side (X_VL_Vap).
    Equations (13-17) from Driesner and Heinrich (2007).

    Parameters:
        T : float
            Temperature in °C
        P : float
            Pressure in bar

    Returns:
        float: NaCl mole fraction on vapor-liquid boundary (vapor side)
    """
    k = np.array([
        -0.235694, -0.188838, 0.004, 0.0552466, 0.66918,
        396.848, 45, -3.2719e-7, 141.699, -0.292631,
        -0.00139991, 1.95965e-6, -7.3653e-10, 0.904411,
        0.000769766, -1.18658e-6
    ])

    j = np.zeros(4)
    j[0] = k[0] + k[1] * exp(-k[2] * T)
    j[1] = k[4] + (k[3] - k[4]) / (1 + exp((T - k[5]) / k[6])) + k[7] * (T + k[8]) ** 2

    j[2] = sum(k[ii + 9] * T ** ii for ii in range(4))
    j[3] = sum(k[ii + 13] * T ** ii for ii in range(3))

    P_NaCl = P_Boil(T) if T >= 800.7 else P_Subl(T)

    X_VL_Lq = X_VL_Liq(T, P)
    P_Crit, _ = X_and_P_crit(T)

    P_Line = (P - P_NaCl) / (P_Crit - P_NaCl)

    Log_K_Line = (
        1 + j[0] * (1 - P_Line) ** j[1]
        + j[2] * (1 - P_Line) + j[3] * (1 - P_Line) ** 2
        - (1 + j[0] + j[2] + j[3]) * (1 - P_Line) ** 3
    )

    Tmp = X_L_Sat(T, P_NaCl)
    Log_K_supScr = Log_K_Line * (log10(P_NaCl / P_Crit) - log10(Tmp)) + log10(Tmp)

    X_VL_Vap = X_VL_Lq / 10 ** Log_K_supScr * P_NaCl / P

    if P <= P_VLH(T):
        X_VL_Vap = X_L_Sat(T, P) / X_VL_Lq * X_VL_Vap

    return X_VL_Vap

def Water_Boiling_Curve(T_C):
    """
    Saturation pressure of pure water at given temperature.

    Parameters:
        T_C : float
            Temperature in °C

    Returns:
        float: Saturation pressure (bar)
    """
    T_K = T_C + 273.15
    sat_water = IAPWS95(T=T_K, x=0)  # saturated liquid
    P_sat_MPa = sat_water.P  # Pressure in MPa
    return P_sat_MPa * 10  # convert MPa to bar

def Rho_Water_Liq_sat(T_C):
    """
    Liquid water density at saturation.

    Parameters:
        T_C : float
            Temperature in °C

    Returns:
        float: Density (kg/m³)
    """
    water = IAPWS95(T=T_C + 273.15, x=0)  # saturated liquid
    return water.rho  # kg/m³

def Rho_Water(T_C, P_bar):
    """
    Water density at given T, P.

    Parameters:
        T_C : float
            Temperature in °C
        P_bar : float
            Pressure in bar

    Returns:
        float: Density (kg/m³)
    """
    water = IAPWS95(T=T_C + 273.15, P=P_bar / 10)  # Pressure in MPa
    return water.rho  # kg/m³


def V_Extrapol(x_in, T_in, P_in):
    """
    Molar volume extrapolation for low and high T-P regions.
    Eq. (17) from Driesner (2007).

    Parameters:
        x_in : float
            Mole fraction of NaCl
        T_in : float
            Temperature in °C
        P_in : float
            Pressure in bar

    Returns:
        float: Extrapolated molar volume (cm³/mol), or 0 if not in extrapolation region.
    """
    if x_in == 0:
        return None  # no extrapolation for pure water

    mH2O = 18.015268
    mNaCl = 58.4428

    T = T_in
    xNaCl = x_in
    P = P_in

    v = X_L_Sat(T, P)

    print(f"Debugging V_Extrapol:")
    print(f"Inputs - xNaCl: {xNaCl}, T: {T}, P: {P}, X_L_Sat(T,P): {v}")

    if P <= Water_Boiling_Curve(T) and T <= 200 and (v - x_in) < 0.01:
        T_star = T_Star_V(xNaCl, T, P)
        Vsat = mH2O / Rho_Water_Liq_sat(T_star) * 1000
        Vwat = mH2O / Rho_Water(T_star, P) * 1000

        print(f"Low T-P region calculation:")
        print(f"T_star: {T_star}, Vsat: {Vsat}, Vwat: {Vwat}")

        if Vsat < Vwat:
            o2 = (2.0125e-7 + 3.29977e-9 * exp(-4.31279 * log10(P))
                  - 1.17748e-7 * log10(P) + 7.58009e-8 * (log10(P)) ** 2)
            v_sat = mH2O / Rho_Water_Liq_sat(T_star) * 1000
            V2 = mH2O / Rho_Water_Liq_sat(T_star - 0.005) * 1000
            o1 = (v_sat - V2) / 0.005 - 3 * o2 * T_star ** 2
            o0 = v_sat - o1 * T_star - o2 * T_star ** 3

            print(f"o0: {o0}, o1: {o1}, o2: {o2}")
            
            volume = o0 + o1 * T_star + o2 * T_star ** 3
            print(f"Calculated molar volume (low region): {volume}")
            return volume

    elif P <= 350 and T >= 600:
        v = X_VL_Liq(T, P)
        print(f"High T-P region calculation, X_VL_Liq(T,P): {v}")

        if np.round(xNaCl, 5) >= np.round(v, 5):
            V1000 = (mH2O * (1 - xNaCl) + mNaCl * xNaCl) / Rh_Br_for_V_extr(xNaCl, T, 1000) * 1000
            v_390 = (mH2O * (1 - xNaCl) + mNaCl * xNaCl) / Rh_Br_for_V_extr(xNaCl, T, 390.147) * 1000
            V2_390 = (mH2O * (1 - xNaCl) + mNaCl * xNaCl) / Rh_Br_for_V_extr(xNaCl, T, 390.137) * 1000

            dVdP390 = (v_390 - V2_390) / 0.01

            o4 = (v_390 - V1000 + dVdP390 * 1609.853) / (np.log(1390.147 / 2000) - 2390.147 / 1390.147)
            o3 = (v_390 - o4 * np.log(1390.147) - 390.147 * dVdP390 + 390.147 / 1390.147 * o4)
            o5 = dVdP390 - o4 / 1390.147

            print(f"o3: {o3}, o4: {o4}, o5: {o5}")

            volume = o3 + o4 * np.log(P + 1000) + o5 * P
            print(f"Calculated molar volume (high region): {volume}")
            return volume

    print("Condition did not match any extrapolation region, returning 0.")
    return 0  # Outside the extrapolation range


def Rh_Br_for_V_extr(xNaCl_frac, T_in_C, P_in_Bar):
    """
    Helper function to estimate density of vapor phase necessary for V_Extrapol function.

    Parameters:
        xNaCl_frac : float
            Mole fraction of NaCl
        T_in_C : float
            Temperature in °C
        P_in_Bar : float
            Pressure in bar

    Returns:
        float: Estimated vapor phase density (kg/m³)
    """
    mH2O = 18.015268
    mNaCl = 58.4428

    T_star = T_Star_V(xNaCl_frac, T_in_C, P_in_Bar)
    V_water = mH2O / Rho_Water(T_star, P_in_Bar) * 1000

    density = (mH2O * (1 - xNaCl_frac) + mNaCl * xNaCl_frac) / V_water * 1000

    return density

def NaCl_Rho_Solid(T_in_C, P_in_Bar):
    """
    Solid NaCl density as a function of temperature and pressure.
    (Driesner, 2007).

    Parameters:
        T_in_C : float
            Temperature in °C
        P_in_Bar : float
            Pressure in bar

    Returns:
        float: Solid NaCl density (kg/m³)
    """
    l = np.array([2170.4, -0.24599, -9.5797e-5, 0.005727, 0.002715, 733.4])

    Rho_Zero = sum(l[i] * T_in_C ** i for i in range(3))
    lParam = l[3] + l[4] * exp(T_in_C / l[5])

    NaCl_Rho_Solid = Rho_Zero + lParam * P_in_Bar

    return NaCl_Rho_Solid


def NaCl_Rho_Liq(T_in_C, P_in_Bar):
    """
    Liquid NaCl density as a function of temperature and pressure.
    (Driesner, 2007).

    Parameters:
        T_in_C : float
            Temperature in °C
        P_in_Bar : float
            Pressure in bar

    Returns:
        float: Liquid NaCl density (kg/m³)
    """
    m = np.array([
        58443, 23.772, 0.018639, -1.9687e-6,
        -1.5259e-5, 5.5058e-8
    ])

    KNaCl = m[4] + m[5] * T_in_C
    Rho_Zero = m[0] / (m[1] + m[2] * T_in_C + m[3] * T_in_C ** 2)
    NaCl_Rho_Liq = Rho_Zero / (1 - 0.1 * np.log(1 + 10 * P_in_Bar * KNaCl))

    return NaCl_Rho_Liq


def NaCl_specific_enthalpy(T_in_C, P_in_Bar):
    """
    Specific enthalpy of NaCl as a function of temperature and pressure.
    (Driesner, 2007).

    Parameters:
        T_in_C : float
            Temperature in °C
        P_in_Bar : float
            Pressure in bar

    Returns:
        float: Specific enthalpy (J/kg)
    """
    t_t = T_in_C - 800.7

    k = np.array([226713, 44.6652, -7.41999e-5])
    r_rec = np.array([
        1148.81,
        0.275774,
        8.8103e-5,
        -0.0017099 - 3.82734e-6 * T_in_C / 2 - 8.65455e-9 * T_in_C ** 2 / 3,
        5.29063e-8 - 9.63084e-11 * T_in_C / 2 + 6.50745e-13 * T_in_C ** 2 / 3
    ])

    NaCl_specific_enthalpy = (
        k[0] +
        r_rec[0] * t_t +
        r_rec[1] * t_t ** 2 +
        r_rec[2] * t_t ** 3 +
        (k[1] + r_rec[3] * T_in_C) * P_in_Bar +
        (k[2] + r_rec[4] * T_in_C) * P_in_Bar ** 2
    )

    return NaCl_specific_enthalpy


def NaCl_isobaric_heat_capacity(T_in_C, P_in_Bar):
    """
    Isobaric heat capacity of NaCl as a function of temperature and pressure.
    (Driesner, 2007).

    Parameters:
        T_in_C : float
            Temperature in °C
        P_in_Bar : float
            Pressure in bar

    Returns:
        float: Isobaric heat capacity (J/kg·K)
    """
    t_t = T_in_C - 800.7

    r = np.array([
        1148.81,
        0.275774,
        8.8103e-5,
        -0.0017099 - 3.82734e-6 * T_in_C - 8.65455e-9 * T_in_C ** 2,
        5.29063e-8 - 9.63084e-11 * T_in_C + 6.50745e-13 * T_in_C ** 2
    ])

    NaCl_isobaric_heat_capacity = (
        r[0] +
        2 * r[1] * t_t +
        3 * r[2] * t_t ** 2 +
        r[3] * P_in_Bar +
        r[4] * P_in_Bar ** 2
    )

    return NaCl_isobaric_heat_capacity

def wtfrac_to_xNaCl(wt_frac_NaCl):
    """
    Convert weight fraction of NaCl to mole fraction of NaCl.
    
    Parameters:
    - wt_frac_NaCl: float (0–1), weight fraction of NaCl

    Returns:
    - xNaCl: float, mole fraction of NaCl
    """
    m_NaCl = 58.4428   # g/mol, molar mass of NaCl
    m_H2O = 18.01528   # g/mol, molar mass of H2O

    n_NaCl = wt_frac_NaCl / m_NaCl
    n_H2O = (1 - wt_frac_NaCl) / m_H2O
    xNaCl = n_NaCl / (n_NaCl + n_H2O)
    return xNaCl

def V_Extrapol_density(x_in, T_in, P_in):
    molar_volume_cm3_mol = V_Extrapol(x_in, T_in, P_in)
    
    if molar_volume_cm3_mol <= 0 or molar_volume_cm3_mol is None:
        raise ValueError("Invalid molar volume from extrapolation.")

    # Molar masses (g/mol)
    M_H2O = 18.015268
    M_NaCl = 58.4428

    # Average molar mass (g/mol)
    M_avg = x_in * M_NaCl + (1 - x_in) * M_H2O

    # Convert to density (kg/m³)
    density_kg_m3 = (M_avg / molar_volume_cm3_mol) * 1000

    return density_kg_m3

def driesner_density(T_C, P_bar, wt_frac_NaCl):
    xNaCl = wtfrac_to_xNaCl(wt_frac_NaCl)
    
    # Check input ranges explicitly
    if not (0 <= wt_frac_NaCl <= 1):
        raise ValueError(f"Invalid wt_frac_NaCl: {wt_frac_NaCl}. Must be between 0 and 1.")
    if not (0 <= xNaCl <= 1):
        raise ValueError(f"Invalid xNaCl: {xNaCl}. Must be between 0 and 1.")
    if not (-50 <= T_C <= 1500):
        raise ValueError(f"Temperature (°C) {T_C} outside supported range.")
    if not (0.01 <= P_bar <= 5000):
        raise ValueError(f"Pressure (bar) {P_bar} outside supported range.")

    # Calculate phase boundaries
    X_L_sat_bound = X_L_Sat(T_C, P_bar)
    P_VLH_bound = P_VLH(T_C)

    if P_bar >= P_VLH_bound and xNaCl <= X_L_sat_bound:
        # Single-phase liquid within correlation limits
        T_star_V = T_Star_V(xNaCl, T_C, P_bar)
        T_K = T_star_V + 273.15
        P_MPa = P_bar * 0.1

        # Check physically meaningful ranges before IAPWS95
        if not (273.15 <= T_K <= 1273.15 and 0.001 <= P_MPa <= 1000):
            print(f"Unphysical conditions for IAPWS95: T={T_K:.2f} K, P={P_MPa:.2f} MPa")
            density = np.nan
        else:
            try:
                water = IAPWS95(T=T_K, P=P_MPa)
                density = water.rho
            except RuntimeError:
                print(f"IAPWS95 convergence failure at T={T_K:.2f} K, P={P_MPa:.2f} MPa")
                density = np.nan
    else:
        # Extrapolation region
        try:
            density = V_Extrapol_density(xNaCl, T_C, P_bar)
        except Exception as e:
            print(f"Extrapolation density calculation failed: {e}")
            density = np.nan

    # Final density validity check
    if density is None or np.isnan(density) or density <= 0:
        print(f"Warning: Invalid density ({density}) at T={T_C}°C, P={P_bar} bar, wt_frac_NaCl={wt_frac_NaCl}. Assigning NaN.")
        density = np.nan  # or assign a small fallback value if preferred
        
    return density