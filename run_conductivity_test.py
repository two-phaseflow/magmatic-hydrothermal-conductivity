#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_conductivity_test.py -- Conductivity analysis using vtu_io + conductivity
==================================================================================

Loads a CSMP++ VTU timestep, computes electrical conductivity via the
production pipeline (conductivity), and plots the result with
temperature contours.

Uses:
    vtu_io.run_conductivity  -> end-to-end VTU loading + conductivity
    conductivity         -> all physics (Watanabe, Sinmyo-Keppler,
                                Archie, surface conduction, melt models)

@author: samuels
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.patheffects as pe
from scipy.interpolate import griddata

from vtu_io import run_conductivity, load_vtu


# =============================================================================
# Helpers
# =============================================================================

def point_on_contour_at_x(CS, level, x_target):
    """Return (x, y) on contour *level* closest to x_target."""
    try:
        i = list(CS.levels).index(level)
    except ValueError:
        return None
    paths = CS.collections[i].get_paths()
    if not paths:
        return None
    p = max(paths, key=lambda P: P.vertices.shape[0])
    V = p.vertices
    idx = np.argmin(np.abs(V[:, 0] - x_target))
    return (float(V[idx, 0]), float(V[idx, 1]))


def add_halo(texts, lw=3.5):
    """White halo on text labels for readability over colormaps."""
    for t in texts:
        t.set_path_effects([pe.withStroke(linewidth=lw, foreground='white')])
        t.set_weight('bold')


# =============================================================================
# Configuration
# =============================================================================

config = {
    # --- Archie parameters (global defaults) ---
    # These apply everywhere unless overridden per region.
    # Tortuosity a = 1.0 following Glover (2009): a != 1 indicates
    # an incorrect m, so a should always be 1 when m is set properly.
    # Saturation exponent n = 2.0 is the standard Archie value.
    'porosity_exponent_m': 1.8,     # global default cementation exponent [-]
    'saturation_exponent_n': 2.0,   # saturation exponent [-]
    'tortuosity_a': 1.0,            # tortuosity factor [-]

    # --- Per-region cementation exponent m ---
    # Overrides the global porosity_exponent_m for specific regions.
    # Values based on:
    #   Intrusive igneous (granite): m = 1.70 +/- 0.02
    #       Revil et al. (2024) GJI 241(2), 1348 (Part 8: intrusive rocks)
    #   Extrusive volcanic (andesite/basalt): m = 2.16 +/- 0.02
    #       Zhang & Revil (2023) GJI 234(3), 2375 (Part 7: stratovolcanoes)
    #   Fractured permeable zones: m -> 1.0-1.5
    #       Glover (2009) TLE 28(1), 82 (m as connectivity measure)
    #
    # Yuzawa region ID mapping:
    #   1  INTRUSION      intrusive body
    #   2  LOWER          deep low-permeability basement
    #   3  OUTER_UPP      outer domain, upper
    #   4  OUTER_MID      outer domain, mid-depth
    #   5  BASEM_NW       crystalline basement (NW)
    #   6  BASEM_SE       crystalline basement (SE)
    #   7  VOLC_NW        volcanic edifice (NW)
    #   8  VOLC_SE        volcanic edifice (SE)
    #   9  SHALLOW        shallow zone above clay cap
    #   10 OUTFLOW        high-permeability outflow zone
    #   11 CLAY_CAP_OUT   outer clay cap
    #   12 CLAY_CAP_CENT  central clay cap (smectite-rich)
    'regions': {
        # Basement / deep crystalline (m ~ 1.7, Revil et al. 2024)
        # Grain density 2750-2800 kg/m3 (typical granodiorite/gneiss)
        1:  {'porosity_exponent_m': 1.7, 'grain_density': 2800.0},
        2:  {'porosity_exponent_m': 1.7, 'grain_density': 2800.0},
        4:  {'porosity_exponent_m': 1.7, 'grain_density': 2750.0},
        5:  {'porosity_exponent_m': 1.7, 'grain_density': 2750.0},
        6:  {'porosity_exponent_m': 1.7, 'grain_density': 2750.0},

        # Volcanic edifice (m ~ 2.1, Zhang & Revil 2023)
        # Grain density 2700 kg/m3 (andesite)
        3:  {'porosity_exponent_m': 2.1, 'grain_density': 2700.0},
        7:  {'porosity_exponent_m': 2.1, 'grain_density': 2700.0},
        8: {'porosity_exponent_m': 2.1, 'grain_density': 2700.0},

        # Outflow zone (fractured, m ~ 1.5)
        10:  {'porosity_exponent_m': 1.6, 'grain_density': 2700.0},

        # Shallow volcanic edifice (high porosity/permeability)
        9:  {'porosity_exponent_m': 1.6, 'grain_density': 2700.0},

        # Clay caps (altered volcanics, m ~ 2.2)
        # Lower grain density (~2600) due to smectite/clay alteration.
        # High CEC from smectite (80-150 meq/100g).
        # f_stern = 0.90 (lower Stern fraction for clays, more
        # counterions in diffuse layer -> higher surface conduction).
        11: {'porosity_exponent_m': 2.2, 'grain_density': 2600.0,
             'CEC_meq_per_100g': 20.0, 'f_stern': 0.90},
        12: {'porosity_exponent_m': 2.2, 'grain_density': 2600.0,
             'CEC_meq_per_100g': 40.0, 'f_stern': 0.90},
    },

    # Default region properties (used where no region-specific override)
    'default_region': {
        'grain_density': 2800.0,        # [kg/m3]
        'f_stern': 0.95,                # Stern layer fraction [-]
        'CEC_meq_per_100g': 2.0,        # cation exchange capacity
    },

    # --- Salinity floor ---
    'min_fluid_salinity_wt_percent': 0.2,

    # --- Two-phase fluid mixing ---
    'mixing_law': 'glover',

    # --- Surface conduction model ---
    'surface_conduction_model': 'hybrid',

    # --- Clay cap identification ---
    'clay_cap_regions': [11, 12],

    # --- Intrusion region ---
    'intrusion_id': 1,
    'spent_magma_min_porosity': 0.02,

    # --- Melt domain ---
    'melt_threshold': 0.0
    ,  # any melt fraction -> magmatic domain
    'magma_composition': {
        'type': 'dacite',
        'SiO2_wt_percent': 67.8,
        'constant_sigma_melt': None,
    },
}


# =============================================================================
# Load VTU and compute conductivity
# =============================================================================

timestep_vtu = '../vtus/yuz_homogenized/Variables_30000.vtu'
initial_vtu = '../vtus/yuz_homogenized/Initial.vtu'

results = run_conductivity(timestep_vtu, initial_vtu, config=config)

x, y = results['coordinates']
triangles = results['triangles']
sigma_bulk = results['sigma_bulk']
melt_mask = results['melt_mask']
clay_cap_mask = results['clay_cap_mask']
phi_melt = results['melt_fractions']

log_conductivity = np.log10(np.maximum(sigma_bulk, 1e-12))

print(f"\nMesh: {len(x)} nodes, {len(triangles)} triangles")
print(f"Bounds: x=[{x.min():.1f}, {x.max():.1f}], "
      f"y=[{y.min():.1f}, {y.max():.1f}] km")
pos = sigma_bulk > 0
if np.any(pos):
    print(f"Conductivity (log10): {log_conductivity[pos].min():.2f} "
          f"to {log_conductivity[pos].max():.2f}")
print(f"Melt nodes: {np.sum(melt_mask)}")
print(f"Clay cap nodes: {np.sum(clay_cap_mask)}")
print(f"Porosity source: {results.get('porosity_source', 'unknown')}")

if np.any(phi_melt > 0):
    print(f"Melt fraction (where > 0): "
          f"min={phi_melt[phi_melt > 0].min():.3f}, "
          f"max={phi_melt.max():.3f}")


# =============================================================================
# Gridded temperature for contours
# =============================================================================

# Reload nodal temperature for contour plotting
vtu_data = load_vtu(timestep_vtu)
T = vtu_data['nodal_data']['temperature']

npts = 600
xi = np.linspace(x.min(), x.max(), npts)
yi = np.linspace(y.min(), y.max() + 0.5, npts)
Xi, Yi = np.meshgrid(xi, yi)
Ti = griddata((x, y), T, (Xi, Yi), method='linear', fill_value=np.nan)


# =============================================================================
# Plot: conductivity with temperature contours
# =============================================================================

plt.rcParams['font.size'] = 16
plt.rcParams['font.family'] = 'Arial'

cond_vmin, cond_vmax = -4.0, 0.0
cond_cmap = plt.cm.RdYlBu_r
cond_norm = colors.Normalize(vmin=cond_vmin, vmax=cond_vmax)

fig, ax = plt.subplots(figsize=(16, 10))

# Nodal tripcolor (smooth interpolation)
log_display = np.clip(log_conductivity, -5.0, None)
tc = ax.tripcolor(x, y, triangles, log_display,
                  cmap=cond_cmap, norm=cond_norm, shading='gouraud')

cbar = plt.colorbar(tc, ax=ax, extend='both', shrink=0.8, pad=0.02)
cbar.set_label(r'$\log_{10}\,\sigma$ [S/m]', fontsize=16)

# Temperature contours
temp_levels = [100, 200, 250, 300, 400, 500, 600, 700, 800]
CS = ax.contour(Xi, Yi, Ti, levels=temp_levels, colors='red', linewidths=2.2)

temp_x_targets = {
    100: 12.0, 200: 12.0, 250: 10.5, 300: 12.5, 400: 13.5,
    500: 14.0, 600: 14.75, 700: 15, 800: 14,
}
temp_manual = [point_on_contour_at_x(CS, lev, xt)
               for lev, xt in temp_x_targets.items()]
temp_manual = [p for p in temp_manual if p is not None]

if temp_manual:
    labels = ax.clabel(CS, levels=list(temp_x_targets.keys()),
                       manual=temp_manual,
                       fmt=lambda v: f"{int(v)}C", inline=True,
                       inline_spacing=5, fontsize=13, colors='red')
    add_halo(labels, lw=4.0)

# Clay cap region boundaries — extracted from element edges
from vtu_io import load_initial_vtu
from scipy.spatial import cKDTree

init_data = load_initial_vtu(initial_vtu)
if 'region_id_initial' in init_data and 'region_id_initial_centers' in init_data:
    init_rid = init_data['region_id_initial']
    cx_init, cy_init = init_data['region_id_initial_centers']
    tree = cKDTree(np.column_stack((cx_init, cy_init)))

    # Map Initial.vtu element region IDs to timestep triangles
    tri_cx = np.mean(x[triangles], axis=1)
    tri_cy = np.mean(y[triangles], axis=1)
    _, idx = tree.query(np.column_stack((tri_cx, tri_cy)))
    region_per_tri = init_rid[idx]

    # Find boundary edges: edges where one triangle is in the target
    # region and the adjacent triangle is not (or is on the mesh boundary).
    # Build edge-to-triangle map.
    from collections import defaultdict
    edge_to_tris = defaultdict(list)
    for i_tri, tri in enumerate(triangles):
        for e in [(tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])]:
            edge = tuple(sorted(e))
            edge_to_tris[edge].append(i_tri)

    for rid, color in [(11, 'grey'), (12, 'grey')]:
        boundary_edges = []
        for edge, tri_list in edge_to_tris.items():
            if len(tri_list) == 1:
                if region_per_tri[tri_list[0]] == rid:
                    boundary_edges.append(edge)
            elif len(tri_list) == 2:
                r0 = region_per_tri[tri_list[0]]
                r1 = region_per_tri[tri_list[1]]
                if (r0 == rid) != (r1 == rid):
                    boundary_edges.append(edge)

        for n0, n1 in boundary_edges:
            ax.plot([x[n0], x[n1]], [y[n0], y[n1]],
                    color=color, linewidth=1.5, solid_capstyle='round',
                    alpha=1)

        print(f"  Region {rid}: {len(boundary_edges)} boundary edges")

# Axes
ax.set_xlabel('Distance (km)', fontsize=16)
ax.set_ylabel('Elevation (km a.s.l.)', fontsize=16)
ax.set_xlim(x.min(), x.max())
ax.set_ylim(y.min(), y.max() + 0.5)
ax.set_aspect('equal')

# Compass labels
kw = dict(transform=ax.transAxes, va='bottom', fontsize=16,
          fontweight='bold', clip_on=False, zorder=20)
halo = [pe.withStroke(linewidth=3, foreground='white')]
ax.text(0.01, 1.02, "NW", ha='left', **kw).set_path_effects(halo)
ax.text(0.99, 1.02, "SE", ha='right', **kw).set_path_effects(halo)

plt.tight_layout()
plt.savefig('conductivity_test_output.png', dpi=300,
            bbox_inches='tight', facecolor='white')
print(f"\nSaved: conductivity_test_output.png")
plt.show()


# =============================================================================
# Summary statistics
# =============================================================================

print(f"\n{'='*60}")
print("CONDUCTIVITY ANALYSIS SUMMARY")
print(f"{'='*60}")
print(f"Model: {results['model_type']}")
print(f"Porosity source: {results.get('porosity_source', 'unknown')}")
print(f"Temperature range: {T.min():.1f} to {T.max():.1f} C")

if np.any(pos):
    print(f"Overall conductivity (log10): "
          f"{log_conductivity[pos].min():.2f} to "
          f"{log_conductivity[pos].max():.2f}")

if np.any(clay_cap_mask):
    clay_cond = log_conductivity[clay_cap_mask & pos]
    hydro_cond = log_conductivity[~clay_cap_mask & ~melt_mask & pos]
    if len(clay_cond) > 0:
        print(f"Clay cap (log10): {clay_cond.min():.2f} to "
              f"{clay_cond.max():.2f}")
    if len(hydro_cond) > 0:
        print(f"Hydrothermal (log10): {hydro_cond.min():.2f} to "
              f"{hydro_cond.max():.2f}")
    if len(clay_cond) > 0 and len(hydro_cond) > 0:
        print(f"Clay cap enhancement: "
              f"{10**(clay_cond.mean() - hydro_cond.mean()):.1f}x")

if np.any(melt_mask):
    melt_cond = log_conductivity[melt_mask & pos]
    if len(melt_cond) > 0:
        print(f"Melt zone (log10): {melt_cond.min():.2f} to "
              f"{melt_cond.max():.2f}")
    print(f"Melt fraction: {phi_melt[melt_mask].min():.3f} to "
          f"{phi_melt[melt_mask].max():.3f}")
    if 'phi_solid' in results:
        print(f"CVF (melt zone): {results['phi_solid'][melt_mask].min():.3f} "
              f"to {results['phi_solid'][melt_mask].max():.3f}")
    if 'phi_vol' in results:
        print(f"VVF (melt zone): {results['phi_vol'][melt_mask].min():.3f} "
              f"to {results['phi_vol'][melt_mask].max():.3f}")
