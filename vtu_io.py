#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
vtu_io.py -- VTU I/O and conductivity pipeline orchestrator
============================================================

Reads CSMP++ VTU output files and feeds them into the conductivity
pipeline (conductivity.calculate_conductivity_nodal). This module
handles all VTK I/O and field name mapping; it contains no physics.

Public interface
----------------
    load_vtu              Read a timestep VTU, return nodal/element data
    load_initial_vtu      Read Initial.vtu for porosity and region_id
    run_conductivity      End-to-end: load VTUs -> compute conductivity

Architecture
------------
This module is a thin I/O layer between CSMP++ VTU files and the
conductivity library:

    VTU files (vtk)  -->  load_vtu / load_initial_vtu
                               |
                          (nodal_data, element_data, coordinates, triangles)
                               |
    conductivity  <--  run_conductivity
                               |
                          calculate_conductivity_nodal(config, ...)
                               |
                          sigma_bulk, sigma_fluid, phase fractions, ...

VTU field name mapping
----------------------
CSMP++ uses space-separated field names in VTU output (e.g.,
"saturation liquid"). This module maps them to underscore-separated
internal keys (e.g., "saturation_liquid") used by conductivity.

The mappings are defined in NODAL_FIELD_MAP and ELEMENT_FIELD_MAP
at module level. If CSMP++ changes its naming convention, update
these dicts -- no other code needs to change.

Usage example
-------------
    from vtu_io import run_conductivity

    config = {
        'porosity_exponent_m': 1.8,
        'magma_composition': {'type': 'dacite', 'T_solidus': 700.0},
        'clay_cap_regions': [11, 12],
    }

    results = run_conductivity(
        timestep_vtu='path/to/yuzawa_Properties_50000.vtu',
        initial_vtu='path/to/yuzawa_Initial.vtu',
        config=config,
    )

    sigma_bulk = results['sigma_bulk']
    x, y = results['coordinates']

References
----------
VTK XML file format:
    https://vtk.org/Wiki/VTK_XML_Formats

@author: samuels
"""

import os
import warnings

import numpy as np
import vtk
from vtk.util.numpy_support import vtk_to_numpy

from conductivity import calculate_conductivity_nodal


# =============================================================================
# VTU FIELD NAME MAPPINGS
# =============================================================================

# Maps internal key -> CSMP++ VTU field name for nodal (point) data.
# Required fields raise ValueError if missing; optional fields warn.
NODAL_FIELD_MAP = {
    # --- Required (ValueError if missing) ---
    'temperature':           'temperature',
    'fluid_pressure':        'fluid pressure',
    'saturation_liquid':     'saturation liquid',
    'saturation_vapor':      'saturation vapor',
    'salt_fraction_liquid':  'salt fraction liquid',
    'salt_fraction_vapor':   'salt fraction vapor',
    'density_liquid':        'density liquid',
    'density_vapor':         'density vapor',
    'salinity':              'salinity',

    # --- Optional (warning if missing) ---
    'saturation_halite':         'saturation halite',
    'water_fraction_melt':       'water fraction melt',
    'water_solubility_melt':     'water solubility melt',
    'injection_location':        'injection location',
    'fluid_density':             'fluid density',
    'crystal_volume_fraction':   'crystal volume fraction',
    'fluid_volume_fraction':     'fluid volume fraction',
}

REQUIRED_NODAL_FIELDS = {
    'temperature', 'fluid_pressure', 'saturation_liquid',
    'saturation_vapor', 'salt_fraction_liquid', 'salt_fraction_vapor',
    'density_liquid', 'density_vapor', 'salinity',
}

# Maps internal key -> CSMP++ VTU field name for element (cell) data.
ELEMENT_FIELD_MAP = {
    'porosity':                  'porosity',
    'region_id':                 'region id',
    'crystal_volume_fraction':   'crystal volume fraction',
    'fluid_volume_fraction':     'fluid volume fraction',
    'permeability':              'permeability',
}


# =============================================================================
# VTU READING HELPERS
# =============================================================================

def _read_vtu_surface(filename):
    """
    Read a VTU file and extract its surface triangulation.

    Uses vtkDataSetSurfaceFilter to extract the surface mesh, which
    gives us triangles for element-based operations and consistent
    node numbering.

    Parameters
    ----------
    filename : str
        Path to the VTU file.

    Returns
    -------
    surface : vtkPolyData
        Surface mesh with point and cell data.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    RuntimeError
        If VTK fails to read the file or returns empty data.
    """
    if not os.path.exists(filename):
        raise FileNotFoundError(f"VTU file not found: {filename}")

    reader = vtk.vtkXMLUnstructuredGridReader()
    reader.SetFileName(filename)
    reader.Update()

    if reader.GetErrorCode() != 0:
        raise RuntimeError(
            f"VTK reader error code {reader.GetErrorCode()}: {filename}")

    output = reader.GetOutput()
    if output is None or output.GetNumberOfPoints() == 0:
        raise RuntimeError(f"VTU file is empty or unreadable: {filename}")

    surface_filter = vtk.vtkDataSetSurfaceFilter()
    surface_filter.SetInputData(output)
    surface_filter.Update()
    surface = surface_filter.GetOutput()

    if surface.GetNumberOfPoints() == 0:
        raise RuntimeError(f"Surface extraction returned no points: {filename}")

    return surface


def _extract_triangles(surface):
    """
    Extract triangle connectivity from a vtkPolyData surface.

    Parameters
    ----------
    surface : vtkPolyData

    Returns
    -------
    triangles : ndarray, shape (n_tri, 3)
        Node indices for each triangle.
    """
    polys = surface.GetPolys().GetData()
    n_tri = polys.GetNumberOfTuples() // 4
    triangles = np.zeros((n_tri, 3), dtype=int)
    for i in range(n_tri):
        triangles[i, 0] = int(polys.GetTuple(4 * i + 1)[0])
        triangles[i, 1] = int(polys.GetTuple(4 * i + 2)[0])
        triangles[i, 2] = int(polys.GetTuple(4 * i + 3)[0])
    return triangles


def _extract_coordinates(surface):
    """
    Extract node coordinates and convert from meters to kilometers.

    Parameters
    ----------
    surface : vtkPolyData

    Returns
    -------
    x, y : ndarray
        Node coordinates [km].
    """
    coords = vtk_to_numpy(surface.GetPoints().GetData())
    x = coords[:, 0] * 1e-3  # m -> km
    y = coords[:, 1] * 1e-3
    return x, y


def _extract_cell_centers(surface):
    """
    Compute cell (element) center coordinates in km.

    Parameters
    ----------
    surface : vtkPolyData

    Returns
    -------
    cx, cy : ndarray
        Cell center coordinates [km].
    """
    centers_filter = vtk.vtkCellCenters()
    centers_filter.SetInputData(surface)
    centers_filter.Update()
    centers = vtk_to_numpy(centers_filter.GetOutput().GetPoints().GetData())
    return centers[:, 0] * 1e-3, centers[:, 1] * 1e-3


def _load_fields(surface, field_map, data_type='point'):
    """
    Extract named fields from VTU point or cell data.

    For each (internal_key, vtu_name) in field_map, tries to find the
    array in the specified data type. If not found, tries the other
    data type as a fallback.

    Parameters
    ----------
    surface : vtkPolyData
    field_map : dict
        {internal_key: vtu_field_name}.
    data_type : str
        'point' for nodal data, 'cell' for element data.

    Returns
    -------
    data : dict
        {internal_key: ndarray}. Missing fields are omitted.
    missing : list of str
        Internal keys that were not found.
    """
    point_data = surface.GetPointData()
    cell_data = surface.GetCellData()

    primary = point_data if data_type == 'point' else cell_data
    fallback = cell_data if data_type == 'point' else point_data

    data = {}
    missing = []

    for key, vtu_name in field_map.items():
        arr = primary.GetArray(vtu_name)
        if arr is None:
            arr = fallback.GetArray(vtu_name)
        if arr is not None:
            data[key] = vtk_to_numpy(arr)
        else:
            missing.append(key)

    return data, missing


# =============================================================================
# PUBLIC FUNCTIONS
# =============================================================================

def load_vtu(filename):
    """
    Read a CSMP++ timestep VTU file and extract all fields.

    Reads the VTU file, extracts surface triangulation, maps VTU field
    names to internal keys, and converts coordinates from meters to
    kilometers.

    Parameters
    ----------
    filename : str
        Path to the timestep VTU file (e.g., yuzawa_Properties_50000.vtu).

    Returns
    -------
    dict with keys:
        'nodal_data' : dict of {field_name: ndarray}
        'element_data' : dict of {field_name: ndarray}
        'coordinates' : tuple of (x, y) ndarrays [km]
        'triangles' : ndarray, shape (n_tri, 3)
        'cell_centers' : tuple of (cx, cy) ndarrays [km]
        'n_points' : int
        'n_elements' : int

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    RuntimeError
        If VTK fails to read the file.
    ValueError
        If required nodal fields are missing.
    """
    print(f"Loading VTU: {os.path.basename(filename)}")

    surface = _read_vtu_surface(filename)
    x, y = _extract_coordinates(surface)
    triangles = _extract_triangles(surface)
    cx, cy = _extract_cell_centers(surface)

    print(f"  {len(x)} nodes, {len(triangles)} elements")
    print(f"  X: [{x.min():.1f}, {x.max():.1f}] km, "
          f"Y: [{y.min():.1f}, {y.max():.1f}] km")

    # Nodal fields
    nodal_data, nodal_missing = _load_fields(
        surface, NODAL_FIELD_MAP, 'point')

    missing_required = [k for k in nodal_missing if k in REQUIRED_NODAL_FIELDS]
    missing_optional = [k for k in nodal_missing if k not in REQUIRED_NODAL_FIELDS]

    if missing_required:
        pd = surface.GetPointData()
        available = [pd.GetArrayName(i) for i in range(pd.GetNumberOfArrays())]
        raise ValueError(
            f"Missing required nodal fields in VTU: {missing_required}. "
            f"Available arrays: {available}")

    if missing_optional:
        warnings.warn(
            f"Optional nodal fields not found: {missing_optional}")

    n_loaded = len(nodal_data)
    print(f"  Loaded {n_loaded} nodal fields, "
          f"{len(missing_optional)} optional fields missing")

    # Element fields
    element_data, elem_missing = _load_fields(
        surface, ELEMENT_FIELD_MAP, 'cell')

    return {
        'nodal_data': nodal_data,
        'element_data': element_data,
        'coordinates': (x, y),
        'triangles': triangles,
        'cell_centers': (cx, cy),
        'n_points': len(x),
        'n_elements': len(triangles),
    }


def load_initial_vtu(filename):
    """
    Read Initial.vtu for porosity and region_id.

    These fields are written only to the initial VTU by CSMP++, not
    to timestep VTUs. The returned dict uses keys expected by
    conductivity._match_porosity and _match_region_ids.

    Parameters
    ----------
    filename : str
        Path to Initial.vtu file.

    Returns
    -------
    dict with keys:
        'nodal_porosity' : ndarray, porosity at nodes.
        'initial_node_coordinates' : tuple of (x, y) [km].
        'region_id_initial' : ndarray of int, region ID per element.
        'region_id_initial_centers' : tuple of (cx, cy) [km].

    Returns None for any field not found (with a warning).

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    RuntimeError
        If VTK fails to read the file.
    """
    print(f"Loading Initial VTU: {os.path.basename(filename)}")

    surface = _read_vtu_surface(filename)
    x, y = _extract_coordinates(surface)
    cx, cy = _extract_cell_centers(surface)

    point_data = surface.GetPointData()
    cell_data = surface.GetCellData()

    result = {
        'initial_node_coordinates': (x, y),
    }

    # --- Nodal porosity ---
    arr = point_data.GetArray('nodal porosity')
    if arr is not None:
        result['nodal_porosity'] = vtk_to_numpy(arr)
        phi = result['nodal_porosity']
        print(f"  Nodal porosity: {len(phi)} values, "
              f"[{phi.min():.4f}, {phi.max():.4f}]")
    else:
        # Try element porosity as fallback
        arr = cell_data.GetArray('porosity')
        if arr is not None:
            result['nodal_porosity'] = vtk_to_numpy(arr)
            warnings.warn(
                "nodal porosity not found; using element porosity as fallback")
        else:
            warnings.warn("No porosity found in Initial VTU")

    # --- Region ID ---
    for name in ['region id', 'region_id']:
        arr = cell_data.GetArray(name)
        if arr is None:
            arr = point_data.GetArray(name)
        if arr is not None:
            result['region_id_initial'] = vtk_to_numpy(arr).astype(int)
            result['region_id_initial_centers'] = (cx, cy)
            unique = np.unique(result['region_id_initial'])
            print(f"  Region ID: {len(result['region_id_initial'])} values, "
                  f"regions {unique}")
            break
    else:
        warnings.warn("No region_id found in Initial VTU")

    return result


def run_conductivity(timestep_vtu, initial_vtu=None, config=None):
    """
    End-to-end conductivity calculation from VTU files.

    Loads the timestep VTU and (optionally) the Initial VTU, then
    calls conductivity.calculate_conductivity_nodal to compute
    bulk electrical conductivity at all mesh nodes.

    Parameters
    ----------
    timestep_vtu : str
        Path to the timestep VTU file.
    initial_vtu : str or None
        Path to Initial.vtu for porosity and region_id. If None,
        looks for '*_Initial.vtu' in the same directory.
    config : dict or None
        Conductivity configuration. Merged with
        conductivity.DEFAULT_CONFIG. See conductivity module
        docstring for available keys.

    Returns
    -------
    dict with keys:
        'sigma_bulk' : ndarray, bulk conductivity [S/m].
        'sigma_fluid' : ndarray, effective fluid conductivity [S/m].
        'melt_fractions' : ndarray, phi_melt [0-1].
        'phi_solid' : ndarray.
        'phi_vol' : ndarray.
        'porosity' : ndarray.
        'porosity_source' : str.
        'melt_mask' : ndarray of bool.
        'clay_cap_mask' : ndarray of bool.
        'model_type' : str.
        'coordinates' : tuple of (x, y) ndarrays [km].
        'triangles' : ndarray, shape (n_tri, 3).

    Raises
    ------
    FileNotFoundError
        If VTU files are not found.
    ValueError
        If required fields are missing.
    """
    if config is None:
        config = {}

    # --- Load timestep VTU ---
    vtu_data = load_vtu(timestep_vtu)
    nodal_data = vtu_data['nodal_data']
    element_data = vtu_data['element_data']
    coordinates = vtu_data['coordinates']
    triangles = vtu_data['triangles']

    # --- Load initial VTU ---
    if initial_vtu is None:
        # Auto-detect: look for *_Initial.vtu in same directory
        vtu_dir = os.path.dirname(timestep_vtu)
        candidates = [f for f in os.listdir(vtu_dir)
                      if f.endswith('_Initial.vtu') or f.endswith('_initial.vtu')]
        if candidates:
            initial_vtu = os.path.join(vtu_dir, candidates[0])
            print(f"  Auto-detected Initial VTU: {candidates[0]}")

    if initial_vtu is not None and os.path.exists(initial_vtu):
        init_data = load_initial_vtu(initial_vtu)

        # Merge initial data into nodal_data / element_data
        if 'nodal_porosity' in init_data:
            nodal_data['nodal_porosity'] = init_data['nodal_porosity']
        if 'initial_node_coordinates' in init_data:
            element_data['initial_node_coordinates'] = \
                init_data['initial_node_coordinates']
        if 'region_id_initial' in init_data:
            element_data['region_id_initial'] = \
                init_data['region_id_initial']
        if 'region_id_initial_centers' in init_data:
            element_data['region_id_initial_centers'] = \
                init_data['region_id_initial_centers']
    elif initial_vtu is not None:
        warnings.warn(f"Initial VTU not found: {initial_vtu}")

    # --- Run conductivity pipeline ---
    print("\nRunning conductivity pipeline...")
    results = calculate_conductivity_nodal(
        config, nodal_data, element_data, coordinates, triangles)

    # Attach coordinates for downstream plotting
    results['coordinates'] = coordinates
    results['triangles'] = triangles

    return results


# =============================================================================
# SELF-TEST
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("vtu_io.py -- self-test")
    print("=" * 70)

    print("\nNodal field mapping:")
    for key, vtu_name in NODAL_FIELD_MAP.items():
        req = "(required)" if key in REQUIRED_NODAL_FIELDS else "(optional)"
        print(f"  {key:30s} <- '{vtu_name}' {req}")

    print("\nElement field mapping:")
    for key, vtu_name in ELEMENT_FIELD_MAP.items():
        print(f"  {key:30s} <- '{vtu_name}'")

    # Try loading a VTU if available
    test_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(test_dir)
    vtu_dir = os.path.join(parent_dir, 'vtus')

    if os.path.isdir(vtu_dir):
        vtus = [f for f in os.listdir(vtu_dir) if f.endswith('.vtu')
                and 'Initial' not in f and 'Properties' in f]
        if vtus:
            test_file = os.path.join(vtu_dir, sorted(vtus)[0])
            print(f"\nTest loading: {os.path.basename(test_file)}")
            try:
                data = load_vtu(test_file)
                print(f"  Loaded: {data['n_points']} nodes, "
                      f"{data['n_elements']} elements")
                print(f"  Nodal fields: {list(data['nodal_data'].keys())}")
                print(f"  Element fields: {list(data['element_data'].keys())}")
            except Exception as e:
                print(f"  Failed: {e}")
        else:
            print(f"\nNo Properties VTU files found in {vtu_dir}")
    else:
        print(f"\nNo vtus directory found at {vtu_dir}")
        print("To test VTU loading, run from a directory with a vtus/ subfolder.")

    print("\nModule ready.")
