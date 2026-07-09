from __future__ import annotations

import awkward as ak
import vector
import numpy as np


def remove_calo_gaps(cells: ak.Array) -> ak.Array:
    """
    Remove calorimeter cells from sampling regions with known eta gaps.

    Parameters
    ----------
    cells : awkward.Array
        Cell records containing ``cell_x``, ``cell_y``, ``cell_z``, and
        ``cell_sampling`` fields.

    Returns
    -------
    awkward.Array
        Input cells with gap-region samplings removed.
    """
    cell_abseta = np.abs(to_3vector(cells).eta)
    psb = (cells.cell_sampling == 0) & (cell_abseta > 1.5)
    eme1 = (cells.cell_sampling == 5) & (cell_abseta < 1.5)
    eme2 = (cells.cell_sampling == 6) & (cell_abseta < 1.5)
    ext0 = (cells.cell_sampling == 18) & (cell_abseta > 1.5)
    mask = (~psb) & (~eme1) & (~eme2) & (~ext0)
    return cells[mask]


def get_layer(sampling: ak.Array | np.ndarray) -> ak.Array | np.ndarray:
    """
    Map ATLAS calorimeter sampling indices onto compact layer indices.

    Parameters
    ----------
    sampling : awkward.Array or array-like
        Calorimeter sampling identifiers.

    Returns
    -------
    awkward.Array or array-like
        Layer identifiers using a compact layer numbering.
    """
    layer_map = {
        0: 0,  # PSB
        1: 1,  # EMB1
        2: 2,  # EMB2
        3: 3,  # EMB3
        4: 0,  # PSE
        5: 1,  # EME1
        6: 2,  # EME2
        7: 3,  # EME3
        8: 4,  # HEC0
        9: 5,  # HEC1
        10: 6,  # HEC2
        11: 7,  # HEC3
        12: 4,  # TileBar0
        13: 5,  # TileBar1
        14: 6,  # TileBar2
        15: 5,  # TileGap1
        16: 6,  # TileGap2
        17: 7,  # TileGap3
        18: 4,  # TileExt0
        19: 5,  # TileExt1
        20: 6,  # TileExt2
        # things get weird in the fcal
        21: 1,  # FCAL0 (EM)
        22: 2,  # FCAL1 (Had)
        23: 3,  # FCAL1 (Had)
    }

    output = ak.copy(sampling)
    for k, v in layer_map.items():
        output = ak.where(output == k, v, output)

    return output


def get_corrected_eta(cell_vectors: vector.Array) -> ak.Array:
    """
    Apply small eta corrections for selected calorimeter cell layers.

    Parameters
    ----------
    cell_vectors : vector.Array
        Cell position vectors with ``eta`` and ``layer`` components.

    Returns
    -------
    awkward.Array
        Corrected eta values.
    """
    cell_eta = cell_vectors.eta
    cell_eta = (
        cell_eta
        + ak.where((cell_eta > 1.5) & (cell_vectors.layer == 2), -0.01, 0)
        + ak.where((cell_eta < -1.5) & (cell_vectors.layer == 2), +0.01, 0)
        + ak.where(
            (cell_eta > 0.1) & (cell_eta < 1.4) & (cell_vectors.layer == 1), -0.005, 0
        )
    )
    return cell_eta


def to_3vector(cells: ak.Array, metre: float = 1e3) -> vector.Array:
    """
    Convert cell Cartesian positions into spatial vector objects.

    Parameters
    ----------
    cells : awkward.Array
        Cell records containing ``cell_x``, ``cell_y``, ``cell_z``, and
        ``cell_sampling`` fields.
    metre : float, default=1e3
        Conversion factor from the input position units to metres.

    Returns
    -------
    vector.Array
        Three-vectors with ``x``, ``y``, ``z``, and compact ``layer`` fields.
    """
    vectors = vector.zip(
        {
            "x": cells.cell_x / metre,
            "y": cells.cell_y / metre,
            "z": cells.cell_z / metre,
            "layer": get_layer(cells.cell_sampling),
        }
    )
    return vectors


def to_4momentum(cells: ak.Array, Et_key: str = "cell_et") -> vector.Array:
    """
    Convert calorimeter cells into massless eta-phi-pT four-vectors.

    Parameters
    ----------
    cells : awkward.Array
        Cell records with position, sampling, and transverse-energy fields.
    Et_key : str, default="cell_et"
        Field name containing the transverse energy to use as vector pT.

    Returns
    -------
    vector.Array
        Four-vectors with ``m``, ``pt``, ``eta``, ``phi``, and ``layer``.
    """
    position = to_3vector(cells)
    cell_eta = get_corrected_eta(position)
    components = {
        "m": ak.zeros_like(cells[Et_key]),
        "pt": cells[Et_key],
        "eta": cell_eta,
        "phi": position.phi,
        "layer": position.layer,
    }
    vectors = vector.zip(components)
    return vectors


def cells_to_vectors(
    cells: ak.Array,
    Et_key: str = "cell_et",
    central_only: bool = True,
    remove_gaps: bool = True,
) -> vector.Array:
    """
    Convert raw calorimeter cell records to analysis-ready cell vectors.

    Parameters
    ----------
    cells : awkward.Array
        Cell records with Cartesian positions, sampling IDs, and transverse
        energy.
    Et_key : str, default="cell_et"
        Field name containing the transverse energy to use as vector pT.
    central_only : bool, default=True
        If True, keep only cells in the central detector region.
    remove_gaps : bool, default=True
        If True, remove cells from selected calorimeter gap regions before
        conversion.

    Returns
    -------
    vector.Array
        Per-event massless cell vectors with compact layer labels.
    """
    if remove_gaps:
        cells = remove_calo_gaps(cells)

    cell_vectors = to_4momentum(cells, Et_key=Et_key)

    # Keep only central cells
    if central_only:
        central = np.abs(cell_vectors) < 2.5
        cell_vectors = cell_vectors[central]

    return cell_vectors
