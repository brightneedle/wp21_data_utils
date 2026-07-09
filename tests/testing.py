from pathlib import Path
import numpy as np


TEST_DIR = Path(__file__).parent
DATA_DIR = TEST_DIR / "data"


def test_imports():
    import pkgutil
    import importlib
    import wp21_data_utils

    package = wp21_data_utils.__name__
    package_path = [str(Path(wp21_data_utils.__file__).parent)]
    for _, module_name, _ in pkgutil.walk_packages(package_path, package + "."):
        importlib.import_module(module_name)


def test_towering():
    import awkward as ak
    import vector
    from wp21_data_utils.image import cell_vectors_to_image

    cell_data = ak.from_parquet(DATA_DIR / "cell_vectors.parquet")

    cell_vectors = vector.zip(
        {
            "pt": cell_data.rho,
            "eta": cell_data.eta,
            "phi": cell_data.phi,
            "m": cell_data.tau,
            "layer": cell_data.layer,
        }
    )

    towers = cell_vectors_to_image(cell_vectors)

    assert towers.ndim == 4
    assert towers.shape[0] == len(cell_vectors)
    assert towers.shape[-1] == 6
    assert np.all(towers >= 0)
    assert towers.sum() > 0


def test_converters():
    import vector
    from wp21_data_utils.image import vectors_to_image, image_to_vectors

    test_vectors = vector.zip(
        {
            "pt": [[2.0, 1.0], [0]],
            "eta": [[0.15, 0.25], [0]],
            "phi": [[np.pi / 2 - np.pi / 64, -np.pi / 4 - np.pi / 64], [0]],
            "m": [[0, 0], [0]],
        }
    )
    towers = vectors_to_image(test_vectors)
    vectors = image_to_vectors(towers)

    assert np.allclose(test_vectors[0].pt, vectors[0].pt)


def test_vectors_to_image_binning_and_event_axis():
    import vector
    from wp21_data_utils.image import vectors_to_image

    vectors = vector.zip(
        {
            "pt": [[2.0, 3.0], [5.0]],
            "eta": [[-0.25, 0.25], [0.25]],
            "phi": [[-0.25, 0.25], [-0.25]],
            "m": [[0.0, 0.0], [0.0]],
        }
    )

    image = vectors_to_image(
        vectors,
        eta_edges=np.array([-0.5, 0.0, 0.5]),
        phi_edges=np.array([-0.5, 0.0, 0.5]),
    )

    assert image.shape == (2, 2, 2, 1)
    assert np.isclose(image[0, 0, 0, 0], 2.0)
    assert np.isclose(image[0, 1, 1, 0], 3.0)
    assert np.isclose(image[1, 1, 0, 0], 5.0)
    assert np.isclose(image.sum(), 10.0)


def test_cell_vectors_to_image_separates_layers():
    import vector
    from wp21_data_utils.image import cell_vectors_to_image

    cell_vectors = vector.zip(
        {
            "pt": [[1.0, 2.0, 3.0]],
            "eta": [[0.25, 0.25, 0.25]],
            "phi": [[0.25, 0.25, 0.25]],
            "m": [[0.0, 0.0, 0.0]],
            "layer": [[0, 2, 5]],
        }
    )

    image = cell_vectors_to_image(
        cell_vectors,
        eta_edges=np.array([0.0, 0.5]),
        phi_edges=np.array([0.0, 0.5]),
    )

    assert image.shape == (1, 1, 1, 6)
    assert np.allclose(image[0, 0, 0], [1.0, 0.0, 2.0, 0.0, 0.0, 3.0])


def test_image_padding_wraps_phi_and_zeros_eta():
    from wp21_data_utils.image import pad

    x = np.arange(1 * 2 * 3 * 1).reshape(1, 2, 3, 1)
    padded = pad(x, 1)

    assert padded.shape == (1, 4, 5, 1)
    assert np.allclose(padded[:, 0, :, :], 0)
    assert np.allclose(padded[:, -1, :, :], 0)
    assert np.allclose(padded[0, 1:-1, 0, 0], x[0, :, -1, 0])
    assert np.allclose(padded[0, 1:-1, -1, 0], x[0, :, 0, 0])
    assert np.allclose(padded[0, 1:-1, 1:-1, 0], x[0, :, :, 0])


def test_zero_pad_and_pad_vectors_sort_by_pt():
    import awkward as ak
    import vector
    from wp21_data_utils.utils import pad_vectors, zero_pad

    values = ak.Array([[1.0, 2.0], [], [3.0, 4.0, 5.0]])
    assert np.allclose(
        zero_pad(values, 3),
        np.array([[1.0, 2.0, 0.0], [0.0, 0.0, 0.0], [3.0, 4.0, 5.0]]),
    )

    vectors = vector.zip(
        {
            "pt": [[1.0, 3.0, 2.0]],
            "eta": [[0.1, 0.3, 0.2]],
            "phi": [[0.0, 0.0, 0.0]],
            "m": [[0.0, 0.0, 0.0]],
        }
    )

    padded = pad_vectors(vectors, max_vectors=4)

    assert np.allclose(ak.to_numpy(padded.pt[0]), [3.0, 2.0, 1.0, 0.0])
    assert np.allclose(ak.to_numpy(padded.eta[0]), [0.3, 0.2, 0.1, 0.0])


def test_to_vector_array_detects_supported_component_names():
    import awkward as ak
    from wp21_data_utils.utils import to_vector_array

    pt_eta_phi_m = to_vector_array(
        ak.Array({"pt": [10.0], "eta": [0.1], "phi": [0.2], "m": [1.0]})
    )
    rho_eta_phi_tau = to_vector_array(
        ak.Array({"rho": [20.0], "eta": [0.3], "phi": [0.4], "tau": [2.0]})
    )
    cartesian = to_vector_array(
        ak.Array({"px": [3.0], "py": [4.0], "pz": [0.0], "E": [5.0]})
    )
    xyzt = to_vector_array(ak.Array({"x": [3.0], "y": [4.0], "z": [0.0], "t": [5.0]}))

    assert np.allclose(pt_eta_phi_m.pt, [10.0])
    assert np.allclose(rho_eta_phi_tau.pt, [20.0])
    assert np.allclose(cartesian.pt, [5.0])
    assert np.allclose(xyzt.pt, [5.0])


def test_to_vector_array_rejects_unknown_layout():
    import awkward as ak
    from wp21_data_utils.utils import to_vector_array

    try:
        to_vector_array(ak.Array({"energy": [1.0]}))
    except ValueError as err:
        assert "could not detect vector component keys" in str(err)
    else:
        raise AssertionError("Expected ValueError for unknown vector layout")


def test_balance_weights_equalises_class_weight_sums():
    from wp21_data_utils.utils import balance_weights

    weights = np.array([1.0, 1.0, 1.0, 3.0])
    labels = np.array([0, 0, 1, 1])

    balanced = balance_weights(weights.copy(), labels)

    assert np.isclose(balanced[labels == 0].sum(), balanced[labels == 1].sum())
    assert np.isclose(balanced.sum(), len(labels))


def test_unflatten_like_preserves_empty_events():
    import awkward as ak
    from wp21_data_utils.utils import unflatten_like

    like = ak.Array([[10, 11], [], [12]])
    out = unflatten_like(np.array([1.0, 2.0, 3.0]), like)

    assert ak.to_list(out) == [[1.0, 2.0], [], [3.0]]


def test_delta_r_matching_returns_close_pairs_and_validates_event_counts():
    import awkward as ak
    import vector
    from wp21_data_utils.utils import delta_R_matching

    recon = vector.zip(
        {
            "pt": [[10.0, 20.0], [30.0]],
            "eta": [[0.0, 1.0], [0.0]],
            "phi": [[0.0, 1.0], [0.0]],
            "m": [[0.0, 0.0], [0.0]],
        }
    )
    truth = vector.zip(
        {
            "pt": [[11.0, 21.0], [31.0]],
            "eta": [[0.05, 2.0], [0.1]],
            "phi": [[0.05, 2.0], [0.1]],
            "m": [[0.0, 0.0], [0.0]],
        }
    )

    matched_recon, matched_truth = delta_R_matching(recon, truth, max_dR=0.2)

    assert ak.to_list(matched_recon.pt) == [[10.0], [30.0]]
    assert ak.to_list(matched_truth.pt) == [[11.0], [31.0]]

    try:
        delta_R_matching(recon[:1], truth, max_dR=0.2)
    except ValueError as err:
        assert "same size along axis=0" in str(err)
    else:
        raise AssertionError("Expected ValueError for mismatched event counts")


def test_cell_layer_mapping_eta_correction_and_gap_removal():
    import awkward as ak
    import vector
    from wp21_data_utils.cells import get_corrected_eta, get_layer, remove_calo_gaps

    assert ak.to_list(get_layer(ak.Array([[0, 1, 5, 18, 23]]))) == [[0, 1, 1, 4, 3]]

    cell_vectors = vector.zip(
        {
            "pt": [[1.0, 1.0, 1.0]],
            "eta": [[1.6, -1.6, 0.5]],
            "phi": [[0.0, 0.0, 0.0]],
            "m": [[0.0, 0.0, 0.0]],
            "layer": [[2, 2, 1]],
        }
    )

    corrected_eta = ak.to_numpy(ak.flatten(get_corrected_eta(cell_vectors)))
    assert np.allclose(corrected_eta, [1.59, -1.59, 0.495])

    cells = ak.Array(
        [
            {
                "cell_x": [0.0, 2000.0, 2000.0],
                "cell_y": [0.0, 0.0, 0.0],
                "cell_z": [3000.0, 0.0, 3000.0],
                "cell_sampling": [0, 0, 1],
            }
        ]
    )

    filtered = remove_calo_gaps(cells)

    assert ak.to_list(filtered.cell_sampling) == [[0, 1]]
