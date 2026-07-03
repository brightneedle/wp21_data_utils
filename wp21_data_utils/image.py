import numpy as np
import skimage
import vector
import awkward as ak

from fastml.utils.cells import get_cell_vectors


def vectors_to_image(
    vectors,
    eta_edges=np.linspace(-2.5, 2.5, 51),
    phi_edges=np.linspace(-np.pi, np.pi, 65),
):
    tower_edges = (np.arange(1 + len(vectors)), eta_edges, phi_edges)

    event_indices = ak.flatten(get_index(vectors))
    flat_vectors = ak.flatten(vectors)

    towers = np.histogramdd(
        (
            ak.to_numpy(event_indices),
            ak.to_numpy(flat_vectors.eta),
            ak.to_numpy(flat_vectors.phi),
        ),
        bins=tower_edges,
        weights=ak.to_numpy(flat_vectors.pt),
    )[0]

    return np.expand_dims(towers, axis=-1)


def cells_to_image(cells, Et_key="cell_et"):
    cell_vectors = get_cell_vectors(cells, Et_key)
    towers = np.concatenate(
        [
            vectors_to_image(cell_vectors[cell_vectors.layer == layer])
            for layer in range(6)
        ],
        axis=-1,
    )
    return towers


def pad(x, pad_size):
    assert x.ndim == 4
    y = np.pad(
        x,
        ((0, 0), (pad_size, pad_size), (0, 0), (0, 0)),
        mode="constant",
        constant_values=0,
    )
    y = np.pad(y, ((0, 0), (0, 0), (pad_size, pad_size), (0, 0)), mode="wrap")
    return y


def sliding_window(x, size):
    assert x.ndim == 4
    px = pad(x, (size - 1) // 2)
    windows = skimage.util.view_as_windows(
        px, window_shape=(1, size, size, px.shape[-1]), step=1
    )
    return windows


def get_tower_eta(X):
    eta_idxs = np.indices(X[..., :1].shape)[1]
    return (eta_idxs - np.median(eta_idxs)) * 0.1


def image_to_vectors(X):
    _, eta_idxs, phi_idxs, _ = np.indices(X.shape)

    eta = (eta_idxs - np.median(eta_idxs)) * 0.1
    phi = (phi_idxs - np.median(phi_idxs)) * np.pi / 32

    vectors = vector.arr(
        {
            "eta": eta,
            "phi": phi,
            "pt": X,
            "m": np.zeros_like(X),
        }
    )

    return vectors


def get_index(vectors):
    return ak.ones_like(vectors.eta, dtype=int) * np.arange(len(vectors))
