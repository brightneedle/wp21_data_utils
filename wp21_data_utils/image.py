import numpy as np
import skimage
import vector
import awkward as ak


from wp21_data_utils.utils import pt_sort, to_jagged_array


def vectors_to_image(
    vectors: vector.Array,
    eta_edges: list = np.linspace(-2.5, 2.5, 51),
    phi_edges: list = np.linspace(-np.pi, np.pi, 65),
):
    """
    Histogram per-event vectors into eta-phi images of summed pT.

    Parameters
    ----------
    vectors : vector.Array
        Jagged per-event vectors with ``pt``, ``eta``, and ``phi`` components.
    eta_edges : array-like, default=np.linspace(-2.5, 2.5, 51)
        Eta bin edges.
    phi_edges : array-like, default=np.linspace(-np.pi, np.pi, 65)
        Phi bin edges.

    Returns
    -------
    numpy.ndarray
        Dense image tensor with shape ``(events, eta, phi, 1)``.
    """
    tower_edges = (np.arange(1 + len(vectors)), eta_edges, phi_edges)

    event_indices = get_index(vectors)

    flat_indices = ak.flatten(event_indices)
    flat_vectors = ak.flatten(vectors)

    towers = np.histogramdd(
        (
            ak.to_numpy(flat_indices),
            ak.to_numpy(flat_vectors.eta),
            ak.to_numpy(flat_vectors.phi),
        ),
        bins=tower_edges,
        weights=ak.to_numpy(flat_vectors.pt),
    )[0]

    return np.expand_dims(towers, axis=-1)


def cell_vectors_to_image(
    cell_vectors,
    eta_edges=np.linspace(-2.5, 2.5, 51),
    phi_edges=np.linspace(-np.pi, np.pi, 65),
):
    """
    Histogram layered cell vectors into multi-channel eta-phi images.

    Parameters
    ----------
    cell_vectors : vector.Array
        Jagged per-event cell vectors with ``pt``, ``eta``, ``phi``, and
        ``layer`` components.
    eta_edges : array-like, default=np.linspace(-2.5, 2.5, 51)
        Eta bin edges.
    phi_edges : array-like, default=np.linspace(-np.pi, np.pi, 65)
        Phi bin edges.

    Returns
    -------
    numpy.ndarray
        Dense image tensor with one channel per compact layer.
    """
    towers = np.concatenate(
        [
            vectors_to_image(
                cell_vectors[cell_vectors.layer == layer],
                eta_edges=eta_edges,
                phi_edges=phi_edges,
            )
            for layer in range(6)
        ],
        axis=-1,
    )
    return towers


def pad(x, pad_size):
    """
    Pad an eta-phi image batch with zero eta padding and wrapped phi padding.

    Parameters
    ----------
    x : numpy.ndarray
        Image tensor with shape ``(events, eta, phi, channels)``.
    pad_size : int
        Number of cells to add to each side of the eta and phi axes.

    Returns
    -------
    numpy.ndarray
        Padded image tensor.
    """
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
    """
    Build local eta-phi windows over an image batch.

    Parameters
    ----------
    x : numpy.ndarray
        Image tensor with shape ``(events, eta, phi, channels)``.
    size : int
        Odd window size along eta and phi.

    Returns
    -------
    numpy.ndarray
        View of local windows over the padded image tensor.
    """
    assert x.ndim == 4
    px = pad(x, (size - 1) // 2)
    windows = skimage.util.view_as_windows(
        px, window_shape=(1, size, size, px.shape[-1]), step=1
    )
    return windows


def get_tower_eta(X):
    """
    Return eta-coordinate values for each tower in an image tensor.

    Parameters
    ----------
    X : numpy.ndarray
        Image tensor with shape ``(events, eta, phi, channels)``.

    Returns
    -------
    numpy.ndarray
        Eta coordinate tensor aligned with ``X[..., :1]``.
    """
    eta_idxs = np.indices(X[..., :1].shape)[1]
    return (eta_idxs - np.median(eta_idxs)) * 0.1


def image_to_vectors(X, deta=0.1, dphi=np.pi / 32, eta0=0, phi0=0, return_sorted=True):
    """
    Convert eta-phi image pixels back into jagged vector arrays.

    Parameters
    ----------
    X : numpy.ndarray
        Image tensor with shape ``(events, eta, phi, channels)`` containing
        pT-like pixel weights.
    deta : float, default=0.1
        Eta spacing between adjacent pixels.
    dphi : float, default=np.pi / 32
        Phi spacing between adjacent pixels.
    eta0 : float, default=0
        Offset added to the centred eta coordinates.
    phi0 : float, default=0
        Offset added to the centred phi coordinates.
    return_sorted : bool, default=True
        If True, sort output vectors by descending pT.

    Returns
    -------
    vector.Array
        Jagged per-event vectors reconstructed from non-zero pixels.
    """
    _, eta_idxs, phi_idxs, _ = np.indices(X.shape)

    eta = (eta_idxs - np.median(eta_idxs)) * deta + eta0
    phi = (phi_idxs - np.median(phi_idxs)) * dphi + phi0

    vectors = vector.arr(
        {
            "eta": eta,
            "phi": phi,
            "pt": X,
            "m": np.zeros_like(X),
        }
    )

    vectors = to_jagged_array(vectors)

    if return_sorted:
        return pt_sort(vectors)
    else:
        return vectors


def get_index(vectors):
    """
    Return per-vector event indices for a jagged vector array.

    Parameters
    ----------
    vectors : vector.Array
        Jagged per-event vector array.

    Returns
    -------
    awkward.Array
        Jagged array containing the parent event index for each vector.
    """
    return ak.ones_like(vectors.eta, dtype=int) * np.arange(len(vectors))
