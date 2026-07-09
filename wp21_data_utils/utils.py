import numpy as np
import awkward as ak
import vector


def zero_pad(x, max_vectors):
    """
    Pad or clip a jagged array along axis 1 and replace missing values with 0.

    Parameters
    ----------
    x : awkward.Array
        Jagged array to pad.
    max_vectors : int
        Target length along the vector axis.

    Returns
    -------
    numpy.ndarray
        Dense padded array.
    """
    return ak.to_numpy(ak.fill_none(ak.pad_none(x, max_vectors, clip=True), 0))


def pt_sort(vectors, ascending=False):
    """
    Sort per-event vectors by transverse momentum.

    Parameters
    ----------
    vectors : vector.Array
        Jagged vector array with a ``rho`` or pT-compatible component.
    ascending : bool, default=False
        If True, sort from low to high pT.

    Returns
    -------
    vector.Array
        Vectors sorted independently within each event.
    """
    return vectors[ak.argsort(vectors.rho, axis=1, ascending=ascending)]


def pad_vectors(vectors, max_vectors, sort_first=True):
    """
    Sort and zero-pad jagged four-vectors to a fixed event length.

    Parameters
    ----------
    vectors : vector.Array
        Jagged vectors with ``m``, ``pt``, ``eta``, and ``phi`` components.
    max_vectors : int
        Target number of vectors per event.
    sort_first : bool, default=True
        If True, sort vectors by descending pT before padding.

    Returns
    -------
    vector.Array
        Dense vector array with exactly ``max_vectors`` entries per event.
    """
    if sort_first:
        vectors = pt_sort(vectors, ascending=False)

    padded_vectors = vector.zip(
        {
            "m": zero_pad(vectors.m, max_vectors),
            "pt": zero_pad(vectors.pt, max_vectors),
            "eta": zero_pad(vectors.eta, max_vectors),
            "phi": zero_pad(vectors.phi, max_vectors),
        }
    )
    return padded_vectors


def to_vector_array(array: ak.Array):
    """
    Convert common momentum record layouts into a vector array.

    The function recognises ``(pt, eta, phi, m)``, ``(rho, eta, phi, tau)``,
    ``(x, y, z, t)``, and ``(px, py, pz, E)`` field conventions.

    Parameters
    ----------
    array : awkward.Array
        Input records with one of the supported component layouts.

    Returns
    -------
    vector.Array
        Vector-backed awkward array.

    Raises
    ------
    ValueError
        If the input fields do not match a supported component convention.
    """
    if "pt" in array.fields:
        return vector.zip(
            {"pt": array.pt, "eta": array.eta, "phi": array.phi, "m": array.m}
        )

    if "rho" in array.fields:
        return vector.zip(
            {"pt": array.rho, "eta": array.eta, "phi": array.phi, "m": array.tau}
        )

    elif "t" in array.fields:
        return vector.zip({"px": array.x, "py": array.y, "pz": array.z, "E": array.t})

    elif "px" in array.fields:
        return vector.zip(
            {"px": array.px, "py": array.py, "pz": array.pz, "E": array.E}
        )

    else:
        raise ValueError("could not detect vector component keys.")


def to_jagged_array(vectors, min_pt=1e-12):
    """
    Convert a dense vector image or grid into a jagged vector array.

    Parameters
    ----------
    vectors : vector.Array
        Dense vector array containing ``m``, ``pt``, ``eta``, and ``phi``.
    min_pt : float, default=1e-12
        Minimum pT threshold for keeping a vector.

    Returns
    -------
    vector.Array
        Jagged vectors containing only entries with ``pt > min_pt``.
    """
    mask = np.asarray(vectors.pt > min_pt)
    counts = ak.from_numpy(np.sum(mask.reshape(mask.shape[0], -1), axis=1))
    m = ak.unflatten(vectors.m[mask], counts)
    pt = ak.unflatten(vectors.pt[mask], counts)
    eta = ak.unflatten(vectors.eta[mask], counts)
    phi = ak.unflatten(vectors.phi[mask], counts)
    return vector.zip({"m": m, "pt": pt, "eta": eta, "phi": phi})


def balance_weights(weights: np.ndarray, labels: np.ndarray):
    """
    Rescale event weights so each label class has equal total weight.

    Parameters
    ----------
    weights : numpy.ndarray
        Per-sample input weights.
    labels : numpy.ndarray
        Per-sample class labels.

    Returns
    -------
    numpy.ndarray
        Reweighted copy-like array with balanced class sums.
    """
    weights_ = np.asarray(weights)
    labels_ = np.asarray(labels)
    classes_ = np.unique(labels_)
    for y in classes_:
        sf = len(labels_) / len(classes_) / np.sum(weights_[labels == y])
        weights_[np.nonzero(labels_ == y)] *= sf

    return weights_


def unflatten_like(x, like):
    """
    Rebuild a jagged array using the event lengths from another array.

    Parameters
    ----------
    x : array-like
        Flat values to distribute across events.
    like : awkward.Array
        Jagged array whose axis-1 counts define the output structure.

    Returns
    -------
    awkward.Array
        Values from ``x`` unflattened to match ``like``.
    """
    out = []
    start_idx = 0
    for length in ak.count(like, axis=1):
        if length > 0:
            out.append(x[start_idx : start_idx + length])

        else:
            out.append([])

        start_idx += length

    return ak.Array(out)


def delta_R_matching(recon, truth, max_dR, unique_pairing=False):
    """
    Match reconstructed and truth objects within a delta-R threshold.

    Parameters
    ----------
    recon : vector.Array
        Per-event reconstructed objects.
    truth : vector.Array
        Per-event truth objects.
    max_dR : float
        Maximum delta-R separation for a candidate match.
    unique_pairing : bool, default=False
        If True, drop duplicated reconstructed-truth index pairs after
        thresholding.

    Returns
    -------
    tuple[vector.Array, vector.Array]
        Matched reconstructed objects and matched truth objects.

    Raises
    ------
    ValueError
        If ``recon`` and ``truth`` do not have the same number of events.
    """

    def drop_duplicate_pairs(a, b):
        max_b = ak.max(b, axis=-1, keepdims=True)
        key = a * (max_b + 1) + b

        order = ak.argsort(key, axis=-1)
        a_sorted = a[order]
        b_sorted = b[order]
        key_sorted = key[order]

        keep = ak.run_lengths(key_sorted) == 1

        a_unique = a_sorted[keep]
        b_unique = b_sorted[keep]

        restore = ak.argsort(order[keep], axis=-1)
        return a_unique[restore], b_unique[restore]

    if len(truth) != len(recon):
        raise ValueError("truth and recon jets must be same size along axis=0.")

    recon_idxs, truth_idxs = ak.unzip(ak.argcartesian([recon, truth]))

    dR = recon[recon_idxs].deltaR(truth[truth_idxs])

    is_close = dR < max_dR

    recon_idxs = recon_idxs[is_close]
    truth_idxs = truth_idxs[is_close]

    if unique_pairing:
        recon_idxs, truth_idxs = drop_duplicate_pairs(recon_idxs, truth_idxs)

    matched_truth = truth[truth_idxs]
    matched_recon = recon[recon_idxs]
    return matched_recon, matched_truth
