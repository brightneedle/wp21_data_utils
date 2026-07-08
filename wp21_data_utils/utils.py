import numpy as np
import awkward as ak
import vector


def zero_pad(x, max_vectors):
    return ak.to_numpy(ak.fill_none(ak.pad_none(x, max_vectors, clip=True), 0))


def pt_sort(vectors, ascending=False):
    return vectors[ak.argsort(vectors.rho, axis=1, ascending=ascending)]


def pad_vectors(vectors, max_vectors, sort_first=True):
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
    mask = np.asarray(vectors.pt > min_pt)
    counts = ak.from_numpy(np.sum(mask.reshape(mask.shape[0], -1), axis=1))
    m = ak.unflatten(vectors.m[mask], counts)
    pt = ak.unflatten(vectors.pt[mask], counts)
    eta = ak.unflatten(vectors.eta[mask], counts)
    phi = ak.unflatten(vectors.phi[mask], counts)
    return vector.zip({"m": m, "pt": pt, "eta": eta, "phi": phi})


def balance_weights(weights: np.ndarray, labels: np.ndarray):
    weights_ = np.asarray(weights)
    labels_ = np.asarray(labels)
    classes_ = np.unique(labels_)
    for y in classes_:
        sf = len(labels_) / len(classes_) / np.sum(weights_[labels == y])
        weights_[np.nonzero(labels_ == y)] *= sf

    return weights_


def unflatten_like(x, like):
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
