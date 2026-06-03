import numpy as np
import awkward as ak

def zero_pad(x, max_vectors):
    return ak.to_numpy(ak.fill_none(ak.pad_none(x, max_vectors, clip=True), 0))

def to_vector_array(array : ak.Array):
    if "pt" in array.fields:
        return vector.zip({"pt": array.pt, "eta": array.eta, "phi": array.phi, "m": array.m})

    if "rho" in array.fields:
        return vector.zip({"pt": array.rho, "eta": array.eta, "phi": array.phi, "m": array.tau})

    elif "t" in array.fields:
        return vector.zip({"px": array.x, "py": array.y, "pz": array.z, "E": array.t})

    elif "px" in array.fields:
        return vector.zip({"px": array.px, "py": array.py, "pz": array.pz, "E": array.E})

    else:
        raise ValueError("could not detect vector component keys.")


def balance_weights(weights, labels):
    classes = np.unique(labels)
    for y in classes:
        weights[np.nonzero(labels == y)] *= (
            len(labels) / len(classes) / np.sum(weights[labels == y])
        )
    return weights