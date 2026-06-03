import awkward as ak
import vector
import numpy as np
import yaml
import os
import uproot
import functools
import pyarrow.parquet as pq
from tqdm import tqdm

from global_data_utils.utils import zero_pad

print = functools.partial(print, flush=True)


def get_config(sample):
    with open("configs/samples.yaml", "r") as f:
        return yaml.safe_load(f)[sample]


def get_data(config, **kwargs):
    files = [
        config["dir"] + "/" + f
        for f in os.listdir(config["dir"])
        if config["keyword"] in f and ".root" in f
    ]
    events = uproot.concatenate(files, **kwargs)

    if "EventWeight" in events.fields:
        weights = ak.to_numpy(events["EventWeight"][:, 0])
        events["weight"] = (
            weights
            * config["xsec"]
            * config["hstp_filter_sf"]
            * config["filter_eff"]
            / weights.sum()
        )
        events = ak.without_field(events, where="EventWeight")

    for var in ["cell_et", "cell_et_mu0"]:
        if var in events.fields:
            events[var] = events[var] / 1000  # MeV --> GeV

    return events


def sparse_to_awkward(arr, min_pt=1e-12):
    mask = np.asarray(arr.pt > min_pt)
    counts = ak.from_numpy(np.sum(mask.reshape(mask.shape[0], -1), axis=1))
    m = ak.unflatten(arr.m[mask], counts)
    pt = ak.unflatten(arr.pt[mask], counts)
    eta = ak.unflatten(arr.eta[mask], counts)
    phi = ak.unflatten(arr.phi[mask], counts)
    return vector.zip({"m": m, "pt": pt, "eta": eta, "phi": phi})


def sort_and_pad(vectors, n):

    vectors = vectors[ak.argsort(vectors.pt, axis=1, ascending=False)]

    padded_vectors = vector.zip(
        {
            "m": zero_pad(vectors.m),
            "pt": zero_pad(vectors.pt),
            "eta": zero_pad(vectors.eta),
            "phi": zero_pad(vectors.phi),
        }
    )

    return padded_vectors


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


def balance_indices(idxs0, idxs1):
    if len(idxs1) > len(idxs0):
        idxs1 = np.random.choice(idxs1, len(idxs0), replace=False)

    elif len(idxs1) < len(idxs0):
        idxs0 = np.random.choice(idxs0, len(idxs1), replace=False)

    return np.concatenate((idxs1, idxs0))


def to_parquet_row_groups(array, path, events_per_group=1000):
    print(f"INFO: Writing {len(array)} events to {path}")
    ak.to_parquet_row_groups(
        [
            array[i : i + events_per_group]
            for i in range(0, len(array), events_per_group)
        ],
        path,
    )


def get_parquet_column_names(path):
    parquet_file = pq.ParquetFile(path)
    return parquet_file.schema_arrow


def get_num_parquet_row_groups(path):
    parquet_file = pq.ParquetFile(path)
    return parquet_file.num_row_groups


def iterate_row_group_batches(path, row_groups_per_batch, verbose=False, **kwargs):
    num_row_groups = get_num_parquet_row_groups(path)
    print("INF0: Found", num_row_groups, "row groups in", path)

    looper = range(0, num_row_groups, row_groups_per_batch)
    if verbose:
        looper = tqdm(looper, total=len(looper))

    for begin_row_group in looper:
        end_row_group = min(begin_row_group + row_groups_per_batch, num_row_groups)
        batch = ak.from_parquet(
            path, row_groups=range(begin_row_group, end_row_group), **kwargs
        )
        yield batch
