from __future__ import annotations

import awkward as ak
import fastjet
import vector


def antikt_jets(
    vectors: ak.Array | vector.Array, min_pt: float, r: float = 0.4
) -> ak.Array:
    """
    Cluster input vectors into anti-kt jets with FastJet.

    Parameters
    ----------
    vectors : awkward.Array or vector.Array
        Per-event input particles or cell/tower four-vectors accepted by
        FastJet.
    min_pt : float
        Minimum transverse momentum for inclusive output jets.
    r : float, default=0.4
        Anti-kt radius parameter.

    Returns
    -------
    awkward.Array
        Inclusive anti-kt jets above ``min_pt``.
    """
    jetdef = fastjet.JetDefinition(fastjet.antikt_algorithm, r)
    cluster = fastjet.ClusterSequence(vectors, jetdef)
    return cluster.inclusive_jets(min_pt)
