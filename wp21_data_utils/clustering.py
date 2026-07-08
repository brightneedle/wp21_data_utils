import fastjet


def antikt_jets(vectors, min_pt, r=0.4):
    jetdef = fastjet.JetDefinition(fastjet.antikt_algorithm, r)
    cluster = fastjet.ClusterSequence(vectors, jetdef)
    return cluster.inclusive_jets(min_pt)
