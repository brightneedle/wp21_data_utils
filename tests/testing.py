import os
from pathlib import Path
import matplotlib.pyplot as plt


plt.rcParams["figure.dpi"] = 150
plt.rcParams["figure.constrained_layout.use"] = True

TEST_DIR = Path(__file__).parent
DATA_DIR = TEST_DIR / "data"
OUTPUT_DIR = TEST_DIR / "test_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def test_imports():
    import pkgutil
    import importlib
    import wp21_data_utils

    package = wp21_data_utils.__name__
    for _, module_name, _ in pkgutil.walk_packages(
        wp21_data_utils.__path__, package + "."
    ):
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

    plt.matshow(towers[0].sum(axis=-1))
    plt.savefig(OUTPUT_DIR / "test_towering.png")
    plt.close


def test_converters():
    import numpy as np
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
