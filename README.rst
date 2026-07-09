wp21_data_utils
===============

Data-handling utilities for Global Trigger studies.

The package provides lightweight helpers for preparing calorimeter cells,
physics vectors, eta-phi images, clustered jets, and simple calibration models.
It is intended to complement ``wp21_ml_utils`` by handling pre-processing and
post-processing steps around ML pipelines.

Features
--------

- Convert raw calorimeter cell records into vector arrays with compact layer
  labels.
- Histogram sparse vectors into dense eta-phi tower images and convert images
  back to jagged vector arrays.
- Cluster anti-kt jets with FastJet.
- Match reconstructed and truth jets in delta-R.
- Fit and apply binned or XGBoost-based jet pT calibrations.

Installation
------------

From the ``wp21_data_utils`` package directory:

.. code-block:: bash

   pip install -e .

For development and testing:

.. code-block:: bash

   pip install -e .[dev]

Dependencies
------------

- ``fastjet``
- ``numpy``
- ``scikit-image``
- ``scikit-learn``
- ``xgboost``

The code also uses ``awkward`` and ``vector`` data structures throughout the
public APIs.

Core modules
------------

- ``cells.py``: remove selected calorimeter gap regions, map sampling IDs to
  compact layer IDs, and convert cell records into massless four-vectors.
- ``image.py``: convert jagged vectors to dense eta-phi images, build
  multi-layer cell images, pad images with wrapped phi boundaries, and convert
  non-zero pixels back to vectors.
- ``clustering.py``: run anti-kt jet clustering through FastJet.
- ``calibration.py``: train and apply ``BinnedCalibration`` and
  ``BDTCalibration`` jet pT corrections.
- ``utils.py``: padding, sorting, vector-layout conversion, class-weight
  balancing, jagged-array reconstruction, and delta-R matching helpers.

Example workflow
----------------

Convert cell records to vectors, build a layered tower image, and cluster jets:

.. code-block:: python

   import awkward as ak

   from wp21_data_utils.cells import cells_to_vectors
   from wp21_data_utils.image import cell_vectors_to_image
   from wp21_data_utils.clustering import antikt_jets

   cells = ak.from_parquet("cells.parquet")

   cell_vectors = cells_to_vectors(cells)
   towers = cell_vectors_to_image(cell_vectors)
   jets = antikt_jets(cell_vectors, min_pt=20.0, r=0.4)

Fit and apply a simple binned calibration:

.. code-block:: python

   from wp21_data_utils.calibration import BinnedCalibration

   calibration = BinnedCalibration(max_dR=0.3)
   calibration.fit(truth_jets, recon_jets)

   calibrated_jets = calibration.predict(recon_jets)
   calibration.save("binned_calibration.json")

   restored = BinnedCalibration().load("binned_calibration.json")

License
-------

BSD 2-clause
