# Reproduce the results

1. Download the two datasets (links in the main README) and add them to Colab or Kaggle.
2. Open `notebooks/fracture_trainer_colab_kaggle.ipynb`. In the settings cell set the dataset paths. Use **Save and Run All** on Kaggle so the output files are kept.
3. Seed is 42; the notebook prints the duplicate counts, split sizes, metrics and per-dataset results.
4. Download `model_*.pt`, `results_*.json`, `ood_*.npz` from the output and put them in `models/`.
5. Network experiments: start `server.py` (see `models/README.md`), then
   `python tele_radiology/experiment.py --fractured <folder> --normal <folder> --n 40 --out channel_results.csv`.
6. Run the tests (no model needed): `pip install pytest numpy pillow` then `pytest -q`.
