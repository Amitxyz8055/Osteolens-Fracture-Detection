# OsteoLens: a digital radiologist for fracture detection from X-rays

A client sends an X-ray over a network to a radiology service. The service answers whether a fracture is likely, shows where the model looked (Grad-CAM), and refuses images it was not built for. Course project for Digital Image Processing (DIP) and Computer Networks (CN), IIIT Raichur.

**Author:** Amit Meena (CS24B1006). *Study project, not a medical device.*

## What is in this repository
| Folder | Contents |
|---|---|
| `notebooks/` | Training notebook (Colab / Kaggle): duplicate removal, group-aware split, DenseNet121 training, evaluation, Grad-CAM, unsupported-image check, external test |
| `app/` | OsteoLens web app (Gradio) with prediction, Grad-CAM and advice text |
| `tele_radiology/` | CN part: TCP client-server, own protocol, channel model (noise, bandwidth limit, JPEG, packet loss), experiment script, server page and client page |
| `models/` | Where the trained model files go, with a guide to run the model (`models/README.md`) |
| `scripts/` | `predict.py`: run the model on one image or a folder from the command line |
| `tests/` | Automated tests (channel model, protocol, TCP round trip); run `pytest -q` |
| `docs/` | Architecture, protocol, methodology, results, how to reproduce |
| `results/` | Result tables used in the report (`report_tables.xlsx`) |

## Method in short
1. Remove exact duplicate images (MD5). Dataset 1: 10,581 files, 3,888 unique.
2. Group-aware 70/15/15 split, seed 42.
3. DenseNet121 (ImageNet weights) fine-tuned: weighted cross-entropy, AdamW lr 1e-4, cosine schedule, 15 epochs, batch 32, 224 px.
4. Decision threshold chosen on validation data for 95% sensitivity.
5. Grad-CAM on denseblock4; Mahalanobis distance on pooled features to flag unsupported images.
6. Tested on a second, independent dataset, then trained on both datasets together.

## Main results
- Internal test (584 images): TP 239, FN 4, TN 340, FP 1. Sensitivity 98.4%, specificity 99.7%, AUC 0.9998.
- Independent dataset before pooling: AUC 0.834, specificity 55.8%. After pooled training, specificity on that dataset: 81.4%.
- JPEG quality 25-50 cuts bytes by 84-89% with no accuracy loss; noise below 30 dB SNR hurts accuracy.
- Preprocessing ablation (CLAHE, median filter): only the baseline run is complete.

## Quick start
```
pip install -r requirements.txt
python scripts/predict.py xray.jpg --models models --heatmaps out/
```
Full instructions for the model folder: [`models/README.md`](models/README.md).

## Run the whole system
```
pip install -r requirements.txt
# 1. Train: open notebooks/fracture_trainer_colab_kaggle.ipynb in Colab or Kaggle
# 2. Tele-radiology server page (starts the TCP server on port 5050 and a web page on 7860)
cd tele_radiology
python server_ui.py --models "<folder with model files>" --region multi_region_combined
# 3. Client page (port 7870), in a second terminal
python client_ui.py
```

## Data and models
- **Datasets are not included.** Dataset 1 and Dataset 2 are public Kaggle X-ray fracture datasets: add the exact links here: `[DATASET 1 LINK]`, `[DATASET 2 LINK]`.
- **Trained model files** (`model_multi_region_combined.pt`, `ood_multi_region_combined.npz`, `results_multi_region_combined.json`, about 28 MB) are put in a `models/` folder. If they are not in this repository, they are available from `[MODEL LINK]`.

## Documentation
[Architecture](docs/ARCHITECTURE.md) · [Protocol](docs/PROTOCOL.md) · [Methodology](docs/METHODOLOGY.md) · [Results](docs/RESULTS.md) · [Reproduce](docs/REPRODUCE.md)

## Tests
`pip install pytest numpy pillow && pytest -q` 

## License
MIT, see `LICENSE`.

## Paper and slides
Research paper: [research_paper.pdf](research_paper.pdf) · Presentation: [presentation.pdf](presentation.pdf)
