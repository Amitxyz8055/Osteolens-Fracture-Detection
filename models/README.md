# Model folder: how to use the trained model

This folder holds the trained model. Put these three files here (all use the same name ending, called the *region*; ours is `multi_region_combined`):

| File | What it is |
|---|---|
| `model_multi_region_combined.pt` | DenseNet121 weights (PyTorch `state_dict`), about 28 MB, 2 outputs: 0 = no fracture, 1 = fracture |
| `results_multi_region_combined.json` | Settings the model needs: `img_size`, `mean`, `std`, `threshold` (chosen for 95% sensitivity on validation), `preprocess` (`none`, `clahe`, `denoise` or `clahe_denoise`) and the saved test metrics |
| `ood_multi_region_combined.npz` | Unsupported-image check: `mu` (mean feature), `prec` (inverse covariance), `thresh` (99th percentile of validation distances) |

The `.npz` file is optional. Without it the model still answers, but it cannot reject unusual images.

If the files are not in this repository, download them from the link in the main README and copy them here.

## 1. Check that it works (one image)
```
pip install -r requirements.txt
python scripts/predict.py path/to/xray.jpg --models models --region multi_region_combined --heatmaps out/
```
Output: `fracture`, `clear` or `unsupported`, the probability, and the threshold. With `--heatmaps` the Grad-CAM picture is saved in `out/`.

## 2. A whole folder, saved as a table
```
python scripts/predict.py my_xrays/ --models models --out predictions.csv
```

## 3. Use it from your own Python code
```python
import sys; sys.path.insert(0, "tele_radiology")
from PIL import Image
from inference import Predictor

pred = Predictor("models", "multi_region_combined", ood_margin=0.9)
r = pred.predict(Image.open("xray.jpg"), heatmap=True)
print(r["label"], r["p"], r["threshold"], r["dist"], r["limit"])
open("gradcam.jpg", "wb").write(r["heatmap"])
```

## 4. As a service (client and server over TCP)
```
cd tele_radiology
python server_ui.py --models ../models --region multi_region_combined     # server page on :7860, TCP on :5050
python client_ui.py                                                       # client page on :7870
```

## 5. Web app
```
cd app
python app.py       # copy or link the three model files into app/models first
```

## How the answer is decided
1. The image is resized to `img_size`, converted to 3-channel grayscale and normalised with `mean` and `std` (plus CLAHE or median filter if `preprocess` says so).
2. DenseNet121 gives P(fracture). If P is at least `threshold`, the answer is `fracture`, otherwise `clear`.
3. The pooled 1024-value feature vector is compared with the training features using the Mahalanobis distance. If it is larger than `thresh * ood_margin`, the answer becomes `unsupported`.
4. Grad-CAM is computed from `denseblock4` and drawn over the image (red = strongest response).

## Train your own model
Run `notebooks/fracture_trainer_colab_kaggle.ipynb` in Colab or Kaggle. It writes the three files above into its output; download them and copy them here. To add another body part, train again with a new region name and place its three files here.

## Notes
- Trained on 3,888 unique images of Dataset 1 plus Dataset 2 (pooled version). Metrics are in `results_*.json` and `docs/RESULTS.md`.
- Study project, not a medical device.
