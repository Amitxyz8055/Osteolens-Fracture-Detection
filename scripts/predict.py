"""Run the trained model on one X-ray or a whole folder, without any web page or network.

  python scripts/predict.py xray.jpg --models models --region multi_region_combined
  python scripts/predict.py my_xrays/ --models models --out predictions.csv --heatmaps heatmaps/

Prints one line per image (answer, probability, unsupported-image distance) and can save a CSV and Grad-CAM pictures.
"""
import argparse, csv, os, sys
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tele_radiology"))
EXT = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path", help="an image file or a folder of images")
    ap.add_argument("--models", default="models", help="folder with model_<region>.pt, results_<region>.json, ood_<region>.npz")
    ap.add_argument("--region", default="multi_region_combined")
    ap.add_argument("--ood-margin", type=float, default=0.9, help="lower = rejects more unusual images")
    ap.add_argument("--out", default=None, help="save results to this CSV file")
    ap.add_argument("--heatmaps", default=None, help="folder to save Grad-CAM overlays")
    a = ap.parse_args()

    from inference import Predictor                      # needs torch
    pred = Predictor(a.models, a.region, a.ood_margin)
    if os.path.isdir(a.path):
        files = sorted(os.path.join(a.path, f) for f in os.listdir(a.path) if f.lower().endswith(EXT))
    else:
        files = [a.path]
    if not files:
        raise SystemExit("no images found in " + a.path)
    if a.heatmaps: os.makedirs(a.heatmaps, exist_ok=True)

    rows = []
    for f in files:
        r = pred.predict(Image.open(f), heatmap=bool(a.heatmaps))
        hm = r.pop("heatmap", b"")
        if hm: open(os.path.join(a.heatmaps, os.path.splitext(os.path.basename(f))[0] + "_gradcam.jpg"), "wb").write(hm)
        rows.append([os.path.basename(f), r["label"], round(r["p"], 4), round(r["threshold"], 4),
                     None if r["dist"] is None else round(r["dist"], 2), None if r["limit"] is None else round(r["limit"], 2)])
        print(f"{rows[-1][0]:35s} {r['label']:12s} p={r['p']:.3f} (threshold {r['threshold']:.3f})")
    if a.out:
        with open(a.out, "w", newline="") as fh:
            w = csv.writer(fh); w.writerow(["image", "answer", "p_fracture", "threshold", "feature_distance", "distance_limit"]); w.writerows(rows)
        print("saved", a.out)


if __name__ == "__main__":
    main()
