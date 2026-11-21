"""Model side (needs torch). Same model, preprocessing and unsupported-image check as the OsteoLens app."""
import glob, io, json, os
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import cv2
from matplotlib import cm

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def dip_preprocess(img, mode, size):
    if mode == "none": return img
    g = np.array(img.convert("L").resize((size, size), Image.BILINEAR))
    if "denoise" in mode: g = cv2.medianBlur(g, 3)
    if "clahe" in mode: g = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(g)
    return Image.fromarray(g).convert("RGB")


class GradCAM:
    def __init__(self, model):
        self.model, self.acts, self.grads = model, None, None
        layer = model.features.denseblock4
        layer.register_forward_hook(lambda m, i, o: setattr(self, "acts", o))
        layer.register_full_backward_hook(lambda m, gi, go: setattr(self, "grads", go[0]))

    def __call__(self, x):
        self.model.eval(); self.model.zero_grad()
        out = self.model(x); out[0, 1].backward()
        w = self.grads.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((w * self.acts).sum(1)).squeeze().detach().cpu().numpy()
        return (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)


class Predictor:
    def __init__(self, models_dir, region, ood_margin=0.9):
        pt = os.path.join(models_dir, f"model_{region}.pt")
        cfg = json.load(open(os.path.join(models_dir, f"results_{region}.json")))
        m = models.densenet121(weights=None)
        m.classifier = nn.Linear(m.classifier.in_features, 2)
        m.load_state_dict(torch.load(pt, map_location=DEVICE)); m.to(DEVICE).eval()
        self.model, self.cfg, self.cam, self.region, self.margin = m, cfg, GradCAM(m), region, ood_margin
        self.tf = transforms.Compose([transforms.Resize((cfg["img_size"], cfg["img_size"])), transforms.Grayscale(3),
                                      transforms.ToTensor(), transforms.Normalize(cfg["mean"], cfg["std"])])
        self.ood = None
        p = os.path.join(models_dir, f"ood_{region}.npz")
        if os.path.exists(p):
            z = np.load(p); self.ood = (z["mu"].astype("float64"), z["prec"].astype("float64"), float(z["thresh"]))

    def predict(self, img, heatmap=False):
        img = img.convert("RGB"); size = self.cfg["img_size"]
        x = self.tf(dip_preprocess(img, self.cfg.get("preprocess", "none"), size)).unsqueeze(0).to(DEVICE)
        res = {"region": self.region, "threshold": float(self.cfg["threshold"]), "dist": None, "limit": None, "unsupported": False}
        if self.ood is not None:
            mu, prec, lim = self.ood
            with torch.no_grad():
                f = F.adaptive_avg_pool2d(F.relu(self.model.features(x)), 1).flatten(1).cpu().numpy()[0].astype("float64")
            res["dist"] = float(np.sqrt(max((f - mu) @ prec @ (f - mu), 0.0))); res["limit"] = lim * self.margin
            res["unsupported"] = res["dist"] > res["limit"]
        if heatmap:
            cam = self.cam(x.requires_grad_(True))
            with torch.no_grad(): p = float(F.softmax(self.model(x), 1)[0, 1])
            cam = np.array(Image.fromarray((cam * 255).astype("uint8")).resize((size, size), Image.BILINEAR)) / 255.0
            base = np.array(img.resize((size, size)).convert("L").convert("RGB")) / 255.0
            ov = (0.6 * base + 0.4 * cm.jet(cam)[..., :3]).clip(0, 1)
            buf = io.BytesIO(); Image.fromarray((ov * 255).astype("uint8")).save(buf, "JPEG", quality=90)
            res["heatmap"] = buf.getvalue()
        else:
            with torch.no_grad(): p = float(F.softmax(self.model(x), 1)[0, 1])
        res["p"] = p
        res["label"] = "unsupported" if res["unsupported"] else ("fracture" if p >= res["threshold"] else "clear")
        return res
