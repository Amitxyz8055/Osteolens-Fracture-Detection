"""OsteoLens - AI-assisted X-ray fracture screening (study project by Amit Meena).

Put each trained region in the models/ folder as a set of files:
    models/model_<region>.pt        (the trained model)
    models/results_<region>.json    (threshold and test scores)
    models/ood_<region>.npz         (optional: "unsupported image" check)
The app lists every region it finds in a dropdown.
"""
import glob, html, inspect, json, os, random
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import cv2
import gradio as gr
import matplotlib
matplotlib.use("Agg")
from matplotlib import cm

from advice import FRACTURE, NO_FRACTURE, UNSUPPORTED

APP_NAME = "OsteoLens"
TAGLINE = "AI-assisted X-ray fracture screening"
FOOTER = "This is a model made by Amit Meena, used for study purposes. It can make mistakes."

HERE = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(HERE, "models")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OOD_MARGIN = 0.85     # raise (e.g. 0.9) if too many good images are rejected, lower to be stricter


# ----------------------------------------------------------------------------- model side
def dip_preprocess(img, mode, size):
    # Same classical DIP preprocessing as in the training notebook. mode: none | clahe | denoise | clahe_denoise
    if mode == "none": return img
    g = np.array(img.convert("L").resize((size, size), Image.BILINEAR))
    if "denoise" in mode: g = cv2.medianBlur(g, 3)
    if "clahe" in mode: g = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(g)
    return Image.fromarray(g).convert("RGB")


def find_regions():
    regions = {}
    for pt in sorted(glob.glob(os.path.join(MODELS_DIR, "model_*.pt"))):
        region = os.path.basename(pt)[len("model_"):-len(".pt")]
        js = os.path.join(MODELS_DIR, f"results_{region}.json")
        if os.path.exists(js):
            regions[region] = (pt, js)
    return regions


class GradCAM:
    def __init__(self, model):
        self.model, self.acts, self.grads = model, None, None
        layer = model.features.denseblock4
        layer.register_forward_hook(lambda m, i, o: setattr(self, "acts", o))
        layer.register_full_backward_hook(lambda m, gi, go: setattr(self, "grads", go[0]))

    def __call__(self, x):
        self.model.eval(); self.model.zero_grad()
        out = self.model(x)
        out[0, 1].backward()                      # class 1 = fracture
        w = self.grads.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((w * self.acts).sum(1)).squeeze().detach().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam, float(F.softmax(out, 1)[0, 1].item())


_cache = {}
def load_region(region):
    if region not in _cache:
        pt, js = find_regions()[region]
        cfg = json.load(open(js))
        model = models.densenet121(weights=None)
        model.classifier = nn.Linear(model.classifier.in_features, 2)
        model.load_state_dict(torch.load(pt, map_location=DEVICE))
        model.to(DEVICE).eval()
        tf = transforms.Compose([
            transforms.Resize((cfg["img_size"], cfg["img_size"])), transforms.Grayscale(3),
            transforms.ToTensor(), transforms.Normalize(cfg["mean"], cfg["std"])])
        ood = None
        ood_path = os.path.join(MODELS_DIR, f"ood_{region}.npz")
        if os.path.exists(ood_path):
            z = np.load(ood_path)
            ood = (z["mu"].astype("float64"), z["prec"].astype("float64"), float(z["thresh"]))
        _cache[region] = (model, GradCAM(model), cfg, tf, ood)
    return _cache[region]


def predict_core(region, image):
    """Returns a dict describing the result."""
    model, cam_tool, cfg, tf, ood = load_region(region)
    img = image.convert("RGB")                                   # original, used for the picture
    x = tf(dip_preprocess(img, cfg.get("preprocess", "none"), cfg["img_size"])).unsqueeze(0).to(DEVICE)
    res = {"status": "ok", "region": region, "cfg": cfg, "dist": None, "limit": None, "overlay": None}

    if ood is not None:
        mu, prec, limit = ood
        with torch.no_grad():
            f = F.adaptive_avg_pool2d(F.relu(model.features(x)), 1).flatten(1).cpu().numpy()[0].astype("float64")
        d = float(np.sqrt(max((f - mu) @ prec @ (f - mu), 0.0)))
        res["dist"], res["limit"] = d, limit * OOD_MARGIN
        if d > res["limit"]:
            res["status"] = "unsupported"
            return res

    cam, p = cam_tool(x.requires_grad_(True))
    size = cfg["img_size"]
    cam = np.array(Image.fromarray((cam * 255).astype("uint8")).resize((size, size), Image.BILINEAR)) / 255.0
    base = np.array(img.resize((size, size)).convert("L").convert("RGB")) / 255.0
    overlay = (0.6 * base + 0.4 * cm.jet(cam)[..., :3]).clip(0, 1)
    res.update(p=p, thr=float(cfg["threshold"]), overlay=Image.fromarray((overlay * 255).astype("uint8")))
    res["status"] = "fracture" if p >= res["thr"] else "clear"
    return res


# ----------------------------------------------------------------------------- UI pieces
LOGO = """<svg viewBox="0 0 64 64" width="46" height="46" xmlns="http://www.w3.org/2000/svg" aria-hidden="true">
<defs><linearGradient id="g" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="#2dd4bf"/><stop offset="1" stop-color="#38bdf8"/></linearGradient></defs>
<rect width="64" height="64" rx="16" fill="url(#g)"/>
<g fill="#fff"><circle cx="18" cy="22" r="6"/><circle cx="24" cy="16" r="6"/><circle cx="46" cy="42" r="6"/><circle cx="40" cy="48" r="6"/>
<path d="M21 19 L43 45 L40 48 L18 22 Z"/></g></svg>"""

HEADER = f"""
<div class="ol-header">
  <div class="ol-brand">{LOGO}
    <div><div class="ol-title">{APP_NAME}</div><div class="ol-tag">{TAGLINE}</div></div></div>
  <div class="ol-pill">Study project</div>
</div>
<div class="ol-steps">
  <div class="ol-step"><span>1</span> Choose body part</div><div class="ol-arrow">&rarr;</div>
  <div class="ol-step"><span>2</span> Upload a clean X-ray</div><div class="ol-arrow">&rarr;</div>
  <div class="ol-step"><span>3</span> Analyze &amp; read the advice</div>
</div>"""

FOOT = f"""
<div class="ol-footer">
  <div class="ol-foot-main">{html.escape(FOOTER)}</div>
  <div class="ol-foot-sub">Not a medical device. It cannot diagnose. If you suspect a real injury, please see a doctor.</div>
</div>"""

EMPTY_RESULT = """<div class="ol-card ol-empty"><div class="ol-empty-icon">&#129460;</div>
<div class="ol-empty-title">Your result will appear here</div>
<div class="ol-empty-sub">Upload an X-ray and press <b>Analyze</b>.</div></div>"""

EMPTY_ASSIST = """<div class="ol-card"><div class="ol-assist-head"><div class="ol-avatar">&#129302;</div>
<div><div class="ol-assist-name">Lens Assistant</div><div class="ol-assist-sub">AI-written general tips</div></div></div>
<div class="ol-bubble ol-bubble-idle">Hi! Analyze an X-ray and I will share some general advice about what to do next.</div></div>"""


def result_html(res):
    st = res["status"]
    if st == "unsupported":
        badge, cls, title = "Unsupported image", "amber", "No answer given"
        body = ("This image looks different from the X-rays this model was trained on "
                "(another body part, a photo of a screen, a watermark or a very different style).")
        chips = [("Image check", f"{res['dist']:.1f}"), ("Limit", f"{res['limit']:.1f}")]
        meter = ""
    else:
        p, thr = res["p"], res["thr"]
        if st == "fracture":
            badge, cls, title = "Possible fracture", "red", "A fracture may be present"
            body = "Next step: please get this checked by an orthopedic doctor soon."
        else:
            badge, cls, title = "No fracture flagged", "green", "Nothing flagged in this image"
            body = "The tool can miss small cracks. If you still have pain or swelling, see a doctor."
        chips = [("Fracture probability", f"{p:.1%}"), ("Decision threshold", f"{thr:.2f}")]
        if res["dist"] is not None:
            chips.append(("Image check", f"{res['dist']:.1f} / {res['limit']:.1f}"))
        meter = (f'<div class="ol-meter"><div class="ol-meter-fill {cls}" style="width:{max(p, 0.01) * 100:.1f}%"></div>'
                 f'<div class="ol-meter-mark" style="left:{thr * 100:.1f}%" title="decision threshold"></div></div>'
                 f'<div class="ol-meter-labels"><span>0%</span><span>white line = decision threshold</span><span>100%</span></div>')
    chip_html = "".join(f'<div class="ol-chip"><div class="ol-chip-k">{html.escape(k)}</div>'
                        f'<div class="ol-chip-v">{html.escape(v)}</div></div>' for k, v in chips)
    cfg = res["cfg"]
    scores = (f'Model <b>{html.escape(res["region"])}</b> &middot; test AUC {cfg["auc"]} &middot; '
              f'sensitivity {cfg["sensitivity"]} &middot; specificity {cfg["specificity"]} '
              f'<span class="ol-muted">(on its own test set, may be optimistic)</span>')
    return (f'<div class="ol-card ol-pop"><div class="ol-badge {cls}">{badge}</div>'
            f'<div class="ol-res-title">{title}</div><div class="ol-res-body">{html.escape(body)}</div>'
            f'{meter}<div class="ol-chips">{chip_html}</div><div class="ol-scores">{scores}</div></div>')


def pool_for(status):
    return {"fracture": FRACTURE, "clear": NO_FRACTURE, "unsupported": UNSUPPORTED}[status]


def assistant_html(status, idx):
    pool = pool_for(status)
    text = html.escape(pool[idx % len(pool)])
    tone = {"fracture": "red", "clear": "green", "unsupported": "amber"}[status]
    return (f'<div class="ol-card ol-pop"><div class="ol-assist-head"><div class="ol-avatar">&#129302;</div>'
            f'<div><div class="ol-assist-name">Lens Assistant</div>'
            f'<div class="ol-assist-sub">AI-written general tips &middot; tip {idx % len(pool) + 1} of {len(pool)}</div></div></div>'
            f'<div class="ol-bubble {tone}">{text}</div>'
            f'<div class="ol-assist-note">General information only. It is not medical advice.</div></div>')


# ----------------------------------------------------------------------------- handlers
def analyze(region, image):
    if not region:
        msg = ('<div class="ol-card ol-empty"><div class="ol-empty-title">No models found</div><div class="ol-empty-sub">'
               'Put model_&lt;region&gt;.pt and results_&lt;region&gt;.json in the models folder.</div></div>')
        return msg, None, EMPTY_ASSIST, None, 0
    if image is None:
        msg = ('<div class="ol-card ol-empty"><div class="ol-empty-title">Please upload an X-ray first</div>'
               '<div class="ol-empty-sub">Drag an image into the box on the left, then press Analyze.</div></div>')
        return msg, None, EMPTY_ASSIST, None, 0
    res = predict_core(region, image)
    idx = random.randrange(len(pool_for(res["status"])))
    return result_html(res), res["overlay"], assistant_html(res["status"], idx), res["status"], idx


def another_tip(status, idx):
    if not status:
        return EMPTY_ASSIST, idx
    pool = pool_for(status)
    new = random.choice([i for i in range(len(pool)) if i != idx % len(pool)])
    return assistant_html(status, new), new


def clear_all():
    return None, EMPTY_RESULT, None, EMPTY_ASSIST, None, 0


# ----------------------------------------------------------------------------- styling
CSS = """
:root { --teal:#2dd4bf; --sky:#38bdf8; --red:#fb7185; --green:#4ade80; --amber:#fbbf24; --ink:#e6f1f5; --mute:#8aa4b2; }
html, body { background: radial-gradient(1200px 600px at 10% -10%, #11425a 0%, #0a1424 48%, #060a12 100%) fixed !important; min-height:100vh; }
.gradio-container { background: transparent !important; color: var(--ink); }
.gradio-container { max-width: 1180px !important; margin: 0 auto !important; font-family: 'Inter', 'Segoe UI', system-ui, sans-serif !important; }
footer { display: none !important; }
.block, .form, .panel { background: transparent !important; border: none !important; box-shadow: none !important; }
.ol-header { display:flex; align-items:center; justify-content:space-between; padding: 22px 6px 8px; }
.ol-brand { display:flex; align-items:center; gap:14px; }
.ol-title { font-size: 30px; font-weight: 800; letter-spacing:.3px; background: linear-gradient(90deg,#5eead4,#7dd3fc); -webkit-background-clip:text; background-clip:text; color: transparent; }
.ol-tag { color: var(--mute); font-size: 14px; margin-top: 2px; }
.ol-pill { border:1px solid rgba(125,211,252,.35); color:#bae6fd; padding:6px 14px; border-radius:999px; font-size:12px; background: rgba(56,189,248,.08); }
.ol-steps { display:flex; align-items:center; gap:10px; flex-wrap:wrap; padding: 6px 6px 18px; color: var(--mute); font-size: 13px; }
.ol-step { display:flex; align-items:center; gap:8px; } .ol-step span { width:22px; height:22px; border-radius:50%; display:grid; place-items:center; font-size:12px; font-weight:700; color:#042f2e; background: linear-gradient(135deg,var(--teal),var(--sky)); }
.ol-arrow { opacity:.5; }
.ol-card { background: rgba(255,255,255,.045); border:1px solid rgba(255,255,255,.09); border-radius: 20px; padding: 20px 22px; backdrop-filter: blur(10px); box-shadow: 0 10px 30px rgba(0,0,0,.25); transition: transform .25s ease, border-color .25s ease; }
.ol-card:hover { border-color: rgba(45,212,191,.35); }
.ol-pop { animation: pop .45s cubic-bezier(.2,.8,.2,1); }
@keyframes pop { from { opacity:0; transform: translateY(10px) scale(.98);} to { opacity:1; transform:none; } }
.ol-empty { text-align:center; padding: 34px 20px; } .ol-empty-icon { font-size: 40px; opacity:.9; } .ol-empty-title { font-size:18px; font-weight:700; margin-top:6px; } .ol-empty-sub { color:var(--mute); margin-top:6px; font-size:14px; }
.ol-badge { display:inline-block; padding:6px 14px; border-radius:999px; font-weight:700; font-size:13px; letter-spacing:.3px; }
.ol-badge.red, .ol-bubble.red { background: rgba(251,113,133,.14); color:#fecdd3; border:1px solid rgba(251,113,133,.45); }
.ol-badge.green, .ol-bubble.green { background: rgba(74,222,128,.12); color:#bbf7d0; border:1px solid rgba(74,222,128,.4); }
.ol-badge.amber, .ol-bubble.amber { background: rgba(251,191,36,.12); color:#fde68a; border:1px solid rgba(251,191,36,.4); }
.ol-res-title { font-size: 22px; font-weight: 800; margin: 12px 0 4px; } .ol-res-body { color:#b6cad4; font-size:14.5px; line-height:1.5; }
.ol-meter { position:relative; height:12px; border-radius:999px; background: rgba(255,255,255,.08); margin-top:18px; overflow:visible; }
.ol-meter-fill { height:100%; border-radius:999px; transition: width .8s cubic-bezier(.2,.8,.2,1); } .ol-meter-fill.red { background: linear-gradient(90deg,#fb923c,#fb7185); } .ol-meter-fill.green { background: linear-gradient(90deg,#2dd4bf,#4ade80); }
.ol-meter-mark { position:absolute; top:-5px; width:3px; height:22px; background:#fff; border-radius:2px; opacity:.85; }
.ol-meter-labels { display:flex; justify-content:space-between; color:var(--mute); font-size:11px; margin-top:6px; }
.ol-chips { display:flex; gap:10px; flex-wrap:wrap; margin-top:14px; } .ol-chip { background: rgba(255,255,255,.05); border:1px solid rgba(255,255,255,.08); border-radius:12px; padding:8px 12px; } .ol-chip-k { color:var(--mute); font-size:11px; } .ol-chip-v { font-weight:700; font-size:16px; }
.ol-scores { margin-top:14px; color:#9db4c0; font-size:12px; line-height:1.5; } .ol-muted { opacity:.7; }
.ol-assist-head { display:flex; align-items:center; gap:12px; } .ol-avatar { width:42px; height:42px; border-radius:50%; display:grid; place-items:center; font-size:22px; background: linear-gradient(135deg,var(--teal),var(--sky)); }
.ol-assist-name { font-weight:800; } .ol-assist-sub { color:var(--mute); font-size:12px; }
.ol-bubble { position:relative; margin-top:14px; padding:14px 16px; border-radius: 4px 18px 18px 18px; font-size:15px; line-height:1.55; }
.ol-bubble-idle { background: rgba(255,255,255,.06); border:1px solid rgba(255,255,255,.1); color:#c9dbe4; }
.ol-assist-note { color:var(--mute); font-size:11.5px; margin-top:10px; }
.ol-footer { text-align:center; padding: 26px 8px 30px; } .ol-foot-main { font-weight:700; color:#c8e3ee; font-size:14px; } .ol-foot-sub { color:var(--mute); font-size:12px; margin-top:6px; }
button.primary, .gr-button-primary { background: linear-gradient(90deg,#14b8a6,#0ea5e9) !important; border:none !important; color:#04222a !important; font-weight:800 !important; border-radius:14px !important; transition: transform .15s ease, box-shadow .2s ease !important; box-shadow: 0 8px 22px rgba(14,165,233,.28) !important; }
button.primary:hover { transform: translateY(-2px); box-shadow: 0 12px 28px rgba(14,165,233,.38) !important; }
button.secondary { background: rgba(255,255,255,.06) !important; border:1px solid rgba(255,255,255,.14) !important; color: var(--ink) !important; border-radius:14px !important; transition: transform .15s ease !important; }
button.secondary:hover { transform: translateY(-2px); }
.ol-upload, .ol-heat { border-radius: 18px !important; overflow:hidden; border: 1px dashed rgba(125,211,252,.35) !important; background: rgba(255,255,255,.03) !important; }
.ol-heat { border-style: solid !important; border-color: rgba(255,255,255,.1) !important; }
@media (max-width: 760px) { .ol-title { font-size: 24px; } .ol-steps { display:none; } }
"""
JS = "() => { document.body.classList.add('dark'); }"


def build():
    regions = list(find_regions().keys())
    sig = inspect.signature(gr.Blocks.__init__).parameters
    blocks_kw, launch_kw = {"title": f"{APP_NAME} | X-ray fracture screening"}, {}
    extras = {"css": CSS, "js": JS, "theme": gr.themes.Soft(primary_hue="teal", neutral_hue="slate")}
    for k, v in extras.items():                    # Gradio 6 moved these to launch()
        (blocks_kw if k in sig else launch_kw)[k] = v
    with gr.Blocks(**blocks_kw) as demo:
        gr.HTML(HEADER)
        status, idx = gr.State(None), gr.State(0)
        with gr.Row(equal_height=False):
            with gr.Column(scale=5):
                region = gr.Dropdown(regions, value=regions[0] if regions else None, label="Body part / model")
                image = gr.Image(type="pil", label="Upload a clean X-ray image (not a photo of a screen)",
                                 height=380, elem_classes="ol-upload")
                with gr.Row():
                    go = gr.Button("Analyze", variant="primary", scale=3)
                    clr = gr.Button("Clear", variant="secondary", scale=1)
            with gr.Column(scale=6):
                out_html = gr.HTML(EMPTY_RESULT)
                out_img = gr.Image(label="Where the model looked (red = most important)", interactive=False,
                                   height=300, elem_classes="ol-heat")
                out_assist = gr.HTML(EMPTY_ASSIST)
                tip = gr.Button("Another tip", variant="secondary")
        gr.HTML(FOOT)
        go.click(analyze, [region, image], [out_html, out_img, out_assist, status, idx])
        tip.click(another_tip, [status, idx], [out_assist, idx])
        clr.click(clear_all, None, [image, out_html, out_img, out_assist, status, idx])
    return demo, launch_kw


if __name__ == "__main__":
    demo, launch_kw = build()
    demo.launch(**launch_kw)
