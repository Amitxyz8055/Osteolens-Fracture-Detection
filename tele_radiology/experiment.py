"""How much channel damage can the model tolerate?  Sends X-rays through the channel to the TCP server, one damage type at a time.
Example:  python experiment.py --fractured "C:\\data\\Fractured" --normal "C:\\data\\Non-Fractured" --n 40
Needs the server running (python server.py ...). Writes channel_results.csv"""
import argparse, csv, os, random, time
import numpy as np
from PIL import Image
from channel import apply_channel
from client import connect, ping, predict

EXT = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
SWEEPS = [
    ("baseline", [("original", {})]),
    ("noise_snr_db", [(str(s), {"snr": s}) for s in (40, 30, 20, 10, 5, 0)]),
    ("bandwidth_cutoff", [(str(c), {"cutoff": c}) for c in (0.5, 0.25, 0.125, 0.0625, 0.03)]),
    ("jpeg_quality", [(str(q), {"jpeg": q}) for q in (95, 75, 50, 25, 10)]),
    ("packet_loss_pct", [(str(l), {"loss": l}) for l in (5, 10, 20, 40)]),
]


def list_images(folder):
    out = []
    for dp, _, fs in os.walk(folder):
        out += [os.path.join(dp, f) for f in fs if f.lower().endswith(EXT)]
    return sorted(out)


def auc(y, p):
    y, p = np.asarray(y), np.asarray(p)
    pos, neg = p[y == 1], p[y == 0]
    if len(pos) == 0 or len(neg) == 0: return float("nan")
    return float((pos[:, None] > neg[None, :]).mean() + 0.5 * (pos[:, None] == neg[None, :]).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fractured", required=True); ap.add_argument("--normal", required=True)
    ap.add_argument("--n", type=int, default=40, help="images per class")
    ap.add_argument("--host", default="127.0.0.1"); ap.add_argument("--port", type=int, default=5050)
    ap.add_argument("--out", default="channel_results.csv"); ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    rnd = random.Random(a.seed)
    fr, nm = list_images(a.fractured), list_images(a.normal)
    assert fr and nm, "no images found in the folders"
    fr, nm = rnd.sample(fr, min(a.n, len(fr))), rnd.sample(nm, min(a.n, len(nm)))
    items = [(p, 1) for p in fr] + [(p, 0) for p in nm]
    imgs = {p: Image.open(p).convert("RGB") for p, _ in items}
    print(f"{len(fr)} fractured + {len(nm)} normal images, {sum(len(l) for _, l in SWEEPS)} channel settings")

    sock, hs = connect(a.host, a.port)
    rtt = ping(sock, 10)
    print(f"TCP handshake {hs:.1f} ms, ping RTT {rtt:.1f} ms")
    rows = []
    for sweep, levels in SWEEPS:
        for level, kw in levels:
            ys, ps, unsup, nbytes, snrs, psnrs, lat, srv = [], [], 0, [], [], [], [], []
            thr = 0.5
            for i, (path, y) in enumerate(items):
                ch = apply_channel(imgs[path], seed=a.seed * 100003 + i, **kw)
                hdr, _, t = predict(sock, ch["payload"])
                if not hdr.get("ok"): raise SystemExit("server error: " + str(hdr.get("error")))
                thr = hdr["threshold"]; ys.append(y); ps.append(hdr["p"]); unsup += hdr["label"] == "unsupported"
                nbytes.append(ch["bytes"]); lat.append(t); srv.append(hdr["server_ms"])
                if np.isfinite(ch["measured_snr_db"]): snrs.append(ch["measured_snr_db"])
                if np.isfinite(ch["psnr_db"]): psnrs.append(ch["psnr_db"])
            ys, ps = np.array(ys), np.array(ps); pred = ps >= thr
            tp, fn = int(((pred == 1) & (ys == 1)).sum()), int(((pred == 0) & (ys == 1)).sum())
            tn, fp = int(((pred == 0) & (ys == 0)).sum()), int(((pred == 1) & (ys == 0)).sum())
            row = {"sweep": sweep, "level": level, "images": len(ys), "TP": tp, "FN": fn, "TN": tn, "FP": fp,
                   "auc": round(auc(ys, ps), 4), "sensitivity": round(tp / max(tp + fn, 1), 4),
                   "specificity": round(tn / max(tn + fp, 1), 4), "accuracy": round((tp + tn) / len(ys), 4),
                   "unsupported_rate": round(unsup / len(ys), 4), "mean_bytes": int(np.mean(nbytes)),
                   "mean_snr_db": round(float(np.mean(snrs)), 2) if snrs else "", "mean_psnr_db": round(float(np.mean(psnrs)), 2) if psnrs else "",
                   "mean_request_ms": round(float(np.mean(lat)), 1), "mean_server_ms": round(float(np.mean(srv)), 1), "mean_network_ms": round(float(np.mean(lat)) - float(np.mean(srv)), 2), "threshold": round(thr, 4), "ping_rtt_ms": round(rtt, 2)}
            rows.append(row)
            print(f"{sweep:18s} {level:>9s}  AUC {row['auc']:.3f}  sens {row['sensitivity']:.3f}  spec {row['specificity']:.3f}  "
                  f"unsupported {row['unsupported_rate']:.0%}  {row['mean_bytes']:>8,d} B")
    sock.close()
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print("saved", a.out)


if __name__ == "__main__":
    main()
