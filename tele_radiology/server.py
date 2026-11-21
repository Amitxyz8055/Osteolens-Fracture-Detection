"""TCP server that runs the fracture model.  Start:  python server.py --models ../app3/models --region multi_region_combined
Protocol (see protocol.py). Operations: ping | regions | predict."""
import argparse, collections, io, socketserver, threading, time
from PIL import Image
from protocol import recv_msg, send_msg

PRED = None
LOG = collections.deque(maxlen=200)      # recent requests, read by server_ui.py
STATS = {"requests": 0, "pings": 0, "bytes": 0, "errors": 0, "started": time.time()}
LOCK = threading.Lock()            # one prediction at a time (Grad-CAM hooks are not thread safe)


class FakePredictor:               # ONLY for testing the network code without torch
    region, margin = "fake", 0.9
    def predict(self, img, heatmap=False):
        import numpy as np
        a = np.asarray(img.convert("L"), dtype=float)
        g = float(np.abs(np.diff(a, axis=1)).mean())
        p = 1 / (1 + np.exp(-(g - 6) / 2))
        r = {"region": "fake", "threshold": 0.5, "dist": 1.0, "limit": 2.0, "unsupported": False, "p": float(p),
             "label": "fracture" if p >= 0.5 else "clear"}
        if heatmap:
            b = io.BytesIO(); img.convert("RGB").save(b, "JPEG"); r["heatmap"] = b.getvalue()
        return r


class Handler(socketserver.BaseRequestHandler):
    def handle(self):
        who = "%s:%d" % self.client_address
        try:
            while True:
                try:
                    hdr, body = recv_msg(self.request)
                except ConnectionError:
                    break                                   # client closed the connection
                t0 = time.perf_counter(); op = hdr.get("op")
                if op == "ping":
                    STATS["pings"] += 1
                    send_msg(self.request, {"ok": True, "op": "pong"})
                elif op == "regions":
                    send_msg(self.request, {"ok": True, "regions": [PRED.region]})
                elif op == "predict":
                    img = Image.open(io.BytesIO(body))
                    with LOCK:
                        res = PRED.predict(img, heatmap=bool(hdr.get("heatmap")))
                    hm = res.pop("heatmap", b"")
                    res.update(ok=True, server_ms=round((time.perf_counter() - t0) * 1000, 1), received_bytes=len(body))
                    send_msg(self.request, res, hm)
                    STATS["requests"] += 1; STATS["bytes"] += len(body)
                    LOG.appendleft([time.strftime("%H:%M:%S"), who, len(body), res["label"], round(res["p"], 3), res["server_ms"]])
                    print(f"[{time.strftime('%H:%M:%S')}] {who}  {len(body):>8d} bytes  -> {res['label']:11s} p={res['p']:.3f}  {res['server_ms']} ms")
                else:
                    send_msg(self.request, {"ok": False, "error": f"unknown op {op!r}"})
        except Exception as e:
            try: send_msg(self.request, {"ok": False, "error": str(e)})
            except Exception: pass
            STATS["errors"] += 1
            print("error from", who, ":", e)


class Server(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True


def serve_background(host, port, predictor):
    """Used by server_ui.py: run the TCP server in a thread of the same process."""
    global PRED
    PRED = predictor
    srv = Server((host, port), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


def main():
    global PRED
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1", help="0.0.0.0 = accept connections from other computers on the network")
    ap.add_argument("--port", type=int, default=5050)
    ap.add_argument("--models", default="../app3/models")
    ap.add_argument("--region", default="multi_region_combined")
    ap.add_argument("--ood-margin", type=float, default=0.9)
    ap.add_argument("--fake", action="store_true", help="test mode without torch")
    a = ap.parse_args()
    if a.fake: PRED = FakePredictor()
    else:
        from inference import Predictor
        PRED = Predictor(a.models, a.region, a.ood_margin)
    with Server((a.host, a.port), Handler) as s:
        print(f"Fracture server listening on {a.host}:{a.port}  (region {PRED.region})  Ctrl+C to stop")
        try: s.serve_forever()
        except KeyboardInterrupt: print("bye")


if __name__ == "__main__":
    main()
