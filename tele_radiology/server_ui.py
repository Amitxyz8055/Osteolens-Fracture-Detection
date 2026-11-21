"""PAGE 1 - SERVER dashboard. Starts the TCP model server AND a web page that shows its status.
  python server_ui.py --models "<models folder>" --region multi_region_combined
Open http://127.0.0.1:7860     (TCP port 5050 is what clients connect to; 7860 is only this web page)"""
import argparse, socket, time
import gradio as gr
import server as S


def build(ui_port, tcp_port, host, region, models, margin, fake):
    if fake: pred = S.FakePredictor()
    else:
        from inference import Predictor
        pred = Predictor(models, region, margin)
    S.serve_background(host, tcp_port, pred)
    try: my_ip = socket.gethostbyname(socket.gethostname())
    except OSError: my_ip = "unknown"

    def status():
        up = int(time.time() - S.STATS["started"])
        md = f"""### Server is **running**
| | |
|---|---|
| TCP address clients use | `{host}:{tcp_port}` (this computer's LAN address: `{my_ip}`) |
| Model | {pred.region} |
| Unsupported-image margin | {pred.margin} |
| Uptime | {up // 60} min {up % 60} s |
| Predictions served | {S.STATS['requests']} |
| Pings answered | {S.STATS['pings']} |
| Image bytes received | {S.STATS['bytes']:,} |
| Errors | {S.STATS['errors']} |
"""
        return md, [list(r) for r in S.LOG]

    with gr.Blocks(title="Tele-radiology SERVER") as demo:
        gr.Markdown("# Tele-radiology server\nThis page only *shows* the server. Clients connect over TCP using the address below. *Study project, not a medical device.*")
        info = gr.Markdown()
        tbl = gr.Dataframe(headers=["time", "client (ip:port)", "bytes", "answer", "P(fracture)", "server ms"], label="Recent requests (newest first)", interactive=False)
        gr.Timer(1.0).tick(status, None, [info, tbl])
        demo.load(status, None, [info, tbl])
    return demo


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1", help="0.0.0.0 = allow other computers")
    ap.add_argument("--port", type=int, default=5050); ap.add_argument("--ui-port", type=int, default=7860)
    ap.add_argument("--models", default="../app3/models"); ap.add_argument("--region", default="multi_region_combined")
    ap.add_argument("--ood-margin", type=float, default=0.9); ap.add_argument("--fake", action="store_true")
    a = ap.parse_args()
    build(a.ui_port, a.port, a.host, a.region, a.models, a.ood_margin, a.fake).launch(server_name="127.0.0.1", server_port=a.ui_port)
