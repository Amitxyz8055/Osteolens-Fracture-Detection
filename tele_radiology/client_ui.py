"""PAGE 2 - CLIENT page. Damages an X-ray like a weak network, sends it to the server over TCP, shows the answer.
  python client_ui.py        then open http://127.0.0.1:7870   (server must already be running)"""
import io, time
import gradio as gr
import numpy as np
from PIL import Image
from channel import apply_channel, shannon_capacity_bps
from client import connect, link_time_s, ping, predict

LINKS = {"Dial-up 56 kbit/s": 56, "Rural 3G 1 Mbit/s": 1000, "Home Wi-Fi 20 Mbit/s": 20000, "Hospital LAN 100 Mbit/s": 100000}


def test_connection(host, port):
    try:
        sock, hs = connect(host, int(port)); rtt = ping(sock, 5); sock.close()
        return f"**Connected.** TCP handshake {hs:.1f} ms, ping RTT {rtt:.1f} ms."
    except OSError as e:
        return f"**Cannot reach the server** at {host}:{int(port)} ({e}). Start `server_ui.py` first."


def send(image, host, port, use_noise, snr, cutoff, use_jpeg, quality, loss, link, delay_ms, want_cam):
    if image is None:
        return None, None, None, "Upload an X-ray first."
    ch = apply_channel(image, snr if use_noise else None, cutoff, loss, int(quality) if use_jpeg else None, seed=1)
    recv_img = Image.fromarray(ch["received"].astype("uint8"))
    try:
        sock, hs = connect(host, int(port)); rtt = ping(sock, 3)
        hdr, hm, t_req = predict(sock, ch["payload"], heatmap=bool(want_cam)); sock.close()
    except OSError as e:
        return recv_img, None, None, f"**Cannot reach the server** at {host}:{int(port)} ({e})."
    if not hdr.get("ok"):
        return recv_img, None, None, "Server error: " + str(hdr.get("error"))
    net_ms = max(t_req - hdr["server_ms"], 0.01); kbps = LINKS[link]
    cap = shannon_capacity_bps(3000, ch["measured_snr_db"]) if np.isfinite(ch["measured_snr_db"]) else float("inf")
    label = {"fracture": "Possible fracture", "clear": "No fracture found", "unsupported": "Unsupported image"}[hdr["label"]]
    cam = Image.open(io.BytesIO(hm)) if hm else None
    answer = f"## {label}\nP(fracture) = **{hdr['p']:.3f}** (threshold {hdr['threshold']:.3f}) · feature distance {hdr['dist']:.1f} / limit {hdr['limit']:.1f}"
    nums = f"""| Channel | |
|---|---|
| Bytes sent ({ch['format'].upper()}) | {ch['bytes']:,} |
| Measured SNR / PSNR | {ch['measured_snr_db']:.1f} dB / {ch['psnr_db']:.1f} dB |
| Packets lost | {ch['packets_lost']} of {ch['packets_total']} |

| Network | |
|---|---|
| TCP handshake / ping RTT | {hs:.1f} ms / {rtt:.1f} ms |
| Total request + reply | {t_req:.1f} ms |
| = server (model{' + Grad-CAM' if want_cam else ''}) | {hdr['server_ms']} ms |
| = network | {net_ms:.1f} ms |
| Upload on **{link}**, {delay_ms:g} ms delay | {link_time_s(ch['bytes'], kbps, delay_ms):.2f} s |
| Shannon limit, 3 kHz channel at this SNR | {cap / 1000:.1f} kbit/s |

*On one computer the network part is loopback, so use the link row for realistic upload time.*"""
    return recv_img, cam, answer, nums


with gr.Blocks(title="Tele-radiology CLIENT") as demo:
    gr.Markdown("# Tele-radiology client\nPick an X-ray, choose how bad the network is, send it to the server over TCP. *Study project, not a medical device.*")
    with gr.Row():
        host = gr.Textbox("127.0.0.1", label="Server host"); port = gr.Number(5050, label="Server TCP port", precision=0)
        chk = gr.Button("Test connection"); status = gr.Markdown()
    chk.click(test_connection, [host, port], status)
    with gr.Row():
        with gr.Column():
            img = gr.Image(type="pil", label="Original X-ray")
            use_noise = gr.Checkbox(True, label="Add noise"); snr = gr.Slider(0, 40, 30, step=1, label="SNR (dB), lower = noisier")
            cutoff = gr.Slider(0.02, 1.0, 1.0, step=0.01, label="Bandwidth kept (1 = full)")
            use_jpeg = gr.Checkbox(True, label="JPEG compression"); quality = gr.Slider(1, 100, 50, step=1, label="JPEG quality")
            loss = gr.Slider(0, 60, 0, step=1, label="Packet loss (%)")
            link = gr.Dropdown(list(LINKS), value="Rural 3G 1 Mbit/s", label="Link type (for upload time)")
            delay = gr.Number(100, label="Link delay (ms)")
            cam = gr.Checkbox(True, label="Ask server for Grad-CAM")
            go = gr.Button("Send to server", variant="primary")
        with gr.Column():
            answer = gr.Markdown()
            with gr.Row():
                out_img = gr.Image(label="What the server received"); out_cam = gr.Image(label="Grad-CAM from server")
            nums = gr.Markdown()
    go.click(send, [img, host, port, use_noise, snr, cutoff, use_jpeg, quality, loss, link, delay, cam], [out_img, out_cam, answer, nums])

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7870)
