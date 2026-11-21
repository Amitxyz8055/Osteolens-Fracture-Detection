"""Client. Applies the channel to an X-ray, sends it over TCP, prints the result and the network numbers.
Example:  python client.py xray.jpg --snr 10 --jpeg 30 --loss 5 --link-kbps 256 --delay-ms 100"""
import argparse, socket, time
from PIL import Image
from channel import apply_channel, shannon_capacity_bps
from protocol import recv_msg, send_msg


def connect(host, port):
    t0 = time.perf_counter()
    s = socket.create_connection((host, port), timeout=60)
    s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    return s, (time.perf_counter() - t0) * 1000          # time for the TCP three-way handshake


def ping(sock, n=5):
    rtts = []
    for _ in range(n):
        t = time.perf_counter(); send_msg(sock, {"op": "ping"}); recv_msg(sock)
        rtts.append((time.perf_counter() - t) * 1000)
    return sum(rtts) / len(rtts)


def predict(sock, payload, heatmap=False):
    t = time.perf_counter()
    send_msg(sock, {"op": "predict", "heatmap": heatmap}, payload)
    hdr, body = recv_msg(sock)
    return hdr, body, (time.perf_counter() - t) * 1000


def link_time_s(n_bytes, link_kbps, delay_ms):
    """Time the upload WOULD take on a slower link: transmission delay (bits / rate) + one-way propagation delay."""
    return n_bytes * 8 / (link_kbps * 1000.0) + delay_ms / 1000.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("--host", default="127.0.0.1"); ap.add_argument("--port", type=int, default=5050)
    ap.add_argument("--snr", type=float, default=None, help="noise: signal-to-noise ratio in dB (lower = noisier)")
    ap.add_argument("--bandwidth", type=float, default=1.0, help="low-pass cut-off, fraction of full bandwidth (0.1 = very limited)")
    ap.add_argument("--jpeg", type=int, default=None, help="JPEG quality 1-100 (omit = lossless PNG)")
    ap.add_argument("--loss", type=float, default=0.0, help="packet loss in percent")
    ap.add_argument("--conceal", choices=["black", "repeat"], default="black")
    ap.add_argument("--link-kbps", type=float, default=1000.0); ap.add_argument("--delay-ms", type=float, default=50.0)
    ap.add_argument("--heatmap-out", default=None, help="save the Grad-CAM picture the server returns")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    ch = apply_channel(Image.open(a.image), a.snr, a.bandwidth, a.loss, a.jpeg, a.seed, a.conceal)
    sock, hs = connect(a.host, a.port)
    rtt = ping(sock)
    hdr, hm, t_req = predict(sock, ch["payload"], heatmap=bool(a.heatmap_out))
    sock.close()
    if not hdr.get("ok"): raise SystemExit("server error: " + str(hdr.get("error")))
    if a.heatmap_out and hm: open(a.heatmap_out, "wb").write(hm); print("heatmap saved to", a.heatmap_out)

    print("\n--- what the channel did ---")
    print(f"payload       : {ch['bytes']:,} bytes ({ch['format'].upper()})")
    print(f"measured SNR  : {ch['measured_snr_db']:.1f} dB | PSNR {ch['psnr_db']:.1f} dB")
    if ch["packets_total"]: print(f"packets lost  : {ch['packets_lost']} of {ch['packets_total']}")
    print("\n--- network numbers ---")
    net_ms = max(t_req - hdr["server_ms"], 0.01)           # time on the network = total time minus the server's thinking time
    print(f"TCP handshake : {hs:.1f} ms | ping RTT {rtt:.1f} ms")
    print(f"request+reply : {t_req:.1f} ms in total = {hdr['server_ms']} ms server (model{' + Grad-CAM' if a.heatmap_out else ''}) + {net_ms:.1f} ms network")
    print(f"throughput    : {ch['bytes'] * 8 / 1000 / (net_ms / 1000):.0f} kbit/s (network part only; on one computer this is loopback, not a real link)")
    print(f"on a {a.link_kbps:g} kbit/s link with {a.delay_ms:g} ms delay this upload takes {link_time_s(ch['bytes'], a.link_kbps, a.delay_ms):.2f} s")
    print("\n--- model answer ---")
    print(f"{hdr['label'].upper()}  (p fracture = {hdr['p']:.3f}, threshold {hdr['threshold']:.3f})")


if __name__ == "__main__":
    main()
