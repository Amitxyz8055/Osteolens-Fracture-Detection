"""Tests that need only numpy and Pillow (no torch, no model files). Run:  pytest -q"""
import os, sys, threading
import numpy as np
from PIL import Image
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tele_radiology"))
import channel, server, client  # noqa: E402


def img(seed=0):
    return Image.fromarray((np.random.default_rng(seed).random((128, 128)) * 255).astype("uint8"))


def test_noise_matches_requested_snr():
    out = channel.apply_channel(img(), snr=20.0, cutoff=1.0, loss=0.0, jpeg=None, seed=1)
    assert abs(out["measured_snr_db"] - 20.0) < 1.5


def test_jpeg_makes_payload_smaller_than_png():
    png = channel.apply_channel(img(), None, 1.0, 0.0, None, 1)["bytes"]
    jpg = channel.apply_channel(img(), None, 1.0, 0.0, 30, 1)["bytes"]
    assert jpg < png


def test_packet_loss_counts():
    out = channel.apply_channel(img(), None, 1.0, 50.0, None, 3)
    assert out["packets_total"] > 0 and 0 < out["packets_lost"] < out["packets_total"]


def test_low_pass_removes_detail():
    a = np.asarray(img(), dtype=float)
    assert np.std(channel.low_pass(a, 0.1)) < np.std(a)


def test_tcp_round_trip_with_fake_model():
    srv = server.serve_background("127.0.0.1", 0, server.FakePredictor())
    port = srv.server_address[1]
    ch = channel.apply_channel(img(), None, 1.0, 0.0, 50, 1)
    sock, _ = client.connect("127.0.0.1", port)
    assert client.ping(sock, 2) >= 0
    hdr, _, _ = client.predict(sock, ch["payload"])
    sock.close(); srv.shutdown()
    assert hdr["ok"] and hdr["received_bytes"] == ch["bytes"] and hdr["label"] in ("fracture", "clear")
