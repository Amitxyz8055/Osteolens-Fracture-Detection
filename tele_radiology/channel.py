"""Channel simulator: damages an X-ray the way a real network/link could.
DIP side: FFT low-pass filtering, additive noise, JPEG compression, block (packet) loss.
CN side : SNR in dB, bandwidth, Shannon capacity, packet loss.
"""
import io
import numpy as np
from PIL import Image


def to_gray(img, max_side=512):
    g = img.convert("L")
    if max(g.size) > max_side:                       # keep experiments fast and packets comparable
        s = max_side / max(g.size)
        g = g.resize((max(1, round(g.width * s)), max(1, round(g.height * s))), Image.BILINEAR)
    return np.asarray(g, dtype=np.float64)


def low_pass(a, cutoff):
    """Ideal low-pass filter in the frequency domain. cutoff = fraction of the Nyquist frequency kept (1.0 = no limit)."""
    if cutoff >= 1.0:
        return a
    h, w = a.shape
    spec = np.fft.fftshift(np.fft.fft2(a))
    yy, xx = np.ogrid[:h, :w]
    r = np.hypot((yy - h / 2) / (h / 2), (xx - w / 2) / (w / 2))
    spec[r > cutoff] = 0
    return np.clip(np.real(np.fft.ifft2(np.fft.ifftshift(spec))), 0, 255)


def add_awgn(a, snr_db, rng):
    """Additive white Gaussian noise so that the signal-to-noise ratio equals snr_db."""
    sigma = np.sqrt(np.mean(a ** 2) / (10 ** (snr_db / 10.0)))
    return np.clip(a + rng.normal(0.0, sigma, a.shape), 0, 255)


def packet_loss(a, loss_pct, rng, rows_per_packet=8, conceal="black"):
    """The image is sent as packets of `rows_per_packet` pixel rows. Each packet is lost with probability loss_pct %.
    conceal = 'black' (lost rows become black) or 'repeat' (copy the last received row)."""
    out, lost, total = a.copy(), 0, 0
    for y in range(0, a.shape[0], rows_per_packet):
        total += 1
        if rng.random() < loss_pct / 100.0:
            lost += 1
            out[y:y + rows_per_packet] = out[y - 1] if (conceal == "repeat" and y > 0) else 0
    return out, lost, total


def encode(a, jpeg_quality=None):
    """Turns pixels into the bytes that are sent. JPEG (lossy, smaller) or PNG (lossless)."""
    buf = io.BytesIO()
    im = Image.fromarray(np.clip(a, 0, 255).astype("uint8"), "L")
    if jpeg_quality is None:
        im.save(buf, "PNG"); fmt = "png"
    else:
        im.save(buf, "JPEG", quality=int(jpeg_quality)); fmt = "jpeg"
    return buf.getvalue(), fmt


def snr_db(ref, test):
    err = np.sum((test - ref) ** 2)
    return float("inf") if err == 0 else float(10 * np.log10(np.sum(ref ** 2) / err))


def psnr_db(ref, test):
    mse = np.mean((test - ref) ** 2)
    return float("inf") if mse == 0 else float(10 * np.log10(255.0 ** 2 / mse))


def shannon_capacity_bps(bandwidth_hz, snr_db_value):
    return bandwidth_hz * np.log2(1 + 10 ** (snr_db_value / 10.0))


def nyquist_rate_bps(bandwidth_hz, levels):
    return 2 * bandwidth_hz * np.log2(levels)


def apply_channel(img, snr=None, cutoff=1.0, loss=0.0, jpeg=None, seed=0, conceal="black", max_side=512):
    """img: PIL image. snr: dB or None (no noise). cutoff: 1.0 = full bandwidth. loss: packet loss %. jpeg: quality or None.
    Returns payload bytes + information about what the channel did."""
    rng = np.random.default_rng(seed)
    ref = to_gray(img, max_side)
    a = low_pass(ref, cutoff)
    if snr is not None:
        a = add_awgn(a, snr, rng)
    lost = total = 0
    if loss and loss > 0:
        a, lost, total = packet_loss(a, loss, rng, conceal=conceal)
    payload, fmt = encode(a, jpeg)
    received = np.asarray(Image.open(io.BytesIO(payload)).convert("L"), dtype=np.float64)
    return {"payload": payload, "format": fmt, "bytes": len(payload), "received": received, "reference": ref,
            "measured_snr_db": snr_db(ref, received), "psnr_db": psnr_db(ref, received),
            "packets_lost": lost, "packets_total": total}
