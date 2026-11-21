# Architecture

```
 client_ui.py / client.py                              server_ui.py / server.py
 ┌──────────────────────────┐   TCP port 5050    ┌──────────────────────────────────┐
 │ X-ray                    │ ─────────────────► │ recv_msg (protocol.py)           │
 │  └ channel.py            │  8-byte length     │  └ inference.py Predictor        │
 │    low-pass, noise,      │  prefix + JSON     │     DenseNet121 → P(fracture)    │
 │    JPEG, packet loss     │  header + bytes    │     Mahalanobis → unsupported?   │
 │                          │ ◄───────────────── │     Grad-CAM (denseblock4)       │
 │ answer + heat map + times│   JSON + JPEG      │ send_msg                         │
 └──────────────────────────┘                    └──────────────────────────────────┘
```

| Part | File | Job |
|---|---|---|
| Channel model | `tele_radiology/channel.py` | Ideal FFT low-pass, additive white Gaussian noise at a target SNR, JPEG or PNG encoding, packet (8-row block) loss, PSNR, Shannon and Nyquist formulas |
| Protocol | `tele_radiology/protocol.py` | Length-prefixed messages: two big-endian 32-bit lengths, a JSON header, a binary body; `recv_exact` loops until all bytes arrive |
| Server | `tele_radiology/server.py` | `ThreadingTCPServer`; operations `ping`, `regions`, `predict`; one prediction at a time (Grad-CAM hooks are not thread safe) |
| Model side | `tele_radiology/inference.py` | Preprocessing, DenseNet121, unsupported-image check, Grad-CAM |
| Server page | `tele_radiology/server_ui.py` | Starts the server in a thread and shows status and a request log |
| Client page | `tele_radiology/client_ui.py` | Sliders for the channel, sends the image, shows answer, heat map, time split |
| Experiments | `tele_radiology/experiment.py` | 21 channel settings over a labelled image set, writes a CSV |

Time split on the client: total request time = server time (model, Grad-CAM) + network time. On one computer the network part is loopback, so the page also computes the upload time on a chosen link: bits / rate + delay.
