# Tele-radiology: DIP + Computer Networks in one project

A clinic with a weak network sends an X-ray to a remote AI server. How much channel damage can the fracture model tolerate?

```
 X-ray --> [ channel.py ]  --TCP-->  [ server.py + model ]
           noise (SNR dB)             DenseNet121, Grad-CAM,
           bandwidth limit (FFT)      unsupported-image check
           JPEG compression   <--TCP--  JSON answer (+ heatmap)
           packet loss
```

| File | What it is |
|---|---|
| `channel.py` | The "bad network": noise at a chosen SNR, FFT low-pass (limited bandwidth), JPEG compression, packet loss, Shannon capacity |
| `protocol.py` | Our own application-layer protocol over TCP (length-prefixed JSON header + binary body) |
| `server.py` + `inference.py` | TCP server that runs the model (same model files as the OsteoLens app) |
| `client.py` | Sends one X-ray through the channel, prints network numbers and the answer |
| `experiment.py` | Runs a whole sweep and writes `channel_results.csv` |
| `channel_lab.py` | Demo UI with sliders (port 7870) |

## Run it (Windows, same Python environment as the app)

Copy this folder next to your app folder (or point `--models` at your `models` folder).

1. **Terminal 1, server**
   `python server.py --models "..\fracture_app\models" --region multi_region_combined`
2. **Terminal 2, one image**
   `python client.py xray.jpg --snr 10 --jpeg 30 --loss 5 --link-kbps 256 --delay-ms 100 --heatmap-out heat.jpg`
3. **Experiment (for the report)**
   `python experiment.py --fractured "C:\...\Fractured" --normal "C:\...\Non-Fractured" --n 40`
4. **Demo UI**
   `python channel_lab.py` and open http://127.0.0.1:7870

No server needed for a quick self-test: `python server.py --fake` uses a dummy model (only to check the network code).

## Run it across two computers (real network)
Server computer: `python server.py --host 0.0.0.0 ...` (allow Python through the Windows firewall for port 5050).
Client computer: add `--host <server IP>`. Join both to the same Wi-Fi or a phone hotspot. Now handshake and RTT are real numbers.

## CN topics used
Client-server model, application layer and our own protocol design, TCP (connection, handshake, byte stream, why `recv_exact` loops), ports and sockets, threads (one per client), latency/RTT, throughput, transmission vs propagation delay, bandwidth-delay product, SNR in dB, Nyquist and Shannon limits, attenuation/noise/distortion, packet loss, (optional) Wireshark capture of the upload.

## DIP topics used
Fourier transform and ideal low-pass filtering (ringing), additive Gaussian noise, JPEG/DCT compression, PSNR, block loss and concealment, plus the CLAHE/median preprocessing inside the model.

## Notes
* On one computer the RTT is about 0 ms, so use the link formula (`--link-kbps`, `--delay-ms`) or two computers for realistic times.
* At very low SNR the measured SNR is a bit above the target because pixel values are clipped to 0-255. The tools report the measured value.
* The model was trained on most of these images, so absolute scores are optimistic. What matters is how much they drop relative to the baseline.

## Two web pages (server page + client page)
Terminal 1 (server page, also starts the TCP server):
    python server_ui.py --models "<your models folder>" --region multi_region_combined
    -> open http://127.0.0.1:7860
Terminal 2 (client page):
    python client_ui.py
    -> open http://127.0.0.1:7870   (press "Test connection", upload an X-ray, press "Send to server")
For two computers: run server_ui.py with --host 0.0.0.0 and type the server computer's IP in the client page.
