# Application protocol (over TCP)

Every message, in both directions:

| Bytes | Field |
|---|---|
| 4 | length of the JSON header (big-endian unsigned) |
| 4 | length of the binary body (big-endian unsigned) |
| n | JSON header (UTF-8) |
| m | binary body (image bytes or heat-map bytes; may be empty) |

TCP is a byte stream with no message boundaries, so the receiver uses `recv_exact`, which keeps reading until the requested number of bytes has arrived.

| Request header | Body | Reply header |
|---|---|---|
| `{"op":"ping"}` | none | `{"ok":true,"op":"pong"}` |
| `{"op":"regions"}` | none | `{"ok":true,"regions":[...]}` |
| `{"op":"predict","heatmap":true}` | image bytes (PNG or JPEG) | `ok`, `label`, `p`, `threshold`, `dist`, `limit`, `unsupported`, `server_ms`, `received_bytes`; body = Grad-CAM JPEG if requested |

Errors: `{"ok":false,"error":"..."}`.
