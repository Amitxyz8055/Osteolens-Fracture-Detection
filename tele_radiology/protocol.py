"""Our own tiny application-layer protocol, carried over TCP.
Every message = 8-byte prefix (header length, body length; both 32-bit big-endian) + JSON header + binary body.
"""
import json
import struct

MAX_BODY = 50 * 1024 * 1024


def recv_exact(sock, n):
    buf = bytearray()
    while len(buf) < n:                                  # TCP is a byte stream: one recv() may return less than asked
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("connection closed before the full message arrived")
        buf += chunk
    return bytes(buf)


def send_msg(sock, header, body=b""):
    hdr = json.dumps(header).encode("utf-8")
    sock.sendall(struct.pack("!II", len(hdr), len(body)) + hdr + body)


def recv_msg(sock):
    hl, bl = struct.unpack("!II", recv_exact(sock, 8))
    if hl > 1_000_000 or bl > MAX_BODY:
        raise ValueError("message too large")
    header = json.loads(recv_exact(sock, hl).decode("utf-8"))
    return header, recv_exact(sock, bl) if bl else b""
