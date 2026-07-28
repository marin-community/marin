"""Cross-cluster TCP reachability probe.

Runs in a hostNetwork pod on each CoreWeave cluster. Each instance both listens
on PORT and dials every peer in TARGETS, so one deployment tests all directions.
Diagnostic only: no bandwidth is measured here.
"""

import os
import socket
import socketserver
import sys
import threading
import time

PORT = int(os.environ.get("PROBE_PORT", "29700"))
TARGETS = [t for t in os.environ.get("PROBE_TARGETS", "").split(",") if t]
SELF = os.environ.get("PROBE_SELF", socket.gethostname())
ROUNDS = int(os.environ.get("PROBE_ROUNDS", "3"))
DIAL_TIMEOUT = float(os.environ.get("PROBE_DIAL_TIMEOUT", "8"))
HOLD = float(os.environ.get("PROBE_HOLD", "60"))


class Handler(socketserver.BaseRequestHandler):
    def handle(self):
        self.request.settimeout(5)
        try:
            self.request.recv(64)
        except OSError:
            pass
        self.request.sendall(f"PONG {SELF}".encode())


class Server(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True


def dial(target: str) -> str:
    host, _, port = target.partition(":")
    port = int(port or PORT)
    t0 = time.monotonic()
    try:
        with socket.create_connection((host, port), timeout=DIAL_TIMEOUT) as s:
            s.settimeout(DIAL_TIMEOUT)
            s.sendall(b"PING")
            reply = s.recv(64).decode(errors="replace")
        return f"OK {reply} rtt_connect_and_reply={time.monotonic() - t0:.3f}s"
    except Exception as exc:  # noqa: BLE001 - diagnostic wants the failure text
        return f"FAIL {type(exc).__name__}: {exc} after={time.monotonic() - t0:.3f}s"


def main() -> int:
    server = Server(("0.0.0.0", PORT), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    print(f"[netprobe] self={SELF} listening 0.0.0.0:{PORT} targets={TARGETS}", flush=True)

    addrs = []
    for line in os.popen("ip -o -4 addr show 2>/dev/null").read().splitlines():
        parts = line.split()
        if len(parts) > 3:
            addrs.append(f"{parts[1]}={parts[3]}")
    print(f"[netprobe] self={SELF} ipv4={' '.join(addrs)}", flush=True)

    for r in range(ROUNDS):
        time.sleep(20)
        for target in TARGETS:
            print(f"[netprobe] round={r} {SELF} -> {target}: {dial(target)}", flush=True)

    print(f"[netprobe] holding listener {HOLD}s for peers", flush=True)
    time.sleep(HOLD)
    print("[netprobe] done", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
