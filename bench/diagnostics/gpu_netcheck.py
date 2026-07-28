"""GPU-node cross-cluster reachability probe.

Runs inside an Iris dev-GPU pod (hostNetwork, so the pod's stack is the GPU
node's stack). Dumps the node's interfaces and kernel routing table, then
listens on PORT while dialing every peer in PEERS, so one invocation on each
side tests both directions.

Diagnostic only. No bulk bytes, no proxy, no relay: it dials the peer's node
address directly.
"""

import fcntl
import os
import socket
import socketserver
import struct
import sys
import threading
import time

PORT = int(os.environ.get("PROBE_PORT", "29700"))
PEERS = [p for p in os.environ.get("PROBE_PEERS", "").split(",") if p]
SELF = os.environ.get("PROBE_SELF", socket.gethostname())
ROUNDS = int(os.environ.get("PROBE_ROUNDS", "3"))
DIAL_TIMEOUT = float(os.environ.get("PROBE_DIAL_TIMEOUT", "10"))
HOLD = float(os.environ.get("PROBE_HOLD", "45"))
SIOCGIFADDR = 0x8915


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


def interfaces() -> list[str]:
    out = []
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    for _, name in sorted(socket.if_nameindex()):
        try:
            packed = fcntl.ioctl(s.fileno(), SIOCGIFADDR, struct.pack("256s", name[:15].encode()))
            out.append(f"{name}={socket.inet_ntoa(packed[20:24])}")
        except OSError:
            continue
    s.close()
    return out


def routes() -> list[str]:
    def quad(hex_le: str) -> str:
        return socket.inet_ntoa(struct.pack("<L", int(hex_le, 16)))

    out = []
    with open("/proc/net/route") as fh:
        next(fh)
        for line in fh:
            f = line.split()
            dest, gw, mask = quad(f[1]), quad(f[2]), quad(f[7])
            out.append(f"{dest}/{mask} via {gw} dev {f[0]}")
    return out


def dial(peer: str) -> str:
    host, _, port = peer.partition(":")
    port = int(port or PORT)
    t0 = time.monotonic()
    try:
        with socket.create_connection((host, port), timeout=DIAL_TIMEOUT) as s:
            s.settimeout(DIAL_TIMEOUT)
            s.sendall(b"PING")
            reply = s.recv(64).decode(errors="replace")
        return f"OK {reply} elapsed={time.monotonic() - t0:.3f}s"
    except Exception as exc:  # noqa: BLE001 - the failure text is the result
        return f"FAIL {type(exc).__name__}: {exc} after={time.monotonic() - t0:.3f}s"


def main() -> int:
    print(f"[gpucheck] self={SELF} host={socket.gethostname()} arch={os.uname().machine}", flush=True)
    print(f"[gpucheck] self={SELF} interfaces: {' '.join(interfaces())}", flush=True)
    for route in routes():
        print(f"[gpucheck] self={SELF} route: {route}", flush=True)

    server = Server(("0.0.0.0", PORT), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    print(f"[gpucheck] self={SELF} listening 0.0.0.0:{PORT} peers={PEERS}", flush=True)

    for r in range(ROUNDS):
        time.sleep(15)
        for peer in PEERS:
            print(f"[gpucheck] round={r} {SELF} -> {peer}: {dial(peer)}", flush=True)

    print(f"[gpucheck] self={SELF} holding listener {HOLD}s", flush=True)
    time.sleep(HOLD)
    print(f"[gpucheck] self={SELF} done", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
