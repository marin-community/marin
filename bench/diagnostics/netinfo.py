"""Dump the node's L3 view: interfaces, routes, IB devices, and egress reachability."""

import json
import socket
import struct
import sys
import urllib.request


def routes() -> list[dict]:
    out = []
    with open("/proc/net/route") as fh:
        next(fh)
        for line in fh:
            f = line.split()
            def ip(h: str) -> str:
                return socket.inet_ntoa(struct.pack("<L", int(h, 16)))
            out.append({"iface": f[0], "dest": ip(f[1]), "gw": ip(f[2]), "mask": ip(f[7])})
    return out


def ifaces() -> list[str]:
    return [n for _, n in socket.if_nameindex()]


def probe(url: str, timeout: float = 8.0) -> str:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return f"OK http={r.status}"
    except Exception as exc:  # noqa: BLE001
        return f"FAIL {type(exc).__name__}: {exc}"


def tcp(host: str, port: int, timeout: float = 8.0) -> str:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return "OK connected"
    except Exception as exc:  # noqa: BLE001
        return f"FAIL {type(exc).__name__}: {exc}"


def main() -> int:
    import os
    print("[netinfo] host=" + socket.gethostname(), flush=True)
    print("[netinfo] ifaces=" + json.dumps(ifaces()), flush=True)
    print("[netinfo] routes=" + json.dumps(routes()), flush=True)
    try:
        ib = sorted(os.listdir("/sys/class/infiniband"))
    except OSError:
        ib = []
    print("[netinfo] infiniband=" + json.dumps(ib), flush=True)
    for target in os.environ.get("NETINFO_HTTP", "").split(","):
        if target:
            print(f"[netinfo] http {target}: {probe(target)}", flush=True)
    for target in os.environ.get("NETINFO_TCP", "").split(","):
        if target:
            host, _, port = target.partition(":")
            print(f"[netinfo] tcp {target}: {tcp(host, int(port))}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
