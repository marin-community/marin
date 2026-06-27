"""S4 latency harness: interactive round-trip vs Poll cadence vs held stream.

Spins the root server on loopback, dials it with the agent, issues N blocking
exec commands per configuration, and reports the distribution of the end-to-end
interactive latency a user would feel (`exec` call -> result in hand).

Run:  uv run python bench.py        (from this directory)
"""

from __future__ import annotations

import random
import statistics
import time

from agent_client import PollAgent, StreamAgent
from root_server import RootState, start_server

N_SAMPLES = 30
AUTH = True
BACKEND = "gcp-tpu-west"


def _summary(latencies: list[float]) -> dict[str, float]:
    s = sorted(latencies)
    return {
        "n": len(s),
        "mean": statistics.mean(s),
        "p50": s[len(s) // 2],
        "p95": s[min(len(s) - 1, int(0.95 * len(s)))],
        "min": s[0],
        "max": s[-1],
    }


def _fmt(label: str, st: dict[str, float]) -> str:
    return (f"{label:<34} n={st['n']:>3}  mean={st['mean']*1000:8.1f}ms  "
            f"p50={st['p50']*1000:8.1f}ms  p95={st['p95']*1000:8.1f}ms  "
            f"min={st['min']*1000:8.1f}ms  max={st['max']*1000:8.1f}ms")


def measure_unary_floor(url: str) -> dict[str, float]:
    """Bare loopback unary RTT (empty Poll) — the network floor under everything."""
    from agent_client import _client
    import remote_agent_pb2 as pb

    c = _client(url, auth=AUTH)
    # warmup
    c.poll(pb.PollRequest(backend_id=BACKEND, last_sync_id=1))
    lats = []
    for _ in range(50):
        t0 = time.perf_counter()
        c.poll(pb.PollRequest(backend_id=BACKEND, last_sync_id=1))
        lats.append(time.perf_counter() - t0)
    c.close()
    return _summary(lats)


def measure_poll(root: RootState, url: str, cadence: float, *, fast_follow: bool) -> dict[str, float]:
    agent = PollAgent(url, BACKEND, cadence, auth=AUTH, fast_follow=fast_follow)
    agent.start()
    while agent.poll_count < 1:
        time.sleep(0.01)
    rng = random.Random(1234)
    lats = []
    for i in range(N_SAMPLES):
        # Sample the phase uniformly across the poll cycle for a fair distribution.
        time.sleep(rng.uniform(0, cadence))
        exec_resp, latency = root.exec_piggyback([f"echo s{i}"], timeout=4 * cadence + 5)
        assert exec_resp.exit_code == 0 and exec_resp.stdout == f"ran: echo s{i}", exec_resp
        lats.append(latency)
    agent.stop()
    return _summary(lats)


def measure_stream(root: RootState, url: str) -> dict[str, float]:
    agent = StreamAgent(url, BACKEND, auth=AUTH)
    agent.start()
    time.sleep(0.1)
    lats = []
    for i in range(N_SAMPLES):
        exec_resp, latency = root.exec_stream([f"echo h{i}"], timeout=10)
        assert exec_resp.exit_code == 0 and exec_resp.stdout == f"ran: echo h{i}", exec_resp
        lats.append(latency)
        time.sleep(0.02)
    agent.stop()
    return _summary(lats)


def main() -> None:
    root = RootState(BACKEND)
    root.add_desired("attempt-aaa", '{"image": "demo"}')
    handle = start_server(root, auth=AUTH)
    url = handle.url
    print(f"root server up at {url}  (auth={'on' if AUTH else 'off'}, identity=system:controller)\n")

    floor = measure_unary_floor(url)
    print(_fmt("loopback unary Poll RTT (floor)", floor))
    print()

    print("== Poll piggyback (command DOWN one Poll, result UP the NEXT Poll) ==")
    for cadence in (0.5, 1.0, 2.0):
        st = measure_poll(root, url, cadence, fast_follow=False)
        print(_fmt(f"cadence={cadence:>3}s  no fast-follow", st))
    print()
    print("== Poll piggyback + fast-follow (extra Poll fired to return the result) ==")
    for cadence in (0.5, 1.0, 2.0):
        st = measure_poll(root, url, cadence, fast_follow=True)
        print(_fmt(f"cadence={cadence:>3}s  fast-follow", st))
    print()

    print("== Held stream (CommandStream push + unary ReportResult) ==")
    st = measure_stream(root, url)
    print(_fmt("held stream (cadence-independent)", st))

    handle.stop()


if __name__ == "__main__":
    main()
