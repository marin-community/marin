# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""Width test: hundreds of concurrent trial-shaped sandboxes (brief section 7, step 5).

    python -m silo.width --image docker.io/library/python@sha256:... --concurrency 250

Each worker thread runs one trial the way the pipeline does: create a
network-blocked sandbox from one snapshot, run a build-like command through a
polled session (``timeout N`` wrapper, poll every 3 s), read cgroup telemetry with
the pipeline's own parser, then delete. A sampler records the broker's ceiling
(``/capacity``) against the sandboxes actually in flight, every few seconds.

The question it answers is not "does it work" (acceptance does that) but "what
stops it": the receipt shows peak concurrency reached, whether any create waited
at capacity, and every failure by class. A provider that tops out near 40 for a
new reason has not solved the problem.
"""

from __future__ import annotations

import argparse
import collections
import concurrent.futures
import json
import os
import statistics
import sys
import threading
import time
import uuid
from types import SimpleNamespace
from typing import Any

from silo.client import SiloClient
from silo.pipeline_contract import daytona_resources, daytona_snapshot, daytona_telemetry

# Build-like and SUSTAINED: recompile the image's stdlib, one worker per visible
# CPU (so it also exercises nproc == profile.cpu), over and over until the
# timeout. A build that finished in ten seconds could never show hundreds of
# sandboxes held at once; the first width run made exactly that mistake.
WORK_SCRIPT = """
import compileall, os, sys

# compileall's process pool uses forkserver, which re-imports __main__: without
# this guard every worker re-runs the compile and the pool breaks (width test 01).
if __name__ == "__main__":
    compileall.compile_dir(os.path.join(sys.prefix, "lib"), quiet=2, force=True, workers=0)
"""
WORKLOAD = "timeout {seconds} /bin/sh -c 'while :; do python3 /tmp/silo-width/work.py; done'"


def _sandbox_params(snapshot: str) -> SimpleNamespace:
    return SimpleNamespace(
        snapshot=snapshot,
        labels={"envgen": "1", "envgen_purpose": "silo-width"},
        ephemeral=True,
        auto_stop_interval=0,
        ttl_minutes=60,
        network_block_all=True,
        network_allow_list=None,
        domain_allow_list=None,
        env_vars=None,
    )


class Tracker:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.in_flight = 0
        self.peak_in_flight = 0
        self.outcomes: collections.Counter[str] = collections.Counter()
        self.create_seconds: list[float] = []
        self.trial_seconds: list[float] = []
        self.errors: list[str] = []
        self.telemetry_finite = 0
        self.print_logs = False

    def enter(self) -> None:
        with self.lock:
            self.in_flight += 1
            self.peak_in_flight = max(self.peak_in_flight, self.in_flight)

    def leave(self) -> None:
        with self.lock:
            self.in_flight -= 1


def _trial(client: SiloClient, snapshot: str, profile: Any, work_seconds: int, tracker: Tracker) -> None:
    started = time.monotonic()
    try:
        sandbox = client.create(_sandbox_params(snapshot), timeout=600)
    except Exception as error:
        with tracker.lock:
            tracker.outcomes[f"create_failed:{type(error).__name__}"] += 1
            tracker.errors.append(f"create: {error}"[:300])
        return
    created = time.monotonic()
    tracker.enter()
    try:
        with tracker.lock:
            tracker.create_seconds.append(created - started)
        sandbox.fs.upload_file(WORK_SCRIPT.encode(), "/tmp/silo-width/work.py")
        session = f"cap-harbor-{uuid.uuid4().hex[:12]}"
        sandbox.process.create_session(session)
        launched = sandbox.process.execute_session_command(
            session, SimpleNamespace(command=WORKLOAD.format(seconds=work_seconds), run_async=True), timeout=60
        )
        deadline = time.monotonic() + work_seconds + 120
        exit_code = None
        while time.monotonic() < deadline:
            exit_code = sandbox.process.get_session_command(session, launched.cmd_id).exit_code
            if exit_code is not None:
                break
            time.sleep(3)
        if tracker.print_logs:
            logs = sandbox.process.get_session_command_logs(session, launched.cmd_id)
            print(
                f"[width] trial exit={exit_code} stdout={logs.stdout[-1500:]!r} stderr={logs.stderr[-1500:]!r}",
                flush=True,
            )
        sandbox.process.delete_session(session)
        telemetry = daytona_telemetry.collect_cgroup_telemetry(lambda command: _exec(sandbox, command), profile)
        finite = all(telemetry["limit_evidence"][k] == "finite_cgroup_limit" for k in ("cpu", "memory"))
        with tracker.lock:
            tracker.telemetry_finite += int(finite)
            # The loop only ends when its timeout fires (124): that is the success case.
            tracker.outcomes["ok" if exit_code == 124 else f"workload_exit:{exit_code}"] += 1
            tracker.trial_seconds.append(time.monotonic() - started)
    except Exception as error:
        with tracker.lock:
            tracker.outcomes[f"trial_failed:{type(error).__name__}"] += 1
            tracker.errors.append(f"trial: {error}"[:300])
    finally:
        tracker.leave()
        try:
            sandbox.delete()
        except Exception as error:
            with tracker.lock:
                tracker.outcomes[f"delete_failed:{type(error).__name__}"] += 1


def _exec(sandbox: Any, command: str) -> dict[str, Any]:
    response = sandbox.process.exec(command, cwd="/", timeout=30)
    return {"exit": response.exit_code, "stdout": response.result}


def _percentiles(values: list[float]) -> dict[str, float]:
    if not values:
        return {}
    ordered = sorted(values)
    pick = lambda q: ordered[min(len(ordered) - 1, int(q * len(ordered)))]  # noqa: E731
    return {
        "p50": round(pick(0.5), 2),
        "p90": round(pick(0.9), 2),
        "p99": round(pick(0.99), 2),
        "max": round(ordered[-1], 2),
        "mean": round(statistics.fmean(ordered), 2),
        "n": len(ordered),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--image", required=True)
    parser.add_argument("--concurrency", type=int, default=250)
    parser.add_argument("--trials", type=int, default=None, help="total trials (default: 2 x concurrency)")
    parser.add_argument("--work-seconds", type=int, default=90)
    parser.add_argument("--sample-seconds", type=float, default=5.0)
    parser.add_argument("--min-hosts", type=int, default=1, help="wait (up to 10 min) for this many live hosts")
    parser.add_argument("--print-logs", action="store_true", help="print each trial's session output (small runs)")
    args = parser.parse_args(argv)
    total = args.trials or 2 * args.concurrency

    client = SiloClient(
        os.environ.get("SILO_BROKER_URL"),
        os.environ["SILO_API_TOKEN"],
        resolve_url=os.environ.get("SILO_BROKER_RESOLVE_URL"),
    )
    # Measure the fleet the run was sized for, not whichever hosts finished
    # booting first; and say so loudly if that fleet never arrives.
    deadline = time.monotonic() + 600
    while (live := client.capacity()["hosts_live"]) < args.min_hosts:
        if time.monotonic() > deadline:
            print(f"[width] only {live} of {args.min_hosts} hosts live after 600s; aborting", flush=True)
            return 2
        print(f"[width] waiting for hosts: {live}/{args.min_hosts}", flush=True)
        time.sleep(10)

    profile = daytona_resources.CANDIDATE_DEFAULT
    recipe = f"FROM {args.image}\n"
    snapshot = daytona_resources.snapshot_name("cap-harbor", recipe, profile)
    image = SimpleNamespace(dockerfile=lambda: recipe)
    try:
        client.snapshot.create(
            SimpleNamespace(
                name=snapshot,
                image=image,
                resources=SimpleNamespace(cpu=profile.cpu, memory=profile.memory_gb, disk=profile.disk_gb),
            ),
            timeout=600,
        )
    except Exception as error:
        if not daytona_snapshot.snapshot_conflict(error):
            raise
    daytona_snapshot.wait_for_snapshot_active(client.snapshot.get, snapshot, recipe)

    tracker = Tracker()
    tracker.print_logs = args.print_logs
    samples: list[dict[str, Any]] = []
    stop = threading.Event()
    t0 = time.monotonic()

    def sample() -> None:
        while not stop.is_set():
            try:
                ceiling = client.capacity()
                with tracker.lock:
                    row = {
                        "t": round(time.monotonic() - t0, 1),
                        "in_flight": tracker.in_flight,
                        "sandboxes_live": ceiling["sandboxes_live"],
                        "hosts_live": ceiling["hosts_live"],
                        "slots_free_candidate": ceiling["slots_free"]["candidate_4cpu_8gb"],
                        "creates_waiting": ceiling["creates_waiting_for_capacity"],
                        "done": sum(tracker.outcomes.values()),
                    }
                samples.append(row)
                print(f"[width] {json.dumps(row)}", flush=True)
            except Exception as error:
                print(f"[width] sample failed: {error}", flush=True)
            stop.wait(args.sample_seconds)

    sampler = threading.Thread(target=sample, daemon=True)
    sampler.start()
    capacity_before = client.capacity()
    with concurrent.futures.ThreadPoolExecutor(args.concurrency) as pool:
        futures = [pool.submit(_trial, client, snapshot, profile, args.work_seconds, tracker) for _ in range(total)]
        concurrent.futures.wait(futures)
    stop.set()
    sampler.join(timeout=30)

    peak_live = max((row["sandboxes_live"] for row in samples), default=0)
    waited = any(row["creates_waiting"] > 0 for row in samples)
    receipt = {
        "schema": "silo-width-v1",
        "image": args.image,
        "profile": profile.receipt(),
        "requested_concurrency": args.concurrency,
        "trials": total,
        "wall_seconds": round(time.monotonic() - t0, 1),
        "outcomes": dict(tracker.outcomes),
        "peak_in_flight_client": tracker.peak_in_flight,
        "peak_sandboxes_live_broker": peak_live,
        "any_create_waited_at_capacity": waited,
        "telemetry_finite": tracker.telemetry_finite,
        "create_seconds": _percentiles(tracker.create_seconds),
        "trial_seconds": _percentiles(tracker.trial_seconds),
        "capacity_before": {k: capacity_before[k] for k in ("hosts_live", "slots_free", "sandboxes_live")},
        "errors_sample": tracker.errors[:20],
        "samples": samples,
    }
    text = json.dumps(receipt, indent=2)
    out_dir = os.environ.get("IRIS_OUTPUT_DIR")
    if out_dir:
        with open(os.path.join(out_dir, "silo-width.json"), "w") as handle:
            handle.write(text)
    print("=== SILO WIDTH RECEIPT ===")
    print(json.dumps({k: v for k, v in receipt.items() if k != "samples"}, indent=2))
    print("=== END SILO WIDTH RECEIPT ===")
    ok = tracker.outcomes.get("ok", 0)
    print(f"[width] {ok}/{total} ok, peak in flight {tracker.peak_in_flight}, peak live {peak_live}")
    return 0 if ok == total else 1


if __name__ == "__main__":
    sys.exit(main())
