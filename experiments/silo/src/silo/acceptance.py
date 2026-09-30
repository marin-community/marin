# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""Acceptance run against a live deployment: brief section 7, steps 2 and 3.

    python -m silo.acceptance --image docker.io/library/python@sha256:...

Runs as an ordinary (unprivileged) Iris job on the same cluster as the hosts,
with SILO_API_TOKEN and SILO_BROKER_RESOLVE_URL in its environment -- exactly
what a pipeline worker would have.

Every verdict is computed with the capability pipeline's OWN code where one
exists (silo.pipeline_contract, vendored byte for byte): recipe identity,
readiness, deletion observation, cgroup telemetry parsing. The isolation proof
runs the same probe inside a sandbox and in this job's own process; the second is
the positive control, without which a failing probe inside proves nothing.

The receipt is printed between markers and written to $IRIS_OUTPUT_DIR. The job
exits non-zero if any check fails, so `iris job describe` shows the tail.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import contextlib
import hashlib
import io
import json
import os
import sys
import time
import uuid
from types import SimpleNamespace
from typing import Any

from silo.client import Sandbox, SiloClient
from silo.pipeline_contract import daytona_resources, daytona_snapshot, daytona_telemetry

# Run inside the sandbox AND in this process. Stdlib python only.
ISOLATION_PROBE = r"""
import json, socket, ssl, urllib.request
out = {}
try:
    socket.getaddrinfo("pypi.org", 443); out["dns"] = "reached"
except Exception as e:
    out["dns"] = "blocked:" + type(e).__name__
try:
    s = socket.create_connection(("1.1.1.1", 443), timeout=5); s.close(); out["tcp_ip"] = "reached"
except Exception as e:
    out["tcp_ip"] = "blocked:" + type(e).__name__
try:
    urllib.request.urlopen("https://pypi.org/simple/", timeout=8).read(64); out["https_name"] = "reached"
except Exception as e:
    out["https_name"] = "blocked:" + type(e).__name__
out["interfaces"] = sorted(n for _, n in socket.if_nameindex()) if hasattr(socket, "if_nameindex") else None
print(json.dumps(out))
"""


class Receipt:
    def __init__(self) -> None:
        self.data: dict[str, Any] = {"schema": "silo-acceptance-v1", "started_at": _now()}
        self.checks: dict[str, bool] = {}

    def check(self, name: str, passed: bool, /, **detail: Any) -> bool:
        self.checks[name] = bool(passed)
        print(f"[acceptance] {'PASS' if passed else 'FAIL'} {name} {json.dumps(detail, default=str)[:600]}", flush=True)
        if detail:
            self.data.setdefault("detail", {})[name] = detail
        return passed


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _snapshot_params(name: str, recipe: str, profile: Any) -> SimpleNamespace:
    image = SimpleNamespace(dockerfile=lambda: recipe)
    return SimpleNamespace(
        name=name,
        image=image,
        resources=SimpleNamespace(cpu=profile.cpu, memory=profile.memory_gb, disk=profile.disk_gb),
    )


def _sandbox_params(snapshot: str, purpose: str) -> SimpleNamespace:
    return SimpleNamespace(
        snapshot=snapshot,
        labels={"envgen": "1", "envgen_purpose": purpose},
        ephemeral=True,
        auto_stop_interval=0,
        ttl_minutes=60,
        network_block_all=True,
        network_allow_list=None,
        domain_allow_list=None,
        env_vars=None,
    )


def _exec(sandbox: Sandbox, command: str, timeout: int = 60) -> dict[str, Any]:
    response = sandbox.process.exec(command, cwd="/", timeout=timeout)
    return {"exit": response.exit_code, "stdout": response.result}


def _make_snapshot(client: SiloClient, receipt: Receipt, prefix: str, image: str, profile: Any) -> str:
    recipe = f"FROM {image}\n"
    name = daytona_resources.snapshot_name(prefix, recipe, profile)
    started = time.monotonic()
    try:
        client.snapshot.create(_snapshot_params(name, recipe, profile), timeout=600)
    except Exception as error:
        if not daytona_snapshot.snapshot_conflict(error):
            raise
    resolved = daytona_snapshot.wait_for_snapshot_active(client.snapshot.get, name, recipe)
    evidence = daytona_snapshot.validate_snapshot_recipe(resolved, expected_name=name, expected_dockerfile=recipe)
    receipt.check(
        f"snapshot_active_and_identity_verified[{prefix}]",
        evidence["evidence"] == "provider-build-info-exact-match",
        snapshot=name,
        seconds=round(time.monotonic() - started, 2),
        identity=evidence,
    )
    return name


def _trial(client: SiloClient, receipt: Receipt, snapshot: str, profile: Any, runtime: str) -> None:
    tag = f"{runtime}/{profile.cpu}cpu"
    started = time.monotonic()
    sandbox = client.create(_sandbox_params(snapshot, "silo-acceptance"), timeout=600, runtime=runtime)
    receipt.check(
        f"create[{tag}]", True, sandbox_id=sandbox.id, host=sandbox.host_id, seconds=round(time.monotonic() - started, 2)
    )
    try:
        observed = client.get(sandbox.id)
        receipt.check(f"network_block_all_attested[{tag}]", observed.network_block_all is True)

        result = _exec(sandbox, 'python3 -c "print(6*7)"')
        receipt.check(f"exec[{tag}]", result["exit"] == 0 and result["stdout"].strip() == "42", **result)

        # Exit codes must arrive exactly: the pipeline reads 124 as "timed out".
        # (nerdctl exec collapses every non-zero exit to 1 -- spike phase0c.)
        seven = _exec(sandbox, "exit 7")
        receipt.check(f"exit_code_exact_one_shot[{tag}]", seven["exit"] == 7, **seven)
        fidelity = f"cap-harbor-{uuid.uuid4().hex[:12]}"
        sandbox.process.create_session(fidelity)
        launched = sandbox.process.execute_session_command(
            fidelity, SimpleNamespace(command="echo out; echo err >&2; exit 124", run_async=True), timeout=60
        )
        deadline = time.monotonic() + 60
        while sandbox.process.get_session_command(fidelity, launched.cmd_id).exit_code is None:
            if time.monotonic() > deadline:
                break
            time.sleep(0.5)
        code = sandbox.process.get_session_command(fidelity, launched.cmd_id).exit_code
        logs = sandbox.process.get_session_command_logs(fidelity, launched.cmd_id)
        sandbox.process.delete_session(fidelity)
        # stderr exactly what the command wrote: no runtime noise appended.
        receipt.check(
            f"exit_code_and_stderr_exact_session[{tag}]",
            code == 124 and logs.stdout == "out\n" and logs.stderr == "err\n",
            exit_code=code,
            stdout=logs.stdout,
            stderr=logs.stderr,
        )

        telemetry = daytona_telemetry.collect_cgroup_telemetry(
            lambda command: _exec(sandbox, command, timeout=30), profile
        )
        evidence = telemetry["limit_evidence"]
        receipt.check(
            f"cgroup_limits_finite[{tag}]",
            evidence["cpu"] == "finite_cgroup_limit" and evidence["memory"] == "finite_cgroup_limit",
            cgroup_version=telemetry["cgroup_version"],
            observed=telemetry["observed"],
            limit_evidence=evidence,
        )
        nproc = _exec(sandbox, "nproc")
        receipt.check(f"nproc_matches_profile[{tag}]", nproc["stdout"].strip() == str(profile.cpu), nproc=nproc)

        session = f"cap-harbor-{uuid.uuid4().hex[:12]}"
        sandbox.process.create_session(session)
        launched = sandbox.process.execute_session_command(
            session,
            SimpleNamespace(command="timeout 60 /bin/sh -c 'sleep 3; echo session-ok'", run_async=True),
            timeout=60,
        )
        polls, deadline = 0, time.monotonic() + 90
        while sandbox.process.get_session_command(session, launched.cmd_id).exit_code is None:
            polls += 1
            if time.monotonic() > deadline:
                break
            time.sleep(1)
        exit_code = sandbox.process.get_session_command(session, launched.cmd_id).exit_code
        logs = sandbox.process.get_session_command_logs(session, launched.cmd_id)
        again = sandbox.process.get_session_command_logs(session, launched.cmd_id)
        sandbox.process.delete_session(session)
        receipt.check(
            f"session_polled_without_consuming_output[{tag}]",
            exit_code == 0 and logs.stdout.strip() == "session-ok" and again.stdout == logs.stdout and polls >= 1,
            exit_code=exit_code,
            polls=polls,
        )

        payload = os.urandom(4 * 1024 * 1024)
        sandbox.fs.upload_file(payload, "/tmp/silo-acceptance/blob.bin")
        back = sandbox.fs.download_file("/tmp/silo-acceptance/blob.bin")
        receipt.check(f"fs_roundtrip_4mib[{tag}]", hashlib.sha256(back).digest() == hashlib.sha256(payload).digest())

        sandbox.fs.upload_file(ISOLATION_PROBE.encode(), "/tmp/silo-acceptance/probe.py")
        inside = _exec(sandbox, "python3 /tmp/silo-acceptance/probe.py", timeout=60)
        verdict = _last_json_line(inside["stdout"])
        # A probe that did not run is not evidence of blocking: it must have
        # produced a verdict, and every verdict must be "blocked".
        blocked = verdict is not None and all(
            str(verdict.get(k, "")).startswith("blocked") for k in ("dns", "tcp_ip", "https_name")
        )
        receipt.check(f"egress_blocked_inside_sandbox[{tag}]", blocked, inside=verdict or inside)
    finally:
        sandbox.delete()
    state, observations = daytona_snapshot.wait_for_sandbox_deletion(client, sandbox.id)
    receipt.check(f"deletion_observed_async[{tag}]", state == "not_found", observations=observations)


def _built_snapshot(client: SiloClient, receipt: Receipt, image: str) -> None:
    """The RUN path: build with network, then run the result with none.

    Same shape as daytona_policy.verifier_snapshot_recipe (USER root + a pinned
    pip install), smaller. The build needs the network (pip fetches the wheel);
    the sandbox started from the result must not have it, yet the package it
    installed must import.
    """
    recipe = f"FROM {image}\nUSER root\nRUN python3 -m pip install --no-cache-dir --no-deps six==1.16.0\n"
    profile = daytona_resources.VERIFIER_DEFAULT
    name = daytona_resources.snapshot_name("cap-verifier", recipe, profile)
    started = time.monotonic()
    # A failed build leaves its snapshot in "error" under the same content-derived
    # name, blocking a rebuild -- Daytona's semantics too, which is why dt.py
    # deletes it after a failure. Do the same: clear an errored record, build once.
    try:
        if client.snapshot.get(name).state == "error":
            client.snapshot.delete(name)
    except Exception as error:
        if not daytona_snapshot.snapshot_not_found(error):
            raise
    try:
        client.snapshot.create(_snapshot_params(name, recipe, profile), timeout=900)
    except Exception as error:
        if not daytona_snapshot.snapshot_conflict(error):
            receipt.check(
                "run_recipe_built",
                False,
                error=f"{type(error).__name__}: {error}"[:800],
                build_logs=client.snapshot.build_logs(name)[-2000:],
            )
            return
    resolved = daytona_snapshot.wait_for_snapshot_active(client.snapshot.get, name, recipe)
    receipt.check(
        "run_recipe_built",
        resolved.state == "active",
        snapshot=name,
        ref=resolved.ref,
        seconds=round(time.monotonic() - started, 1),
    )
    sandbox = client.create(_sandbox_params(name, "silo-acceptance-built"), timeout=600)
    try:
        result = _exec(sandbox, 'python3 -c "import six; print(six.__version__)"')
        receipt.check("run_recipe_package_usable_offline", result["stdout"].strip() == "1.16.0", **result)
    finally:
        sandbox.delete()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--image", required=True, help="pinned digest, e.g. the slot-3 python image")
    parser.add_argument("--runtimes", default="runsc,runc")
    parser.add_argument("--fanout", type=int, default=6, help="concurrent sandboxes for the uniqueness check")
    parser.add_argument("--min-hosts", type=int, default=1, help="wait (up to 10 min) for this many live hosts")
    args = parser.parse_args(argv)

    receipt = Receipt()
    client = SiloClient(
        os.environ.get("SILO_BROKER_URL"),
        os.environ["SILO_API_TOKEN"],
        resolve_url=os.environ.get("SILO_BROKER_RESOLVE_URL"),
    )
    # Judge a running fleet, not one still booting (acceptance silo3-02 started
    # before its only host had registered).
    deadline = time.monotonic() + 600
    while client.capacity()["hosts_live"] < args.min_hosts and time.monotonic() < deadline:
        time.sleep(10)
    receipt.data["capacity_before"] = client.capacity()
    receipt.check(
        "broker_reachable_and_hosts_live",
        receipt.data["capacity_before"]["hosts_live"] >= 1,
        hosts_live=receipt.data["capacity_before"]["hosts_live"],
        slots_free=receipt.data["capacity_before"]["slots_free"],
    )

    # Positive control first: the probe CAN reach the network from outside a sandbox.
    namespace: dict[str, Any] = {}
    control = _capture(lambda: exec(ISOLATION_PROBE, namespace))
    control_verdict = _last_json_line(control) or {}
    receipt.check(
        "positive_control_network_reachable_outside",
        all(control_verdict.get(k) == "reached" for k in ("dns", "tcp_ip", "https_name")),
        outside=control_verdict,
    )

    candidate = daytona_resources.CANDIDATE_DEFAULT
    verifier = daytona_resources.VERIFIER_DEFAULT
    candidate_snapshot = _make_snapshot(client, receipt, "cap-harbor", args.image, candidate)
    verifier_snapshot = _make_snapshot(client, receipt, "cap-verifier", args.image, verifier)

    for runtime in args.runtimes.split(","):
        for snapshot, profile in ((candidate_snapshot, candidate), (verifier_snapshot, verifier)):
            try:
                _trial(client, receipt, snapshot, profile, runtime)
            except Exception as error:
                receipt.check(
                    f"trial_completed[{runtime}/{profile.cpu}cpu]", False, error=f"{type(error).__name__}: {error}"[:800]
                )

    try:
        _built_snapshot(client, receipt, args.image)
    except Exception as error:
        receipt.check("run_recipe_completed", False, error=f"{type(error).__name__}: {error}"[:800])

    # brief 3.2: fresh, unique sandboxes, created concurrently from one snapshot.
    started = time.monotonic()
    with concurrent.futures.ThreadPoolExecutor(args.fanout) as pool:
        sandboxes = list(
            pool.map(lambda _: client.create(_sandbox_params(verifier_snapshot, "silo-fanout")), range(args.fanout))
        )
    ids = [s.id for s in sandboxes]
    receipt.check(
        "concurrent_ids_unique", len(set(ids)) == len(ids), ids=ids, seconds=round(time.monotonic() - started, 2)
    )
    for sandbox in sandboxes:
        sandbox.delete()

    receipt.data["capacity_after"] = client.capacity()
    receipt.data["finished_at"] = _now()
    receipt.data["checks"] = receipt.checks
    receipt.data["passed"] = all(receipt.checks.values())
    text = json.dumps(receipt.data, indent=2, default=str)
    out_dir = os.environ.get("IRIS_OUTPUT_DIR")
    if out_dir:
        with open(os.path.join(out_dir, "silo-acceptance.json"), "w") as handle:
            handle.write(text)
    print("=== SILO ACCEPTANCE RECEIPT ===")
    print(text)
    print("=== END SILO ACCEPTANCE RECEIPT ===")
    failed = [name for name, ok in receipt.checks.items() if not ok]
    print(f"[acceptance] {len(receipt.checks) - len(failed)}/{len(receipt.checks)} checks passed; failed: {failed}")
    return 0 if not failed else 1


def _last_json_line(text: str) -> dict[str, Any] | None:
    for line in reversed(text.strip().splitlines()):
        try:
            value = json.loads(line)
        except ValueError:
            continue
        if isinstance(value, dict):
            return value
    return None


def _capture(fn) -> str:
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        fn()
    return buffer.getvalue()


if __name__ == "__main__":
    sys.exit(main())
