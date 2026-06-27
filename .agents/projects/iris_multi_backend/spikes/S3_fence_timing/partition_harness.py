#!/usr/bin/env python
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""SPIKE S3 — GATED live partition-injection harness.

This measures the TWO fence terms the offline harness cannot: a real
worker-daemon self-fence over a real network partition, and k8s pod-create /
pod-self-fence / pod-delete latency against a real kube-apiserver.

!!! SAFETY !!!
AGENTS.md forbids partitioning or bouncing live Iris clusters without the user's
express permission. This script therefore REFUSES to do anything destructive
unless ALL of:

    --i-have-approval                 (operator asserts they have user approval)
    --confirm DESTROY-<target>        (echoes the exact target back)
    an explicit --target / --namespace / --pod / --worker

With no flags it prints a DRY-RUN: the exact commands it WOULD run and what each
measures. The dry-run touches nothing. Run the dry-run first; paste it for review;
only then re-run with the approval flags against a DISPOSABLE TEST target (a
scratch slice / scratch namespace), never a production job.

Usage (safe):
    python partition_harness.py --mode worker-daemon            # dry-run plan
    python partition_harness.py --mode k8s-pod                  # dry-run plan

Usage (gated, only after approval, on a test target):
    python partition_harness.py --mode worker-daemon \
        --worker <vm> --zone <zone> --controller-ip <ip> \
        --heartbeat-timeout 20 --i-have-approval --confirm DESTROY-<vm>

    python partition_harness.py --mode k8s-pod \
        --namespace iris-spike --kubeconfig <path> \
        --i-have-approval --confirm DESTROY-iris-spike
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass, field


@dataclass
class Step:
    """One measurement step: a human description plus the shell command it runs."""

    description: str
    command: list[str]
    measures: str = ""
    destructive: bool = False


@dataclass
class Plan:
    title: str
    steps: list[Step] = field(default_factory=list)

    def add(self, description: str, command: str, measures: str = "", destructive: bool = False) -> None:
        self.steps.append(Step(description, shlex.split(command), measures, destructive))


# --------------------------------------------------------------------------- #
# Worker-daemon partition plan
# --------------------------------------------------------------------------- #
def worker_daemon_plan(args: argparse.Namespace) -> Plan:
    """Partition one worker daemon from the controller; time its self-fence.

    The worker self-resets after ``heartbeat_timeout`` with no controller
    contact (worker.py:572-580), wiping its containers. We block the worker's
    egress to the controller (one-way is enough — a reconcile that can't be
    answered is a missed heartbeat), watch its running container vanish, and time
    it. Run with a SHORT --heartbeat-timeout (e.g. 20s) so the measurement takes
    seconds, then read it back as `latency ~= heartbeat_timeout + <=1s + reset`.
    """
    w = args.worker or "<worker-vm>"
    zone = args.zone or "<zone>"
    cip = args.controller_ip or "<controller-ip>"
    ssh = f"gcloud compute ssh {w} --zone {zone} --tunnel-through-iap --command"

    plan = Plan(f"WORKER-DAEMON PARTITION — worker={w} zone={zone} controller={cip}")
    plan.add(
        "Record the running iris container id + start time (t0 baseline)",
        f'{ssh} "docker ps --filter label=iris.managed=true --format {{{{.ID}}}}\\ {{{{.CreatedAt}}}}"',
        measures="baseline: confirm an attempt is RUNNING before the partition",
    )
    plan.add(
        "Note last-reconcile time from the worker log (t0 for self-fence clock)",
        f'{ssh} "journalctl -u iris-worker --since \\"-2 min\\" | grep -i reconcile | tail -1"',
        measures="t0 = last successful controller contact",
    )
    plan.add(
        "INJECT PARTITION: drop worker->controller egress",
        f'{ssh} "sudo iptables -A OUTPUT -d {cip} -j DROP"',
        measures="starts the lost-contact clock",
        destructive=True,
    )
    plan.add(
        "POLL until the running container is gone (worker self-reset)",
        f'{ssh} "while docker ps --filter label=iris.managed=true --format {{{{.ID}}}} | grep -q .; '
        f'do sleep 0.5; done; date +%s.%N"',
        measures="t_selffence = when the worker wiped its containers",
        destructive=True,
    )
    plan.add(
        "HEAL PARTITION: restore egress",
        f'{ssh} "sudo iptables -D OUTPUT -d {cip} -j DROP"',
        measures="lets the worker re-register; restores the slice",
        destructive=True,
    )
    plan.add(
        "Confirm re-registration",
        f'{ssh} "journalctl -u iris-worker --since \\"-1 min\\" | grep -i register | tail -1"',
        measures="self_fence_latency = t_selffence - t0; expect ~= heartbeat_timeout + <=1s + reset",
    )
    return plan


# --------------------------------------------------------------------------- #
# k8s pod plan
# --------------------------------------------------------------------------- #
def k8s_pod_plan(args: argparse.Namespace) -> Plan:
    """Measure pod-create, pod-self-fence, and pod-delete latency.

    Three numbers the offline harness can't reach:
      * pod-create: kubectl apply -> Running (scheduling + image pull dominate).
      * pod-self-fence: today a partitioned pod does NOT self-kill unless
        activeDeadlineSeconds fires (tasks.py:796), and that is SKIPPED for
        Kueue-gated gangs. This run proves whether a lease-less pod self-fences
        at all, and measures activeDeadlineSeconds enforcement granularity.
      * pod-delete: delete -> gone, dominated by terminationGracePeriodSeconds
        (unset -> k8s default 30s) and the agent's force-delete path.
    """
    ns = args.namespace or "<test-namespace>"
    kc = f"--kubeconfig {args.kubeconfig}" if args.kubeconfig else ""
    kubectl = f"kubectl {kc} -n {ns}".strip()

    plan = Plan(f"K8S POD LIFECYCLE — namespace={ns}")
    # A throwaway pod with a short activeDeadlineSeconds to measure self-fence.
    pod_yaml = (
        "apiVersion: v1\nkind: Pod\nmetadata:\n  name: s3-fence-probe\n  labels: {iris.spike: s3}\n"
        "spec:\n  activeDeadlineSeconds: 30\n  terminationGracePeriodSeconds: 10\n  restartPolicy: Never\n"
        "  containers:\n  - name: sleep\n    image: busybox\n    command: [sleep, '3600']\n"
    )
    plan.add(
        "CREATE probe pod, time apply -> Running",
        f"bash -c \"printf %s {shlex.quote(pod_yaml)} | {kubectl} apply -f - && "
        f"t0=$(date +%s.%N); {kubectl} wait --for=condition=Ready pod/s3-fence-probe --timeout=120s; "
        f'echo create_latency=$(echo "$(date +%s.%N) - $t0" | bc)"',
        measures="pod-create latency (kubectl-apply); split first-pull vs warm-cache by re-running",
        destructive=True,
    )
    plan.add(
        "SELF-FENCE: watch the pod terminate on its own (activeDeadlineSeconds=30)",
        f"bash -c \"t0=$(date +%s.%N); "
        f"while [ \\\"$({kubectl} get pod s3-fence-probe -o jsonpath='{{.status.phase}}')\\\" = Running ]; "
        f'do sleep 0.5; done; echo self_fence_latency=$(echo "$(date +%s.%N) - $t0" | bc)"',
        measures="does a lease-less pod die on its own? granularity of activeDeadlineSeconds",
        destructive=True,
    )
    plan.add(
        "PARTITION proxy: cut the agent's apiserver access, confirm a RUNNING pod survives",
        f"# (manual) revoke the agent SA token / NetworkPolicy-deny egress to the apiserver, then: {kubectl} get pod s3-fence-probe",
        measures="confirms the pod KEEPS RUNNING with no agent -> the k8s self-fence GAP",
    )
    plan.add(
        "DELETE: time delete -> gone (graceful)",
        f'bash -c "t0=$(date +%s.%N); {kubectl} delete pod s3-fence-probe --wait=true; '
        f'echo delete_latency=$(echo "$(date +%s.%N) - $t0" | bc)"',
        measures="pod-delete (kill) latency; dominated by terminationGracePeriodSeconds",
        destructive=True,
    )
    plan.add(
        "DELETE (force): time force-delete (the agent's terminal-pod GC path)",
        f'bash -c "t0=$(date +%s.%N); {kubectl} delete pod s3-fence-probe --grace-period=0 --force --wait=true; '
        f'echo force_delete_latency=$(echo "$(date +%s.%N) - $t0" | bc)"',
        measures="force-delete latency (tasks.py:1972 terminal GC uses force=True, grace=0)",
        destructive=True,
    )
    return plan


# --------------------------------------------------------------------------- #
# Runner
# --------------------------------------------------------------------------- #
def print_dry_run(plan: Plan) -> None:
    print(f"\n### DRY-RUN PLAN: {plan.title}\n")
    print("This printed nothing destructive. Review, get approval, then re-run gated.\n")
    for i, step in enumerate(plan.steps, 1):
        flag = " [DESTRUCTIVE]" if step.destructive else ""
        print(f"[{i}]{flag} {step.description}")
        if step.measures:
            print(f"     measures: {step.measures}")
        print(f"     $ {' '.join(shlex.quote(c) for c in step.command)}\n")


def run_gated(plan: Plan, args: argparse.Namespace) -> None:
    expected = f"DESTROY-{args.confirm_target()}"
    if not args.i_have_approval or args.confirm != expected:
        print(
            f"REFUSED: gated execution requires --i-have-approval and "
            f"--confirm {expected} (got --confirm={args.confirm!r}).",
            file=sys.stderr,
        )
        sys.exit(2)
    print(f"\n### GATED EXECUTION (approved): {plan.title}\n")
    results: list[str] = []
    for i, step in enumerate(plan.steps, 1):
        print(f"[{i}] {step.description}\n     $ {' '.join(shlex.quote(c) for c in step.command)}")
        if step.command and step.command[0].startswith("#"):
            print("     (manual step — skipping)\n")
            continue
        t0 = time.monotonic()
        proc = subprocess.run(step.command, capture_output=True, text=True)
        dt = time.monotonic() - t0
        out = (proc.stdout or "").strip()
        err = (proc.stderr or "").strip()
        print(f"     rc={proc.returncode} wall={dt:.3f}s")
        if out:
            print(f"     stdout: {out}")
        if err:
            print(f"     stderr: {err}")
        results.append(f"{step.description}: rc={proc.returncode} wall={dt:.3f}s out={out!r}")
        print()
    print("### SUMMARY")
    for r in results:
        print("  " + r)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", required=True, choices=["worker-daemon", "k8s-pod"])
    # worker-daemon
    parser.add_argument("--worker", help="target worker VM name (TEST slice only)")
    parser.add_argument("--zone", help="GCP zone of the worker")
    parser.add_argument("--controller-ip", help="controller IP to partition the worker from")
    parser.add_argument("--heartbeat-timeout", type=float, help="test heartbeat_timeout (s) the worker was launched with")
    # k8s
    parser.add_argument("--namespace", help="TEST namespace for the probe pod")
    parser.add_argument("--kubeconfig", help="kubeconfig path (test cluster)")
    # gating
    parser.add_argument("--i-have-approval", action="store_true", help="assert user-granted approval")
    parser.add_argument("--confirm", default="", help="must equal DESTROY-<target>")
    args = parser.parse_args()

    def _confirm_target() -> str:
        return (args.worker if args.mode == "worker-daemon" else args.namespace) or "UNSET"

    args.confirm_target = _confirm_target  # type: ignore[attr-defined]

    plan = worker_daemon_plan(args) if args.mode == "worker-daemon" else k8s_pod_plan(args)

    gated = args.i_have_approval or args.confirm
    if not gated:
        print_dry_run(plan)
        return
    run_gated(plan, args)


if __name__ == "__main__":
    main()
