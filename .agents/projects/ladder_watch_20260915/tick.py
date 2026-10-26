# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""One compact STATUS line for the seven ladder children, plus ALERT lines for anything needing action.

usage: uv run --offline --no-sync python tick.py
STATUS <HH:MMZ> key=state,f=n/limit,pend/run,step/total(pct%) ; ...
ALERT  ... emitted when a child is failed/exhausted (needs a scoped retry) or an eval child changes state.
Exits 0 always; a probe failure becomes a PROBEERR line so the watch keeps running.
"""

import re
import subprocess
from datetime import UTC, datetime

# parent -> per-child failure budget recorded at submission time
PARENTS = {
    "/calvinxu/dm-delphi-frozen-unconstrained-scaling-v6e-20260908-retry5": 4,
    "/calvinxu/dm-delphi-frozen-unconstrained-scaling-v6e-20260908-retry6": 4,
    "/calvinxu/dm-delphi-frozen-unconstrained-scaling-v6e-20260908-retry8": 12,
    "/calvinxu/dm-delphi-frozen-unconstrained-scaling-v6e-20260908-retry9": 12,
    # eval-only parent carrying the resumable Table-9 evaluator; waits behind retry9 on the step lock
    "/calvinxu/dm-delphi-frozen-unconstrained-scaling-v6e-20260908-retry10": 12,
    # terminal (7-day execution timeout) but still holds the landed OLM-U-3e20 and the eval children
    "/calvinxu/dm-delphi-matched-olmix-scaling-v6e-20260910": 4,
    "/calvinxu/dm-delphi-matched-olmix-scaling-v6e-20260910-retry2": 12,
    "/calvinxu/dm-delphi-matched-olmix-scaling-v6e-20260910-retry1": 12,
    # took over both 1e21 steps from retry1 at 13:23 PDT 2026-09-21; clock runs to Sep 28
    "/calvinxu/dm-delphi-matched-olmix-scaling-v6e-20260910-retry3": 12,
    # retry4 hung in startup with no output for 46 min and was cancelled before taking over
    "/calvinxu/dm-delphi-matched-olmix-scaling-v6e-20260910-retry5": 12,
}
IRIS = ["uv", "run", "iris", "--config", "lib/iris/config/marin.yaml"]
# the three 1e21 children each want a 16-host v6e-64 gang in us-east5-b; when that scale group
# holds no workers the jobs cannot be placed at any priority, so track supply next to demand
SUPPLY_SQL_SLICES = (
    "SELECT scale_group, lifecycle, COUNT(*) FROM slices "
    "WHERE scale_group LIKE '%v6e-preemptible%us-east5-b' GROUP BY scale_group, lifecycle"
)
SUPPLY_SQL_WORKERS = (
    "SELECT scale_group, COUNT(*) FROM workers WHERE scale_group LIKE '%v6e-preemptible%us-east5-b' GROUP BY scale_group"
)
SUPPLY_SQL_GROUPS = (
    # only last_scale_up_ms is still written; consecutive_failures/backoff/quota are dead
    # columns left over from the pre-AIMD controller (see lib/iris/OPS.md)
    "SELECT name, last_scale_up_ms FROM scaling_groups WHERE name LIKE '%v6e-preemptible%us-east5-b'"
)
# only the sizes the ladder can actually use; a v6e-4 slice cannot host a 16-host gang
SUPPLY_SIZES = ("64", "32", "4")
ROOTS = {
    "lwspu": "gs://marin-us-east5/pinlin_calvin_xu/data_mixture/delphi_frozen_procedure_scaling_v6e_20260908",
    "olmixq": "gs://marin-us-east5/pinlin_calvin_xu/data_mixture/delphi_matched_olmix_scaling_v6e_20260910",
}
TEMP_PREFIX = "marin-us-east5/tmp/ttl=14d/checkpoints-temp/marin-us-east5/"
# the finish line is the manifest's expected_checkpoint_step (train_steps - 1), which the audit asserts;
# 1e21 runs end at 22056, not the 22057 train_steps figure
TOTAL = {"3e20": 23531, "1e21": 22056}
# the seven still in flight; key -> short label used in the STATUS line
WATCHED = {
    "lwspu_u_snc_cap06_1e21_seed666206": "MAR-U-1e21",
    "lwspu_t9_snc_cap08_3e20_seed662003": "MAR-T-3e20",
    "lwspu_t9_snc_cap08_1e21_seed662005": "MAR-T-1e21",
    "olmixq_u_kl0p05_cap04_3e20_seed666204": "OLM-U-3e20",
    "olmixq_t9_kl0p005_cap04_3e20_seed662003": "OLM-T-3e20",
    "olmixq_u_kl0p05_cap04_1e21_seed666206": "OLM-U-1e21",
    "olmixq_t9_kl0p005_cap04_1e21_seed662005": "OLM-T-1e21",
}
# a child listed under several parents is owned by the last parent that still has it non-failed
PARENT_ORDER = list(PARENTS)
# states from which a child never resumes on its own; each needs a scoped retry parent
TERMINAL = {"failed", "cancelled", "timeout", "killed", "worker_failed", "preempted"}


def run(args, timeout=240):
    try:
        p = subprocess.run(args, capture_output=True, text=True, timeout=timeout)
        return p.stdout
    except Exception as exc:
        return f"ERR {exc}"


def checkpoint_steps():
    import fsspec  # noqa: PLC0415 - optional dependency

    fs = fsspec.filesystem("gcs")
    out = {}
    for root in ROOTS.values():
        for d in fs.ls(root):
            name = d.rstrip("/").rsplit("/", 1)[-1]
            if not re.search(r"(3e20|1e21)", name):
                continue
            steps = []
            rel = d.rstrip("/").split("marin-us-east5/", 1)[1]
            for path, anchored in ((d + "/checkpoints", False), (TEMP_PREFIX + rel + "/checkpoints", True)):
                try:
                    for c in fs.ls(path):
                        m = re.search(r"step-(\d+)/?$" if anchored else r"step-(\d+)", c.rstrip("/"))
                        if m:
                            steps.append(int(m.group(1)))
                except FileNotFoundError:
                    pass
                except Exception:
                    pass
            out[re.sub(r"-[0-9a-f]{6}$", "", name)] = max(steps) if steps else 0
    return out


def _csv_rows(sql):
    out = run([*IRIS, "query", "-f", "csv", sql], timeout=120)
    if out.startswith("ERR"):
        return None
    rows = [r.split(",") for r in out.strip().splitlines() if r and not r.startswith(("I2", "W2", "Not all"))]
    return rows[1:] if rows else []


def _ago(ms, now_ms):
    if not ms:
        return "never"
    d = (now_ms - int(ms)) / 1000
    return f"{d:.0f}s" if d < 120 else (f"{d / 60:.0f}m" if d < 7200 else f"{d / 3600:.1f}h")


def v6e_supply():
    """One line of accelerator supply for us-east5-b, so a stall reads as stockout vs scheduling.

    A gang of 16 hosts can only come from a ready v6e-64 slice. A slice that sits in `booting` and
    then vanishes is a create LRO that came back RESOURCE_EXHAUSTED (zone stockout); the row is
    deleted the same tick it is marked failed, so error_message is never visible here.
    """
    now_ms = int(datetime.now(UTC).timestamp() * 1000)
    slices, workers, groups = _csv_rows(SUPPLY_SQL_SLICES), _csv_rows(SUPPLY_SQL_WORKERS), _csv_rows(SUPPLY_SQL_GROUPS)
    if slices is None or workers is None or groups is None:
        return None, []
    by_size = {sz: {"w": 0, "life": {}, "up": 0} for sz in SUPPLY_SIZES}

    def size_of(name):
        tail = name.split("_")[-1].split("-us-")[0]
        return tail if tail in by_size else None

    for row in slices:
        sz = size_of(row[0])
        if sz:
            by_size[sz]["life"][row[1]] = by_size[sz]["life"].get(row[1], 0) + int(row[2])
    for row in workers:
        sz = size_of(row[0])
        if sz:
            by_size[sz]["w"] += int(row[1])
    for row in groups:
        sz = size_of(row[0])
        if sz:
            by_size[sz]["up"] = int(row[1])

    bits, alerts = [], []
    for sz in SUPPLY_SIZES:
        g = by_size[sz]
        life = ",".join(f"{n}{k}" for k, n in sorted(g["life"].items())) or "none"
        bits.append(f"{sz}:w{g['w']},{life},up{_ago(g['up'], now_ms)}")
    # the one transition worth waking up for: a 16-host gang is finally purchasable
    if by_size["64"]["w"] > 0:
        alerts.append(f"ALERT v6e-64 us-east5-b has {by_size['64']['w']} workers - gang placement possible")
    return "v6e-east5b " + " ; ".join(bits), alerts


def dead(state):
    return state in TERMINAL


def main():
    stamp = datetime.now(UTC).strftime("%H:%MZ")
    try:
        ckpts = checkpoint_steps()
    except Exception as exc:
        print(f"PROBEERR {stamp}: gcs {exc}")
        ckpts = {}

    # child key -> best (non-failed preferred, latest parent wins)
    children = {}
    evals = {}
    parent_states = {}
    for parent, limit in PARENTS.items():
        listing = run([*IRIS, "job", "list", "--prefix", parent, "--limit", "100"])
        if listing.strip() == "No jobs found.":
            # The controller prunes terminal jobs after a retention window. A pruned parent's rungs
            # fall through to the checkpoint check below, which reports them landed or MISSING.
            continue
        if listing.startswith("ERR") or not any(l.startswith("/calvinxu") for l in listing.splitlines()):
            print(f"PROBEERR {stamp}: list {parent.rsplit('/', 1)[-1]} returned no jobs: {listing[:120]!r}")
            continue
        # a parent that is preempted off the CPU pool kills its children, so track the parent too
        for row in listing.splitlines():
            parts = row.split()
            if len(parts) >= 2 and parts[0] == parent:
                reason = row.split(parts[1], 1)[1].strip()
                reason = reason.split(None, 1)[1].strip() if len(reason.split(None, 1)) > 1 else ""
                parent_states[parent] = (parts[1], reason[:110])
        for row in listing.splitlines():
            parts = row.split()
            if len(parts) < 2 or not parts[0].startswith("/") or parts[0] == parent:
                continue
            job, state = parts[0], parts[1]
            if "/manifest" in job or job in PARENTS:
                continue
            if "olmo-base-eval" in job:
                evals[job.rsplit("/", 1)[-1]] = state
                continue
            m = re.search(r"-((?:lwspu|olmixq)_\w+?_(?:3e18|2e19|3e20|1e21)_seed\d+)", job)
            if not m or m.group(1) not in WATCHED:
                continue
            key = m.group(1)
            desc = run([*IRIS, "job", "describe", job])
            d = re.search(r"State: (\w+)\s+exit=(-?\d+)\s+failures=(\d+)\s+preemptions=(\d+)", desc)
            t = re.search(r"Tasks: (\d+)/(\d+) completed(.*)", desc)
            counts = dict(re.findall(r"(\w+)=(\d+)", t.group(3))) if t else {}
            err = re.search(r"Error: (.*)", desc)
            rec = {
                "parent": parent,
                "limit": limit,
                "state": d.group(1) if d else state,
                "failures": int(d.group(3)) if d else -1,
                "preemptions": int(d.group(4)) if d else -1,
                "done": int(t.group(1)) if t else -1,
                "tasks": int(t.group(2)) if t else -1,
                "running": int(counts.get("running", 0)),
                "pending": int(counts.get("pending", 0)),
                # startup phase between placement and running; without it a starting job reads as p0r0
                "building": int(counts.get("building", 0)),
                "error": err.group(1)[:120] if err else "",
            }
            prev = children.get(key)
            # prefer a live record over any terminal one; among equals, the latest parent wins
            if (
                prev is None
                or (dead(prev["state"]) and not dead(rec["state"]))
                or (
                    dead(prev["state"]) == dead(rec["state"])
                    and PARENT_ORDER.index(parent) >= PARENT_ORDER.index(prev["parent"])
                )
            ):
                children[key] = rec

    bits, alerts = [], []
    for key, label in WATCHED.items():
        rec = children.get(key)
        rung = "3e20" if "3e20" in key else "1e21"
        total = TOTAL[rung]
        step = ckpts.get(key, 0)
        pct = f"{100 * step / total:.0f}%" if step else "?"
        if rec is None:
            # The controller prunes terminal job records after a retention window, so a rung that
            # finished days ago loses its record. The GCS checkpoint is the durable truth: at the
            # final step the rung has landed and a missing record is expected, not an alert.
            if step >= total:
                bits.append(f"{label}=landed(record-pruned),{step}/{total}(100%)")
                continue
            bits.append(f"{label}=MISSING,{step}/{total}({pct})")
            alerts.append(f"ALERT {stamp} {label}: no child job found under any watched parent")
            continue
        bits.append(
            # x = preemptions: a clean host loss that does NOT burn the failure budget, unlike a
            # sibling-preemption SIGSEGV, which increments failures instead. Both look like a drop
            # off the accelerators, so print each separately.
            f"{label}={rec['state']},f={rec['failures']}/{rec['limit']},x={rec['preemptions']},"
            f"p{rec['pending']}r{rec['running']}"
            + (f"b{rec['building']}" if rec["building"] else "")
            + f",{step}/{total}({pct})"
        )
        if dead(rec["state"]):
            alerts.append(
                f"ALERT {stamp} {label} {rec['state']} under {rec['parent'].rsplit('/', 1)[-1]} "
                f"f={rec['failures']}/{rec['limit']} ckpt={step}/{total} :: {rec['error']}"
            )
        elif rec["failures"] >= rec["limit"] - 1 and rec["limit"] - rec["failures"] <= 1:
            alerts.append(f"ALERT {stamp} {label} one failure from its budget (f={rec['failures']}/{rec['limit']})")
        elif rec["state"] == "succeeded":
            alerts.append(f"ALERT {stamp} {label} training SUCCEEDED at ckpt={step}/{total}")

    for parent, (pstate, reason) in sorted(parent_states.items()):
        if pstate != "running":
            alerts.append(f"ALERT {stamp} parent {parent.rsplit('/', 1)[-1]} state={pstate} :: {reason}")

    print(f"STATUS {stamp} " + " ; ".join(bits))
    supply, supply_alerts = v6e_supply()
    print(f"SUPPLY {stamp} {supply}" if supply else f"PROBEERR {stamp}: supply query failed")
    alerts.extend(supply_alerts)
    for name, state in sorted(evals.items()):
        if state not in ("succeeded",):
            alerts.append(f"ALERT {stamp} eval {name} state={state}")
        else:
            alerts.append(f"EVALOK {stamp} {name}")
    for a in alerts:
        print(a)


if __name__ == "__main__":
    main()
