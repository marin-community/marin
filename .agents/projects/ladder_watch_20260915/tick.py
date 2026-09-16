"""One compact STATUS line for the seven ladder children, plus ALERT lines for anything needing action.

usage: uv run --offline --no-sync python tick.py
STATUS <HH:MMZ> key=state,f=n/limit,pend/run,step/total(pct%) ; ...
ALERT  ... emitted when a child is failed/exhausted (needs a scoped retry) or an eval child changes state.
Exits 0 always; a probe failure becomes a PROBEERR line so the watch keeps running.
"""
import json
import re
import subprocess
import sys
from datetime import datetime, timezone

# parent -> per-child failure budget recorded at submission time
PARENTS = {
    "/calvinxu/dm-delphi-frozen-unconstrained-scaling-v6e-20260908-retry5": 4,
    "/calvinxu/dm-delphi-frozen-unconstrained-scaling-v6e-20260908-retry6": 4,
    "/calvinxu/dm-delphi-frozen-unconstrained-scaling-v6e-20260908-retry8": 12,
    "/calvinxu/dm-delphi-matched-olmix-scaling-v6e-20260910": 4,
    "/calvinxu/dm-delphi-matched-olmix-scaling-v6e-20260910-retry1": 12,
}
# Call the venv's iris directly rather than through `uv run`: an unresolved merge can leave uv.lock
# with conflict markers, and every `uv run` then dies on a TOML parse error, blinding the whole watch.
IRIS = ["/Users/calvinxu/Projects/Work/Marin/marin/.venv/bin/iris", "--config", "lib/iris/config/marin.yaml"]
ROOTS = {
    "lwspu": "gs://marin-us-east5/pinlin_calvin_xu/data_mixture/delphi_frozen_procedure_scaling_v6e_20260908",
    "olmixq": "gs://marin-us-east5/pinlin_calvin_xu/data_mixture/delphi_matched_olmix_scaling_v6e_20260910",
}
TEMP_PREFIX = "marin-us-east5/tmp/ttl=14d/checkpoints-temp/marin-us-east5/"
TOTAL = {"3e20": 23531, "1e21": 22057}
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
    except Exception as exc:  # noqa: BLE001 - the watch must keep going
        return f"ERR {exc}"


def checkpoint_steps():
    import fsspec

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
                except Exception:  # noqa: BLE001
                    pass
            out[re.sub(r"-[0-9a-f]{6}$", "", name)] = max(steps) if steps else 0
    return out


def dead(state):
    return state in TERMINAL


def main():
    stamp = datetime.now(timezone.utc).strftime("%H:%MZ")
    try:
        ckpts = checkpoint_steps()
    except Exception as exc:  # noqa: BLE001
        print(f"PROBEERR {stamp}: gcs {exc}")
        ckpts = {}

    # child key -> best (non-failed preferred, latest parent wins)
    children = {}
    evals = {}
    parent_states = {}
    for parent, limit in PARENTS.items():
        listing = run([*IRIS, "job", "list", "--prefix", parent, "--limit", "100"])
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
            bits.append(f"{label}=MISSING,{step}/{total}({pct})")
            alerts.append(f"ALERT {stamp} {label}: no child job found under any watched parent")
            continue
        bits.append(
            # x = preemptions: a clean host loss that does NOT burn the failure budget, unlike a
            # sibling-preemption SIGSEGV, which increments failures instead. Both look like a drop
            # off the accelerators, so print each separately.
            f"{label}={rec['state']},f={rec['failures']}/{rec['limit']},x={rec['preemptions']},"
            f"p{rec['pending']}r{rec['running']},{step}/{total}({pct})"
        )
        if dead(rec["state"]):
            alerts.append(
                f"ALERT {stamp} {label} {rec['state']} under {rec['parent'].rsplit('/', 1)[-1]} "
                f"f={rec['failures']}/{rec['limit']} ckpt={step}/{total} :: {rec['error']}"
            )
        elif rec["failures"] >= rec["limit"] - 1 and rec["limit"] - rec["failures"] <= 1:
            alerts.append(
                f"ALERT {stamp} {label} one failure from its budget (f={rec['failures']}/{rec['limit']})"
            )
        elif rec["state"] == "succeeded":
            alerts.append(f"ALERT {stamp} {label} training SUCCEEDED at ckpt={step}/{total}")

    for parent, (pstate, reason) in sorted(parent_states.items()):
        if pstate != "running":
            alerts.append(f"ALERT {stamp} parent {parent.rsplit('/', 1)[-1]} state={pstate} :: {reason}")

    print(f"STATUS {stamp} " + " ; ".join(bits))
    for name, state in sorted(evals.items()):
        if state not in ("succeeded",):
            alerts.append(f"ALERT {stamp} eval {name} state={state}")
        else:
            alerts.append(f"EVALOK {stamp} {name}")
    for a in alerts:
        print(a)


if __name__ == "__main__":
    main()
