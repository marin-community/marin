"""One-line-per-child snapshot of the scaling-ladder Iris jobs plus checkpoint steps on GCS.

usage: uv run --offline --no-sync python snapshot.py [--gcs]
Prints lines `KEY|state|failures|preemptions|done/total|running|pending|ckpt` for every training child and
eval child under the watched parents, sorted by key. Exit 0 even when a query fails (the failure is a line).
"""
import re
import subprocess
import sys

PARENTS = [
    "/calvinxu/dm-delphi-frozen-unconstrained-scaling-v6e-20260908-retry5",
    "/calvinxu/dm-delphi-frozen-unconstrained-scaling-v6e-20260908-retry6",
    "/calvinxu/dm-delphi-frozen-unconstrained-scaling-v6e-20260908-retry7",
    "/calvinxu/dm-delphi-frozen-unconstrained-scaling-v6e-20260908-retry8",
    "/calvinxu/dm-delphi-matched-olmix-scaling-v6e-20260910",
    "/calvinxu/dm-delphi-matched-olmix-scaling-v6e-20260910-retry1",
    "/calvinxu/table9-accuracy-choices-v6e4-full-20260915",
    "/calvinxu/table9-accuracy-generation-v6e4-full-20260915",
    "/calvinxu/table9-accuracy-choices-v6e4-canary-20260915-r3",
    "/calvinxu/table9-accuracy-generation-v6e4-canary-20260915-r3",
]
IRIS = ["uv", "run", "iris", "--config", "lib/iris/config/marin.yaml"]
TEMP_PREFIX = "marin-us-east5/tmp/ttl=14d/checkpoints-temp/marin-us-east5/"
GCS_ROOTS = {
    "lwspu": "gs://marin-us-east5/pinlin_calvin_xu/data_mixture/delphi_frozen_procedure_scaling_v6e_20260908",
    "olmixq": "gs://marin-us-east5/pinlin_calvin_xu/data_mixture/delphi_matched_olmix_scaling_v6e_20260910",
}


def run(args):
    try:
        return subprocess.run(args, capture_output=True, text=True, timeout=240).stdout
    except Exception as exc:  # noqa: BLE001 - the watch must keep going
        return f"ERR {exc}"


def short(job):
    leaf = job.rsplit("/", 1)[-1]
    m = re.search(r"-((?:lwspu|olmixq)_\w+?_(?:3e18|2e19|3e20|1e21)_seed\d+)", leaf)
    if m:
        return m.group(1)
    return leaf


def checkpoint_steps():
    try:
        import fsspec
        fs = fsspec.filesystem("gcs")
    except Exception as exc:  # noqa: BLE001
        return {"ERR": str(exc)}
    out = {}
    for family, root in GCS_ROOTS.items():
        try:
            for d in fs.ls(root):
                name = d.rsplit("/", 1)[-1]
                if not re.search(r"(3e20|1e21)", name):
                    continue
                steps = []
                try:
                    for c in fs.ls(d + "/checkpoints"):
                        m = re.search(r"step-(\d+)", c)
                        if m:
                            steps.append(int(m.group(1)))
                except Exception:  # noqa: BLE001
                    pass
                # Rolling 10-minute checkpoints live under the 14-day temp prefix; the run resumes from them.
                # gcsfs.ls returns bucket-relative paths with no gs:// scheme, so splice on the bucket name.
                temp = TEMP_PREFIX + d.rstrip("/").split("marin-us-east5/", 1)[1] + "/checkpoints"
                try:
                    for c in fs.ls(temp):
                        m = re.search(r"step-(\d+)$", c)
                        if m:
                            steps.append(int(m.group(1)))
                except Exception:  # noqa: BLE001
                    pass
                key = re.sub(r"-[0-9a-f]{6}$", "", name)
                out[key] = max(steps) if steps else 0
        except Exception as exc:  # noqa: BLE001
            out[family] = f"ERR {exc}"
    return out


def main():
    want_gcs = "--gcs" in sys.argv
    ckpts = checkpoint_steps() if want_gcs else {}
    lines = []
    for parent in PARENTS:
        listing = run([*IRIS, "job", "list", "--prefix", parent, "--limit", "100"])
        if listing.startswith("ERR"):
            lines.append(f"{parent.rsplit('/',1)[-1]}|LISTERR|{listing[:80]}")
            continue
        for row in listing.splitlines():
            parts = row.split()
            if len(parts) < 2 or not parts[0].startswith("/"):
                continue
            job, state = parts[0], parts[1]
            if job == parent:
                lines.append(f"{parent.rsplit('/',1)[-1]}|{state}|parent")
                continue
            if job in PARENTS or "/manifest" in job:
                continue
            key = short(job)
            if "olmo-base-eval" in job:
                lines.append(f"{key}|{state}|eval")
                continue
            desc = run([*IRIS, "job", "describe", job])
            m = re.search(r"State: (\w+)\s+exit=(-?\d+)\s+failures=(\d+)\s+preemptions=(\d+)", desc)
            t = re.search(r"Tasks: (\d+)/(\d+) completed(.*)", desc)
            counts = {}
            if t:
                for k, v in re.findall(r"(\w+)=(\d+)", t.group(3)):
                    counts[k] = v
            err = re.search(r"Error: (.*)", desc)
            ck = ckpts.get(key, "")
            lines.append(
                f"{key}|{m.group(1) if m else state}|f={m.group(3) if m else '?'}|p={m.group(4) if m else '?'}"
                f"|{t.group(1) if t else '?'}/{t.group(2) if t else '?'}|run={counts.get('running','0')}|pend={counts.get('pending','0')}"
                f"|ckpt={ck}" + (f"|ERR={err.group(1)[:90]}" if err else "")
            )
    for line in sorted(set(lines)):
        print(line)


if __name__ == "__main__":
    main()
