# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Compare two rehearsal runs bit for bit: per-step metrics from their JSON logs, and checkpoint arrays.

    python compare_runs.py --logs-a K-0.*.log --logs-b R-0.*.log --ckpt-a K/checkpoints/step-400 \\
        --ckpt-b R/checkpoints/step-400

Logs are process 0's output with ``JsonLoggerConfig`` lines; a run restarted after a crash logs some steps twice,
and the later attempt's value is the one kept (the files are read in the order given). Exits 1 on any difference.
"""

import argparse
import json
import sys

import numpy as np
import tensorstore as ts

KEYS = ("train/loss", "train/cross_entropy_loss")


def step_metrics(paths: list[str]) -> tuple[dict[int, dict[str, float]], list[tuple[int, str, float, float]]]:
    """Each step's last-logged values of ``KEYS``, and every (step, key, first, later) pair a restart logged twice
    with different values."""
    metrics: dict[int, dict[str, float]] = {}
    relogged: list[tuple[int, str, float, float]] = []
    for path in paths:
        with open(path, errors="replace") as f:
            for line in f:
                start = line.find('{"tracker": "json_logger"')
                if start < 0:
                    continue
                record = json.loads(line[start:])
                if record["event"] != "log":
                    continue
                for key in KEYS:
                    if key not in record["metrics"]:
                        continue
                    value = float(record["metrics"][key])
                    previous = metrics.setdefault(record["step"], {}).get(key)
                    if previous is not None and previous != value:
                        relogged.append((record["step"], key, previous, value))
                    metrics[record["step"]][key] = value
    return metrics, relogged


def checkpoint_arrays(path: str) -> dict[str, np.ndarray]:
    kvstore = {"driver": "ocdbt", "base": {"driver": "file", "path": path}}
    keys = [key.decode() for key in ts.KvStore.open(kvstore).result().list().result()]
    arrays = {}
    for key in keys:
        if not key.endswith("zarr.json"):
            continue
        name = key[: -len("/zarr.json")]
        spec = {"driver": "zarr3", "kvstore": {**kvstore, "path": name + "/"}}
        arrays[name] = ts.open(spec, read=True).result().read().result()
    return arrays


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--logs-a", nargs="+", required=True)
    parser.add_argument("--logs-b", nargs="+", required=True)
    parser.add_argument("--ckpt-a", nargs="*", default=[])
    parser.add_argument("--ckpt-b", nargs="*", default=[])
    args = parser.parse_args()

    differ = False
    a, a_relogged = step_metrics(args.logs_a)
    b, _ = step_metrics(args.logs_b)
    common = sorted(set(a) & set(b))
    mismatched = [step for step in common if a[step] != b[step]]
    print(
        f"steps logged: a {len(a)}, b {len(b)}, common {len(common)} ({common[0] if common else '-'}-"
        f"{common[-1] if common else '-'}); steps whose {', '.join(KEYS)} differ: {len(mismatched)}"
    )
    for step in mismatched[:10]:
        print(f"  step {step}: a {a[step]} b {b[step]}")
    print(f"steps a logged twice with different values (before and after its restart): {len(a_relogged)}")
    for entry in a_relogged[:10]:
        print(f"  step {entry[0]} {entry[1]}: first {entry[2]!r} after restart {entry[3]!r}")
    differ |= bool(mismatched) or bool(a_relogged)

    for path_a, path_b in zip(args.ckpt_a, args.ckpt_b, strict=True):
        arrays_a, arrays_b = checkpoint_arrays(path_a), checkpoint_arrays(path_b)
        if set(arrays_a) != set(arrays_b):
            print(f"{path_a} vs {path_b}: different array sets: {sorted(set(arrays_a) ^ set(arrays_b))[:10]}")
            differ = True
            continue
        unequal = []
        for name in sorted(arrays_a):
            x, y = arrays_a[name], arrays_b[name]
            if x.shape != y.shape or x.dtype != y.dtype or x.tobytes() != y.tobytes():
                diff = np.max(np.abs(x.astype(np.float64) - y.astype(np.float64))) if x.shape == y.shape else None
                unequal.append((name, diff))
        elements = sum(arr.size for arr in arrays_a.values())
        print(f"{path_a} vs {path_b}: {len(arrays_a)} arrays, {elements:,} elements, {len(unequal)} arrays differ")
        for name, diff in unequal[:10]:
            print(f"  {name}: max abs diff {diff}")
        differ |= bool(unequal)
    sys.exit(1 if differ else 0)


if __name__ == "__main__":
    main()
