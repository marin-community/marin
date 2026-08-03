# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""A/B the non-expert FSDP layout on real GPUs, through the ordinary training path.

The `after` arm is this branch's layout, which shards attention, dense-MLP and LM-head
weights over `("data", "expert")`. The `before` arm sets `SCALE_FSDP_LAYOUT=pre_ep`,
which rebinds those groups to the single-axis ones the template used before expert
parallelism, so both arms run from one commit and one image.

`submit` launches `--repeats` draws per arm through `launch_cw_scale.py`. Repeats matter:
placement across nodes moves MFU by a couple of points on this cluster, enough to swamp
the effect being measured from a single pair.

`summarize` reads the JSON metric lines the runs emit and reports, per arm, the median
over the post-warmup window of `throughput/mfu`, `throughput/tokens_per_second` and
`train/router/capacity_overflow_rate_mean`, plus per-device parameter bytes.

The drop rate is a control, not a result: no commit on the PR branch touches routing or
the capacity factor, so it has to stay flat. A moved drop rate means the run differed for
some other reason and the throughput comparison should not be trusted.

The batch-spec fix in grug_moe.py is deliberately out of scope. Inside `jit` the value
reaching `moe_mlp` is a tracer with no `.sharding`, so `_batch_spec_from_x` already
returned the canonical spec; the fix changes eager-path results only and cannot move
throughput.
"""

import argparse
import collections
import json
import math
import os
import re
import statistics
import subprocess
import sys

_ARMS = ("before", "after")
_ARM_ENV = {"before": "pre_ep", "after": "ep"}
_METRICS = {
    "mfu": "throughput/mfu",
    "tokens_per_second": "throughput/tokens_per_second",
    "drop_rate": "train/router/capacity_overflow_rate_mean",
}
_METRIC_LINE = re.compile(r"\{.*\}")
# Emitted once per run by train.py, from inside the train task.
_LAYOUT_LINE = re.compile(r"BENCH_FSDP_LAYOUT resolved fsdp=(\(.*?\)) lm_head=(\(.*?\))")


def _run_env(args, arm: str, draw: int) -> dict[str, str]:
    env = os.environ.copy()
    env.update(
        SCALE_FSDP_LAYOUT=_ARM_ENV[arm],
        SCALE_GPU_REPLICAS=str(args.nodes),
        SCALE_EXPERT_AXIS=str(args.expert),
        SCALE_REPLICA_AXIS="1",
        SCALE_HIDDEN_DIM=str(args.hidden_dim),
        SCALE_NUM_LAYERS=str(args.num_layers),
        SCALE_NUM_EXPERTS=str(args.num_experts),
        SCALE_TOP_K=str(args.top_k),
        SCALE_SEQ_LEN=str(args.seq_len),
        SCALE_BATCH=str(args.batch),
        SCALE_STEPS=str(args.steps),
        SCALE_TRACKER="json_logger",
        RUN_ID=f"{args.run_prefix}-{arm}-{draw}",
    )
    return env


def submit(args) -> None:
    for draw in range(args.repeats):
        for arm in _ARMS:
            run_id = f"{args.run_prefix}-{arm}-{draw}"
            print(f"submitting {run_id}", flush=True)
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "experiments.grug.moe.launch_cw_scale",
                    "--version",
                    args.version,
                    "--run",
                ],
                env=_run_env(args, arm, draw),
                text=True,
                capture_output=True,
                check=False,
            )
            print(result.stdout[-2000:] or result.stderr[-2000:], flush=True)
            if result.returncode != 0:
                raise SystemExit(f"{run_id} failed to submit: {result.stderr[-4000:]}")


def _metric_series(log_path: str) -> dict[str, list[tuple[int, float]]]:
    series: dict[str, list[tuple[int, float]]] = collections.defaultdict(list)
    with open(log_path) as handle:
        for line in handle:
            match = _METRIC_LINE.search(line)
            if not match:
                continue
            try:
                payload = json.loads(match.group(0))
            except json.JSONDecodeError:
                continue
            step = payload.get("step")
            if step is None:
                continue
            for key in _METRICS.values():
                value = payload.get(key)
                if isinstance(value, int | float):
                    series[key].append((int(step), float(value)))
    return series


def _resolved_layout(log_path: str) -> str | None:
    """The shard groups the train task actually used, as logged by train.py."""
    with open(log_path) as handle:
        for line in handle:
            match = _LAYOUT_LINE.search(line)
            if match:
                return f"fsdp={match.group(1)} lm_head={match.group(2)}"
    return None


def _median_after_warmup(points: list[tuple[int, float]], warmup: int) -> float | None:
    values = [value for step, value in sorted(points) if step >= warmup]
    return statistics.median(values) if values else None


def summarize(args) -> None:
    rows = {}
    layouts: dict[str, set[str]] = {arm: set() for arm in _ARMS}
    for arm in _ARMS:
        per_draw = collections.defaultdict(list)
        for path in sorted(args.logs):
            if f"-{arm}-" not in os.path.basename(path):
                continue
            layout = _resolved_layout(path)
            if layout is None:
                raise SystemExit(
                    f"{path}: no BENCH_FSDP_LAYOUT line. The train task never recorded which shard "
                    "groups it used, so this run cannot be attributed to an arm."
                )
            layouts[arm].add(layout)
            series = _metric_series(path)
            for name, key in _METRICS.items():
                value = _median_after_warmup(series.get(key, []), args.warmup)
                if value is not None:
                    per_draw[name].append(value)
        rows[arm] = per_draw

    # SCALE_FSDP_LAYOUT is read in the train task, which does not inherit the
    # submitter's shell. If forwarding regresses, both arms resolve to the default and
    # the comparison silently measures nothing; refuse to print a result in that case.
    if layouts["before"] == layouts["after"]:
        raise SystemExit(
            f"both arms ran the same shard groups ({layouts['before']}); "
            "SCALE_FSDP_LAYOUT did not reach the train task, so there is no A/B to report"
        )
    for arm in _ARMS:
        print(f"{arm} layout: {' | '.join(sorted(layouts[arm]))}")
    print()
    print(f"{'metric':<22}{'before':>16}{'after':>16}{'delta':>16}")
    for name in _METRICS:
        before, after = rows["before"].get(name, []), rows["after"].get(name, [])
        if not before or not after:
            print(f"{name:<22}{'n/a':>16}{'n/a':>16}{'n/a':>16}")
            continue
        b, a = statistics.median(before), statistics.median(after)
        delta = f"{a - b:+.4g}" if not math.isclose(b, 0.0) else "n/a"
        print(f"{name:<22}{b:>16.6g}{a:>16.6g}{delta:>16}")
    for arm in _ARMS:
        draws = {name: [round(v, 6) for v in vals] for name, vals in rows[arm].items()}
        print(f"\n{arm} per-draw: {json.dumps(draws)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("submit", help="launch both arms on the CoreWeave H100 cluster")
    run.add_argument("--nodes", type=int, default=4, help="8 H100 per node")
    run.add_argument("--expert", type=int, default=8)
    run.add_argument("--hidden-dim", type=int, default=3072)
    run.add_argument("--num-layers", type=int, default=48)
    run.add_argument("--num-experts", type=int, default=128)
    run.add_argument("--top-k", type=int, default=4)
    run.add_argument("--seq-len", type=int, default=2048)
    run.add_argument("--batch", type=int, default=256)
    run.add_argument("--steps", type=int, default=60)
    run.add_argument("--repeats", type=int, default=3)
    run.add_argument("--run-prefix", default="ep-fsdp-bench")
    run.add_argument("--version", default="2026.08.03", help="artifact calendar version")
    run.set_defaults(func=submit)

    report = sub.add_parser("summarize", help="compare metric logs from both arms")
    report.add_argument("logs", nargs="+", help="log files named <prefix>-<arm>-<draw>*")
    report.add_argument("--warmup", type=int, default=20, help="ignore steps below this")
    report.set_defaults(func=summarize)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
