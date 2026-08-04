# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""A/B the non-expert FSDP layout on real GPUs, through the ordinary training path.

The `after` arm is this branch's layout, which shards attention, dense-MLP and LM-head
weights over `("data", "expert")`. The `before` arm is the single-axis layout the template
used before expert parallelism.

`submit` launches ONE job whose train task runs every arm in turn, driven by
`SCALE_FSDP_SWEEP` (see `_run_grug_local`). One job rather than one per arm is not a
convenience: on a busy cluster a 4-node gang waits far longer for admission than it spends
training, and arms submitted separately land on different nodes, where placement alone
moves MFU by a couple of points. A single reservation pins every arm to the same four
nodes, so the repeats measure run-to-run noise instead of node assignment.

`summarize` splits one log into per-arm segments on the `BENCH_RUN` markers and reports,
per arm, the median over the post-warmup window of `throughput/mfu`,
`throughput/tokens_per_second` and `train/router/capacity_overflow_rate_mean`.

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
import re
import statistics
import subprocess

_ARMS = ("before", "after")
_METRICS = {
    "mfu": "throughput/mfu",
    "tokens_per_second": "throughput/tokens_per_second",
    "drop_rate": "train/router/capacity_overflow_rate_mean",
}
_METRIC_LINE = re.compile(r"\{\"tracker\".*\}")
# Emitted by the train task around each arm, and once inside it. Iris prefixes every log
# line with the emitting task, and the four tasks interleave, so segments are tracked per
# task rather than by position in the file.
_TASK = re.compile(r"task=(\S+)")
_RUN_BEGIN = re.compile(r"BENCH_RUN begin index=(\d+) arm=(\w+)")
_LAYOUT_LINE = re.compile(r"BENCH_FSDP_LAYOUT resolved fsdp=(\(.*?\)) lm_head=(\(.*?\))")


# The launcher itself runs as a CPU task on the cluster rather than locally. Its artifact
# lookups resolve the tokenized SlimPajama cache out of the CoreWeave bucket, and task
# pods already carry MARIN_PREFIX and that bucket's credentials via `task_env` in
# lib/iris/config/cw-us-east-02a.yaml. Run it here instead and it needs an object-storage
# key pair on the submitting machine, and without one it silently retargets the cache to a
# local path and tries to re-tokenize the corpus from scratch. Everything the benchmark
# reports comes back through the job log, so nothing needs to read the bucket locally.
# Matches the shape the probe launcher ran at; >=4 GB RAM needs --enable-extra-resources.
_LAUNCHER_RESOURCES = ("--cpu", "2", "--memory", "8GB", "--disk", "30GB", "--enable-extra-resources")


def _sweep_env(args) -> dict[str, str]:
    return {
        "SCALE_FSDP_SWEEP": ",".join(_ARMS * args.repeats),
        "SCALE_DEVICE": args.device,
        "SCALE_GPU_REPLICAS": str(args.nodes),
        "SCALE_EXPERT_AXIS": str(args.expert),
        "SCALE_REPLICA_AXIS": "1",
        "SCALE_HIDDEN_DIM": str(args.hidden_dim),
        "SCALE_NUM_LAYERS": str(args.num_layers),
        "SCALE_NUM_EXPERTS": str(args.num_experts),
        "SCALE_TOP_K": str(args.top_k),
        "SCALE_SEQ_LEN": str(args.seq_len),
        "SCALE_BATCH": str(args.batch),
        "SCALE_STEPS": str(args.steps),
        "SCALE_TRACKER": "json_logger",
        # The default s3 checkpointer saved at step 1 and step 2 of the probe run, ~25
        # minutes of blocking upload each, dwarfing the 80 steps being measured and
        # costing bucket traffic for state no one reads. `local` writes to the pod's
        # /tmp with save_interval=None, so only the forced final save happens.
        "SCALE_CHECKPOINTS": "local",
        "RUN_ID": args.run_id,
    }


def submit(args) -> None:
    env = _sweep_env(args)
    command = [
        "uv",
        "run",
        "iris",
        "--config",
        args.iris_config,
        "job",
        "run",
        "--target-cluster",
        args.target_cluster,
        "--job-name",
        args.run_id,
        "--user",
        args.user,
        "--priority",
        "interactive",
        "--timeout",
        str(args.timeout),
        "--no-wait",
        *_LAUNCHER_RESOURCES,
        # The sandbox blocks streaming to W&B; the benchmark reads the job log anyway.
        "-e",
        "WANDB_MODE",
        "offline",
    ]
    for key, value in env.items():
        command += ["-e", key, value]
    command += ["--", "python", "-m", "experiments.grug.moe.launch_cw_scale", "--version", args.version, "--run"]

    print(f"submitting {args.run_id} with sweep [{env['SCALE_FSDP_SWEEP']}]", flush=True)
    result = subprocess.run(command, text=True, capture_output=True, check=False)
    print(result.stdout[-2000:] or result.stderr[-2000:], flush=True)
    if result.returncode != 0:
        raise SystemExit(f"{args.run_id} failed to submit: {result.stderr[-4000:]}")


def _segments(log_path: str) -> list[dict]:
    """Split the interleaved task log into one record per arm run, in sweep order."""
    segments: dict[int, dict] = {}
    current: dict[str, int] = {}  # task -> segment index
    with open(log_path, errors="replace") as handle:
        for line in handle:
            task_match = _TASK.search(line)
            task = task_match.group(1) if task_match else ""

            begin = _RUN_BEGIN.search(line)
            if begin:
                index = int(begin.group(1))
                current[task] = index
                segments.setdefault(index, {"arm": begin.group(2), "layouts": set(), "series": {}})
                continue

            index = current.get(task)
            if index is None:
                continue
            segment = segments[index]

            layout = _LAYOUT_LINE.search(line)
            if layout:
                segment["layouts"].add(f"fsdp={layout.group(1)} lm_head={layout.group(2)}")
                continue

            metric_match = _METRIC_LINE.search(line)
            if not metric_match:
                continue
            try:
                payload = json.loads(metric_match.group(0))
            except json.JSONDecodeError:
                continue
            step = payload.get("step")
            # The json_logger nests values under "metrics" and keeps "step" beside it.
            metrics = payload.get("metrics")
            if step is None or not isinstance(metrics, dict):
                continue
            for name, key in _METRICS.items():
                value = metrics.get(key)
                if isinstance(value, int | float):
                    # All four tasks log the same value for a step; keyed by step so the
                    # repeats do not count once per task.
                    segment["series"].setdefault(name, {})[int(step)] = float(value)
    return [segments[index] for index in sorted(segments)]


def _median_after_warmup(series: dict[int, float], warmup: int) -> float | None:
    values = [value for step, value in sorted(series.items()) if step >= warmup]
    return statistics.median(values) if values else None


def summarize(args) -> None:
    segments = _segments(args.log)
    if not segments:
        raise SystemExit(f"{args.log}: no BENCH_RUN markers; this log holds no sweep to report")

    per_arm: dict[str, dict[str, list[float]]] = {arm: collections.defaultdict(list) for arm in _ARMS}
    layouts: dict[str, set[str]] = {arm: set() for arm in _ARMS}
    for index, segment in enumerate(segments):
        arm = segment["arm"]
        if not segment["layouts"]:
            raise SystemExit(
                f"{args.log}: run index={index} arm={arm} never logged BENCH_FSDP_LAYOUT, so the shard "
                "groups it used are unknown and it cannot be attributed to an arm."
            )
        layouts[arm] |= segment["layouts"]
        for name in _METRICS:
            value = _median_after_warmup(segment["series"].get(name, {}), args.warmup)
            if value is not None:
                per_arm[arm][name].append(value)

    # The arm is chosen inside the train task, which does not inherit the submitter's
    # shell. If forwarding regresses, every arm resolves to the default and the comparison
    # silently measures nothing; refuse to print a result in that case.
    if layouts["before"] == layouts["after"]:
        raise SystemExit(
            f"both arms ran the same shard groups ({layouts['before']}); "
            "SCALE_FSDP_SWEEP did not reach the train task, so there is no A/B to report"
        )

    expected = collections.Counter(segment["arm"] for segment in segments)
    measured = {arm: len(per_arm[arm]["mfu"]) for arm in _ARMS}
    for arm in _ARMS:
        if measured[arm] < expected[arm]:
            print(f"warning: arm {arm} produced metrics for {measured[arm]} of {expected[arm]} runs")
        print(f"{arm} layout: {' | '.join(sorted(layouts[arm]))}")
    print()
    print(f"{'metric':<22}{'before':>16}{'after':>16}{'delta':>16}{'ratio':>10}")
    for name in _METRICS:
        before, after = per_arm["before"].get(name, []), per_arm["after"].get(name, [])
        if not before or not after:
            print(f"{name:<22}{'n/a':>16}{'n/a':>16}{'n/a':>16}{'n/a':>10}")
            continue
        b, a = statistics.median(before), statistics.median(after)
        delta = f"{a - b:+.4g}"
        ratio = "n/a" if math.isclose(b, 0.0) else f"{a / b:.4f}"
        print(f"{name:<22}{b:>16.6g}{a:>16.6g}{delta:>16}{ratio:>10}")
    for arm in _ARMS:
        draws = {name: [round(v, 6) for v in values] for name, values in per_arm[arm].items()}
        print(f"\n{arm} per-draw: {json.dumps(draws)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("submit", help="launch the sweep on a CoreWeave GPU cluster")
    run.add_argument("--device", default="GB200", choices=("GB200", "H100"))
    run.add_argument("--nodes", type=int, default=16, help="GB200: 4 GPUs/node, one rack = 16 nodes")
    run.add_argument("--expert", type=int, default=64, help="expert-parallel axis; 64 is one full GB200 rack")
    run.add_argument("--hidden-dim", type=int, default=3072)
    run.add_argument("--num-layers", type=int, default=48)
    run.add_argument("--num-experts", type=int, default=128)
    run.add_argument("--top-k", type=int, default=4)
    run.add_argument("--seq-len", type=int, default=2048)
    run.add_argument("--batch", type=int, default=512)
    run.add_argument("--steps", type=int, default=80)
    run.add_argument("--repeats", type=int, default=3, help="paired before/after draws")
    run.add_argument("--run-id", default="ep-fsdp-sweep")
    # The GB200 fleet is reached through the federation controller with --target-cluster,
    # the shape experiments/grug/moe_hero_ep/README.md uses for the one-rack gate.
    run.add_argument("--iris-config", default="lib/iris/config/marin.yaml")
    run.add_argument("--target-cluster", default="cw-us-east-08a")
    run.add_argument("--timeout", type=int, default=21600, help="launcher job timeout")
    run.add_argument("--user", default="mwittmann", help="iris user prefix for the job")
    run.add_argument("--version", default="2026.08.03", help="artifact calendar version")
    run.set_defaults(func=submit)

    report = sub.add_parser("summarize", help="compare the arms in one sweep log")
    report.add_argument("log", help="the sweep job's log")
    report.add_argument("--warmup", type=int, default=20, help="ignore steps below this")
    report.set_defaults(func=summarize)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
