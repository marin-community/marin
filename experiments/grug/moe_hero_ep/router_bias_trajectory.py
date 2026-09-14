# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Track the MoE ``router_bias`` across hero checkpoints with a minimal S3 read.

``router_bias`` is the only learned bias in the grug hero; it is stored as one stacked
``[num_layers, num_experts] = [48, 384]`` array per checkpoint. This job opens ONLY that
tensorstore leaf (zarr3 inside the checkpoint's OCDBT kvstore, ~74 KB) for every permanent
checkpoint on the hero lineage -- it never materializes the multi-TB state. It stacks the
biases into ``[n_checkpoints, 48, 384]``, writes that to S3, and prints a compact JSON of a
grid of ``(layer, expert)`` trajectories (plus per-checkpoint summary stats) to stdout so the
trajectories can be pulled from the logs and plotted.

    uv run iris --config lib/iris/config/marin.yaml job run --no-wait --enable-extra-resources \\
        --target-cluster cw-us-east-08a --priority interactive --cpu 4 --memory 8GB \\
        -- python -m experiments.grug.moe_hero_ep.router_bias_trajectory --run
"""

import io
import json
import logging

import click
import fsspec
import numpy as np
import tensorstore as ts
from levanter.tensorstore_serialization import build_kvstore_spec

logger = logging.getLogger(__name__)

# The only learned bias in the model: MoEMLP.router_bias, stacked over layers by ArrayStacked.
ROUTER_BIAS_LEAF = "params/stacked_blocks/stacked/mlp/router_bias"

# Hero lineage, newest first: the current ragged continuation, the wd fork it resumed from,
# then the original hero. For each step the first base that has it wins, so late steps come
# from ragged, mid from wd, early from the original hero.
DEFAULT_BASES = (
    "s3://marin-us-east-02a/marin/grug/hero-ragged_a2a-nccl2307-ep-step81k/2026.08.19.2/checkpoints",
    "s3://marin-us-east-02a/marin/grug/hero-wd-gate-router-p02-step58k/2026.08.19.2/checkpoints",
    "s3://marin-us-east-02a/marin/grug/hero-12d8b6f0-dee637/2026.08.19.2/checkpoints",
)
DEFAULT_OUT = "s3://marin-us-east-02a/marin/grug/samplegen-hero/analysis/router_bias_trajectory.npz"
# Every-6k permanent cadence plus the one-off 55000 pin (#8818).
DEFAULT_STEPS = tuple(sorted({*range(6000, 114001, 6000), 55000}))


def _read_router_bias(checkpoint_root: str) -> np.ndarray:
    """Read only the router_bias leaf ([num_layers, num_experts]) from one checkpoint."""
    spec = {
        "driver": "zarr3",
        "kvstore": {"driver": "ocdbt", "base": build_kvstore_spec(checkpoint_root), "path": ROUTER_BIAS_LEAF},
    }
    return np.asarray(ts.open(spec, open=True).result().read().result())


def _collect(bases: tuple[str, ...], steps: tuple[int, ...]) -> tuple[list[int], list[str], np.ndarray]:
    kept_steps: list[int] = []
    kept_source: list[str] = []
    arrays: list[np.ndarray] = []
    for step in steps:
        for base in bases:
            ckpt = f"{base}/step-{step}"
            try:
                bias = _read_router_bias(ckpt)
            except Exception:  # missing checkpoint at this base -> try the next base
                continue
            kept_steps.append(step)
            kept_source.append(base.split("/grug/")[-1].split("/")[0])
            arrays.append(bias)
            logger.info("read step-%d from %s shape=%s", step, kept_source[-1], bias.shape)
            break
        else:
            logger.info("step-%d not found on any base", step)
    if not arrays:
        raise RuntimeError("no router_bias arrays read from any checkpoint")
    return kept_steps, kept_source, np.stack(arrays)


@click.command()
@click.option("--bases", default=",".join(DEFAULT_BASES), help="comma-separated checkpoint base paths, priority order")
@click.option("--steps", default=",".join(map(str, DEFAULT_STEPS)), help="comma-separated candidate steps")
@click.option("--out", default=DEFAULT_OUT, help="S3 path for the [n, L, E] npz")
@click.option("--run", is_flag=True, default=False, help="required to actually run")
def main(bases: str, steps: str, out: str, run: bool) -> None:
    logging.basicConfig(level=logging.INFO)
    if not run:
        raise SystemExit("pass --run")
    base_list = tuple(b.strip() for b in bases.split(",") if b.strip())
    step_list = tuple(int(s) for s in steps.split(",") if s.strip())
    kept_steps, kept_source, data = _collect(base_list, step_list)  # data: [n, L, E]

    buf = io.BytesIO()
    np.savez_compressed(buf, steps=np.array(kept_steps), source=np.array(kept_source), bias=data)
    buf.seek(0)
    with fsspec.open(out, "wb") as handle:
        handle.write(buf.read())
    logger.info("wrote %s  shape=%s", out, data.shape)

    n_layers, n_experts = data.shape[1], data.shape[2]
    layers = sorted(set(np.linspace(0, n_layers - 1, 7).astype(int).tolist()))
    experts = sorted(set(np.linspace(0, n_experts - 1, 7).astype(int).tolist()))
    series = {f"L{layer}E{expert}": [float(v) for v in data[:, layer, expert]] for layer in layers for expert in experts}
    payload = {
        "steps": kept_steps,
        "source": kept_source,
        "series": series,
        "summary": {
            "mean_abs": [float(np.abs(data[i]).mean()) for i in range(len(kept_steps))],
            "std": [float(data[i].std()) for i in range(len(kept_steps))],
            "max_abs": [float(np.abs(data[i]).max()) for i in range(len(kept_steps))],
        },
    }
    # Single-line marker so the trajectories can be recovered from the job logs.
    print("BIAS_JSON " + json.dumps(payload), flush=True)


if __name__ == "__main__":
    main()
