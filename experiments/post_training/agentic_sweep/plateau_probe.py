# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""How many tokens would plateau dropping drop at epoch 3 of the 1e-6 sweep run?

Scores per-token cross-entropy on fixed rows with three models: v0 (the parent, the parameters of
visit 1), the epoch-1 checkpoint (step-547, visit 2) and the epoch-2 checkpoint (step-1095, visit 3).
In real training a row's loss is measured somewhere inside its epoch; the start-of-epoch
parameters are the approximation. Rows: the pilot held-out files (TMax and SWE-smith) and a seeded
500-row sample of the training rows. Reports the share of supervised tokens the plateau rule would
drop at visit 3 for relative-change thresholds 5%, 10% and 20% (floor 0.05 nats).

    python -m experiments.post_training.agentic_sweep.plateau_probe --output <s3 prefix> [--run]

Without --run it prints the plan. With --run (inside an Iris CPU job) it submits one 8xH100 child.
"""

import argparse
import gzip
import io
import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

import haliax as hax
import jax
import jax.numpy as jnp
import jmp
import numpy as np
from fray.current_client import current_client, set_current_client
from fray.iris_backend import FrayIrisClient
from fray.types import Entrypoint, JobRequest, ResourceConfig, create_environment
from haliax import Axis
from haliax.partitioning import named_jit, round_axis_for_partitioning
from iris.client.client import IrisClient, IrisContext, iris_ctx_scope
from iris.cluster.client.job_info import get_job_info
from levanter.layers.attention_mask import AttentionMask
from levanter.model_loading import load_levanter_checkpoint
from levanter.models.lm_model import LmExample
from levanter.models.snowball import SnowballConfig
from levanter.plateau import token_cross_entropy
from levanter.tracker.json_logger import JsonLoggerConfig
from levanter.trainer import TrainerConfig, initialize
from levanter.utils.tree_utils import inference_mode
from rigging.filesystem.storage_path import StoragePath, prefix_join
from rigging.timing import Duration

from experiments.post_training.agentic_sweep.train import CONTEXT, GPU_LOG_ENV, TOKENIZER
from experiments.post_training.agentic_sweep.validate import OFFLINE_MESH

logger = logging.getLogger(__name__)

PREPARED = "s3://marin-us-east-02a/marin/teacher-traces/mopd2/task202-sweep-prepared-20261016d"
PARENT = (
    "s3://marin-us-east-02a/marin/models/"
    "open-athena--Grug-67B-A2B-Merge-Campaign3-DARE-Linear-02-2026-10-12--4eeb756b404db25bc04e4daf49dfb9817bceb741"
)
RUN = "s3://marin-us-east-02a/marin/experiments/task203-sweep/checkpoints/karan/v0-agentic-pilot-lr1e-06"
MODELS = (
    ("v0", "hf", PARENT),
    ("step-547", "native", f"{RUN}/2026.10.17.3/checkpoints/step-547"),
    ("step-1095", "native", f"{RUN}/2026.10.17.4/checkpoints/step-1095"),
)
THRESHOLDS = (0.05, 0.1, 0.2)
FLOOR = 0.05
LEARNED = 0.1
BATCH = 8


@dataclass
class PlateauProbeConfig:
    output_path: str
    train_rows: int = 500
    seed: int = 20261018
    trainer: TrainerConfig = field(
        default_factory=lambda: TrainerConfig(
            id="plateau-probe",
            mesh=OFFLINE_MESH,
            use_explicit_mesh_axes=True,
            # The sweep's training precision; the H100 attention kernel accepts only bf16 or fp16.
            mp=jmp.get_policy("p=f32,c=bfloat16"),
            per_device_eval_parallelism=1,
            tracker=JsonLoggerConfig(),
            log_jaxprs=False,
            log_xla_hlo=False,
        )
    )


def read_rows(train_rows: int, seed: int) -> list[dict]:
    """Held-out rows of both families and a seeded sample of training rows, padding rows removed."""
    rows = []
    for family in ("tmax", "swesmith"):
        data = StoragePath(prefix_join(PREPARED, f"validation-{family}-prebuilt.jsonl.gz")).read_bytes()
        with gzip.open(io.BytesIO(data), "rt") as lines:
            for line in lines:
                row = json.loads(line)
                if any(row["loss_weights"]):
                    rows.append({"group": f"heldout-{family}", **row})
    index = [
        json.loads(line) for line in StoragePath(prefix_join(PREPARED, "train-index.jsonl")).read_text().splitlines()
    ]
    eligible = [entry["row"] for entry in index if entry["split"] == "train"]
    chosen = {
        int(row): next(entry["family"] for entry in index if entry["row"] == row)
        for row in np.random.default_rng(seed).choice(eligible, size=train_rows, replace=False)
    }
    data = StoragePath(prefix_join(PREPARED, "train-prebuilt.jsonl.gz")).read_bytes()
    with gzip.open(io.BytesIO(data), "rt") as lines:
        for number, line in enumerate(lines):
            if number in chosen:
                rows.append({"group": f"train-{chosen[number]}", **json.loads(line)})
    return rows


def supervised_positions(weights: list[float]) -> np.ndarray:
    weights = np.asarray(weights, dtype=np.float32)
    weights[-1] = 0
    return np.flatnonzero(weights)


def plateau_share(losses: list[np.ndarray], threshold: float) -> np.ndarray:
    """True where both visit-to-visit relative changes are below the threshold."""
    first, second, third = losses
    change = lambda earlier, later: np.abs(later - earlier) / np.maximum(earlier, FLOOR)  # noqa: E731
    return (change(first, second) < threshold) & (change(second, third) < threshold)


def summarize(rows: list[dict], losses: dict[str, list[np.ndarray]]) -> dict:
    labels = [label for label, _, _ in MODELS]
    groups = [*sorted({row["group"] for row in rows}), "all-train", "all-heldout"]
    summary = {}
    for group in groups:
        members = [
            index
            for index, row in enumerate(rows)
            if row["group"] == group or group == f"all-{row['group'].split('-')[0]}"
        ]
        per_model = [np.concatenate([losses[label][index] for index in members]) for label in labels]
        entry = {
            "rows": len(members),
            "supervised_tokens": int(per_model[0].size),
            "mean_loss": {label: float(values.mean()) for label, values in zip(labels, per_model, strict=True)},
        }
        for threshold in THRESHOLDS:
            dropped = plateau_share(per_model, threshold)
            learned = per_model[-1] < LEARNED
            entry[f"dropped_{threshold:g}"] = float(dropped.mean())
            entry[f"dropped_learned_{threshold:g}"] = float((dropped & learned).mean())
            entry[f"dropped_stuck_{threshold:g}"] = float((dropped & ~learned).mean())
        last_change = np.abs(per_model[2] - per_model[1]) / np.maximum(per_model[1], FLOOR)
        entry["relative_change_epoch2_to_3_quantiles"] = {
            str(q): float(np.quantile(last_change, q)) for q in (0.1, 0.25, 0.5, 0.75, 0.9)
        }
        entry["share_below_0.1_nats_at_step1095"] = float((per_model[-1] < LEARNED).mean())
        summary[group] = entry
    return summary


def scorer(trainer: TrainerConfig):
    """A jitted function from (model, tokens[batch, position]) to per-position cross-entropy."""

    @named_jit(axis_resources=trainer.compute_axis_mapping)
    def score(model, tokens):
        model = trainer.mp.cast_to_compute(inference_mode(model, True))
        example = LmExample(tokens=tokens, loss_weight=None, attn_mask=AttentionMask.causal())
        return token_cross_entropy(model, example)[0]

    return score


def load_model(kind: str, path: str, converter, model_config, trainer: TrainerConfig, vocab: Axis):
    """Load an HF export or a native Levanter checkpoint, sharded for scoring. Call under the trainer's mesh."""
    if kind == "hf":
        return converter.load_pretrained(
            model_config.model_type,
            ref=path,
            config=model_config,
            axis_mapping=trainer.parameter_axis_mapping,
            dtype=trainer.mp.compute_dtype,
        )
    return load_levanter_checkpoint(
        model_config, path, Vocab=vocab, axis_mapping=trainer.parameter_axis_mapping, key=jax.random.PRNGKey(0)
    )


def probe(config: PlateauProbeConfig) -> dict:
    initialize(config)
    trainer = config.trainer
    rows = read_rows(config.train_rows, config.seed)
    logger.info("Scoring %d rows", len(rows))
    converter = SnowballConfig().hf_checkpoint_converter(PARENT).replaced(tokenizer=TOKENIZER)
    model_config = converter.config_from_hf_config(converter.default_hf_config)
    tokenizer = converter.tokenizer
    Batch = Axis("batch", BATCH)
    Pos = Axis("position", CONTEXT)
    losses: dict[str, list[np.ndarray]] = {}
    with trainer.use_device_mesh(), hax.axis_mapping(trainer.compute_axis_mapping):
        vocab = round_axis_for_partitioning(Axis("vocab", len(tokenizer)), trainer.compute_axis_mapping)

        score = scorer(trainer)
        for label, kind, path in MODELS:
            model = load_model(kind, path, converter, model_config, trainer, vocab)
            losses[label] = []
            for start in range(0, len(rows), BATCH):
                batch = rows[start : start + BATCH]
                ids = np.zeros((BATCH, CONTEXT), dtype=np.int32)
                for offset, row in enumerate(batch):
                    ids[offset] = row["input_ids"]
                tokens = hax.shard(hax.named(jnp.asarray(ids), (Batch, Pos)))
                token_loss = np.asarray(jax.device_get(score(model, tokens).array))
                for offset, row in enumerate(batch):
                    losses[label].append(token_loss[offset, supervised_positions(row["loss_weights"])])
            logger.info("Scored %s: mean %.4f", label, float(np.concatenate(losses[label]).mean()))
            del model
    summary = {
        "models": [{"label": label, "kind": kind, "path": path} for label, kind, path in MODELS],
        "floor": FLOOR,
        "learned_below": LEARNED,
        "train_rows": config.train_rows,
        "seed": config.seed,
        "groups": summarize(rows, losses),
    }
    if jax.process_index() == 0:
        StoragePath(prefix_join(config.output_path, "summary.json")).write_text(json.dumps(summary, indent=2))
        buffer = io.BytesIO()
        np.savez_compressed(
            buffer,
            groups=np.array([row["group"] for row in rows]),
            lengths=np.array([len(values) for values in losses[MODELS[0][0]]]),
            **{label: np.concatenate(values) for label, values in losses.items()},
        )
        StoragePath(prefix_join(config.output_path, "token-losses.npz")).write_bytes(buffer.getvalue())
        logger.info("Plateau probe summary: %s", json.dumps(summary["groups"]))
    return summary


def submit(config: PlateauProbeConfig) -> None:
    environment = {key: os.environ[key] for key in ("CW_KEY_ID", "CW_KEY_SECRET", "MARIN_PREFIX")}
    environment.update(GPU_LOG_ENV)
    request = JobRequest(
        name="plateau-probe",
        entrypoint=Entrypoint.from_callable(probe, args=(config,)),
        resources=ResourceConfig.with_gpu(
            "H100", count=8, replicas=1, cpu=64, ram="1024g", disk="512g", preemptible=False
        ),
        environment=create_environment(extras=["gpu"], env_vars=environment),
        timeout=Duration.from_seconds(80 * 60),
        max_retries_failure=0,
        max_retries_preemption=0,
    )
    current_client().submit(request, adopt_existing=False).wait(raise_on_failure=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", required=True)
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    config = PlateauProbeConfig(output_path=args.output)
    if not args.run:
        print(json.dumps({"models": MODELS, "output": args.output, "rows": "heldout + 500 train"}, indent=2))
        return
    info = get_job_info()
    assert info is not None and info.controller_address is not None
    with IrisClient.in_cluster(info.controller_address, workspace=Path.cwd()) as client:
        context = IrisContext.from_job_info(info, client=client)
        with iris_ctx_scope(context), set_current_client(FrayIrisClient.from_iris_client(client)):
            submit(config)
    print(json.dumps({"completed": "plateau-probe", "output": args.output}), flush=True)


if __name__ == "__main__":
    main()
