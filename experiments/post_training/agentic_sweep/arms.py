# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Train one SFT shootout arm (plain, RS-multi, recovery or relay) on the row builder's per-epoch files (task #235).

Epoch k trains exactly the builder's ``train-epoch-<k>-prebuilt.jsonl.gz``, in file order. Each epoch file gets its
own Levanter cache, and the training mixture has one stage per epoch that draws only from that epoch's cache, with
mixture block 16 (one block per batch) and no shuffle. So batch b of epoch k is rows 16b..16b+15 of that file; only
the order inside a batch is permuted, and the batch loss does not depend on it. There is no global shuffle and no
nominal block-2048 epoch.

Before any GPU is requested the submitter checks every epoch file against the arm manifest: file hash, rows_sha256
of the decompressed rows, padded row count and steps, supervised and context tokens. It checks that the replay rows
are byte-identical to the plain arm's in every epoch, and that every cache row matches its file row, in order.

Recipe of the #203 sweep: AdamW, constant LR, betas 0.9/0.999, WD 0, clip 1.0, router bias frozen, batch 16 x 32k,
mean loss over supervised tokens, no in-loop validation, NCCL WARN. A permanent checkpoint at every epoch end, HF
exports after epoch 2 and at the end, and per-epoch supervised and context tokens checked against the manifest
within 0.1% (epoch-metrics.jsonl). ``--plateau`` turns on plateau dropping (task #226).

    python -m experiments.post_training.agentic_sweep.arms --arm-data <prefix> --manifest-sha256 <sha256> \\
        --plain-data <plain arm prefix> --parent-uri <parent> --learning-rate 1e-6 --epochs 2 [--only train] [--run]

Without --run it prints the plan from the manifest. With --run (inside an Iris CPU job) it verifies the files,
builds the caches (--only cache) or verifies the caches and submits the 4x8 H100 training job (--only train).
"""

import argparse
import gzip
import hashlib
import io
import itertools
import json
import os
from collections.abc import Sequence
from dataclasses import asdict, dataclass, replace
from functools import partial
from pathlib import Path
from typing import cast

import numpy as np
from fray.current_client import current_client, set_current_client
from fray.iris_backend import FrayIrisClient
from fray.types import ANY_REGION, Entrypoint, JobRequest, ResourceConfig, create_environment
from iris.client.client import IrisClient, IrisContext, iris_ctx_scope
from iris.cluster.client.job_info import get_job_info
from levanter.data.text.datasets import DatasetComponent, LmDataConfig
from levanter.main.train_lm import TrainLmConfig
from levanter.plateau import PlateauDroppingConfig, row_key
from levanter.store.cache import TreeCache
from levanter.tracker.json_logger import JsonLoggerConfig
from levanter.tracker.wandb import WandbConfig
from levanter.utils.mesh import MeshConfig
from marin.execution.lazy import ArtifactStep, StepContext, lower, run
from marin.processing.tokenize.tokenize import TokenizeConfig, TokenizedCache, tokenize
from marin.training.training import LevanterCheckpoint, TrainLmOnPodConfig
from rigging.filesystem.storage_path import StoragePath, prefix_join
from rigging.timing import Duration

from experiments.post_training.agentic_sweep.callbacks import EpochTrackerConfig, SweepExportsConfig
from experiments.post_training.agentic_sweep.train import (
    BATCH_SIZE,
    CONTEXT,
    FORMAT,
    GPU_LOG_ENV,
    MODEL,
    REVISION,
    TOKENIZER,
    optimizer,
    run_worker,
)
from experiments.sft.launcher import HFModel, SFTSpec

ARMS = ("plain", "rs-multi", "recovery", "relay")
VERSION = "2026.10.18.1"
EXPORT_EPOCHS = [2]
EXEMPLAR = {"input_ids": np.zeros(0, dtype=np.int32), "loss_weights": np.zeros(0, dtype=np.float32)}


@dataclass(frozen=True)
class ArmEpoch:
    """One epoch file as the manifest describes it."""

    file: str
    rows: int
    updates: int
    supervised_tokens: int
    context_tokens: int


def epoch_file(epoch: int) -> str:
    return f"train-epoch-{epoch}-prebuilt.jsonl.gz"


def arm_epochs(manifest: dict, epochs: int) -> list[ArmEpoch]:
    """The first ``epochs`` epoch files of the arm, with their padded rows, updates and token totals."""
    if manifest["arm"] not in ARMS:
        raise ValueError(f"Not a training arm: {manifest['arm']}")
    if manifest["batch_size"] != BATCH_SIZE:
        raise ValueError(f"The arm was built for batch {manifest['batch_size']}, not {BATCH_SIZE}")
    if not 1 <= epochs <= sum(name.startswith("epoch-") for name in manifest["splits"]):
        raise ValueError(f"The arm has no epoch file for each of {epochs} epochs")
    result = []
    for epoch in range(epochs):
        split = manifest["splits"][f"epoch-{epoch}"]
        rows = split["padded_rows"]
        if rows % BATCH_SIZE or split["steps"] != rows // BATCH_SIZE or rows != split["rows"] + split["padding_rows"]:
            raise ValueError(f"Epoch {epoch} is not padded to whole batches of {BATCH_SIZE}")
        result.append(
            ArmEpoch(
                file=epoch_file(epoch),
                rows=rows,
                updates=split["steps"],
                supervised_tokens=split["supervised_total"],
                context_tokens=sum(split["context_tokens"].values()),
            )
        )
    return result


def row_context(input_ids: np.ndarray, pad_token_id: int) -> int:
    """Positions up to the last token that is not padding, as the trainer's context_tokens metric counts them."""
    filled = np.flatnonzero(input_ids != pad_token_id)
    return int(filled[-1]) + 1 if filled.size else 0


def read_epoch(root: StoragePath, manifest: dict, epoch: int, replay_rows: set[int]) -> tuple[list[str], str]:
    """Check one epoch file against the manifest; return its row keys in order and the digest of its replay rows."""
    name = epoch_file(epoch)
    split = manifest["splits"][f"epoch-{epoch}"]
    data = (root / name).read_bytes()
    if hashlib.sha256(data).hexdigest() != manifest["output_hashes"][name]:
        raise ValueError(f"{name} differs from the manifest's output hash")
    rows_digest = hashlib.sha256()
    keys, replay = [], []
    supervised = context = 0
    pad = manifest["pad_token_id"]
    with gzip.open(io.BytesIO(data), "rb") as lines:
        for number, line in enumerate(lines):
            rows_digest.update(line)
            row = json.loads(line)
            input_ids = np.asarray(row["input_ids"], dtype=np.int32)
            loss_weights = np.asarray(row["loss_weights"], dtype=np.float32)
            if len(input_ids) != manifest["max_sequence_tokens"] or loss_weights.shape != input_ids.shape:
                raise ValueError(f"{name} row {number} has the wrong length")
            if not np.isin(loss_weights, [0, 1]).all() or loss_weights[-1] != 0:
                raise ValueError(f"{name} row {number} has a malformed loss mask")
            keys.append(row_key(input_ids, loss_weights))
            supervised += int(loss_weights.sum())
            context += row_context(input_ids, pad)
            if number in replay_rows:
                replay.append(hashlib.sha256(line).hexdigest())
    if rows_digest.hexdigest() != split["rows_sha256"]:
        raise ValueError(f"{name} rows differ from the manifest's rows_sha256")
    if len(keys) != split["padded_rows"]:
        raise ValueError(f"{name} holds {len(keys)} rows, the manifest {split['padded_rows']}")
    if supervised != split["supervised_total"]:
        raise ValueError(f"{name} holds {supervised} supervised tokens, the manifest {split['supervised_total']}")
    expected = sum(split["context_tokens"].values())
    if abs(context - expected) > 0.001 * expected:
        raise ValueError(f"{name} holds {context} context tokens, the manifest {expected}")
    if len(replay) != len(replay_rows):
        raise ValueError(f"{name} is missing replay rows")
    return keys, hashlib.sha256("".join(sorted(replay)).encode()).hexdigest()


def replay_rows_by_epoch(root: StoragePath) -> dict[int, set[int]]:
    rows: dict[int, set[int]] = {}
    for line in (root / "train-index.jsonl").read_text().splitlines():
        entry = json.loads(line)
        if entry["family"] == "replay":
            rows.setdefault(entry["epoch"], set()).add(entry["row"])
    return rows


def verify_arm(prefix: str, plain_prefix: str, manifest: dict, epochs: int) -> list[list[str]]:
    """Check the epoch files and the replay rows against plain's; return each epoch's row keys in file order."""
    root, plain_root = StoragePath(prefix), StoragePath(plain_prefix)
    plain_manifest = json.loads((plain_root / "manifest.json").read_text())
    if plain_manifest["arm"] != "plain":
        raise ValueError("The replay reference must be the plain arm")
    if manifest["replay"]["rows_sha256"] != plain_manifest["replay"]["rows_sha256"]:
        raise ValueError("The replay cut differs from the plain arm's")
    replay = replay_rows_by_epoch(root)
    _, plain_replay = read_epoch(plain_root, plain_manifest, 0, replay_rows_by_epoch(plain_root)[0])
    keys = []
    for epoch in range(epochs):
        epoch_keys, epoch_replay = read_epoch(root, manifest, epoch, replay.get(epoch, set()))
        if len(replay.get(epoch, ())) != manifest["replay"]["rows"] or epoch_replay != plain_replay:
            raise ValueError(f"Epoch {epoch} replay rows are not byte-identical to the plain arm's")
        keys.append(epoch_keys)
    return keys


def verify_caches(cache_paths: Sequence[str], keys: Sequence[Sequence[str]]) -> None:
    """Every cache row must be its epoch file's row at the same position."""
    for epoch, (path, epoch_keys) in enumerate(zip(cache_paths, keys, strict=True)):
        cache = TreeCache.load(path, EXEMPLAR)
        if len(cache) != len(epoch_keys):
            raise ValueError(f"Epoch {epoch} cache holds {len(cache)} rows, the file {len(epoch_keys)}")
        for start in range(0, len(epoch_keys), BATCH_SIZE):
            rows = cache.get_batch_sync(list(range(start, start + BATCH_SIZE)))
            for offset, row in enumerate(rows):
                if row_key(row["input_ids"], row["loss_weights"]) != epoch_keys[start + offset]:
                    raise ValueError(f"Epoch {epoch} cache row {start + offset} differs from the file")
        print(json.dumps({"verified_cache": epoch, "rows": len(epoch_keys)}), flush=True)


def arm_data(cache_paths: Sequence[str], epochs: Sequence[ArmEpoch], tokenizer: str) -> LmDataConfig:
    """Stage k of the mixture reads only epoch k's cache, in order, one 16-row block per batch."""
    names = [f"epoch-{epoch}" for epoch in range(len(epochs))]
    starts = [0, *itertools.accumulate(epoch.updates for epoch in epochs)][:-1]
    return LmDataConfig(
        tokenizer=tokenizer,
        components={
            name: DatasetComponent(cache_dir=path, format=FORMAT, split="train", flat_cache=True, pack=False)
            for name, path in zip(names, cache_paths, strict=True)
        },
        train_weights=[
            (start, {other: float(other == name) for other in names}) for start, name in zip(starts, names, strict=True)
        ],
        shuffle=False,
        mixture_block_size=BATCH_SIZE,
        enforce_eos=False,
        auto_build_caches=False,
        block_cross_document_attention=False,
    )


def arm_settings(
    config: TrainLmConfig,
    epochs: Sequence[ArmEpoch],
    output_path: str,
    pad_token_id: int,
    plateau: bool,
    start_step: int = 0,
) -> TrainLmConfig:
    """Epoch accounting, a checkpoint per epoch, HF exports after epoch 2 and at the end, optional plateau dropping."""
    updates = [epoch.updates for epoch in epochs]
    tracker = EpochTrackerConfig(
        output_path,
        updates,
        [epoch.supervised_tokens for epoch in epochs],
        [epoch.context_tokens for epoch in epochs],
        start_step=start_step,
    )
    trackers = config.trainer.tracker if isinstance(config.trainer.tracker, tuple) else (config.trainer.tracker,)
    return replace(
        config,
        trainer=replace(
            config.trainer,
            num_train_steps=sum(updates),
            tracker=(*trackers, tracker),
            checkpointer=replace(config.trainer.checkpointer, keep=[], save_interval=None),
        ),
        adapter=SweepExportsConfig(None, updates, config.train_seq_len or CONTEXT, EXPORT_EPOCHS),
        context_pad_token_id=pad_token_id,
        plateau_dropping=PlateauDroppingConfig() if plateau else None,
    )


def run_training(
    config: TrainLmOnPodConfig, cache_paths: list[str], keys: list[list[str]], preemption_retries: int = 0
) -> None:
    if not os.environ.get("WANDB_API_KEY"):
        raise ValueError("A nonempty W&B key is required before requesting GPUs")
    verify_caches(cache_paths, keys)
    environment = {key: os.environ[key] for key in ("CW_KEY_ID", "CW_KEY_SECRET", "MARIN_PREFIX", "WANDB_API_KEY")}
    environment.update(GPU_LOG_ENV)
    if "HF_TOKEN" in os.environ:
        environment["HF_TOKEN"] = os.environ["HF_TOKEN"]
    request = JobRequest(
        name="snowball-v0-arm-sft",
        entrypoint=Entrypoint.from_callable(run_worker, args=(config,)),
        resources=config.resources,
        environment=create_environment(extras=["gpu"], env_vars=environment),
        timeout=Duration.from_seconds(12 * 60 * 60),
        max_retries_failure=0,
        # A retry restarts from the job's --resume-checkpoint (or from scratch), never from a mid-epoch checkpoint.
        max_retries_preemption=preemption_retries,
    )
    current_client().submit(request, adopt_existing=False).wait(raise_on_failure=True)


def pipeline(
    manifest: dict,
    manifest_sha256: str,
    data_source: str,
    parent_uri: str,
    learning_rate: float,
    epochs: list[ArmEpoch],
    plateau: bool,
    keys: list[list[str]] | None = None,
    resume_checkpoint: str | None = None,
    preemption_retries: int = 0,
) -> dict[str, ArtifactStep | list[ArtifactStep]]:
    if manifest["student_revision"] != REVISION or manifest["student_model"] != MODEL:
        raise ValueError("The arm was built for a different parent")
    if manifest["max_sequence_tokens"] != CONTEXT:
        raise ValueError("The arm changes the frozen context")
    if learning_rate <= 0:
        raise ValueError("The learning rate must be positive")
    arm = manifest["arm"]
    ends = list(itertools.accumulate(epoch.updates for epoch in epochs))
    start_step = 0
    if resume_checkpoint is not None:
        metadata = json.loads(StoragePath(prefix_join(resume_checkpoint, "metadata.json")).read_text())
        start_step = int(metadata["step"]) + 1
        if start_step not in ends[:-1]:
            raise ValueError("Arms resume only from an epoch-end checkpoint")
    source = ArtifactStep.adopt(
        f"datasets/karan/arm-{arm}-{manifest_sha256[:12]}", VERSION, data_source, config=manifest
    )
    parent = ArtifactStep.adopt(
        "checkpoints/karan/v0-arm-parent", VERSION, parent_uri, config={"model": MODEL, "revision": REVISION}
    )

    def cache_config(ctx: StepContext, epoch: int) -> TokenizeConfig:
        return TokenizeConfig(
            train_paths=[prefix_join(ctx.artifact_path(source), epoch_file(epoch))],
            validation_paths=[],
            allow_test_in_train=False,
            cache_path=ctx.output_path,
            tokenizer=TOKENIZER,
            format=FORMAT,
            max_workers=1,
            levanter_batch_size=8,
            worker_resources=ResourceConfig.with_cpu(cpu=4, ram="16g", disk="32g"),
            tags=["sft-arms", "task235", arm],
        )

    caches = [
        ArtifactStep(
            name=f"tokenized/karan/arm-{arm}-{manifest_sha256[:12]}-epoch-{epoch}",
            version=VERSION,
            artifact_type=TokenizedCache,
            run=tokenize,
            build_config=partial(cache_config, epoch=epoch),
            deps=(source,),
        )
        for epoch in range(len(epochs))
    ]
    model = HFModel(model_ref=f"{MODEL}@{REVISION}", tokenizer_path=TOKENIZER, model_type="snowball")
    resources = ResourceConfig.with_gpu(
        "H100", count=8, replicas=4, cpu=32, ram="512g", disk="512g", regions=[ANY_REGION], preemptible=False
    )
    name = f"arm-{arm}-lr{learning_rate:g}-{len(epochs)}ep" + ("-plateau" if plateau else "")
    spec = SFTSpec(
        name=f"checkpoints/karan/{name}",
        version=VERSION,
        model=model,
        chat_template="",
        datasets=[],
        optimizer=optimizer(learning_rate),
        mesh=MeshConfig(
            axes={"data": 1, "replica": 1, "model": 1, "expert": 8},
            dcn_axes={"replica_dcn": 1, "context": 4},
            compute_mapping={"batch": ["replica_dcn", "data", "expert"], "position": "context", "vocab": "model"},
        ),
        seq_len=CONTEXT,
        pack=False,
        batch_size=BATCH_SIZE,
        num_train_steps=ends[-1],
        hf_save_dtype="bfloat16",
        wandb_project="snowball-v0-sft-arms",
    )

    def cache_paths(ctx: StepContext) -> list[str]:
        return [prefix_join(ctx.artifact_path(cache), "train") for cache in caches]

    def train_config(ctx: StepContext) -> TrainLmOnPodConfig:
        data = arm_data(cache_paths(ctx), epochs, TOKENIZER)
        pod = replace(model, model_ref=ctx.artifact_path(parent)).build_train_config(
            ctx, spec, data, resources, ends[-1]
        )
        config = cast(TrainLmConfig, pod.train_config)
        wandb = replace(
            cast(WandbConfig, config.trainer.tracker),
            entity="marin-community",
            name=name,
            # A resumed run (and a preemption retry of it) logs to its own W&B run, so its steps never collide
            # with the interrupted run's.
            resume="never" if resume_checkpoint is None else "allow",
            replicate_path=ctx.output_path,
            mode="online",
            save_code=False,
        )
        config = replace(
            config,
            trainer=replace(
                config.trainer,
                id=name + "-20261018" + (f"-from{start_step}" if resume_checkpoint is not None else ""),
                seed=42,
                load_checkpoint=resume_checkpoint is not None,
                load_checkpoint_path=resume_checkpoint,
                per_device_parallelism=2,
                per_device_eval_parallelism=2,
                tracker=(wandb, JsonLoggerConfig()),
            ),
            data_seed=42,
        )
        config = arm_settings(config, epochs, ctx.output_path, manifest["pad_token_id"], plateau, start_step)
        return replace(pod, train_config=config, env_vars={"WANDB_MODE": "online", **GPU_LOG_ENV})

    def train(config: TrainLmOnPodConfig) -> None:
        assert keys is not None, "Training needs the verified row keys"
        train_data = cast(TrainLmConfig, config.train_config).data
        paths = [cast(DatasetComponent, component).cache_dir for component in train_data.components.values()]
        run_training(config, cast(list[str], paths), keys, preemption_retries)

    trained = ArtifactStep(
        name=spec.name,
        version=spec.version,
        artifact_type=LevanterCheckpoint,
        run=train,
        build_config=train_config,
        deps=(*caches, parent),
    )
    return {"caches": caches, "train": trained}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--arm-data", required=True, help="built arm prefix holding manifest.json")
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--plain-data", required=True, help="built plain arm prefix, the replay reference")
    parser.add_argument("--parent-uri", required=True)
    parser.add_argument("--learning-rate", type=float, required=True)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--plateau", action="store_true")
    parser.add_argument("--only", choices=("cache", "train"), default="cache")
    parser.add_argument("--resume-checkpoint")
    parser.add_argument("--preemption-retries", type=int, default=0)
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    manifest_bytes = (StoragePath(args.arm_data) / "manifest.json").read_bytes()
    if hashlib.sha256(manifest_bytes).hexdigest() != args.manifest_sha256:
        raise ValueError("Arm manifest hash mismatch")
    manifest = json.loads(manifest_bytes)
    epochs = arm_epochs(manifest, args.epochs)
    if not args.run:
        artifacts = pipeline(
            manifest, args.manifest_sha256, args.arm_data, args.parent_uri, args.learning_rate, epochs, args.plateau
        )
        print(lower(cast(ArtifactStep, artifacts["train"])))
        print(json.dumps({"arm": manifest["arm"], "epochs": [asdict(epoch) for epoch in epochs]}, indent=1))
        return
    keys = verify_arm(args.arm_data, args.plain_data, manifest, args.epochs)
    artifacts = pipeline(
        manifest,
        args.manifest_sha256,
        args.arm_data,
        args.parent_uri,
        args.learning_rate,
        epochs,
        args.plateau,
        keys,
        args.resume_checkpoint,
        args.preemption_retries,
    )
    info = get_job_info()
    assert info is not None and info.controller_address is not None
    with IrisClient.in_cluster(info.controller_address, workspace=Path.cwd()) as client:
        context = IrisContext.from_job_info(info, client=client)
        with iris_ctx_scope(context), set_current_client(FrayIrisClient.from_iris_client(client)):
            if args.only == "cache":
                results = run(*cast(list[ArtifactStep], artifacts["caches"]), max_concurrent=1)
                verify_caches([prefix_join(result.path, "train") for result in results], keys)
                print(json.dumps({"completed": "cache", "outputs": [result.path for result in results]}), flush=True)
                return
            result = run(cast(ArtifactStep, artifacts["train"]), max_concurrent=1)[0]
            print(json.dumps({"completed": "train", "output": result.path}), flush=True)


if __name__ == "__main__":
    main()
