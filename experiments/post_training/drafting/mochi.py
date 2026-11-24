# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run Mochi teacher prefill and cached DSpark training on an Iris GPU gang."""

import fcntl
import json
import logging
import shutil
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from enum import StrEnum
from functools import partial
from pathlib import Path
from tempfile import TemporaryDirectory
from time import monotonic

import requests
from datasets import Dataset, load_from_disk
from marin.execution.lazy import ArtifactStep, StepContext, artifact_identity, lower
from marin.execution.step_runner import StepRunner, step_is_built
from marin.inference.config import InferenceModelConfig, ServingGeometry
from marin.inference.iris_vllm import (
    IrisVllmLaunch,
    iris_vllm_followers,
    iris_vllm_launch,
    notify_iris_vllm_stopped,
    wait_for_iris_vllm_shutdown,
)
from marin.inference.vllm_server import IsolatedCudaVllm, VllmEnvironment, VllmLauncherWithEnvironment, VllmType
from marin.training.drafting.activation_store import activation_path
from marin.training.drafting.dspark_data import (
    ARROW_DATASET_DIR,
    DRAFT_CONFIG_FILE,
    PREPARED_DIR,
    SPLITS_FILE,
    TOKEN_FREQUENCIES_FILE,
    PreparedSplits,
    supervised_token_ids,
)
from marin.training.drafting.dspark_distillation import (
    HIDDEN_STATES_DIR,
    DraftRecipe,
    DSparkExport,
    extraction_config,
    train_cached_and_export,
)
from rigging.filesystem.buckets import filesystem_for
from rigging.filesystem.cluster_config import marin_temp_bucket
from rigging.filesystem.storage_path import StoragePath
from rigging.timing import Duration, ExponentialBackoff
from safetensors.torch import load_file

logger = logging.getLogger(__name__)
_GPUS_PER_NODE = 4
_SMOKE_TRAIN_ROWS = 12
_SMOKE_VALIDATION_ROWS = 4
_SMOKE_MAX_TOKENS = 2048
_SMOKE_MAX_ANCHORS = 128
_SMOKE_ROW_LENGTH = range(720, 1537)
_ACTIVATION_TTL_DAYS = 3
_MODEL_ALIAS = "mochi"
_SHORT_CONV_COMPILATION = json.dumps(
    {
        "mode": 3,
        "cudagraph_mode": "PIECEWISE",
        "splitting_ops": [
            "vllm::unified_attention_with_output",
            "vllm::unified_kv_cache_update",
            "vllm::grug_moe_short_conv",
        ],
        "inductor_compile_config": {"enable_auto_functionalized_v2": False},
    }
)
_STREAMER = json.dumps({"distributed": True, "concurrency": 4, "memory_limit": 8 * 1024**3})


class Sample(StrEnum):
    SMOKE = "smoke"
    ALL = "all"


class DraftSize(StrEnum):
    TWO_LAYERS = "two-layer"
    ONE_LAYER = "one-layer"

    @property
    def architecture(self) -> tuple[int, int]:
        """Return the layer count and FFN width."""
        if self is DraftSize.ONE_LAYER:
            return 1, 4096
        return 2, 6144


@dataclass(frozen=True)
class CachedPrefill:
    """Run-time state for one distributed prefill and single-host trainer."""

    scratch: Path
    hidden_states: StoragePath
    split: PreparedSplits


@dataclass(frozen=True)
class CachedTraining:
    output: str
    teacher: str
    teacher_identity: str
    prepared: str
    recipe: DraftRecipe
    sample: Sample


def cached_recipe(recipe: DraftRecipe, sample: Sample) -> DraftRecipe:
    if sample is Sample.SMOKE:
        return replace(
            recipe,
            data=replace(recipe.data, max_sequence_length=min(recipe.data.max_sequence_length, _SMOKE_MAX_TOKENS)),
            max_anchors=min(recipe.max_anchors, _SMOKE_MAX_ANCHORS),
        )
    return recipe


def checkpoint_step(
    teacher: ArtifactStep,
    teacher_subdirectory: str,
    prepared: ArtifactStep,
    recipe: DraftRecipe,
    version: str,
    sample: Sample,
) -> ArtifactStep[DSparkExport]:
    """Bind Mochi's cached training phase to the teacher and prepared mix."""

    identity = artifact_identity(teacher)

    def config(ctx: StepContext) -> CachedTraining:
        return CachedTraining(
            output=ctx.output_path,
            teacher=str(StoragePath(ctx.artifact_path(teacher)) / teacher_subdirectory),
            teacher_identity=identity,
            prepared=ctx.artifact_path(prepared),
            recipe=recipe,
            sample=sample,
        )

    return ArtifactStep(
        name="checkpoints/dspark/mochi",
        version=version,
        artifact_type=DSparkExport,
        run=partial(_train_cached, cache=None),
        build_config=config,
        deps=(teacher, prepared),
    )


def _train_cached(config: CachedTraining, *, cache: CachedPrefill | None) -> DSparkExport:
    if cache is None:
        raise ValueError("Mochi training must collect the teacher's hidden states in the Iris gang first")
    return train_cached_and_export(
        output=config.output,
        teacher=config.teacher,
        teacher_identity=config.teacher_identity,
        recipe=config.recipe,
        scratch=cache.scratch,
        hidden_states=cache.hidden_states,
        split=cache.split,
    )


def _select_smoke_data(
    data: Dataset, split: PreparedSplits, source: Path, destination: Path
) -> tuple[Dataset, PreparedSplits]:
    train, validation = [], []
    for index, length in enumerate(data["seq_len"]):
        if length in _SMOKE_ROW_LENGTH:
            (train if index < split.train else validation).append(index)
    train, validation = train[:_SMOKE_TRAIN_ROWS], validation[:_SMOKE_VALIDATION_ROWS]
    if len(train) < _SMOKE_TRAIN_ROWS or not validation:
        raise ValueError(f"The Mochi smoke needs more rows with token lengths in {_SMOKE_ROW_LENGTH}")
    subset = data.select(train + validation)
    subset.with_format("torch").save_to_disk(str(destination / ARROW_DATASET_DIR))
    frequencies = Counter(
        token
        for row in subset.select(range(len(train)))
        for token in supervised_token_ids(row["input_ids"], row["loss_mask"])
    )
    (destination / TOKEN_FREQUENCIES_FILE).write_text(json.dumps(frequencies))
    shutil.copyfile(source / DRAFT_CONFIG_FILE, destination / DRAFT_CONFIG_FILE)
    return subset, PreparedSplits.from_counts(len(train), len(validation))


def _stage_data(prepared: str, scratch: Path, sample: Sample) -> tuple[Dataset, PreparedSplits]:
    source = StoragePath(prepared) / PREPARED_DIR
    data = scratch / "data"
    staged = scratch / "prepared" if sample is Sample.SMOKE else data
    source.download_to(str(staged), recursive=True)
    dataset = load_from_disk(str(staged / ARROW_DATASET_DIR)).with_format(None)
    assert isinstance(dataset, Dataset)
    split = PreparedSplits(**json.loads((staged / SPLITS_FILE).read_text()))
    if sample is Sample.SMOKE:
        dataset, split = _select_smoke_data(dataset, split, staged, data)
    return dataset, split


def _cache_rows(url: str, dataset: Dataset, hidden_states: StoragePath, recipe: DraftRecipe) -> None:
    def cache_row(row_index: int) -> tuple[int, float]:
        ids = dataset[row_index]["input_ids"]
        response = requests.post(
            f"{url}/completions",
            headers={"X-data-parallel-rank": "0"},
            json={"model": _MODEL_ALIAS, "prompt": ids, "max_tokens": 1, "temperature": 0, "detokenize": False},
            timeout=180,
        )
        response.raise_for_status()
        path = Path(response.json()["kv_transfer_params"]["hidden_states_path"])
        lock_path = Path(f"{path}.lock")
        ExponentialBackoff(initial=0.25, maximum=2).wait_until_or_raise(
            lock_path.is_file,
            timeout=Duration.from_seconds(120),
            error_message=f"Mochi did not start writing hidden states for row {row_index}",
        )
        with lock_path.open() as lock:
            fcntl.flock(lock, fcntl.LOCK_SH)
            tensors = load_file(path)
        expected = (len(ids), len(recipe.extraction_taps), recipe.data.shape.hidden_size)
        if tensors["token_ids"].tolist() != ids or tuple(tensors["hidden_states"].shape) != expected:
            raise ValueError(f"Mochi returned incorrect tokens or tap dimensions for row {row_index}")
        if not bool(tensors["hidden_states"].isfinite().all()):
            raise ValueError(f"Mochi returned non-finite features for row {row_index}")
        fs, key = filesystem_for(str(activation_path(hidden_states, row_index)))
        upload_start = monotonic()
        fs.put_file(str(path), key)
        upload_seconds = monotonic() - upload_start
        path.unlink()
        lock_path.unlink()
        return len(ids), upload_seconds

    # Pin requests to a local rank so the extracted tensors stay on the trainer's worker.
    started = monotonic()
    cached_tokens = 0
    total_upload_seconds = 0.0
    with ThreadPoolExecutor(max_workers=8) as pool:
        for count, (tokens, upload_seconds) in enumerate(pool.map(cache_row, range(len(dataset))), start=1):
            cached_tokens += tokens
            total_upload_seconds += upload_seconds
            if count % 32 == 0 or count == len(dataset):
                elapsed = monotonic() - started
                logger.info(
                    "Cached %d/%d Mochi rows for taps %s (%.2f rows/s, %.0f prompt tokens/s, %.2f s mean upload)",
                    count,
                    len(dataset),
                    recipe.extraction_taps,
                    count / elapsed,
                    cached_tokens / elapsed,
                    total_upload_seconds / count,
                )


def _teacher_environment(teacher: str, recipe: DraftRecipe, launch: IrisVllmLaunch, hidden: Path) -> VllmEnvironment:
    speculative, transfer = extraction_config(recipe, hidden)
    length = recipe.data.max_sequence_length + 1
    launcher = VllmLauncherWithEnvironment(
        IsolatedCudaVllm(source=VllmType.MARIN_FORK),
        {**launch.subprocess_env, "MAX_JOBS": "2", "PYTHONPATH": ""},
    )
    return VllmEnvironment(
        InferenceModelConfig(
            _MODEL_ALIAS,
            teacher,
            {"dtype": "bfloat16", "max_model_len": length, "gpu_memory_utilization": 0.94},
        ),
        launcher=launcher,
        timeout_seconds=2400,
        wait_for_ready=launch.is_leader,
        extra_args=[
            *launch.extra_cli_args,
            *(["--api-server-count=1"] if launch.is_leader else []),
            f"--served-model-name={_MODEL_ALIAS}",
            "--skip-tokenizer-init",
            "--enable-expert-parallel",
            "--attention-backend=FLASHINFER",
            "--moe-backend=triton",
            "--no-enable-prefix-caching",
            "--max-num-seqs=8",
            f"--max-num-batched-tokens={length}",
            f"--model-loader-extra-config={_STREAMER}",
            f"--compilation-config={_SHORT_CONV_COMPILATION}",
            f"--speculative-config={json.dumps(speculative)}",
            f"--kv-transfer-config={json.dumps(transfer)}",
        ],
    )


def train_mochi(
    *,
    teacher: str,
    prepared: ArtifactStep,
    draft: ArtifactStep[DSparkExport],
    recipe: DraftRecipe,
    prefix: str,
    sample: Sample,
) -> bool:
    """Run the gang and return whether this task is the leader reporting the checkpoint."""

    geometry = ServingGeometry(
        tensor_parallel_size=1,
        data_parallel_size=recipe.teacher_replicas,
        pipeline_parallel_size=1,
        gpus_per_task=_GPUS_PER_NODE,
    )
    if recipe.teacher_replicas < 8 or recipe.trainer_ranks > _GPUS_PER_NODE:
        raise ValueError("Mochi needs at least eight teacher GPUs and at most four trainer ranks on the leader")
    with TemporaryDirectory(prefix="mochi-dspark-") as directory, iris_vllm_launch(geometry) as launch:
        scratch = Path(directory)
        if launch.is_leader:
            checkpoint = lower(draft)
            if step_is_built(checkpoint):
                with iris_vllm_followers(launch):
                    StepRunner().run([checkpoint])
                return True
            StepRunner().run([lower(prepared)])
            dataset, split = _stage_data(prepared.path(prefix), scratch, sample)
            cache = StoragePath(
                marin_temp_bucket(_ACTIVATION_TTL_DAYS, prefix=f"dspark/{scratch.name}", source_prefix=prefix)
            )
            if cache.scheme not in ("s3", "gs"):
                raise ValueError(f"Mochi needs a temporary object-storage bucket, got {cache}")
            logger.info("Staging Mochi teacher activations in %s", cache)
        hidden = scratch / HIDDEN_STATES_DIR
        hidden.mkdir()
        with _teacher_environment(teacher, recipe, launch, hidden) as server:
            if launch.is_leader:
                with iris_vllm_followers(launch):
                    _cache_rows(server.server_url, dataset, cache, recipe)
            else:
                wait_for_iris_vllm_shutdown(launch, server.check_alive)
        if not launch.is_leader:
            notify_iris_vllm_stopped(launch)
            return False
        ready = replace(draft, run=partial(_train_cached, cache=CachedPrefill(scratch, cache, split)))
        StepRunner().run([lower(ready)])
        return True
