# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Prefill a Mochi activation cache and fit DSpark drafts on a separate worker."""

import fcntl
import hashlib
import json
import logging
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from dataclasses import dataclass, replace
from enum import StrEnum
from functools import partial
from pathlib import Path
from tempfile import TemporaryDirectory
from time import monotonic

import requests
from datasets import Dataset, load_from_disk
from marin.execution.fingerprint import canonical_json
from marin.execution.lazy import ArtifactStep, StepContext, artifact_identity, lower
from marin.execution.step_runner import StepRunner
from marin.external_dependencies import VLLM_GPU_RELEASE
from marin.inference.config import InferenceModelConfig, ServingGeometry
from marin.inference.iris_vllm import (
    IrisVllmLaunch,
    iris_vllm_followers,
    iris_vllm_launch,
    notify_iris_vllm_stopped,
    start_iris_vllm_followers,
    wait_for_iris_vllm_shutdown,
    wait_for_iris_vllm_start_decision,
)
from marin.inference.vllm_server import IsolatedCudaVllm, VllmEnvironment, VllmLauncherWithEnvironment, VllmType
from marin.training.drafting.activation_store import (
    activation_path,
    get_or_build_activation_cache,
    read_activation_cache,
)
from marin.training.drafting.dspark_data import (
    ARROW_DATASET_DIR,
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
    train_cached_draft,
)
from rigging.filesystem.buckets import filesystem_for
from rigging.filesystem.cluster_config import marin_temp_bucket
from rigging.filesystem.storage_path import StoragePath
from rigging.timing import Duration, ExponentialBackoff
from safetensors.torch import load_file

from vendor.skyrl.marinskyrl.model_manifest import MODEL_MANIFEST_FILENAME, snapshot_model_manifest

logger = logging.getLogger(__name__)
_GPUS_PER_NODE = 4
_SMOKE_TRAIN_ROWS = 12
_SMOKE_VALIDATION_ROWS = 4
_SMOKE_MAX_TOKENS = 2048
_SMOKE_MAX_ANCHORS = 128
_SMOKE_ROW_LENGTH = range(720, 1537)
_ACTIVATION_TTL_DAYS = 3
_ACTIVATION_REUSE_WINDOW = Duration.from_hours(48)
_ACTIVATION_FORMAT_VERSION = 1
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
class CachedTraining:
    output: str
    teacher: str
    teacher_identity: str
    prepared: str
    recipe: DraftRecipe
    sample: Sample


@dataclass(frozen=True)
class MochiPrefillInputs:
    teacher_identity: str
    prepared_identity: str
    sample: Sample
    sequence_length: int
    extraction_taps: tuple[int, ...]
    teacher_replicas: int
    vllm_source: str = VLLM_GPU_RELEASE.source_commit
    format_version: int = _ACTIVATION_FORMAT_VERSION


def activation_cache_key(teacher_identity: str, prepared: ArtifactStep, recipe: DraftRecipe, sample: Sample) -> str:
    """Identify the exact teacher tensors independently of the draft being trained."""
    inputs = MochiPrefillInputs(
        teacher_identity,
        artifact_identity(prepared),
        sample,
        recipe.data.max_sequence_length,
        recipe.extraction_taps,
        recipe.teacher_replicas,
    )
    return hashlib.sha256(canonical_json(inputs).encode()).hexdigest()


def _activation_cache_root(key: str, prefix: str) -> StoragePath:
    root = StoragePath(marin_temp_bucket(_ACTIVATION_TTL_DAYS, prefix="dspark/mochi", source_prefix=prefix)) / key
    if root.scheme not in ("s3", "gs"):
        raise ValueError(f"Mochi needs a temporary object-storage bucket, got {root}")
    return root


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
    key = activation_cache_key(identity, prepared, recipe, sample)

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
        run=partial(_train_cached, cache_key=key),
        build_config=config,
        deps=(teacher, prepared),
    )


def _publish_draft(export: Path, destination: StoragePath) -> str:
    manifest = snapshot_model_manifest(export, None, None, tokenizer_mode="policy")
    destination.mkdirs()
    marker = destination / MODEL_MANIFEST_FILENAME
    # The artifact record write can fail after the draft has been uploaded.
    if marker.exists():
        marker.rm()
    for entry in manifest.files:
        (destination / entry.path).upload_from(str(export / entry.path))
    marker.write_text(manifest.model_dump_json(indent=2) + "\n")
    return manifest.identity


def _train_cached(config: CachedTraining, *, cache_key: str) -> DSparkExport:
    root = _activation_cache_root(cache_key, config.output)
    cache = read_activation_cache(root, reusable_for=_ACTIVATION_REUSE_WINDOW)
    if cache is None:
        raise ValueError(f"No reusable Mochi teacher activations at {root}. Run --stage prefill with the same inputs.")
    logger.info("Training on Mochi teacher activations in %s", cache)
    with TemporaryDirectory(prefix="mochi-dspark-fit-") as directory:
        scratch = Path(directory)
        _, split = _stage_data(config.prepared, scratch, config.sample)
        result = train_cached_draft(
            teacher=config.teacher,
            teacher_identity=config.teacher_identity,
            recipe=config.recipe,
            scratch=scratch,
            hidden_states=cache,
            split=split,
        )
        destination = StoragePath(config.output) / result.hf_subdirectory
        identity = _publish_draft(scratch / result.hf_subdirectory, destination)
        return result.model_copy(update={"source_identity": identity})


def _select_smoke_data(data: Dataset, split: PreparedSplits, destination: Path) -> tuple[Dataset, PreparedSplits]:
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
        dataset, split = _select_smoke_data(dataset, split, data)
    return dataset, split


def _cache_rows(url: str, dataset: Dataset, hidden_states: StoragePath, recipe: DraftRecipe) -> None:
    def cache_row(row_index: int) -> tuple[int, float]:
        """Cache a row and return the prompt token count and upload duration in seconds."""
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


def _teacher_environment(teacher: str, recipe: DraftRecipe, launch: IrisVllmLaunch, scratch: Path) -> VllmEnvironment:
    hidden = scratch / HIDDEN_STATES_DIR
    hidden.mkdir()
    speculative, transfer = extraction_config(recipe, hidden)
    length = recipe.data.max_sequence_length + 1
    launcher = VllmLauncherWithEnvironment(
        IsolatedCudaVllm(source=VllmType.MARIN_FORK, release=VLLM_GPU_RELEASE),
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


def _cache_activations(
    teacher: str,
    recipe: DraftRecipe,
    launch: IrisVllmLaunch,
    prepared: str,
    sample: Sample,
    root: StoragePath,
    scratch: Path,
) -> StoragePath:
    with ExitStack() as teacher_runtime, iris_vllm_followers(launch):

        def populate(cache: StoragePath) -> None:
            logger.info("Caching Mochi teacher activations in %s", cache)
            dataset, _ = _stage_data(prepared, scratch, sample)
            start_iris_vllm_followers(launch)
            server = teacher_runtime.enter_context(_teacher_environment(teacher, recipe, launch, scratch))
            _cache_rows(server.server_url, dataset, cache, recipe)

        return get_or_build_activation_cache(root, reusable_for=_ACTIVATION_REUSE_WINDOW, populate=populate)


def prefill_mochi(
    *,
    teacher: str,
    teacher_identity: str,
    prepared: ArtifactStep,
    recipe: DraftRecipe,
    prefix: str,
    sample: Sample,
) -> StoragePath | None:
    """Build the activation cache on the teacher gang and return its path on the leader."""

    geometry = ServingGeometry(
        tensor_parallel_size=1,
        data_parallel_size=recipe.teacher_replicas,
        pipeline_parallel_size=1,
        gpus_per_task=_GPUS_PER_NODE,
    )
    if recipe.teacher_replicas < 8:
        raise ValueError("Mochi needs at least eight teacher GPUs")
    with TemporaryDirectory(prefix="mochi-dspark-prefill-") as directory, iris_vllm_launch(geometry) as launch:
        scratch = Path(directory)
        if not launch.is_leader:
            if wait_for_iris_vllm_start_decision(launch):
                with _teacher_environment(teacher, recipe, launch, scratch) as server:
                    wait_for_iris_vllm_shutdown(launch, server.check_alive)
            notify_iris_vllm_stopped(launch)
            return None

        StepRunner().run([lower(prepared)])
        key = activation_cache_key(teacher_identity, prepared, recipe, sample)
        root = _activation_cache_root(key, prefix)
        return _cache_activations(teacher, recipe, launch, prepared.path(prefix), sample, root, scratch)
