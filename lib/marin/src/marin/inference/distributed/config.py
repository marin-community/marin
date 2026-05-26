# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Configuration dataclasses for the distributed inference library.

`ModelSpec` carries the model identifier and engine kwargs. The ``model``
field accepts three URI shapes:

* an HF model id (e.g. ``"meta-llama/Llama-3-8B"``) — vLLM downloads at startup
* a ``marin://`` URI (e.g. ``"marin://checkpoints/qwen3-8b/hf/step-1318"``) —
  resolved per worker region to the corresponding ``gs://marin-{region}/…``
* an explicit ``gs://`` URI — hard-pinned, rejected at worker startup if it
  is not in the worker's region

`SamplingParams` mirrors the most-commonly-used vLLM sampling fields plus a
typed ``extra`` mapping for less-common settings.

`InferenceConfig` is the user-facing config. Defaults match upstream Zephyr;
inference callers explicitly raise the Zephyr per-context fields for
long-running workloads.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from rigging.filesystem import REGION_TO_DATA_BUCKET
from zephyr.execution import MAX_SHARD_FAILURES, MAX_SHARD_INFRA_FAILURES

from marin.rl.placement import marin_prefix_for_region

# Default TPU shapes for inference workloads. Every entry is single-host and
# shares `vm_count=1` and `chips_per_vm=4`, which `ResourceConfig.with_tpu`
# requires for cross-shape alternates (`v6e-8` / `v5litepod-8` use 8 chips
# per VM and cannot be mixed with these 4-chip shapes; callers that want
# 8-chip workers should set `tpu_shapes=("v6e-8", "v5litepod-8")` themselves).
DEFAULT_TPU_SHAPES: tuple[str, ...] = (
    "v5p-8",
    "v6e-4",
    "v5litepod-4",
)

# Heartbeat timeout default — matches upstream Zephyr `ZephyrContext` default.
DEFAULT_HEARTBEAT_TIMEOUT: float = 120.0


@dataclass(frozen=True)
class ModelSpec:
    """Model identifier + vLLM engine kwargs.

    The ``model`` field is polymorphic; see `resolve_for_region`. Use the
    ``marin://`` form when the same model is replicated to every region's
    ``marin-{region}`` bucket and workers should load from their local
    replica. Use an explicit ``gs://`` URI when the model is pinned to one
    region (workers in other regions will crash at startup).
    """

    model: str
    engine_kwargs: Mapping[str, object] = field(default_factory=dict)

    def resolve_for_region(self, region: str) -> str:
        """Return a concrete model identifier usable in the given region.

        - HF id passes through unchanged (vLLM will download at startup).
        - ``marin://path`` is expanded to ``gs://marin-{region}/path``.
        - ``gs://...`` is validated against ``region``; raises if the bucket
          is in a different region.
        """
        if self.model.startswith("marin://"):
            suffix = self.model.removeprefix("marin://").lstrip("/")
            return f"{marin_prefix_for_region(region)}/{suffix}"
        if self.model.startswith("gs://"):
            bucket = self.model.removeprefix("gs://").split("/", 1)[0]
            expected = REGION_TO_DATA_BUCKET.get(region.lower())
            if expected is None:
                raise ValueError(
                    f"No Marin data bucket configured for region {region!r}; "
                    f"cannot validate model path {self.model!r}."
                )
            if bucket != expected:
                raise ValueError(
                    f"Model path {self.model!r} is in bucket {bucket!r}, but worker "
                    f"is in region {region!r} (expected bucket {expected!r}). "
                    f"Cross-region model loads cost egress; use a 'marin://' URI to "
                    f"resolve per region, or specify a path in the worker's region."
                )
            return self.model
        return self.model


@dataclass(frozen=True)
class SamplingParams:
    """vLLM sampling parameters.

    Common fields are explicit so they autocomplete and type-check. The
    ``extra`` mapping forwards any other vLLM ``SamplingParams`` field
    (e.g. ``logits_processors``) without the library having to track every
    addition vLLM ships.
    """

    temperature: float = 0.0
    max_tokens: int = 1024
    min_tokens: int = 0
    top_p: float = 1.0
    top_k: int = -1
    repetition_penalty: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    n: int = 1
    stop: tuple[str, ...] = ()
    stop_token_ids: tuple[int, ...] = ()
    seed: int | None = None
    logprobs: int | None = None
    prompt_logprobs: int | None = None
    skip_special_tokens: bool = True
    extra: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class InferenceConfig:
    """Top-level configuration for a distributed inference run.

    Args:
        regions: Worker regions. The library launches one Zephyr job per region.
        results_region: Where shard outputs are written. All regional jobs
            write to ``gs://marin-{results_region}/<job_name>/<run_id>/outputs/``.
            Defaults to ``"us-central1"``.
        tpu_shapes: TPU variants the workers may run on. Must share vm_count
            and chips_per_vm (enforced by `ResourceConfig.with_tpu`).
        max_workers_per_region: Cap on concurrent worker actors per region.
        shard_size: Records per Zephyr shard. Controls failure granularity:
            preemption loses at most one shard's worth of work. For inline
            (in-memory) input, this also chunks the materialized JSONL files
            on the way out, since the pipeline maps one input file to one
            content shard. For path/glob input, the pre-existing file layout
            determines sharding and ``shard_size`` is unused.
        sampling: Sampling parameters forwarded to vLLM.
        job_name: Prefix used in Fray job names and GCS output paths.
        worker_preemptible: Whether worker tasks are preemptible.
        heartbeat_timeout: Per-context Zephyr heartbeat timeout (seconds).
            Default matches upstream Zephyr; raise for long-running inference
            with cold XLA compile windows.
        max_shard_failures: Per-context Zephyr task-error cap. Default matches
            upstream Zephyr.
        max_shard_infra_failures: Per-context Zephyr infra-failure cap. Default
            matches upstream Zephyr.
        chunk_size: Zephyr coordinator checkpoint stride.
        compile_cache_uri_template: Optional GCS prefix template for the vLLM
            XLA compile cache. ``{region_bucket}`` and ``{model_hash}`` are
            substituted per region/model. None disables external compile-cache
            sharing.
    """

    regions: Sequence[str]
    results_region: str = "us-central1"
    tpu_shapes: Sequence[str] = DEFAULT_TPU_SHAPES
    max_workers_per_region: int = 16
    shard_size: int = 500
    sampling: SamplingParams = field(default_factory=SamplingParams)
    job_name: str = "marin-inference"
    worker_preemptible: bool = True
    heartbeat_timeout: float = DEFAULT_HEARTBEAT_TIMEOUT
    max_shard_failures: int = MAX_SHARD_FAILURES
    max_shard_infra_failures: int = MAX_SHARD_INFRA_FAILURES
    chunk_size: int = 2000
    compile_cache_uri_template: str | None = None
    # Extras installed on each TPU worker job (forwarded to Iris via Fray's
    # ``EnvironmentConfig.extras``). ``marin:vllm`` brings in the inference
    # engine; ``marin:tpu`` pins matching JAX / libtpu / torch versions.
    # Override when a worker needs a different stack (different vLLM version,
    # CPU-only mode for tests, etc.).
    worker_extras: tuple[str, ...] = ("marin:vllm", "marin:tpu")

    def __post_init__(self) -> None:
        if not self.regions:
            raise ValueError("InferenceConfig.regions must be non-empty.")
        for region in self.regions:
            if region.lower() not in REGION_TO_DATA_BUCKET:
                raise ValueError(
                    f"Unknown region {region!r} in InferenceConfig.regions; "
                    f"known regions: {sorted(REGION_TO_DATA_BUCKET)}."
                )
        if self.results_region.lower() not in REGION_TO_DATA_BUCKET:
            raise ValueError(
                f"Unknown results_region {self.results_region!r}; " f"known regions: {sorted(REGION_TO_DATA_BUCKET)}."
            )
        if self.shard_size <= 0:
            raise ValueError(f"shard_size must be positive (got {self.shard_size}).")
        if self.max_workers_per_region <= 0:
            raise ValueError(f"max_workers_per_region must be positive (got {self.max_workers_per_region}).")
