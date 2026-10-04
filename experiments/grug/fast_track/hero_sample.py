# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Prepare a bounded, per-component sample of the current hero mixture."""

from __future__ import annotations

import asyncio
import hashlib
import itertools
import json
import logging
import math
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import jax
import numpy as np
from fray.types import ResourceConfig
from haliax import Axis
from levanter.data.dataset import AsyncDataset
from levanter.data.mixture import MixtureDataset, StopStrategy
from levanter.data.text.datasets import (
    DEFAULT_LM_DATA_SHUFFLE,
    DatasetComponent,
    LmDataConfig,
    dataset_for_component,
)
from levanter.data.text.formats import TextLmDatasetFormat
from levanter.store.cache import CacheLedger, write_levanter_cache
from levanter.tokenizers import load_tokenizer, tokenizer_content_hash
from levanter.utils.jax_utils import key_iterator
from marin.execution.artifact import Artifact
from marin.execution.build_context import resolve_version
from marin.execution.fingerprint import canonical_json
from marin.execution.lazy import ArtifactStep, StepContext
from marin.experiment.namespacing import user_namespaced_name
from marin.processing.tokenize.tokenize import TokenizedCache
from pydantic import Field
from rigging.filesystem.storage_path import prefix_join
from zephyr.context import ZephyrContext
from zephyr.dataset import Dataset
from zephyr.runners import SubprocessRunner

from experiments.grug.fast_track.contracts import (
    FrozenBaselineComponent,
    FrozenBaselineManifest,
    ResolvedTrainingBudget,
)
from experiments.grug.fast_track.launch import FlatCacheTrainingSource, maximum_h100_ladder_tokens
from experiments.grug.moe.launch_datakit_moe_mix import _MIXTURE_BLOCK_SIZE
from experiments.marin_tokenizer import marin_tokenizer

logger = logging.getLogger(__name__)

HERO_RECIPE = Path(__file__).parents[1] / "moe_hero_ep" / "harrier_mix_2026_08_18.json"
HERO_SOURCE_STORE = "s3://marin-us-east-02a/marin/datakit/store_4d2e363d"
HERO_TARGET_TOKENIZER = "hero-bpe-v16384"
HERO_TARGET_TOKENS = maximum_h100_ladder_tokens()
HERO_SEQUENCE_LENGTH = 4096
SAMPLE_TASK_RESOURCES = ResourceConfig(cpu=2, ram="16g", disk="16g")
HERO_LOADER_POLICY = "levanter-block-shuffle-io256-window512-mixture-block49152-v1"


class PreparedHeroSample(Artifact):
    """Measured per-component caches and the fixed hero recipe that made them.

    The cache preserves each component's seeded loader order. Retokenization can change token IDs and
    sequence boundaries, so the artifact guarantees component exposure and stable document order.
    """

    baseline: FrozenBaselineManifest
    source_store: str
    source_tokenizer: str
    source_tokenizer_hash: str
    target_tokenizer_hash: str
    recipe_sha256: str = Field(pattern="^[0-9a-f]{64}$")
    phase: str
    loader_policy: str
    sequence_length: int
    data_seed: int
    requested_tokens: int
    actual_tokens: int
    component_tokens: dict[str, int]


@dataclass(frozen=True)
class HeroSampleConfig:
    """Inputs and output path for one fixed hero sample."""

    source_store: str
    output_path: str
    requested_tokens: int
    data_seed: int
    source_tokenizer_hash: str
    target_tokenizer_hash: str


@dataclass(frozen=True)
class _SampleTask:
    name: str
    source_cache: str
    output_cache: str
    source_tokenizer: str
    target_tokenizer: str
    target_tokens: int
    data_seed: int
    component_index: int
    source_tokenizer_hash: str
    target_tokenizer_hash: str


@dataclass(frozen=True)
class _SampleResult:
    name: str
    tokens: int
    rows: int


@dataclass(frozen=True)
class _HeroRecipe:
    weights: dict[str, float]
    sha256: str


class _ComponentMarker(AsyncDataset[str]):
    def __init__(self, name: str, length: int):
        self.name = name
        self.length = length

    async def async_len(self) -> int:
        return self.length

    def is_finite(self) -> bool:
        return True

    async def get_batch(self, indices: Sequence[int]) -> Sequence[str]:
        return [self.name] * len(indices)


def _sample_pool(name: str) -> ZephyrContext:
    """Create a small CPU pool for the independent component cache writes."""
    return ZephyrContext(
        name=name,
        resources=SAMPLE_TASK_RESOURCES,
        max_workers=8,
        max_concurrent_pipelines=8,
        stage_runner_factory=SubprocessRunner,
    )


def _mixture_component_examples(weights: Mapping[str, float], total_tokens: int) -> Counter[str]:
    """Bound Levanter mixture exposure with full blocks, including the last partial block."""
    total_examples = math.ceil(total_tokens / HERO_SEQUENCE_LENGTH)
    block_count = math.ceil(total_examples / _MIXTURE_BLOCK_SIZE)
    mixture = MixtureDataset(
        datasets={name: _ComponentMarker(name, _MIXTURE_BLOCK_SIZE) for name in weights},
        weights=dict(weights),
        block_size=_MIXTURE_BLOCK_SIZE,
        key=0,
        stop_strategy=StopStrategy.RESTART_STRATEGY,
    )
    counts: Counter[str] = Counter(asyncio.run(mixture.get_batch(range(mixture.block_size))))
    return Counter({name: count * block_count for name, count in counts.items()})


def _recipe() -> _HeroRecipe:
    raw_bytes = HERO_RECIPE.read_bytes()
    raw = json.loads(raw_bytes)
    if raw["candidate_store_uri"] != HERO_SOURCE_STORE:
        raise ValueError("hero recipe store does not match the pinned source store")
    if raw["tokenizer"] != marin_tokenizer:
        raise ValueError("hero recipe tokenizer does not match the pinned source tokenizer")
    available_tokens = {name: int(count) for name, count in raw["available_tokens"].items()}
    main_weights = {name: float(weight) for name, weight in raw["phases"][1]["weights"].items()}
    if set(available_tokens) != set(main_weights) or not math.isclose(sum(main_weights.values()), 1.0, abs_tol=1e-9):
        raise ValueError("hero main-phase weights do not match its available component counts")
    if any(weight < 0 or not math.isfinite(weight) for weight in main_weights.values()):
        raise ValueError("hero main-phase weights must be finite and non-negative")
    if any(count < 1 for count in available_tokens.values()):
        raise ValueError("hero recipe source token counts must be positive")
    if not any(weight > 0 for weight in main_weights.values()):
        raise ValueError("hero main-phase weights must contain a positive component")
    return _HeroRecipe(main_weights, hashlib.sha256(raw_bytes).hexdigest())


def _component_shuffle_index(recipe: dict, name: str) -> int:
    """Return the cell's key position in production's all-phase component order."""
    active_names = {
        component for phase in recipe["phases"] for component, weight in phase["weights"].items() if weight > 0
    }
    ordered_names = [component for component in recipe["available_tokens"] if component in active_names]
    return ordered_names.index(name)


def _component_cache_path(root: str, name: str) -> str:
    return prefix_join(root, f"components/{name}/train")


def hero_sample_manifest(root: str) -> FrozenBaselineManifest:
    """Return the fixed phase-one component weights at one sample artifact path."""
    weights = _recipe().weights
    return FrozenBaselineManifest(
        tokenizer=HERO_TARGET_TOKENIZER,
        components=tuple(
            FrozenBaselineComponent(name, _component_cache_path(root, name), weight)
            for name, weight in weights.items()
            if weight > 0
        ),
    )


def _source_component_config(
    *,
    source_cache: str,
    tokenizer: str,
    name: str,
) -> LmDataConfig:
    component = DatasetComponent(
        source=None,
        cache_dir=source_cache,
        format=TextLmDatasetFormat(),
        tags=[name],
        flat_cache=True,
    )
    return LmDataConfig(
        tokenizer=tokenizer,
        cache_dir=None,
        components={name: component},
        train_weights={name: 1.0},
        auto_build_caches=False,
    )


def _production_component_dataset(
    config: LmDataConfig,
    *,
    name: str,
    position: Axis,
    key: jax.Array,
) -> AsyncDataset:
    """Load and shuffle one cell with the production text component loader."""
    component = config.components[name]
    if not isinstance(component, DatasetComponent):
        raise TypeError(f"hero component {name} is not a cached text component")
    cache = config.build_caches("train")[name]
    dataset = dataset_for_component(
        component,
        position,
        cache,
        eos_id=config.the_tokenizer.eos_token_id,
        block_cross_document_attention=config.block_cross_document_attention,
    )
    shuffle = DEFAULT_LM_DATA_SHUFFLE
    return dataset.block_shuffle(
        io_block_size=shuffle.io_block_size,
        window_blocks=shuffle.window_blocks,
        key=key,
        perm_type=shuffle.perm_type,
    )


def _sample_component(task_data: dict) -> dict:
    task = _SampleTask(**task_data)
    if tokenizer_content_hash(task.source_tokenizer) != task.source_tokenizer_hash:
        raise ValueError(f"source tokenizer content changed for hero component {task.name}")
    if tokenizer_content_hash(task.target_tokenizer) != task.target_tokenizer_hash:
        raise ValueError(f"target tokenizer content changed for hero component {task.name}")
    source_tokenizer = load_tokenizer(task.source_tokenizer)
    target_tokenizer = load_tokenizer(task.target_tokenizer)
    config = _source_component_config(source_cache=task.source_cache, tokenizer=task.source_tokenizer, name=task.name)
    position = Axis("position", HERO_SEQUENCE_LENGTH)

    # LmDataConfig.train_set splits the data key into mixture and shuffle keys.
    # Its train_sets method then gives each component the next key in insertion order.
    _, shuffle_key = jax.random.split(jax.random.PRNGKey(task.data_seed))
    component_keys = key_iterator(shuffle_key)
    component_key = next(itertools.islice(component_keys, task.component_index, None))
    dataset = _production_component_dataset(config, name=task.name, position=position, key=component_key)
    sync_dataset = dataset.as_sync_dataset()
    available_examples = len(sync_dataset)
    if available_examples == 0:
        raise ValueError(f"hero component {task.name} has no source examples")

    tokens_written = 0
    rows_written = 0

    def token_rows():
        nonlocal tokens_written, rows_written
        read_size = 256
        for start in range(0, available_examples, read_size):
            examples = sync_dataset.get_batch(range(start, min(start + read_size, available_examples)))
            for example in examples:
                source_ids = np.asarray(example.tokens).tolist()
                if source_tokenizer.eos_token_id is None:
                    text = source_tokenizer.decode(source_ids, skip_special_tokens=False)
                    target_ids = target_tokenizer.encode(text, add_special_tokens=False)
                else:
                    if target_tokenizer.eos_token_id is None:
                        raise ValueError(f"target tokenizer for hero component {task.name} has no EOS token")
                    target_ids = []
                    segment_start = 0
                    for index, token_id in enumerate(source_ids):
                        if token_id == source_tokenizer.eos_token_id:
                            text = source_tokenizer.decode(source_ids[segment_start:index], skip_special_tokens=False)
                            target_ids.extend(target_tokenizer.encode(text, add_special_tokens=False))
                            target_ids.append(target_tokenizer.eos_token_id)
                            segment_start = index + 1
                    if segment_start < len(source_ids):
                        text = source_tokenizer.decode(source_ids[segment_start:], skip_special_tokens=False)
                        target_ids.extend(target_tokenizer.encode(text, add_special_tokens=False))
                ids = np.asarray(target_ids, dtype=np.int32)
                if len(ids) == 0:
                    continue
                rows_written += 1
                tokens_written += len(ids)
                yield {"input_ids": ids}
                if tokens_written >= task.target_tokens:
                    return
        raise ValueError(
            f"hero component {task.name} has {tokens_written:,} target tokens; "
            f"the sample requires {task.target_tokens:,}"
        )

    write_levanter_cache(
        token_rows(),
        task.output_cache,
        metadata=TextLmDatasetFormat().build_preprocessor(target_tokenizer).metadata,
    )
    ledger = CacheLedger.load(task.output_cache)
    measured_tokens = ledger.field_counts.get("input_ids", 0)
    if ledger.total_num_rows != rows_written or measured_tokens != tokens_written:
        raise ValueError(f"hero component {task.name} cache counts differ from the sampled token counts")
    return asdict(_SampleResult(name=task.name, tokens=measured_tokens, rows=ledger.total_num_rows))


def prepare_hero_sample(
    config: HeroSampleConfig,
    *,
    pool_factory: Callable[[str], ZephyrContext] = _sample_pool,
) -> PreparedHeroSample:
    """Sample each hero cell in its stable production loader order.

    The sampler decodes only the selected source examples and encodes them with the fast-track tokenizer.
    Thus, it preserves component order and exposure, but it does not preserve packed token sequences.
    """
    if config.source_store != HERO_SOURCE_STORE:
        raise ValueError("hero sample source must match the pinned production store")
    if not 0 < config.requested_tokens <= HERO_TARGET_TOKENS:
        raise ValueError(f"requested_tokens must be between 1 and {HERO_TARGET_TOKENS:,}")
    if config.data_seed < 0:
        raise ValueError("data_seed must be non-negative")
    parsed_recipe = _recipe()
    weights = parsed_recipe.weights
    recipe = json.loads(HERO_RECIPE.read_text())
    source_tokenizer = recipe["tokenizer"]
    source_tokenizer_hash = config.source_tokenizer_hash
    target_tokenizer_hash = config.target_tokenizer_hash
    component_names = [name for name, weight in weights.items() if weight > 0]
    required_examples = _mixture_component_examples(weights, config.requested_tokens)
    tasks: list[_SampleTask] = []
    for name in component_names:
        target_tokens = max(1, required_examples[name]) * HERO_SEQUENCE_LENGTH
        tasks.append(
            _SampleTask(
                name=name,
                source_cache=prefix_join(config.source_store, f"cluster={int(name[1:3])}/quality={name[-1]}"),
                output_cache=_component_cache_path(config.output_path, name),
                source_tokenizer=source_tokenizer,
                target_tokenizer=HERO_TARGET_TOKENIZER,
                target_tokens=target_tokens,
                data_seed=config.data_seed,
                component_index=_component_shuffle_index(recipe, name),
                source_tokenizer_hash=source_tokenizer_hash,
                target_tokenizer_hash=target_tokenizer_hash,
            )
        )

    with pool_factory("fast-track-hero-sample") as pool:
        outcomes = pool.execute(
            Dataset.from_list([asdict(task) for task in tasks]).map(_sample_component),
            verbose=True,
            map_task_resources=SAMPLE_TASK_RESOURCES,
        )
    measured = {result["name"]: result["tokens"] for result in outcomes.results}
    manifest = FrozenBaselineManifest(
        tokenizer=HERO_TARGET_TOKENIZER,
        components=tuple(
            FrozenBaselineComponent(name, _component_cache_path(config.output_path, name), weights[name])
            for name in component_names
        ),
    )
    actual_tokens = sum(measured.values())
    logger.info("Prepared hero sample: %d components, %d target tokens", len(measured), actual_tokens)
    return PreparedHeroSample(
        baseline=manifest,
        source_store=config.source_store,
        source_tokenizer=source_tokenizer,
        source_tokenizer_hash=source_tokenizer_hash,
        target_tokenizer_hash=target_tokenizer_hash,
        recipe_sha256=parsed_recipe.sha256,
        phase="main",
        loader_policy=HERO_LOADER_POLICY,
        sequence_length=HERO_SEQUENCE_LENGTH,
        data_seed=config.data_seed,
        requested_tokens=config.requested_tokens,
        actual_tokens=actual_tokens,
        component_tokens=measured,
    )


def hero_sample_step(
    *,
    source_store: str = HERO_SOURCE_STORE,
    requested_tokens: int = HERO_TARGET_TOKENS,
    data_seed: int = 0,
    version: str | None = None,
) -> ArtifactStep[PreparedHeroSample]:
    """Bind the pinned hero sample to one immutable artifact identity."""
    if not 0 < requested_tokens <= HERO_TARGET_TOKENS:
        raise ValueError(f"requested_tokens must be between 1 and {HERO_TARGET_TOKENS:,}")
    if source_store != HERO_SOURCE_STORE:
        raise ValueError("hero sample source must match the pinned production store")
    if data_seed < 0:
        raise ValueError("data_seed must be non-negative")
    recipe_sha256 = _recipe().sha256
    source_tokenizer_hash = tokenizer_content_hash(marin_tokenizer)
    target_tokenizer_hash = tokenizer_content_hash(HERO_TARGET_TOKENIZER)
    identity = {
        "source_store": source_store,
        "recipe_sha256": recipe_sha256,
        "phase": "main",
        "loader_policy": HERO_LOADER_POLICY,
        "sequence_length": HERO_SEQUENCE_LENGTH,
        "tokenizer": HERO_TARGET_TOKENIZER,
        "source_tokenizer_hash": source_tokenizer_hash,
        "target_tokenizer_hash": target_tokenizer_hash,
        "requested_tokens": requested_tokens,
        "data_seed": data_seed,
    }
    digest = hashlib.sha256(canonical_json(identity).encode()).hexdigest()[:20]
    name = f"fast-track/hero-sample/{digest}"
    resolved_version = resolve_version(name, version)
    return ArtifactStep(
        name=user_namespaced_name(name, resolved_version),
        version=resolved_version,
        artifact_type=PreparedHeroSample,
        run=prepare_hero_sample,
        build_config=lambda ctx: HeroSampleConfig(
            source_store=source_store,
            output_path=ctx.output_path,
            requested_tokens=requested_tokens,
            data_seed=data_seed,
            source_tokenizer_hash=source_tokenizer_hash,
            target_tokenizer_hash=target_tokenizer_hash,
        ),
    )


@dataclass(frozen=True)
class HeroTrainingSource:
    """Use a prepared hero sample and retain its per-component order.

    The sample stores the production per-component block shuffle. This source disables a second component
    shuffle, then lets the training loader apply the production mixture schedule once.
    """

    sample: ArtifactStep[PreparedHeroSample]

    def dependencies(self) -> tuple[ArtifactStep, ...]:
        return (self.sample,)

    def data_config(
        self,
        *,
        ctx: StepContext,
        validation: Sequence[ArtifactStep[TokenizedCache]],
        tokenizer: str,
        budget: ResolvedTrainingBudget,
    ) -> LmDataConfig:
        if ctx.is_fingerprint:
            manifest = hero_sample_manifest(ctx.artifact_path(self.sample))
        else:
            prepared = ctx.resolved(self.sample)
            if tokenizer_content_hash(tokenizer) != prepared.target_tokenizer_hash:
                raise ValueError("training tokenizer content differs from the prepared hero sample")
            if prepared.requested_tokens < budget.token_count:
                raise ValueError(
                    f"hero sample covers {prepared.requested_tokens:,} tokens; "
                    f"the training run requires {budget.token_count:,}"
                )
            required_examples = _mixture_component_examples(
                {component.name: component.weight for component in prepared.baseline.components}, budget.token_count
            )
            for component in prepared.baseline.components:
                available_examples = prepared.component_tokens[component.name] // budget.sequence_length
                if available_examples < required_examples[component.name]:
                    raise ValueError(
                        f"hero component {component.name} has {available_examples:,} usable sequences; "
                        f"the run requires {required_examples[component.name]:,}"
                    )
            manifest = prepared.baseline
        if budget.sequence_length != HERO_SEQUENCE_LENGTH:
            raise ValueError(f"hero sample uses sequence length {HERO_SEQUENCE_LENGTH}")
        training_data = FlatCacheTrainingSource(manifest=manifest).data_config(
            ctx=ctx,
            validation=validation,
            tokenizer=tokenizer,
            budget=budget,
        )
        return replace(training_data, shuffle=False, mixture_block_size=_MIXTURE_BLOCK_SIZE)
