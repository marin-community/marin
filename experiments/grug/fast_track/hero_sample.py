# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Prepare a bounded, per-component sample of the current hero mixture."""

from __future__ import annotations

import asyncio
import hashlib
import itertools
import logging
import math
from collections import Counter
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, replace
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
from levanter.store.cache import write_levanter_cache
from levanter.tokenizers import MarinTokenizer, load_tokenizer, tokenizer_content_hash
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
    require_tokenizer_content,
)
from experiments.grug.fast_track.launch import FlatCacheTrainingSource, maximum_h100_ladder_tokens
from experiments.grug.moe.launch_datakit_moe_mix import _MIXTURE_BLOCK_SIZE
from experiments.grug.moe_hero_ep.harrier_mix_2026_08_18 import _SPEC

logger = logging.getLogger(__name__)

HERO_RECIPE = Path(__file__).parents[1] / "moe_hero_ep" / "harrier_mix_2026_08_18.json"
HERO_SOURCE_STORE = _SPEC.candidate_store_uri
HERO_TARGET_TOKENIZER = "hero-bpe-v16384"
HERO_TARGET_TOKENS = maximum_h100_ladder_tokens()
HERO_SEQUENCE_LENGTH = 4096
SAMPLE_TASK_RESOURCES = ResourceConfig(cpu=2, ram="16g", disk="16g")
SAMPLE_READ_EXAMPLES = 256
# Change the version when the retokenized rows change for the same source examples.
HERO_RESAMPLER = "resample-v2"
HERO_LOADER_POLICY = (
    f"block-shuffle-io{DEFAULT_LM_DATA_SHUFFLE.io_block_size}-window{DEFAULT_LM_DATA_SHUFFLE.window_blocks}"
    f"-{DEFAULT_LM_DATA_SHUFFLE.perm_type}-mixture-block{_MIXTURE_BLOCK_SIZE}-{HERO_RESAMPLER}"
)


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


@dataclass(frozen=True)
class _HeroRecipe:
    weights: dict[str, float]
    """Positive main-phase weights, in production component order."""
    sha256: str
    source_tokenizer: str
    shuffle_order: tuple[str, ...]
    """Components with a positive weight in any phase: the order of production shuffle keys."""


def _load_recipe() -> _HeroRecipe:
    active = {name for phase in _SPEC.phase_weights for name, weight in phase if weight > 0}
    return _HeroRecipe(
        weights={name: weight for name, weight in _SPEC.phase_weights[1] if weight > 0},
        sha256=hashlib.sha256(HERO_RECIPE.read_bytes()).hexdigest(),
        source_tokenizer=_SPEC.tokenizer,
        shuffle_order=tuple(name for name, _ in _SPEC.available_tokens if name in active),
    )


_HERO_RECIPE = _load_recipe()


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
        stage_runner_factory=SubprocessRunner,
    )


def _mixture_component_examples(weights: Mapping[str, float], total_tokens: int) -> Counter[str]:
    """Bound Levanter mixture exposure with full blocks, including the last partial block.

    ``weights`` must be in the loader's component order, because Levanter gives the block remainder to the
    first of the largest components.
    """
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


def _component_cache_path(root: str, name: str) -> str:
    return prefix_join(root, f"components/{name}/train")


def hero_sample_manifest(root: str) -> FrozenBaselineManifest:
    """Return the main-phase Hero manifest rooted at one sample artifact path."""
    return FrozenBaselineManifest(
        tokenizer=HERO_TARGET_TOKENIZER,
        components=tuple(
            FrozenBaselineComponent(name, _component_cache_path(root, name), weight)
            for name, weight in _HERO_RECIPE.weights.items()
        ),
    )


def _shuffled_component_dataset(*, cache_dir: str, tokenizer: str, name: str, key: jax.Array) -> AsyncDataset:
    """Load one flat-cache cell with the production text loader and its block shuffle."""
    component = DatasetComponent(
        source=None,
        cache_dir=cache_dir,
        format=TextLmDatasetFormat(),
        tags=[name],
        flat_cache=True,
    )
    config = LmDataConfig(
        tokenizer=tokenizer,
        cache_dir=None,
        components={name: component},
        train_weights={name: 1.0},
        auto_build_caches=False,
    )
    dataset = dataset_for_component(
        component,
        Axis("position", HERO_SEQUENCE_LENGTH),
        config.build_caches("train")[name],
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


def _retokenize(
    examples: Sequence[np.ndarray],
    *,
    source: MarinTokenizer,
    target: MarinTokenizer,
    target_bos: int,
    target_eos: int,
) -> list[np.ndarray]:
    """Return one target row for each source example.

    Each source segment (split on the source EOS) becomes the target BOS if the segment starts with the
    source BOS, then the segment text, then the target EOS. Thus a document that the source example cuts
    also ends with EOS, and the flat target cache never joins two unrelated examples.
    """
    owners: list[int] = []
    starts_document: list[bool] = []
    texts: list[str] = []
    for index, tokens in enumerate(examples):
        ends = np.flatnonzero(tokens == source.eos_token_id)
        for start, stop in zip([0, *(ends + 1)], [*ends, len(tokens)], strict=True):
            segment = tokens[start:stop]
            if len(segment) == 0:
                continue
            owners.append(index)
            starts_document.append(bool(segment[0] == source.bos_token_id))
            texts.append(source.decode(segment.tolist(), skip_special_tokens=True))

    rows: list[list[int]] = [[] for _ in examples]
    encoded = target.encode_batch(texts, add_special_tokens=False)
    for index, has_bos, ids in zip(owners, starts_document, encoded, strict=True):
        if has_bos:
            rows[index].append(target_bos)
        rows[index].extend(ids)
        rows[index].append(target_eos)
    return [np.asarray(row, dtype=np.int32) for row in rows]


def _sample_component(task: _SampleTask) -> _SampleResult:
    owner = f"hero component {task.name}"
    require_tokenizer_content(task.source_tokenizer, task.source_tokenizer_hash, owner=owner)
    require_tokenizer_content(task.target_tokenizer, task.target_tokenizer_hash, owner=owner)
    source_tokenizer = load_tokenizer(task.source_tokenizer)
    target_tokenizer = load_tokenizer(task.target_tokenizer)
    target_bos = target_tokenizer.bos_token_id
    target_eos = target_tokenizer.eos_token_id
    if target_bos is None or target_eos is None:
        raise ValueError(f"target tokenizer {task.target_tokenizer!r} must define BOS and EOS tokens")

    # LmDataConfig.train_set splits the data key into mixture and shuffle keys.
    # Its train_sets method then gives each component the next key in insertion order.
    _, shuffle_key = jax.random.split(jax.random.PRNGKey(task.data_seed))
    component_key = next(itertools.islice(key_iterator(shuffle_key), task.component_index, None))
    dataset = _shuffled_component_dataset(
        cache_dir=task.source_cache, tokenizer=task.source_tokenizer, name=task.name, key=component_key
    ).as_sync_dataset()
    available_examples = len(dataset)
    tokens_written = 0

    def token_rows() -> Iterator[dict[str, np.ndarray]]:
        nonlocal tokens_written
        for start in range(0, available_examples, SAMPLE_READ_EXAMPLES):
            examples = dataset.get_batch(range(start, min(start + SAMPLE_READ_EXAMPLES, available_examples)))
            rows = _retokenize(
                [np.asarray(example.tokens) for example in examples],
                source=source_tokenizer,
                target=target_tokenizer,
                target_bos=target_bos,
                target_eos=target_eos,
            )
            for ids in rows:
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
        # Rows are BOS + text + EOS documents, the format that the training loader expects.
        metadata=TextLmDatasetFormat().build_preprocessor(target_tokenizer).metadata,
    )
    return _SampleResult(name=task.name, tokens=tokens_written)


def prepare_hero_sample(config: HeroSampleConfig) -> PreparedHeroSample:
    """Sample each hero cell in its stable production loader order.

    The sampler decodes only the selected source examples and encodes them with the fast-track tokenizer.
    Thus, it preserves component order and exposure, but it does not preserve packed token sequences.
    """
    required_examples = _mixture_component_examples(_HERO_RECIPE.weights, config.requested_tokens)
    tasks = [
        _SampleTask(
            name=name,
            source_cache=prefix_join(HERO_SOURCE_STORE, f"cluster={int(name[1:3])}/quality={name[-1]}"),
            output_cache=_component_cache_path(config.output_path, name),
            source_tokenizer=_HERO_RECIPE.source_tokenizer,
            target_tokenizer=HERO_TARGET_TOKENIZER,
            target_tokens=max(1, required_examples[name]) * HERO_SEQUENCE_LENGTH,
            data_seed=config.data_seed,
            component_index=_HERO_RECIPE.shuffle_order.index(name),
            source_tokenizer_hash=config.source_tokenizer_hash,
            target_tokenizer_hash=config.target_tokenizer_hash,
        )
        for name in _HERO_RECIPE.weights
    ]

    with _sample_pool("fast-track-hero-sample") as pool:
        outcomes = pool.execute(
            Dataset.from_list(tasks).map(_sample_component),
            verbose=True,
            map_task_resources=SAMPLE_TASK_RESOURCES,
        )
    measured = {result.name: result.tokens for result in outcomes.results}
    actual_tokens = sum(measured.values())
    logger.info("Prepared hero sample: %d components, %d target tokens", len(measured), actual_tokens)
    return PreparedHeroSample(
        baseline=hero_sample_manifest(config.output_path),
        source_store=HERO_SOURCE_STORE,
        source_tokenizer=_HERO_RECIPE.source_tokenizer,
        source_tokenizer_hash=config.source_tokenizer_hash,
        target_tokenizer_hash=config.target_tokenizer_hash,
        recipe_sha256=_HERO_RECIPE.sha256,
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
    requested_tokens: int = HERO_TARGET_TOKENS,
    data_seed: int = 0,
    version: str | None = None,
) -> ArtifactStep[PreparedHeroSample]:
    """Bind the pinned hero sample to one immutable artifact identity."""
    source_tokenizer_hash = tokenizer_content_hash(_HERO_RECIPE.source_tokenizer)
    target_tokenizer_hash = tokenizer_content_hash(HERO_TARGET_TOKENIZER)
    identity = {
        "source_store": HERO_SOURCE_STORE,
        "recipe_sha256": _HERO_RECIPE.sha256,
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
        data = self.mixture_config(ctx=ctx, validation=validation, tokenizer=tokenizer, budget=budget)
        self.require_capacity(data, ctx=ctx, budget=budget)
        return data

    def mixture_config(
        self,
        *,
        ctx: StepContext,
        validation: Sequence[ArtifactStep[TokenizedCache]],
        tokenizer: str,
        budget: ResolvedTrainingBudget,
    ) -> LmDataConfig:
        """Return the Hero mixture without the capacity check, for a caller that changes the weights."""
        if budget.sequence_length != HERO_SEQUENCE_LENGTH:
            raise ValueError(f"hero sample uses sequence length {HERO_SEQUENCE_LENGTH}")
        if ctx.is_fingerprint:
            manifest = hero_sample_manifest(ctx.artifact_path(self.sample))
        else:
            prepared = ctx.resolved(self.sample)
            require_tokenizer_content(tokenizer, prepared.target_tokenizer_hash, owner="hero sample training")
            manifest = prepared.baseline
        training_data = FlatCacheTrainingSource(manifest=manifest).data_config(
            ctx=ctx,
            validation=validation,
            tokenizer=tokenizer,
            budget=budget,
        )
        return replace(training_data, shuffle=False, mixture_block_size=_MIXTURE_BLOCK_SIZE)

    def require_capacity(self, data: LmDataConfig, *, ctx: StepContext, budget: ResolvedTrainingBudget) -> None:
        """Fail when ``data`` reads more sequences of a Hero cell than the sample holds.

        A component with a ``max_train_sequences`` entry repeats by design, so it is not checked.
        """
        if ctx.is_fingerprint:
            return
        prepared = ctx.resolved(self.sample)
        weights = data.train_weights
        assert isinstance(weights, dict), "the Hero mixture has one stage"
        required_examples = _mixture_component_examples(
            {name: weights.get(name, 0.0) for name in data.components}, budget.token_count
        )
        repeated = data.max_train_sequences or {}
        for name, tokens in prepared.component_tokens.items():
            available_examples = tokens // budget.sequence_length
            if name not in repeated and available_examples < required_examples[name]:
                raise ValueError(
                    f"hero component {name} has {available_examples:,} usable sequences; "
                    f"the run requires {required_examples[name]:,}"
                )
