# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import dataclasses
import shutil
from collections import Counter
from collections.abc import Callable
from pathlib import Path

import haliax as hax
import jax
import numpy as np
import pytest
from fray.local_backend import LocalClient
from fray.types import ResourceConfig
from levanter.data.text.datasets import LengthSkewDatasetComponent, LmDataConfig
from levanter.data.text.formats import TextLmDatasetFormat, preprocessor_for_format
from levanter.schedule import BatchSchedule
from levanter.store.cache import write_levanter_cache
from levanter.tokenizers import load_tokenizer
from marin.execution.lazy import StepContext
from zephyr.context import ZephyrContext

from experiments.grug.moe_hero_ep import harrier_long_context_mix, launch_scaling_ladder
from experiments.grug.moe_hero_ep.harrier_length_stats import write_length_stats
from experiments.grug.moe_hero_ep.harrier_long_context_mix import (
    harrier_long_context_data_config,
    read_length_tokens,
)
from experiments.grug.moe_hero_ep.harrier_mix_2026_08_18 import HarrierContextPhase
from experiments.grug.moe_hero_ep.launch_diagnostics import build_diagnostic_run
from experiments.grug.moe_hero_ep.launch_scaling_ladder import build_ladder_run
from experiments.grug.moe_hero_ep.train import build_train_dataset

SEQ_LEN = 8
START_STEP = 390_251
# Bucket: short tokens, long tokens. Token ids encode the bucket and partition.
TOKENS = {"c01q0": (832, 160), "c02q3": (240, 0)}
WEIGHTS = {"c01q0": 0.75, "c02q3": 0.25}
FIRST_ID = {("c01q0", "lte_64k"): 1000, ("c01q0", "gt_64k"): 3000, ("c02q3", "lte_64k"): 5000}
TOKENIZER_FILES = ("gpt2_tokenizer.json", "tokenizer.json"), ("gpt2_tokenizer_config.json", "tokenizer_config.json")
LEVANTER_TESTS = Path(__file__).parents[3] / "lib/levanter/tests"


@dataclasses.dataclass(frozen=True)
class HeroStore:
    store: Path
    stats_path: Path
    data_config: Callable[[], LmDataConfig]


def _fingerprint_config(step):
    return step.build_config(StepContext.for_fingerprint(step.runtime_args, step.deps))


def _training_components(config):
    weights = config.data.train_weights[0][1]
    return [config.data.components[name] for name, weight in weights.items() if weight > 0]


@pytest.fixture
def zephyr_context(tmp_path):
    client = LocalClient()
    context = ZephyrContext(
        client=client,
        max_workers=2,
        resources=ResourceConfig(cpu=1, ram="512m"),
        chunk_storage_prefix=str(tmp_path / "chunks"),
        name="length-stats-test",
    )
    yield context
    context.shutdown()
    client.shutdown(wait=True)


@pytest.fixture
def fake_store(tmp_path, monkeypatch, zephyr_context):
    """A two-bucket store on disk, with the mix restricted to those buckets."""
    tokenizer_dir = tmp_path / "tokenizer"
    tokenizer_dir.mkdir()
    for source, target in TOKENIZER_FILES:
        shutil.copy(LEVANTER_TESTS / source, tokenizer_dir / target)
    metadata = preprocessor_for_format(
        TextLmDatasetFormat(), load_tokenizer(str(tokenizer_dir)), enforce_bos=True, enforce_eos=True
    ).metadata
    store = tmp_path / "store"
    for cell in TOKENS:
        docs = []
        for partition in ("lte_64k", "gt_64k"):
            first_id = FIRST_ID.get((cell, partition))
            if first_id is None:
                continue
            tokens = TOKENS[cell][partition == "gt_64k"]
            doc_len = 32 if partition == "gt_64k" else 8
            ids = np.arange(first_id, first_id + tokens, dtype=np.int32)
            docs.extend({"input_ids": ids[start : start + doc_len]} for start in range(0, tokens, doc_len))
        path = store / f"cluster={int(cell[1:3])}/quality={cell[4]}"
        write_levanter_cache(docs, str(path), metadata=metadata)
    stats_path = tmp_path / "stats"
    write_length_stats(
        str(stats_path), str(store), tuple(TOKENS), thresholds=(8, 16, 32), chunk_rows=8, zephyr_context=zephyr_context
    )
    monkeypatch.setattr(harrier_long_context_mix, "_COOLDOWN_WEIGHTS", WEIGHTS)
    monkeypatch.setattr(harrier_long_context_mix, "LONG_CONTEXT_THRESHOLD", 16)

    def config():
        built = harrier_long_context_data_config(
            ctx=StepContext.for_fingerprint(),
            start_step=START_STEP,
            max_seq_len=SEQ_LEN,
            validation=(),
            store=str(store),
            stats_path=str(stats_path),
        )
        return dataclasses.replace(built, tokenizer=str(tokenizer_dir), mixture_block_size=16)

    return HeroStore(store, stats_path, config)


def _partition(token: int) -> tuple[str, str]:
    first_id = max(first for first in FIRST_ID.values() if first <= token)
    return next(key for key, value in FIRST_ID.items() if value == first_id)


@pytest.mark.parametrize(
    ("skew", "block_size", "expected_sequences"),
    [
        (1, 124, {("c01q0", "lte_64k"): 78, ("c01q0", "gt_64k"): 15, ("c02q3", "lte_64k"): 31}),
        (4, 92, {("c01q0", "lte_64k"): 39, ("c01q0", "gt_64k"): 30, ("c02q3", "lte_64k"): 23}),
    ],
)
def test_statistics_weight_bucket_sampling_and_preserve_cell_share(fake_store, skew, block_size, expected_sequences):
    data = fake_store.data_config()
    data = dataclasses.replace(
        data,
        components={
            name: dataclasses.replace(component, long_weight=skew) for name, component in data.components.items()
        },
        mixture_block_size=block_size,
    )

    dataset = build_train_dataset(
        data, max_seq_len=SEQ_LEN, batch_schedule=BatchSchedule(1), key=jax.random.PRNGKey(0)
    ).as_sync_dataset()
    examples = dataset.get_batch(list(range(START_STEP, START_STEP + block_size)))
    # EOS and padding sit outside the store's token ids.
    tokens = Counter(
        _partition(int(token)) for example in examples for token in np.asarray(example.tokens) if 1000 <= token < 50_000
    )

    assert tokens == {partition: sequences * SEQ_LEN for partition, sequences in expected_sequences.items()}


def test_virtual_buckets_contain_each_source_token_once(fake_store):
    data = fake_store.data_config()
    sources = data.train_sets(hax.Axis("position", SEQ_LEN), key=jax.random.PRNGKey(0))
    assert set(sources) == {"c01q0/short", "c01q0/long", "c02q3/short"}
    for (cell, partition), first_id in FIRST_ID.items():
        bucket = sources[f"{cell}/{'long' if partition == 'gt_64k' else 'short'}"].as_sync_dataset()
        examples = bucket.get_batch(list(range(len(bucket))))
        actual = np.concatenate([np.asarray(example.tokens) for example in examples])
        expected = np.arange(first_id, first_id + TOKENS[cell][partition == "gt_64k"])
        np.testing.assert_array_equal(np.sort(actual), expected)
    assert data.validation_sets(hax.Axis("position", SEQ_LEN)) == {}


def test_read_length_tokens(fake_store):
    assert read_length_tokens(str(fake_store.stats_path)) == TOKENS


def _set_hero_seq_len(monkeypatch, seq_len):
    monkeypatch.setattr(
        launch_scaling_ladder,
        "HERO_MODEL_CONFIG",
        dataclasses.replace(launch_scaling_ladder.HERO_MODEL_CONFIG, max_seq_len=seq_len),
    )


def test_hero_past_the_threshold_starts_the_store_where_its_context_history_ends(monkeypatch):
    _set_hero_seq_len(monkeypatch, 262_144)
    monkeypatch.setattr(launch_scaling_ladder, "HERO_CONTEXT_AXIS_SIZE", 4)

    step = build_ladder_run(run_id="test-long-context-on", size="d6144", version="dev")
    config = _fingerprint_config(step)

    phases = launch_scaling_ladder.HERO_CONTEXT_PHASES
    assert config.data.mixture_start_step == phases[-1].end_step
    assert all(isinstance(component, LengthSkewDatasetComponent) for component in _training_components(config))


@pytest.mark.parametrize(
    ("seq_len", "history"),
    [
        (4096, ()),
        (
            65536,
            (HarrierContextPhase(end_step=257_553, seq_len=4096), HarrierContextPhase(end_step=372_000, seq_len=16384)),
        ),
    ],
)
def test_hero_contexts_up_to_the_threshold_read_harrier(monkeypatch, seq_len, history):
    _set_hero_seq_len(monkeypatch, seq_len)

    config = _fingerprint_config(
        build_ladder_run(
            run_id="test-harrier",
            size="d6144",
            context_phases=history,
            flops_baseline_total=2e24 if history else None,
            version="dev",
        )
    )

    assert config.data.mixture_start_step is None
    assert not any(isinstance(component, LengthSkewDatasetComponent) for component in config.data.components.values())


def _training_data(data: LmDataConfig) -> LmDataConfig:
    """Drop the held-out components, which train at zero weight and leave the mixture's sources unchanged."""
    ((stage, weights),) = data.train_weights
    train_weights = {name: weight for name, weight in weights.items() if weight > 0}
    return dataclasses.replace(
        data,
        components={name: data.components[name] for name in train_weights},
        train_weights=[(stage, train_weights)],
    )


def test_hero_and_diagnostics_select_the_same_data():
    hero = _fingerprint_config(build_ladder_run(run_id="test-long-context-hero", size="d6144", version="dev"))
    start_step = launch_scaling_ladder.HERO_CONTEXT_PHASES[-1].end_step
    diagnostic = _fingerprint_config(
        build_diagnostic_run(
            run_id="test-long-context-diagnostic",
            dp_racks=1,
            num_steps=start_step + 1,
            schedule_steps=start_step,
            long_context_start_step=start_step,
            restore_from=f"s3://bucket/checkpoints/step-{start_step}",
            eval_every=100,
            version="dev",
        )
    )

    assert _training_data(hero.data) == _training_data(diagnostic.data)
    assert hero.trainer.trainer.seed == diagnostic.trainer.trainer.seed
    assert hero.trainer.data_seed == diagnostic.trainer.data_seed


def test_diagnostic_arms_select_the_same_data_on_any_rack_count():
    def arm(racks: int):
        return _fingerprint_config(
            build_diagnostic_run(
                run_id=f"test-long-context-{racks}",
                dp_racks=racks,
                num_steps=390_251,
                schedule_steps=390_251,
                max_seq_len=262_144,
                batch_size=16 * racks,
                context_axis_size=4,
                expert_axis_size=16,
                long_context_start_step=390_000,
                restore_from="s3://bucket/checkpoints/step-390000",
                version="dev",
            )
        )

    one_rack, four_racks = arm(1), arm(4)

    assert one_rack.data == four_racks.data
