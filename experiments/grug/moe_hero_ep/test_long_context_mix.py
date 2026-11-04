# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import dataclasses
import shutil
from collections import Counter
from pathlib import Path

import haliax as hax
import jax
import numpy as np
import pytest
from levanter.data.text.formats import TextLmDatasetFormat, preprocessor_for_format
from levanter.schedule import BatchSchedule
from levanter.store.cache import write_levanter_cache
from levanter.tokenizers import load_tokenizer
from marin.execution.lazy import StepContext

from experiments.grug.moe_hero_ep import harrier_long_context_mix, launch_scaling_ladder
from experiments.grug.moe_hero_ep.harrier_long_context_mix import (
    LONG_CONTEXT_SKEW,
    LONG_CONTEXT_TAG,
    harrier_long_context_data_config,
    long_document_share,
    long_partition_passes,
    overall_long_partition_passes,
    read_length_tokens,
)
from experiments.grug.moe_hero_ep.harrier_mix_2026_08_18 import (
    _SPEC,
    HARRIER_MIX_2026_08_18_STORE,
    harrier_mix_2026_08_18_data_config,
)
from experiments.grug.moe_hero_ep.launch_diagnostics import build_diagnostic_run
from experiments.grug.moe_hero_ep.launch_scaling_ladder import build_ladder_run

SEQ_LEN = 8
START_STEP = 390_251
# Bucket: short tokens, long tokens. Token ids encode the bucket and partition.
TOKENS = {"c01q0": (800, 160), "c02q3": (240, 0)}
WEIGHTS = {"c01q0": 0.75, "c02q3": 0.25}
FIRST_ID = {("c01q0", "lte_64k"): 1000, ("c01q0", "gt_64k"): 3000, ("c02q3", "lte_64k"): 5000}
HERO_LENGTH_TOKENS = {cell: (10**12, 10**11) for cell, _ in _SPEC.available_tokens}
TOKENIZER_FILES = ("gpt2_tokenizer.json", "tokenizer.json"), ("gpt2_tokenizer_config.json", "tokenizer_config.json")
LEVANTER_TESTS = Path(__file__).parents[3] / "lib/levanter/tests"


def _fingerprint_config(step):
    return step.build_config(StepContext.for_fingerprint(step.runtime_args, step.deps))


@pytest.fixture
def fake_store(tmp_path, monkeypatch):
    """A two-bucket store on disk, with the mix restricted to those buckets."""
    tokenizer_dir = tmp_path / "tokenizer"
    tokenizer_dir.mkdir()
    for source, target in TOKENIZER_FILES:
        shutil.copy(LEVANTER_TESTS / source, tokenizer_dir / target)
    metadata = preprocessor_for_format(
        TextLmDatasetFormat(), load_tokenizer(str(tokenizer_dir)), enforce_bos=True, enforce_eos=True
    ).metadata
    store = tmp_path / "store"
    for (cell, partition), first_id in FIRST_ID.items():
        tokens = TOKENS[cell][partition == "gt_64k"]
        ids = np.arange(first_id, first_id + tokens, dtype=np.int32)
        docs = ({"input_ids": ids[start : start + 16]} for start in range(0, tokens, 16))
        path = store / f"cluster={int(cell[1:3])}/quality={cell[4]}/length={partition}"
        write_levanter_cache(docs, str(path), metadata=metadata)
    monkeypatch.setattr(harrier_long_context_mix, "_COOLDOWN_WEIGHTS", WEIGHTS)

    def config(**kwargs):
        built = harrier_long_context_data_config(
            ctx=StepContext.for_fingerprint(),
            start_step=START_STEP,
            max_seq_len=SEQ_LEN,
            validation=(),
            store=str(store),
            length_tokens=TOKENS,
            **kwargs,
        )
        return dataclasses.replace(built, tokenizer=str(tokenizer_dir), mixture_block_size=16)

    return store, config


def _partition(token: int) -> tuple[str, str]:
    first_id = max(first for first in FIRST_ID.values() if first <= token)
    return next(key for key, value in FIRST_ID.items() if value == first_id)


def test_skew_multiplies_long_mass_inside_the_bucket_and_keeps_outer_weights(fake_store):
    _, config = fake_store
    data = config()
    bucket_sequences = {"c01q0": (800 + LONG_CONTEXT_SKEW * 160) // SEQ_LEN, "c02q3": 240 // SEQ_LEN}
    block_reads = {"c01q0": 12, "c02q3": 4}
    # Whole blocks that read c01q0 exactly once through its concatenated partitions.
    blocks = bucket_sequences["c01q0"] // block_reads["c01q0"]

    dataset = data.train_set(
        hax.Axis("position", SEQ_LEN), BatchSchedule(1), key=jax.random.PRNGKey(0)
    ).as_sync_dataset()
    examples = dataset.get_batch(list(range(START_STEP, START_STEP + blocks * data.mixture_block_size)))
    # EOS and padding sit outside the store's token ids.
    tokens = Counter(
        _partition(int(token))
        for example in examples
        for token in np.asarray(example.tokens.array)
        if 1000 <= token < 50_000
    )

    assert (
        sum(count for (cell, _), count in tokens.items() if cell == "c01q0") / SEQ_LEN == blocks * block_reads["c01q0"]
    )
    assert (
        sum(count for (cell, _), count in tokens.items() if cell == "c02q3") / SEQ_LEN == blocks * block_reads["c02q3"]
    )
    assert tokens["c01q0", "gt_64k"] / tokens["c01q0", "lte_64k"] == pytest.approx(LONG_CONTEXT_SKEW * 160 / 800)


def test_bucket_without_long_tokens_reads_only_short_documents(fake_store):
    _, config = fake_store
    data = config()

    assert set(data.components["c02q3"].children) == {"c02q3-lte_64k"}
    assert len(data.components["c01q0"].children) == 1 + LONG_CONTEXT_SKEW
    bucket = data.train_sets(hax.Axis("position", SEQ_LEN), key=jax.random.PRNGKey(0))["c02q3"].as_sync_dataset()
    assert len(bucket) == 240 // SEQ_LEN


def test_store_counts_read_from_ledgers_with_an_absent_partition_as_empty(fake_store):
    store, _ = fake_store

    assert read_length_tokens(str(store), list(TOKENS)) == TOKENS


def test_reported_share_and_passes(monkeypatch):
    monkeypatch.setattr(harrier_long_context_mix, "_COOLDOWN_WEIGHTS", {"a": 0.5, "b": 0.5})
    tokens = {"a": (300, 100), "b": (100, 0)}

    assert long_document_share(tokens, 1) == pytest.approx(0.125)
    assert long_document_share(tokens, 4) == pytest.approx(0.5 * 400 / 700)
    # Half of 1000 tokens read a 700-token concatenation holding the long partition four times.
    assert long_partition_passes(tokens, 4, 1000) == pytest.approx({"a": 500 * 4 / 700})
    assert overall_long_partition_passes(tokens, 4, 1000) == pytest.approx(500 * 4 / 700)


def test_hero_off_builds_the_harrier_config_it_always_built():
    step = build_ladder_run(run_id="test-long-context-off", size="d6144", version="dev")
    ctx = StepContext.for_fingerprint(step.runtime_args, step.deps)
    config = step.build_config(ctx)

    assert config.data == harrier_mix_2026_08_18_data_config(
        ctx=ctx,
        total_steps=config.trainer.trainer.num_train_steps,
        batch_size=config.trainer.trainer.train_batch_size,
        max_seq_len=config.model.max_seq_len,
        enable_simulated_epoching=False,
        validation=[dep for dep in step.deps if dep is not HARRIER_MIX_2026_08_18_STORE],
        prior_context_phases=launch_scaling_ladder.HERO_CONTEXT_PHASES,
    )
    assert config.data.mixture_start_step is None
    assert LONG_CONTEXT_TAG not in config.trainer.trainer.tracker.tags


def test_hero_on_starts_the_store_where_its_context_history_ends(monkeypatch):
    monkeypatch.setattr(launch_scaling_ladder, "HERO_USES_LONG_CONTEXT_MIX", True)
    monkeypatch.setattr(harrier_long_context_mix, "_load_length_tokens", lambda: HERO_LENGTH_TOKENS)

    step = build_ladder_run(run_id="test-long-context-on", size="d6144", version="dev")
    config = _fingerprint_config(step)

    phases = launch_scaling_ladder.HERO_CONTEXT_PHASES
    assert config.data.mixture_start_step == phases[-1].end_step
    assert config.data.prior_context_phases == []
    assert config.data.target_budget is None
    # The FLOPs baseline keeps following the context history.
    assert config.flops_baseline.completed_steps == phases[-1].end_step
    train_weights = config.data.train_weights
    assert len(train_weights) == 1
    cooldown = dict(_SPEC.phase_weights[-1])
    assert {name: weight for name, weight in train_weights[0][1].items() if weight > 0} == {
        cell: weight for cell, weight in cooldown.items() if weight > 0
    }
    validation = {dep.name for dep in step.deps} & train_weights[0][1].keys()
    assert validation and all(train_weights[0][1][name] == 0 for name in validation)
    assert LONG_CONTEXT_TAG in config.trainer.trainer.tracker.tags


def test_diagnostic_arms_select_the_same_data_on_any_rack_count(monkeypatch):
    monkeypatch.setattr(harrier_long_context_mix, "_load_length_tokens", lambda: HERO_LENGTH_TOKENS)

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
                version="dev",
            )
        )

    one_rack, four_racks = arm(1), arm(4)

    assert one_rack.data == four_racks.data
    assert one_rack.data.mixture_start_step == 390_000
    assert one_rack.data.shuffle.io_block_size == 4
    assert LONG_CONTEXT_TAG in one_rack.trainer.trainer.tracker.tags
