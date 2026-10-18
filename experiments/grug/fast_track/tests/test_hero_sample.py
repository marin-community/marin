# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json

import jax
import numpy as np
import pytest
from haliax import Axis
from levanter.schedule import BatchSchedule
from levanter.store.cache import TreeCache, write_levanter_cache
from levanter.tokenizers import load_tokenizer, tokenizer_content_hash
from marin.execution.artifact import write_artifact
from marin.execution.lazy import ArtifactStep, StepContext
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace

import experiments.grug.fast_track.hero_sample as hero_sample
from experiments.grug.fast_track.contracts import FrozenBaselineComponent, FrozenBaselineManifest, ResolvedTrainingBudget
from experiments.grug.fast_track.hero_sample import (
    _mixture_component_examples,
    _sample_component,
    _SampleTask,
    _shuffled_component_dataset,
)
from experiments.grug.moe_hero_ep.harrier_mix_2026_08_18 import (
    HARRIER_MIX_2026_08_18_STORE,
    harrier_mix_2026_08_18_data_config,
)

ALPHA = 1


def test_hero_sample_capacity_matches_mixture_block_rounding():
    weights = {"largest": 0.50001, "other": 0.49999}

    examples = _mixture_component_examples(weights, 4096)

    assert examples == {"largest": 24_577, "other": 24_575}


def test_hero_sample_artifact_identity_tracks_tokenizer_files(monkeypatch):
    hashes = {"marin-community/marin-tokenizer": "sha256:source", "hero-bpe-v16384": "sha256:first"}
    monkeypatch.setattr(hero_sample, "tokenizer_content_hash", hashes.__getitem__)

    first = hero_sample.hero_sample_step(version="2026.10.04")
    hashes["hero-bpe-v16384"] = "sha256:second"
    second = hero_sample.hero_sample_step(version="2026.10.04")

    assert first.name != second.name
    first_config = first.build_config(StepContext.for_fingerprint((), first.deps))
    second_config = second.build_config(StepContext.for_fingerprint((), second.deps))
    assert first_config.target_tokenizer_hash != second_config.target_tokenizer_hash


def _write_test_tokenizer(path, *, bos_token: str, eos_token: str) -> None:
    path.mkdir()
    vocab = {"[UNK]": 0, "alpha": ALPHA, bos_token: 2, eos_token: 3}
    tokenizer = Tokenizer(WordLevel(vocab, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer.add_special_tokens([bos_token, eos_token])
    tokenizer.save(str(path / "tokenizer.json"))
    (path / "tokenizer_config.json").write_text(json.dumps({"bos_token": bos_token, "eos_token": eos_token}))


def _sample_one_component(tmp_path, source_tokens: np.ndarray, *, target_tokens: int) -> _SampleTask:
    source_tokenizer_path = tmp_path / "source-tokenizer"
    target_tokenizer_path = tmp_path / "target-tokenizer"
    _write_test_tokenizer(source_tokenizer_path, bos_token="<|source_begin|>", eos_token="<|source_end|>")
    _write_test_tokenizer(target_tokenizer_path, bos_token="<|target_begin|>", eos_token="<|target_end|>")
    source_cache = tmp_path / "source-cache"
    write_levanter_cache([{"input_ids": source_tokens}], str(source_cache), metadata={})
    task = _SampleTask(
        name="c00q0",
        source_cache=str(source_cache),
        output_cache=str(tmp_path / "target-cache"),
        source_tokenizer=str(source_tokenizer_path),
        target_tokenizer=str(target_tokenizer_path),
        target_tokens=target_tokens,
        data_seed=0,
        component_index=0,
        source_tokenizer_hash=tokenizer_content_hash(str(source_tokenizer_path)),
        target_tokenizer_hash=tokenizer_content_hash(str(target_tokenizer_path)),
    )
    _sample_component(task)
    return task


def _target_rows(task: _SampleTask) -> list[list[int]]:
    cache = TreeCache.load(task.output_cache, {"input_ids": np.zeros((0,), dtype=np.int32)})
    return [row["input_ids"].tolist() for row in cache]


def test_hero_sample_closes_every_source_segment_and_keeps_only_document_starts(tmp_path):
    source_bos, source_eos = 2, 3
    # A cut tail of an earlier document, one whole document, and a document that the example cuts.
    head = [ALPHA, ALPHA, source_eos]
    whole = [source_bos, ALPHA, source_eos]
    cut_length = 4096 - len(head) - len(whole) - 1
    source = np.array([*head, *whole, source_bos, *[ALPHA] * cut_length], dtype=np.int32)

    task = _sample_one_component(tmp_path, source, target_tokens=1)

    target = load_tokenizer(task.target_tokenizer)
    bos, eos = target.bos_token_id, target.eos_token_id
    assert _target_rows(task) == [[ALPHA, ALPHA, eos, bos, ALPHA, eos, bos, *[ALPHA] * cut_length, eos]]


def test_hero_sample_round_trip_preserves_eos_through_training_loader(tmp_path, monkeypatch):
    source_sequence = np.tile(np.array([ALPHA, 3], dtype=np.int32), 2048)
    task = _sample_one_component(tmp_path, source_sequence, target_tokens=4096)
    target_tokenizer_path = task.target_tokenizer

    expected = np.tile(np.array([ALPHA, load_tokenizer(target_tokenizer_path).eos_token_id], dtype=np.int32), 2048)
    np.testing.assert_array_equal(_target_rows(task)[0], expected)

    training_dataset = _shuffled_component_dataset(
        cache_dir=task.output_cache, tokenizer=target_tokenizer_path, name="c00q0", key=jax.random.PRNGKey(0)
    )
    training_example = training_dataset.as_sync_dataset().get_batch([0])[0]
    np.testing.assert_array_equal(np.asarray(training_example.tokens), expected)

    # A one-sequence mixture block lets the artifact loader consume this small real cache.
    monkeypatch.setattr(hero_sample, "_MIXTURE_BLOCK_SIZE", 1)
    artifact_path = str(tmp_path / "hero-artifact")
    write_artifact(
        hero_sample.PreparedHeroSample(
            baseline=FrozenBaselineManifest(
                tokenizer=target_tokenizer_path,
                components=(FrozenBaselineComponent("c00q0", task.output_cache, 1.0),),
            ),
            source_store=task.source_cache,
            source_tokenizer=task.source_tokenizer,
            source_tokenizer_hash=task.source_tokenizer_hash,
            target_tokenizer_hash=task.target_tokenizer_hash,
            recipe_sha256="a" * 64,
            phase="main",
            loader_policy=hero_sample.HERO_LOADER_POLICY,
            sequence_length=4096,
            data_seed=0,
            requested_tokens=4096,
            actual_tokens=4096,
            component_tokens={"c00q0": 4096},
        ),
        artifact_path,
    )
    sample = ArtifactStep.adopt("hero-test", "2026.10.04", artifact_path, kind=hero_sample.PreparedHeroSample)
    source = hero_sample.HeroTrainingSource(sample)
    context = StepContext.for_run(output_path="unused", prefix=str(tmp_path), deps=source.dependencies())
    budget = ResolvedTrainingBudget(1, 1, 4096)
    data = source.data_config(ctx=context, validation=(), tokenizer=target_tokenizer_path, budget=budget)
    loaded = (
        data.train_set(Axis("position", 4096), BatchSchedule(1), key=jax.random.PRNGKey(0))
        .as_sync_dataset()
        .get_batch([0])[0]
    )
    np.testing.assert_array_equal(np.asarray(loaded.tokens.array), expected)

    changed_tokenizer = Tokenizer.from_file(f"{target_tokenizer_path}/tokenizer.json")
    changed_tokenizer.add_tokens(["new-token"])
    changed_tokenizer.save(f"{target_tokenizer_path}/tokenizer.json")
    with pytest.raises(ValueError, match=r"hero sample training: tokenizer .* content is"):
        source.data_config(ctx=context, validation=(), tokenizer=target_tokenizer_path, budget=budget)


def test_hero_shuffle_keys_follow_production_component_order():
    data = harrier_mix_2026_08_18_data_config(
        ctx=StepContext.for_fingerprint((), (HARRIER_MIX_2026_08_18_STORE,)),
        total_steps=390_251,
        batch_size=512,
        max_seq_len=4096,
        experiment_flops=1e24,
        validation=(),
    )
    stages = data.train_weights
    assert isinstance(stages, list)
    production_order = tuple(name for name in data.components if any(weights.get(name, 0) > 0 for _, weights in stages))

    assert hero_sample._HERO_RECIPE.shuffle_order == production_order
    # c22q0 has weight only before the main phase, but it still takes a production shuffle key.
    assert "c22q0" in production_order
    assert "c22q0" not in hero_sample._HERO_RECIPE.weights
