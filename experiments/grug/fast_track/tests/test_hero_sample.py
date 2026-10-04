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
    _component_shuffle_index,
    _mixture_component_examples,
    _production_component_dataset,
    _sample_component,
    _source_component_config,
)


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


def _write_test_tokenizer(path, *, eos_token: str, eos_id: int):
    path.mkdir()
    vocab = {"[UNK]": 0, "alpha": 1, "reserved": 2, eos_token: eos_id}
    tokenizer = Tokenizer(WordLevel(vocab, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer.add_special_tokens([eos_token])
    tokenizer.save(str(path / "tokenizer.json"))
    (path / "tokenizer_config.json").write_text(json.dumps({"eos_token": eos_token}))


def test_hero_sample_round_trip_preserves_eos_through_training_loader(tmp_path, monkeypatch):
    source_tokenizer_path = tmp_path / "source-tokenizer"
    target_tokenizer_path = tmp_path / "target-tokenizer"
    _write_test_tokenizer(source_tokenizer_path, eos_token="<|source_end|>", eos_id=2)
    _write_test_tokenizer(target_tokenizer_path, eos_token="<|target_end|>", eos_id=3)
    source_tokenizer = load_tokenizer(str(source_tokenizer_path))
    target_tokenizer = load_tokenizer(str(target_tokenizer_path))
    source_cache = tmp_path / "source-cache"
    source_sequence = np.tile(np.array([1, source_tokenizer.eos_token_id], dtype=np.int32), 2048)
    write_levanter_cache([{"input_ids": source_sequence}], str(source_cache), metadata={})

    output_cache = tmp_path / "target-cache"
    task = {
        "name": "c00q0",
        "source_cache": str(source_cache),
        "output_cache": str(output_cache),
        "source_tokenizer": str(source_tokenizer_path),
        "target_tokenizer": str(target_tokenizer_path),
        "target_tokens": 4096,
        "data_seed": 0,
        "component_index": 0,
        "source_tokenizer_hash": tokenizer_content_hash(str(source_tokenizer_path)),
        "target_tokenizer_hash": tokenizer_content_hash(str(target_tokenizer_path)),
    }
    _sample_component(task)

    expected = np.tile(np.array([1, target_tokenizer.eos_token_id], dtype=np.int32), 2048)
    target_tree_cache = TreeCache.load(str(output_cache), {"input_ids": np.zeros((0,), dtype=np.int32)})
    np.testing.assert_array_equal(target_tree_cache.get_batch_sync([0])[0]["input_ids"], expected)

    training_config = _source_component_config(
        source_cache=str(output_cache), tokenizer=str(target_tokenizer_path), name="c00q0"
    )
    training_dataset = _production_component_dataset(
        training_config,
        name="c00q0",
        position=Axis("position", 4096),
        key=jax.random.PRNGKey(0),
    )
    training_example = training_dataset.as_sync_dataset().get_batch([0])[0]
    np.testing.assert_array_equal(np.asarray(training_example.tokens), expected)

    # A one-sequence mixture block lets the artifact loader consume this small real cache.
    monkeypatch.setattr(hero_sample, "_MIXTURE_BLOCK_SIZE", 1)
    artifact_path = str(tmp_path / "hero-artifact")
    write_artifact(
        hero_sample.PreparedHeroSample(
            baseline=FrozenBaselineManifest(
                tokenizer=str(target_tokenizer_path),
                components=(FrozenBaselineComponent("c00q0", str(output_cache), 1.0),),
            ),
            source_store=str(source_cache),
            source_tokenizer=str(source_tokenizer_path),
            source_tokenizer_hash=task["source_tokenizer_hash"],
            target_tokenizer_hash=task["target_tokenizer_hash"],
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
    data = source.data_config(
        ctx=context, validation=(), tokenizer=str(target_tokenizer_path), budget=ResolvedTrainingBudget(1, 1, 4096)
    )
    loaded = (
        data.train_set(Axis("position", 4096), BatchSchedule(1), key=jax.random.PRNGKey(0))
        .as_sync_dataset()
        .get_batch([0])[0]
    )
    np.testing.assert_array_equal(np.asarray(loaded.tokens.array), expected)

    changed_tokenizer = Tokenizer.from_file(str(target_tokenizer_path / "tokenizer.json"))
    changed_tokenizer.add_tokens(["new-token"])
    changed_tokenizer.save(str(target_tokenizer_path / "tokenizer.json"))
    with pytest.raises(ValueError, match="training tokenizer content differs"):
        source.data_config(
            ctx=context, validation=(), tokenizer=str(target_tokenizer_path), budget=ResolvedTrainingBudget(1, 1, 4096)
        )


def test_hero_component_shuffle_index_keeps_rare_prior_phase_component():
    recipe = json.loads(hero_sample.HERO_RECIPE.read_text())

    assert recipe["phases"][0]["weights"]["c22q0"] > 0
    assert recipe["phases"][1]["weights"]["c22q0"] == 0
    assert _component_shuffle_index(recipe, "c23q0") == list(
        name
        for name in recipe["available_tokens"]
        if any(phase["weights"].get(name, 0) > 0 for phase in recipe["phases"])
    ).index("c23q0")
