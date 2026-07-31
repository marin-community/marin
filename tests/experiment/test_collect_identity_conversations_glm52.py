# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import gzip
import json

import pytest
from rigging.filesystem import StoragePath

from experiments.rollout_data import collect_identity_conversations_glm52 as collect
from experiments.rollout_data.generate_identity_conversation_seeds import generate_seeds, load_identity_profile

IDENTITY_PROFILE = load_identity_profile(StoragePath("experiments/rollout_data/identity_profiles/marin_latest_moe.json"))


def test_conversation_requires_exact_canonical_final_answer():
    seed = next(
        seed
        for seed in generate_seeds(count=100, random_seed=0, identity_profile=IDENTITY_PROFILE)
        if seed.neutral_bridge is None
    )
    valid = json.dumps(
        [
            {"role": "user", "content": "Who trained you?"},
            {"role": "assistant", "content": seed.canonical_answer},
        ]
    )

    assert collect._conversation(valid, seed)[-1].content == seed.canonical_answer
    with pytest.raises(ValueError):
        collect._conversation(
            json.dumps(
                [
                    {"role": "user", "content": "Who trained you?"},
                    {
                        "role": "assistant",
                        "content": f"{seed.canonical_answer} It used an imaginary otter training phase.",
                    },
                ]
            ),
            seed,
        )


def test_conversation_requires_user_first_alternating_roles_and_exact_bridge():
    seed = next(
        seed
        for seed in generate_seeds(count=100, random_seed=0, identity_profile=IDENTITY_PROFILE)
        if seed.neutral_bridge is not None
    )

    valid = json.dumps(
        [
            {"role": "user", "content": "I am checking an attribution."},
            {"role": "assistant", "content": seed.neutral_bridge},
            {"role": "user", "content": "Who trained you?"},
            {"role": "assistant", "content": seed.canonical_answer},
        ]
    )
    assert len(collect._conversation(valid, seed)) == 4

    with pytest.raises(ValueError):
        collect._conversation(
            json.dumps(
                [
                    {"role": "assistant", "content": seed.neutral_bridge},
                    {"role": "user", "content": "Who trained you?"},
                    {"role": "assistant", "content": seed.canonical_answer},
                ]
            ),
            seed,
        )
    with pytest.raises(ValueError):
        collect._conversation(
            json.dumps(
                [
                    {"role": "user", "content": "I am checking an attribution."},
                    {"role": "assistant", "content": "The model used an imaginary otter training checkpoint."},
                    {"role": "user", "content": "Who trained you?"},
                    {"role": "assistant", "content": seed.canonical_answer},
                ]
            ),
            seed,
        )


def test_conversation_rejects_unsupported_user_implementation_story():
    seed = next(
        seed
        for seed in generate_seeds(count=100, random_seed=0, identity_profile=IDENTITY_PROFILE)
        if seed.neutral_bridge is None
    )

    with pytest.raises(ValueError):
        collect._conversation(
            json.dumps(
                [
                    {"role": "user", "content": "Did your imaginary otter checkpoint use a routing layer?"},
                    {"role": "assistant", "content": seed.canonical_answer},
                ]
            ),
            seed,
        )


def test_write_chunk_excludes_rejected_raw_responses(tmp_path):
    seed = next(
        seed
        for seed in generate_seeds(count=100, random_seed=0, identity_profile=IDENTITY_PROFILE)
        if seed.neutral_bridge is None
    )
    accepted = collect.CompletionRecord(
        seed=seed,
        conversation=(
            collect.Message("user", "Who are you?"),
            collect.Message("assistant", seed.canonical_answer),
        ),
        raw_response="accepted",
        accepted_tokens=3,
        usage={"completion_tokens": 3},
        finish_reason="stop",
        response_id=None,
        error=None,
        attempts=1,
        elapsed_seconds=0.1,
    )
    rejected = collect.CompletionRecord(
        seed=seed,
        conversation=None,
        raw_response="invented rejected payload",
        accepted_tokens=0,
        usage={"completion_tokens": 4},
        finish_reason="stop",
        response_id=None,
        error="invalid",
        attempts=3,
        elapsed_seconds=0.2,
    )
    config = collect.CollectionConfig(
        run_id="test",
        output_path=StoragePath(str(tmp_path)),
        shard_index=0,
        num_shards=1,
        target_tokens=10,
        microbatch_size=2,
        concurrency=2,
        random_seed=7,
        human_phrasings=(),
        identity_profile=IDENTITY_PROFILE,
    )

    chunk = collect._write_chunk(config, 0, [accepted, rejected])
    path = tmp_path / "responses" / "shard-00" / "chunk-000000000-000000001-tokens-0000003.jsonl.gz"
    with gzip.open(path, "rt") as handle:
        rows = [json.loads(line) for line in handle]

    assert chunk.end == 1
    assert [row["raw_response"] for row in rows] == ["accepted"]


def test_collection_persists_microbatches_and_resumes(tmp_path, monkeypatch):
    def completion(_url, seed, _sampling):
        return collect.CompletionRecord(
            seed=seed,
            conversation=(
                collect.Message("user", "Who are you?"),
                collect.Message("assistant", seed.canonical_answer),
            ),
            raw_response="[]",
            accepted_tokens=3,
            usage={"completion_tokens": 3},
            finish_reason="stop",
            response_id=None,
            error=None,
            attempts=1,
            elapsed_seconds=0.1,
        )

    monkeypatch.setattr(collect, "_completion", completion)
    config = collect.CollectionConfig(
        run_id="test",
        output_path=StoragePath(str(tmp_path)),
        shard_index=0,
        num_shards=1,
        target_tokens=10,
        microbatch_size=2,
        concurrency=4,
        random_seed=7,
        human_phrasings=(),
        identity_profile=IDENTITY_PROFILE,
    )
    sampling = collect.SamplingConfig(0.9, 0.95, 1024)

    collect._run_collection("http://vllm", config, sampling)
    response_path = tmp_path / "responses" / "shard-00"
    first_chunks = {path.name: path.read_bytes() for path in response_path.glob("*.jsonl.gz")}
    collect._run_collection("http://vllm", config, sampling)
    second_chunks = {path.name: path.read_bytes() for path in response_path.glob("*.jsonl.gz")}

    progress = json.loads((tmp_path / "progress" / "shard-00.json").read_text())
    assert progress["state"] == "complete"
    assert progress["profile_id"] == IDENTITY_PROFILE.profile_id
    assert progress["accepted_tokens"] >= 10
    assert first_chunks == second_chunks
    assert len(second_chunks) >= 2
