# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import asyncio
from collections import Counter

from levanter.data.dataset import ListAsyncDataset
from levanter.data.mixture import MixtureDataset

from experiments.sft_new_mix.mixture_weights import candidate_weights


def test_native_sampling_preserves_replay_repetition_after_cleanup():
    sources = []
    jobs = []
    for name, component, tokens, mode, path in (
        ("replay/bucket/short", "short", 100, "full_token", "short-cache"),
        ("replay/bucket/long", "long", 50, "full_token", "long-cache"),
        ("nemotron_sft_v3/instruction_following_chat_v3/chat", "chat", 50, "chat_assistant_only", "chat-cache"),
    ):
        sources.append({"source": name, "component": component, "counts": {"output_tokens": tokens}, "shards": [path]})
        jobs.append({"source": name, "loss_mode": mode, "input_path": path, "expected_tokens": 100})
    control = {
        "pretrain/bucket": {
            "type": "concat",
            "children": {
                "short": {"cache_dir": "short-cache"},
                "long0": {"cache_dir": "long-cache"},
                "long1": {"cache_dir": "long-cache"},
            },
        }
    }
    result = candidate_weights(
        {"complete": True, "sources": sources},
        {"jobs": jobs},
        control,
        [[0, {"pretrain/bucket": 0.2, "chat": 0.8}]],
        {"replay": 0.2, "instruction_chat": 0.8},
        updates=2,
    )
    # Half-retention of a twice-repeated long cache balances one intact short cache.
    mixture = MixtureDataset({name: ListAsyncDataset([name]) for name in result.weights}, result.weights, 100, key=0)
    actual = asyncio.run(mixture.get_batch(list(range(100))))
    assert Counter(actual) == {"short": 10, "long": 10, "chat": 80}


def test_ranged_replay_jobs_share_one_replay_basis():
    sources = [
        {
            "source": "replay/bucket/short",
            "component": "short",
            "counts": {"output_tokens": 100},
            "shards": ["short-cache"],
        },
        {"source": "replay/bucket/long", "component": "long", "counts": {"output_tokens": 50}, "shards": ["long-cache"]},
        {
            "source": "nemotron_sft_v3/instruction_following_chat_v3/chat",
            "component": "chat",
            "counts": {"output_tokens": 50},
            "shards": ["chat-cache"],
        },
    ]
    # The long cache was split into ranged children that share its input path.
    jobs = [
        {
            "source": "replay/bucket/short",
            "loss_mode": "full_token",
            "input_path": "short-cache",
            "expected_tokens": 100,
        },
        {"source": "replay/bucket/long", "loss_mode": "full_token", "input_path": "long-cache", "expected_tokens": 20},
        {"source": "replay/bucket/long", "loss_mode": "full_token", "input_path": "long-cache", "expected_tokens": 30},
        {
            "source": "nemotron_sft_v3/instruction_following_chat_v3/chat",
            "loss_mode": "chat_assistant_only",
            "input_path": "chat-cache",
            "expected_tokens": 100,
        },
    ]
    control = {
        "pretrain/bucket": {
            "type": "concat",
            "children": {
                "short": {"cache_dir": "short-cache"},
                "long0": {"cache_dir": "long-cache"},
                "long1": {"cache_dir": "long-cache"},
            },
        }
    }
    arguments = (
        {"complete": True, "sources": sources},
        {"jobs": jobs},
        control,
        [[0, {"pretrain/bucket": 0.2, "chat": 0.8}]],
        {"replay": 0.2, "instruction_chat": 0.8},
        2,
    )
    ranged = candidate_weights(*arguments)
    intact = candidate_weights(
        {"complete": True, "sources": sources},
        {
            "jobs": (
                [job for job in arguments[1]["jobs"] if job["input_path"] != "long-cache"]
                + [
                    {
                        "source": "replay/bucket/long",
                        "loss_mode": "full_token",
                        "input_path": "long-cache",
                        "expected_tokens": 50,
                    }
                ]
            )
        },
        *arguments[2:],
    )
    # Partitioning one cache into ranged jobs must not change relative weights.
    assert ranged.weights == intact.weights
