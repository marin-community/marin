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
