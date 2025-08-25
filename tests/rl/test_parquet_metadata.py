"""Tests Parquet serialization of Turn with InferenceMetadata dataclass, tokens, and timestamp."""

import time
import dataclasses

import numpy as np

from marin.rl.datatypes import InferenceMetadata, RolloutGroup, RolloutRecord, Turn
from marin.rl.parquet_store import iter_rollout_groups, write_rollout_groups


def test_parquet_inference_metadata_and_tokens_roundtrip(tmp_path):
    ts = time.time()

    meta = InferenceMetadata(model_version="m1", finish_reason="stop", usage={"prompt_tokens": 5}, input_seed=7)

    r = RolloutRecord(
        environment="env",
        example_id="ex",
        policy_version="v1",
        rollout_uid="uid",
        replica_id="rep",
        turns=[
            Turn(
                message="hello",
                role="assistant",
                tokens=["he", "llo"],
                logprobs=np.array([-0.1, -0.2], dtype=float),
                reward=1.0,
                inference_metadata=meta,
                timestamp=123456789,
            )
        ],
        metadata={"k": "v"},
        created_ts=ts,
    )

    g = RolloutGroup(
        id="gid",
        environment="env",
        example_id="ex",
        policy_version="v1",
        rollouts=[r],
        sealed_ts=ts,
        metadata={"m": 1},
    )

    write_rollout_groups([g], str(tmp_path))
    read_back = list(iter_rollout_groups(str(tmp_path)))
    assert len(read_back) == 1
    g2 = read_back[0]

    t2 = g2.rollouts[0].turns[0]
    # inference_metadata is read back as a dict
    assert isinstance(t2.inference_metadata, dict)
    assert t2.inference_metadata == dataclasses.asdict(meta)
    assert t2.tokens == ["he", "llo"]
    assert t2.timestamp == 123456789
