# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from dataclasses import replace

import pytest
from taskforge.llm.policy import LLMPolicy

from experiments.post_training.capability_driven_envs.open_ended_proposal_source import OpenEndedProposalSource
from experiments.post_training.capability_driven_envs.open_ended_proposal_store import (
    load_or_propose,
    proposal_batch_identity,
    read_proposal_batch,
)
from experiments.post_training.capability_driven_envs.tests.scripted_model import ScriptedModel
from experiments.post_training.capability_driven_envs.tests.test_open_ended import bundle
from experiments.post_training.capability_driven_envs.tests.test_open_ended_proposal_source import (
    CONFIG,
    filled_document,
    plan,
    slot,
)


async def test_proposal_batch_is_persisted_before_retry_and_not_resampled(tmp_path):
    model = ScriptedModel()
    model.reply(plan(slot(1)))
    model.reply(filled_document(1))
    source = OpenEndedProposalSource(model, LLMPolicy(), CONFIG)
    path = tmp_path / "proposal-batch.json"

    generated = await load_or_propose(path, source, bundle(), 1)
    request_count = len(model.requests)
    restored = await load_or_propose(path, source, bundle(), 1)

    assert restored == generated
    assert len(model.requests) == request_count
    record = read_proposal_batch(path)
    assert record.batch() == generated
    assert record.identity == proposal_batch_identity(bundle(), source, 1)


async def test_proposal_batch_path_rejects_changed_generation_identity(tmp_path):
    model = ScriptedModel()
    model.reply(plan(slot(1)))
    model.reply(filled_document(1))
    source = OpenEndedProposalSource(model, LLMPolicy(), CONFIG)
    path = tmp_path / "proposal-batch.json"
    await load_or_propose(path, source, bundle(), 1)
    changed_source = OpenEndedProposalSource(model, replace(LLMPolicy(), temperature=0.5), CONFIG)

    with pytest.raises(ValueError, match="cache key"):
        await load_or_propose(path, changed_source, bundle(), 1)
