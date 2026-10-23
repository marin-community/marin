# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import copy
import io

import pytest
import torch
from skyrl_train.rollouts.loader import PromptLoader, PromptLoaderState, SeededPasses

NOTHING_LIVE: frozenset[str] = frozenset()
UIDS = ["0", "1", "2"]


class _Prompts:
    def __len__(self) -> int:
        return len(UIDS)

    def __getitem__(self, index: int) -> str:
        return UIDS[index]

    def uid(self, index: int) -> str:
        return UIDS[index]

    def collate_fn(self, items: list[str]) -> list[dict]:
        return [{"uid": uid} for uid in items]


def _loader() -> PromptLoader:
    return PromptLoader(_Prompts(), SeededPasses(len(UIDS), seed=0, shuffle=False), batch_size=1, max_stale_retries=1)


def _next_uid(loader: PromptLoader) -> str:
    return loader.next_prompt(NOTHING_LIVE)["uid"]


def _checkpoint_round_trip(state: PromptLoaderState) -> PromptLoaderState:
    buffer = io.BytesIO()
    torch.save(state, buffer)
    buffer.seek(0)
    return torch.load(buffer, weights_only=False)


def _written_before_counts(state: PromptLoaderState) -> PromptLoaderState:
    legacy = copy.copy(state)
    vars(legacy).pop("stale_retries")
    return legacy


def test_a_prompt_that_keeps_missing_the_staleness_window_waits_for_its_next_turn_in_the_order():
    loader = _loader()
    offered = [_next_uid(loader)]
    loader.retry({"uid": "0"})
    offered += [_next_uid(loader) for _ in range(4)]
    for _ in range(2):
        loader.retry({"uid": "0"})
        offered.append(_next_uid(loader))

    assert offered == ["0", "0", "1", "2", "0", "0", "1"]


@pytest.mark.parametrize(
    ("checkpoint", "offered_after_resume"),
    [
        (lambda state: state, ["0", "1"]),
        (_written_before_counts, ["0", "0"]),
    ],
    ids=["current", "written_before_counts"],
)
def test_resume_restores_regeneration_counts_from_the_checkpoint(checkpoint, offered_after_resume):
    loader = _loader()
    _next_uid(loader)
    loader.retry({"uid": "0"})
    resumed = _loader()
    resumed.load_state_dict(_checkpoint_round_trip(checkpoint(loader.state_dict())))

    offered = [_next_uid(resumed)]
    resumed.retry({"uid": "0"})
    offered.append(_next_uid(resumed))

    assert offered == offered_after_resume
