# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from skyrl_train.rollouts.loader import STALE_RETRIES_EXHAUSTED_METRIC, PromptLoader, SeededPasses

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


def _next_uid(loader: PromptLoader) -> str:
    return loader.next_prompt(NOTHING_LIVE)["uid"]


def test_a_prompt_that_keeps_missing_the_staleness_window_waits_for_its_next_turn_in_the_order():
    loader = PromptLoader(
        _Prompts(), SeededPasses(len(UIDS), seed=0, shuffle=False), batch_size=1, max_stale_retries=1
    )
    offered = [_next_uid(loader)]
    loader.retry({"uid": "0"})
    offered.append(_next_uid(loader))
    loader.retry({"uid": "0"})
    exhausted = [loader.observe([])[STALE_RETRIES_EXHAUSTED_METRIC], loader.observe([])[STALE_RETRIES_EXHAUSTED_METRIC]]
    offered += [_next_uid(loader) for _ in range(3)]
    loader.retry({"uid": "0"})
    offered += [_next_uid(loader) for _ in range(4)]
    loader.retry({"uid": "0"})
    offered.append(_next_uid(loader))

    assert offered == ["0", "0", "1", "2", "0", "0", "1", "2", "0", "0"]
    assert exhausted == [1.0, 0.0]
