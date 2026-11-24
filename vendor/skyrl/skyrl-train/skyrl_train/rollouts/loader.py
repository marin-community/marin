"""The training dataset as an endless sequence of prompts."""

from __future__ import annotations

import collections
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

import torch
from loguru import logger

from skyrl_train.group_admission import AdmissionRejection

STALE_RETRIES_EXHAUSTED_METRIC = "async/stale_retries_exhausted"
QUARANTINED_PROMPTS_METRIC = "async/quarantined_prompts"
# Training steps a quarantined prompt sits out before the loader offers it again.
QUARANTINE_STEPS = 10
# Rejections of a group training cannot use: no loss tokens, or too few graded samples for its baseline. A broken
# grader causes them, and so does an infrastructure failure, whose samples are masked.
UNUSABLE_GROUP_REJECTIONS = frozenset(
    {AdmissionRejection.FULLY_MASKED.value, AdmissionRejection.BELOW_MINIMUM_GROUP_SIZE.value}
)
# Dispositions that say nothing about whether the prompt's groups are usable.
NEUTRAL_DISPOSITIONS = frozenset({AdmissionRejection.STALE.value, AdmissionRejection.DUPLICATE_UID.value})


class PromptGroupDataset(Protocol):
    def __len__(self) -> int: ...

    def __getitem__(self, index: int) -> Any: ...

    def uid(self, index: int) -> str:
        """The uid of row ``index``'s prompt group, without loading the row."""
        ...

    def label(self, index: int) -> str:
        """A name for row ``index`` that identifies its source in logs, such as a task id; the uid when it has none."""
        ...

    def collate_fn(self, items: list[Any]) -> list[dict]: ...


@dataclass(frozen=True)
class JudgedGroup:
    """A prompt group that batch selection judged in one step, kept or discarded, with its per-sample rewards.

    ``uid`` is the dataset row the group was generated from.
    """

    uid: str
    rewards: tuple[float, ...]


class PromptOrder(Protocol):
    """The dataset rows the loader offers, in order; an adaptive order learns from each step's judged groups."""

    def next_index(self) -> int: ...

    def observe(self, groups: Sequence[JudgedGroup]) -> dict[str, float]:
        """Fold one training step's judged groups into the order and return its metrics."""
        ...

    def state_dict(self) -> dict[str, Any]: ...

    def load_state_dict(self, state: Mapping[str, Any]) -> None: ...


class SeededPasses:
    """Endless passes over the dataset, each in a fresh seeded order."""

    def __init__(self, num_rows: int, *, seed: int, shuffle: bool):
        self._num_rows = num_rows
        self._seed = seed
        self._shuffle = shuffle
        self._epoch = 0
        self._position = 0
        self._order = self._pass_order(0)

    def next_index(self) -> int:
        if self._position == self._num_rows:
            self._epoch += 1
            self._order = self._pass_order(self._epoch)
            self._position = 0
        index = self._order[self._position]
        self._position += 1
        return index

    def observe(self, groups: Sequence[JudgedGroup]) -> dict[str, float]:
        return {}

    def state_dict(self) -> dict[str, Any]:
        return {"epoch": self._epoch, "position": self._position}

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        self._epoch = state["epoch"]
        self._order = self._pass_order(self._epoch)
        self._position = state["position"]

    def _pass_order(self, epoch: int) -> list[int]:
        if not self._shuffle:
            return list(range(self._num_rows))
        generator = torch.Generator()
        generator.manual_seed(self._seed + epoch)
        return torch.randperm(self._num_rows, generator=generator).tolist()


@dataclass(frozen=True)
class PromptLoaderState:
    """The prompt order's position and the prompts awaiting regeneration."""

    order: dict[str, Any]
    retries: list[dict]


class PromptLoader:
    """Yield one prompt at a time, re-offering prompts whose rollouts must be regenerated first.

    A batch trains at most one group per uid, so the loader never offers a uid that is live: already generating,
    or committed and not yet trained or discarded. A draw whose uid is live is dropped rather than generated as a
    copy the batch would reject.

    A prompt whose group missed the staleness window is regenerated at most ``max_stale_retries`` times in a row;
    after that it waits until the prompt order offers it again.

    A prompt whose last ``max_unusable_groups`` groups were all unusable, for example because its grader failed on
    every sample, is quarantined: the loader does not offer it for the next ``QUARANTINE_STEPS`` training steps, then
    offers it again with a fresh count. The same rejections follow an infrastructure failure, so a prompt must be
    able to return. Quarantine never leaves fewer usable prompts than one batch needs; a batch that still cannot fill
    ends at the admission stall timeout. Retry counts and quarantine restart empty when a run resumes.

    The sequence never ends; the trainer decides when to stop.
    """

    def __init__(
        self,
        dataset: PromptGroupDataset,
        order: PromptOrder,
        *,
        batch_size: int,
        max_stale_retries: int,
        max_unusable_groups: int,
    ):
        if max_stale_retries < 0:
            raise ValueError(f"max_stale_retries must be non-negative, got {max_stale_retries}")
        if max_unusable_groups < 1:
            raise ValueError(f"max_unusable_groups must be positive, got {max_unusable_groups}")
        self._uid_rows = {dataset.uid(index): index for index in reversed(range(len(dataset)))}
        self._uid_count = len(self._uid_rows)
        if self._uid_count < batch_size:
            raise ValueError(
                f"the training dataset has {self._uid_count} distinct prompt uids, fewer than one batch of "
                f"{batch_size}; a batch trains each uid at most once"
            )
        self._dataset = dataset
        self._order = order
        self._batch_size = batch_size
        self._draw_limit = 2 * batch_size
        self._max_stale_retries = max_stale_retries
        self._max_unusable_groups = max_unusable_groups
        self._unusable_groups: dict[str, int] = {}
        # Quarantined uid to the training step that releases it.
        self._quarantined: dict[str, int] = {}
        self._step = 0
        self._retries: collections.deque[dict] = collections.deque()
        self._stale_retries: dict[str, int] = {}
        self._retries_exhausted = 0
        self._scan_start = 0

    def next_prompt(self, live: Collection[str]) -> dict | None:
        """The next prompt whose uid is neither in ``live`` nor quarantined, or None when no such uid remains.

        Prompts awaiting regeneration come first, then up to two batches' worth of the prompt order's draws. If those
        draws find only live or quarantined uids, a scan of the dataset from where the last scan stopped takes the
        first other row, so an order that keeps drawing such rows cannot stall dispatch; the row may be one the order
        weights at zero.
        """
        for position, prompt in enumerate(self._retries):
            if self._available(prompt["uid"], live):
                del self._retries[position]
                return prompt
        # Every usable dataset uid is live, so drawing would only advance the order. A quarantined uid is never live.
        if len(live) + len(self._quarantined) >= self._uid_count:
            return None
        for _ in range(self._draw_limit):
            index = self._order.next_index()
            if self._available(self._dataset.uid(index), live):
                return self._offer(index)
        num_rows = len(self._dataset)
        for offset in range(num_rows):
            index = (self._scan_start + offset) % num_rows
            if self._available(self._dataset.uid(index), live):
                self._scan_start = index + 1
                return self._offer(index)
        return None

    def retry(self, prompt: dict) -> None:
        """Queue a prompt whose group missed the staleness window, unless it used up its regenerations."""
        uid = prompt["uid"]
        count = self._stale_retries.pop(uid, 0)
        if count >= self._max_stale_retries:
            self._retries_exhausted += 1
            return
        self._stale_retries[uid] = count + 1
        self._retries.append(prompt)

    def record_disposition(self, uid: str, disposition: str) -> None:
        """Count a group that left the buffer toward its prompt's quarantine."""
        if disposition in NEUTRAL_DISPOSITIONS:
            return
        if disposition not in UNUSABLE_GROUP_REJECTIONS:
            self._unusable_groups.pop(uid, None)
            return
        count = self._unusable_groups.get(uid, 0) + 1
        if count < self._max_unusable_groups:
            self._unusable_groups[uid] = count
            return
        self._unusable_groups.pop(uid, None)
        usable = self._uid_count - len(self._quarantined)
        if usable <= self._batch_size:
            logger.warning(
                "Prompt {} produced {} unusable groups in a row (last: {}) and stays in rotation: a batch of {} needs "
                "all {} usable prompts",
                self._log_name(uid),
                count,
                disposition,
                self._batch_size,
                usable,
            )
            return
        self._quarantined[uid] = self._step + QUARANTINE_STEPS
        logger.warning(
            "Quarantined prompt {} for {} steps after {} unusable groups in a row (last: {}); {} usable prompts remain",
            self._log_name(uid),
            QUARANTINE_STEPS,
            count,
            disposition,
            usable - 1,
        )

    def observe(self, groups: Sequence[JudgedGroup]) -> dict[str, float]:
        """Report one training step's judged groups to the prompt order and return its metrics.

        Each call is one training step: prompts whose quarantine has run out become available again. The metrics
        include how many stale prompts used up their regenerations since the previous call and how many prompts are
        quarantined.
        """
        self._step += 1
        self._quarantined = {uid: release for uid, release in self._quarantined.items() if release > self._step}
        exhausted, self._retries_exhausted = self._retries_exhausted, 0
        return {
            **self._order.observe(groups),
            STALE_RETRIES_EXHAUSTED_METRIC: float(exhausted),
            QUARANTINED_PROMPTS_METRIC: float(len(self._quarantined)),
        }

    def state_dict(self) -> PromptLoaderState:
        return PromptLoaderState(self._order.state_dict(), list(self._retries))

    def load_state_dict(self, state: PromptLoaderState) -> None:
        self._order.load_state_dict(state.order)
        self._retries = collections.deque(state.retries)

    def _log_name(self, uid: str) -> str:
        label = self._dataset.label(self._uid_rows[uid])
        return uid if label == uid else f"{uid} ({label})"

    def _available(self, uid: str, live: Collection[str]) -> bool:
        return uid not in live and uid not in self._quarantined

    def _offer(self, index: int) -> dict:
        (prompt,) = self._dataset.collate_fn([self._dataset[index]])
        self._stale_retries.pop(prompt["uid"], None)
        return prompt
