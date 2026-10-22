"""The training dataset as an endless sequence of prompts."""

from __future__ import annotations

import collections
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

import torch


class PromptGroupDataset(Protocol):
    def __len__(self) -> int: ...

    def __getitem__(self, index: int) -> Any: ...

    def uid(self, index: int) -> str:
        """The uid of row ``index``'s prompt group, without loading the row."""
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

    The sequence never ends; the trainer decides when to stop.
    """

    def __init__(self, dataset: PromptGroupDataset, order: PromptOrder, *, batch_size: int):
        self._uid_count = len({dataset.uid(index) for index in range(len(dataset))})
        if self._uid_count < batch_size:
            raise ValueError(
                f"the training dataset has {self._uid_count} distinct prompt uids, fewer than one batch of "
                f"{batch_size}; a batch trains each uid at most once"
            )
        self._dataset = dataset
        self._order = order
        self._draw_limit = 2 * batch_size
        self._retries: collections.deque[dict] = collections.deque()
        self._scan_start = 0

    def next_prompt(self, live: Collection[str]) -> dict | None:
        """The next prompt whose uid is not in ``live``, or None when every uid is live.

        Prompts awaiting regeneration come first, then up to two batches' worth of the prompt order's draws. If those
        draws find only live uids, a scan of the dataset from where the last scan stopped takes a free row, so an
        order that keeps drawing live rows cannot stall dispatch; the row may be one the order weights at zero.
        """
        for position, prompt in enumerate(self._retries):
            if prompt["uid"] not in live:
                del self._retries[position]
                return prompt
        # Every dataset uid is live, so drawing would only advance the order.
        if len(live) >= self._uid_count:
            return None
        for _ in range(self._draw_limit):
            index = self._order.next_index()
            if self._dataset.uid(index) not in live:
                return self._group(index)
        num_rows = len(self._dataset)
        for offset in range(num_rows):
            index = (self._scan_start + offset) % num_rows
            if self._dataset.uid(index) not in live:
                self._scan_start = index + 1
                return self._group(index)
        return None

    def retry(self, prompt: dict) -> None:
        self._retries.append(prompt)

    def observe(self, groups: Sequence[JudgedGroup]) -> dict[str, float]:
        """Report one training step's judged groups to the prompt order and return its metrics."""
        return self._order.observe(groups)

    def state_dict(self) -> PromptLoaderState:
        return PromptLoaderState(self._order.state_dict(), list(self._retries))

    def load_state_dict(self, state: PromptLoaderState) -> None:
        self._order.load_state_dict(state.order)
        self._retries = collections.deque(state.retries)

    def _group(self, index: int) -> dict:
        (prompt,) = self._dataset.collate_fn([self._dataset[index]])
        return prompt
