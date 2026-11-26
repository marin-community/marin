"""The training dataset as an endless sequence of entries."""

from __future__ import annotations

import collections
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

import torch
from loguru import logger

from skyrl_train.config.rollout_validation import FAMILY_COLUMN
from skyrl_train.group_admission import AdmissionRejection

STALE_RETRIES_EXHAUSTED_METRIC = "async/stale_retries_exhausted"
QUARANTINED_ENTRIES_METRIC = "async/quarantined_prompts"
# Training steps a quarantined entry sits out before the selector offers it again.
QUARANTINE_STEPS = 10
# Rejections of a group training cannot use: no loss tokens, or too few graded samples for its baseline. A broken
# grader causes them, and so does an infrastructure failure, whose samples are masked.
UNUSABLE_GROUP_REJECTIONS = frozenset(
    {AdmissionRejection.FULLY_MASKED.value, AdmissionRejection.BELOW_MINIMUM_GROUP_SIZE.value}
)
# Dispositions that say nothing about whether the entry's groups are usable.
NEUTRAL_DISPOSITIONS = frozenset({AdmissionRejection.STALE.value, AdmissionRejection.DUPLICATE_UID.value})


class RolloutDataset(Protocol):
    def __len__(self) -> int: ...

    def __getitem__(self, index: int) -> Any: ...

    def uid(self, index: int) -> str:
        """The uid of the entry at row ``index``, without loading the row."""
        ...

    def label(self, index: int) -> str:
        """A name for row ``index`` that identifies its source in logs, such as a task id; the uid when it has none."""
        ...

    def collate_fn(self, items: list[Any]) -> list[dict]: ...


@dataclass(frozen=True)
class JudgedGroup:
    """A rollout group that batch selection judged in one step, kept or discarded, with its per-sample rewards.

    ``uid`` is the dataset row the group was generated from.
    """

    uid: str
    rewards: tuple[float, ...]


class DatasetEntryOrder(Protocol):
    """The dataset rows the selector offers, in order; an adaptive order learns from each step's judged groups."""

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

    def for_pool(self, num_rows: int) -> SeededPasses:
        """Start an independent order with the same seed and shuffle policy."""
        return SeededPasses(num_rows, seed=self._seed, shuffle=self._shuffle)

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
class DatasetEntrySelectorState:
    """The entry order's position and the entries awaiting regeneration."""

    order: dict[str, Any]
    retries: list[dict]


# Old torch.save checkpoints resolve this name when unpickling selector state.
PromptLoaderState = DatasetEntrySelectorState


class DatasetEntrySelector:
    """Yield one entry at a time, re-offering entries whose rollouts must be regenerated first.

    A batch trains at most one group per uid, so the selector never offers a uid that is live: already generating,
    or submitted and not yet trained or discarded. A draw whose uid is live is dropped rather than generated as a
    copy the batch would reject.

    The selector re-offers an entry whose group missed the staleness window up to ``max_stale_retries`` times in
    a row. After that the entry returns to its usual selection order.

    An entry whose last ``max_unusable_groups`` groups were all unusable, for example because its grader failed on
    every sample, is quarantined: the selector does not offer it for the next ``QUARANTINE_STEPS`` training steps, then
    offers it again with a fresh count. The same rejections follow an infrastructure failure, so an entry must be
    able to return. Quarantine keeps enough eligible entries to fill the open batches and one spare in each pool.
    Retry counts and quarantine restart empty when a run resumes.

    The sequence never ends; the trainer decides when to stop.
    """

    def __init__(
        self,
        dataset: RolloutDataset,
        order: DatasetEntryOrder,
        *,
        batch_size: int,
        minimum_pool_size: int,
        max_stale_retries: int,
        max_unusable_groups: int,
        families: Mapping[str, str] | None = None,
        family_minimums: Mapping[str, int] | None = None,
    ):
        if max_stale_retries < 0:
            raise ValueError(f"max_stale_retries must be non-negative, got {max_stale_retries}")
        if max_unusable_groups < 1:
            raise ValueError(f"max_unusable_groups must be positive, got {max_unusable_groups}")
        self._uid_rows = {dataset.uid(index): index for index in reversed(range(len(dataset)))}
        self._uid_count = len(self._uid_rows)
        if self._uid_count < minimum_pool_size:
            raise ValueError(
                f"the training dataset has {self._uid_count} distinct entries and needs {minimum_pool_size} "
                "to fill the open batches and keep one spare"
            )
        self._dataset = dataset
        self._families = dict(families or {})
        self._family_minimums = dict(family_minimums or {})
        self._order = order
        self._family_rows: dict[str, list[int]] = collections.defaultdict(list)
        for uid in sorted(self._families, key=self._uid_rows.__getitem__):
            self._family_rows[self._families[uid]].append(self._uid_rows[uid])
        self._family_orders = {}
        if self._family_rows:
            if not isinstance(order, SeededPasses):
                raise ValueError("family entry pools require seeded passes")
            self._family_orders = {family: order.for_pool(len(rows)) for family, rows in self._family_rows.items()}
        self._minimum_pool_size = minimum_pool_size
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

    def next_entry(self, live: Collection[str], family: str | None = None) -> dict | None:
        """Offer an entry in the requested family whose uid is neither live nor quarantined.

        Entries awaiting regeneration come first. A family lease advances only that family's seeded pass. Without
        a family lease, two batches' worth of draws precede a scan that can pick a row the order weights at zero.
        """

        def eligible(uid: str) -> bool:
            return self._available(uid, live) and (family is None or self._families.get(uid) == family)

        for position, entry in enumerate(self._retries):
            if eligible(entry["uid"]):
                del self._retries[position]
                return self._label_family(entry)
        if family is not None:
            return self._next_family_entry(live, family)
        # Every usable dataset uid is live, so drawing would only advance the order. A quarantined uid is never live.
        if len(live) + len(self._quarantined) >= self._uid_count:
            return None
        for _ in range(self._draw_limit):
            index = self._order.next_index()
            if eligible(self._dataset.uid(index)):
                return self._offer(index)
        num_rows = len(self._dataset)
        for offset in range(num_rows):
            index = (self._scan_start + offset) % num_rows
            if eligible(self._dataset.uid(index)):
                self._scan_start = index + 1
                return self._offer(index)
        return None

    def _next_family_entry(self, live: Collection[str], family: str) -> dict | None:
        rows = self._family_rows.get(family)
        if not rows:
            return None
        if len(live) + len(self._quarantined) >= len(rows) and all(
            not self._available(self._dataset.uid(index), live) for index in rows
        ):
            return None
        order = self._family_orders[family]
        # At most the tail of the current pass and one complete pass can contain an available row.
        for _ in range(2 * len(rows) - 1):
            index = rows[order.next_index()]
            if self._available(self._dataset.uid(index), live):
                return self._offer(index)
        return None

    def retry(self, entry: dict) -> None:
        """Queue an entry whose group missed the staleness window, unless it used up its regenerations."""
        uid = entry["uid"]
        count = self._stale_retries.pop(uid, 0)
        if count >= self._max_stale_retries:
            self._retries_exhausted += 1
            return
        self._stale_retries[uid] = count + 1
        self._retries.append(entry)

    def record_disposition(self, uid: str, disposition: str) -> None:
        """Count a group that left the buffer toward its entry's quarantine."""
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
        family = self._families.get(uid)
        if family is not None and family in self._family_minimums:
            minimum = self._family_minimums[family]
            eligible = len(self._family_rows[family]) - sum(
                self._families.get(uid) == family for uid in self._quarantined
            )
            scope = f" in family {family!r}"
        else:
            minimum = self._minimum_pool_size
            eligible = self._uid_count - len(self._quarantined)
            scope = ""
        if eligible <= minimum:
            logger.warning(
                "Entry {} produced {} unusable groups in a row (last: {}) and stays in rotation: "
                "requires {} eligible entries{}",
                self._log_name(uid),
                count,
                disposition,
                minimum,
                scope,
            )
            return
        self._quarantined[uid] = self._step + QUARANTINE_STEPS
        logger.warning(
            "Quarantined entry {} for {} steps after {} unusable groups in a row (last: {}). "
            "{} eligible entries remain{}",
            self._log_name(uid),
            QUARANTINE_STEPS,
            count,
            disposition,
            eligible - 1,
            scope,
        )

    def observe(self, groups: Sequence[JudgedGroup]) -> dict[str, float]:
        """Report one training step's judged groups to the entry order and return its metrics.

        Each call is one training step: entries whose quarantine has run out become available again. The metrics
        include how many stale entries used up their regenerations since the previous call and how many entries are
        quarantined.
        """
        self._step += 1
        self._quarantined = {uid: release for uid, release in self._quarantined.items() if release > self._step}
        exhausted, self._retries_exhausted = self._retries_exhausted, 0
        return {
            **self._order.observe(groups),
            STALE_RETRIES_EXHAUSTED_METRIC: float(exhausted),
            QUARANTINED_ENTRIES_METRIC: float(len(self._quarantined)),
        }

    def state_dict(self) -> DatasetEntrySelectorState:
        order = self._order.state_dict()
        if self._family_orders:
            order["family_orders"] = {family: passes.state_dict() for family, passes in self._family_orders.items()}
        return DatasetEntrySelectorState(order, list(self._retries))

    def load_state_dict(self, state: DatasetEntrySelectorState) -> None:
        self._order.load_state_dict(state.order)
        saved_families = state.order.get("family_orders", {})
        for family, order in self._family_orders.items():
            if family in saved_families:
                order.load_state_dict(saved_families[family])
        self._retries = collections.deque(state.retries)

    def _log_name(self, uid: str) -> str:
        label = self._dataset.label(self._uid_rows[uid])
        return uid if label == uid else f"{uid} ({label})"

    def _available(self, uid: str, live: Collection[str]) -> bool:
        return uid not in live and uid not in self._quarantined

    def _offer(self, index: int) -> dict:
        (entry,) = self._dataset.collate_fn([self._dataset[index]])
        self._stale_retries.pop(entry["uid"], None)
        return self._label_family(entry)

    def _label_family(self, entry: dict) -> dict:
        return {**entry, FAMILY_COLUMN: self._families[entry["uid"]]} if self._families else entry
