"""The training dataset as an endless sequence of prompts."""

from __future__ import annotations

import collections
from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import dataclass
from functools import partial
from typing import Any, Protocol

import torch
from loguru import logger

from skyrl_train.config.rollout_validation import FAMILY_COLUMN
from skyrl_train.group_admission import AdmissionRejection
from skyrl_train.rollouts.prompt_order import JudgedGroup, PromptOrder, PromptSelection

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


class SeededPasses(PromptOrder):
    """Endless passes over the dataset, each in a fresh seeded order."""

    @property
    def supports_family_leases(self) -> bool:
        return True

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
class PromptEligibilityState:
    """Availability, retry budgets, and fallback scan position."""

    step: int
    quarantined: dict[str, int]
    unusable_groups: dict[str, int]
    stale_retries: dict[str, int]
    retries_exhausted: int
    scan_start: int = 0


@dataclass(frozen=True)
class PromptLoaderState:
    """The prompt order's position and the prompts awaiting regeneration."""

    order: dict[str, Any]
    retries: list[dict]
    eligibility: PromptEligibilityState | None = None


class PromptLoader:
    """Yield one eligible prompt at a time for rollout.

    A batch trains at most one group per uid, so the loader never offers a uid that is live: already generating,
    or committed and not yet trained or discarded. A draw whose uid is live is dropped rather than generated as a
    copy the batch would reject.

    Source passes count each dispatch as a visit, including stale groups. Other orders give stale groups
    up to ``max_stale_retries`` consecutive retries before returning the prompt to regular selection.

    A prompt whose last ``max_unusable_groups`` groups were all unusable, for example because its grader failed on
    every sample, is quarantined: the loader does not offer it for the next ``QUARANTINE_STEPS`` training steps, then
    offers it again with a fresh count. The same rejections follow an infrastructure failure, so a prompt must be
    able to return. Quarantine keeps enough eligible prompts to fill the open batches and one spare per pool.
    Checkpoints preserve availability and retry budgets for every order.

    The sequence never ends; the trainer decides when to stop.
    """

    def __init__(
        self,
        dataset: PromptGroupDataset,
        order: PromptOrder,
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
                f"the training dataset has {self._uid_count} distinct prompts and needs {minimum_pool_size} "
                "to fill the open batches and keep one spare"
            )
        self._dataset = dataset
        self._families = dict(families or {})
        self._family_minimums = dict(family_minimums or {})
        self.order = order
        self._family_rows: dict[str, list[int]] = collections.defaultdict(list)
        for uid in sorted(self._families, key=self._uid_rows.__getitem__):
            self._family_rows[self._families[uid]].append(self._uid_rows[uid])
        if self._family_rows and not order.supports_family_leases:
            raise ValueError("the prompt order does not support family leases")
        self._family_orders = {
            family: pool for family, rows in self._family_rows.items() if (pool := order.for_pool(len(rows))) is not None
        }
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

    def next_prompt(self, live: Collection[str], family: str | None = None) -> dict | None:
        """Offer a prompt in the requested family whose uid is neither live nor quarantined.

        Source passes select among the least-visited eligible prompts in the chosen source. Other orders
        offer queued retries first, then draw from the leased family or the full dataset.
        """

        def available(uid: str) -> bool:
            return uid not in live and uid not in self._quarantined
        if self.order.retry_stale:
            for position, prompt in enumerate(self._retries):
                uid = prompt["uid"]
                if available(uid) and (family is None or self._families.get(uid) == family):
                    del self._retries[position]
                    return self._label_family(prompt)

        family_rows = self._family_rows.get(family, ()) if family is not None else None
        if family_rows is not None and not family_rows:
            return None
        rows = range(len(self._dataset)) if family_rows is None else family_rows

        def eligible(row: int) -> bool:
            return available(self._dataset.uid(row))

        selector = self._family_orders.get(family, self.order) if family is not None else self.order
        row = selector.select_index(
            PromptSelection(
                family,
                rows,
                live,
                self._quarantined.keys(),
                eligible,
                partial(self._draw_index, eligible, len(live) + len(self._quarantined), family_rows),
            )
        )
        return self._offer(row) if row is not None else None

    def _draw_index(
        self,
        eligible: Callable[[int], bool],
        excluded_count: int,
        family_rows: Sequence[int] | None,
        next_index: Callable[[], int],
    ) -> int | None:
        if family_rows is not None:
            if excluded_count >= len(family_rows) and not any(map(eligible, family_rows)):
                return None
            # The rest of this pass and one complete pass cover every family row.
            for _ in range(2 * len(family_rows) - 1):
                index = family_rows[next_index()]
                if eligible(index):
                    return index
            return None
        if excluded_count >= self._uid_count:
            return None
        for _ in range(self._draw_limit):
            index = next_index()
            if eligible(index):
                return index
        num_rows = len(self._dataset)
        for offset in range(num_rows):
            index = (self._scan_start + offset) % num_rows
            if eligible(index):
                self._scan_start = index + 1
                return index
        return None

    def retry(self, prompt: dict) -> None:
        """Queue a stale prompt for a bounded retry when its order allows immediate regeneration."""
        if not self.order.retry_stale:
            # The stale dispatch already visited this prompt. It becomes eligible again through the pass.
            return
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
                "Prompt {} produced {} unusable groups in a row (last: {}) and stays in rotation: "
                "requires {} eligible prompts{}",
                self._log_name(uid),
                count,
                disposition,
                minimum,
                scope,
            )
            return
        self._quarantined[uid] = self._step + QUARANTINE_STEPS
        logger.warning(
            "Quarantined prompt {} for {} steps after {} unusable groups in a row (last: {}). "
            "{} eligible prompts remain{}",
            self._log_name(uid),
            QUARANTINE_STEPS,
            count,
            disposition,
            eligible - 1,
            scope,
        )

    def observe(self, groups: Sequence[JudgedGroup]) -> dict[str, float]:
        """Advance quarantine and notify any sampler that learns at admission.

        Each call is one training step: prompts whose quarantine has run out become available again. The metrics
        include how many stale prompts used up their regenerations since the previous call and how many prompts are
        quarantined.
        """
        self._step += 1
        self._quarantined = {uid: release for uid, release in self._quarantined.items() if release > self._step}
        exhausted, self._retries_exhausted = self._retries_exhausted, 0
        return {
            **self.order.observe(groups),
            STALE_RETRIES_EXHAUSTED_METRIC: float(exhausted),
            QUARANTINED_PROMPTS_METRIC: float(len(self._quarantined)),
        }

    def state_dict(self) -> PromptLoaderState:
        order = self.order.state_dict()
        if self._family_orders:
            order["family_orders"] = {family: passes.state_dict() for family, passes in self._family_orders.items()}
        eligibility = PromptEligibilityState(
            self._step,
            self._quarantined.copy(),
            self._unusable_groups.copy(),
            self._stale_retries.copy(),
            self._retries_exhausted,
            self._scan_start,
        )
        return PromptLoaderState(order, list(self._retries), eligibility)

    def load_state_dict(self, state: PromptLoaderState) -> None:
        self.order.load_state_dict(state.order)
        saved_families = state.order.get("family_orders", {})
        for family, order in self._family_orders.items():
            if family in saved_families:
                order.load_state_dict(saved_families[family])
        self._retries = collections.deque(state.retries if self.order.retry_stale else ())
        # Older loaders saved queued prompts without the eligibility counters.
        eligibility = state.eligibility or PromptEligibilityState(0, {}, {}, {}, 0)
        self._step = eligibility.step
        self._quarantined = eligibility.quarantined.copy()
        self._unusable_groups = eligibility.unusable_groups.copy()
        self._stale_retries = eligibility.stale_retries.copy()
        self._retries_exhausted = eligibility.retries_exhausted
        self._scan_start = eligibility.scan_start

    def _log_name(self, uid: str) -> str:
        label = self._dataset.label(self._uid_rows[uid])
        return uid if label == uid else f"{uid} ({label})"

    def _offer(self, index: int) -> dict:
        (prompt,) = self._dataset.collate_fn([self._dataset[index]])
        self._stale_retries.pop(prompt["uid"], None)
        return self._label_family(prompt)

    def _label_family(self, prompt: dict) -> dict:
        return {**prompt, FAMILY_COLUMN: self._families[prompt["uid"]]} if self._families else prompt
