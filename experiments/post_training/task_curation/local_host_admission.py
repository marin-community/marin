# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Campaign-wide LOCAL trial slots shared across pod mount namespaces."""

import json
import logging
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field
from rigging.filesystem.conditional_object import ConditionalObject, ConditionalWriteError, conditional_object
from rigging.filesystem.storage_path import StoragePath
from rigging.timing import ExponentialBackoff
from taskcompendium.identity import canonical_sha256

logger = logging.getLogger(__name__)


class LocalHostAdmissionConfig(BaseModel):
    """One immutable shared contract used by every participating LOCAL worker."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    control_root: str = Field(pattern=r"^s3://")
    max_trials_per_host: int = Field(ge=1)


class SlotStatus(StrEnum):
    HELD = "held"
    RELEASED = "released"


class HostSlotRecord(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    status: SlotStatus
    claim_id: str
    worker_id: str
    task_id: str


@dataclass(frozen=True)
class LocalHostSlot:
    control: ConditionalObject
    version: str
    record: HostSlotRecord

    def release(self) -> None:
        """Release exactly this claim, retaining its identity in the control record."""
        released = self.record.model_copy(update={"status": SlotStatus.RELEASED})
        self.control.write(released.model_dump_json().encode(), expected_version=self.version)


@dataclass(frozen=True)
class LocalHostAdmission:
    config: LocalHostAdmissionConfig
    boot_id: UUID
    node_id: str
    worker_id: str
    object_factory: Callable[[str], ConditionalObject] = conditional_object

    @classmethod
    def current(cls, config: LocalHostAdmissionConfig, node_id: str, worker_id: str) -> "LocalHostAdmission":
        """Resolve the physical kernel identity once at the worker boundary."""
        boot_id = UUID(Path("/proc/sys/kernel/random/boot_id").read_text().strip())
        return cls(config, boot_id, node_id, worker_id)

    def control_path(self, name: str) -> str:
        return str(StoragePath(self.config.control_root) / canonical_sha256(self.node_id) / name)

    def require_host_contract(self) -> None:
        control = self.object_factory(self.control_path("limits.json"))
        expected = {
            "node_id": self.node_id,
            "boot_id": str(self.boot_id),
            "max_trials_per_host": self.config.max_trials_per_host,
        }
        current = control.read()
        if current is None:
            try:
                control.write(json.dumps(expected).encode(), expected_version=None)
                return
            except ConditionalWriteError:
                current = control.read()
        if current is None or json.loads(current.data) != expected:
            raise ValueError("LOCAL workers disagree on the immutable physical-host admission contract")

    def try_claim(self, task_id: str) -> LocalHostSlot | None:
        """Atomically claim one slot; held claims never expire or get stolen."""
        self.require_host_contract()
        for index in range(self.config.max_trials_per_host):
            control = self.object_factory(self.control_path(f"slot-{index:04d}.json"))
            current = control.read()
            if current is not None and HostSlotRecord.model_validate_json(current.data).status is SlotStatus.HELD:
                continue
            record = HostSlotRecord(
                status=SlotStatus.HELD, claim_id=str(uuid4()), worker_id=self.worker_id, task_id=task_id
            )
            try:
                version = control.write(
                    record.model_dump_json().encode(), expected_version=current.version if current else None
                )
            except ConditionalWriteError:
                continue
            return LocalHostSlot(control, version, record)
        return None

    @contextmanager
    def claim(
        self,
        task_id: str,
        *,
        deadline: float,
        now: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> Iterator[LocalHostSlot]:
        """Wait before dispatch; the caller releases only after confirmed resource cleanup."""
        backoff = ExponentialBackoff(initial=1, maximum=120, factor=2, jitter=0.5)
        while now() < deadline:
            slot = self.try_claim(task_id)
            if slot is not None:
                if now() >= deadline:
                    slot.release()
                    raise TimeoutError("LOCAL host admission deadline elapsed before source dispatch")
                yield slot
                return
            delay = min(backoff.next_interval(), max(0, deadline - now()))
            logger.info("LOCAL host admission waiting: boot=%s worker=%s task=%s", self.boot_id, self.worker_id, task_id)
            sleep(delay)
        raise TimeoutError("LOCAL host admission deadline elapsed before source dispatch")
