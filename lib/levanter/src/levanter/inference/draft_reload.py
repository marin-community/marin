# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Stage the online EAGLE trainer's immutable checkpoint before atomic publication."""

import threading
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

import haliax as hax
from rigging.filesystem.factory import url_to_fs

from levanter.compat.hf_checkpoints import load_safetensors_state_dict
from levanter.inference.weight_reload import WeightPublication
from levanter.models.eagle3 import Eagle3Draft
from levanter.trainer import TrainerConfig
from levanter.utils.byte_budget import HostByteBudget


class DraftPublicationContext(Protocol):
    def snapshot_draft(self) -> tuple[Eagle3Draft, int]: ...

    def reload_draft(self, source: Callable[[Eagle3Draft], Eagle3Draft], *, expected_version: int) -> None: ...


@dataclass
class _StagedDraft:
    publication: WeightPublication
    draft: Eagle3Draft
    candidate: Eagle3Draft | None = None


class DraftReloadSession:
    """Accept one complete draft checkpoint; generation continues during staging."""

    def __init__(self, context: DraftPublicationContext, trainer: TrainerConfig, max_checkpoint_bytes: int):
        if max_checkpoint_bytes <= 0:
            raise ValueError("Draft checkpoint byte limit must be positive")
        self.context = context
        self.trainer = trainer
        self.max_checkpoint_bytes = max_checkpoint_bytes
        self.lock = threading.Lock()
        self.staged: _StagedDraft | None = None

    def begin(self) -> WeightPublication:
        with self.lock:
            draft, version = self.context.snapshot_draft()
            publication = WeightPublication(uuid.uuid4().hex, version)
            self.staged = _StagedDraft(publication, draft)
            return publication

    def _current(self, publication: WeightPublication) -> _StagedDraft:
        if self.staged is None or self.staged.publication != publication:
            raise ValueError("Draft publication is no longer active")
        return self.staged

    def receive(self, publication: WeightPublication, weights_path: str) -> None:
        """Read one exact immutable safetensors URI with the shared streaming loader."""
        with self.lock:
            staged = self._current(publication)
            try:
                if staged.candidate is not None:
                    raise ValueError("Draft publication already contains a checkpoint")
                fs, path = url_to_fs(weights_path)
                if fs.size(path) > self.max_checkpoint_bytes:
                    raise ValueError("Draft checkpoint exceeds its configured byte limit")
                with self.trainer.use_device_mesh(), hax.axis_mapping(self.trainer.compute_axis_mapping):
                    tensors = load_safetensors_state_dict(
                        path, fs=fs, staging_budget=HostByteBudget(self.max_checkpoint_bytes)
                    )
                    staged.candidate = staged.draft.with_trainable_state_dict(tensors)
            except Exception:
                self.staged = None
                raise

    def finish(self, publication: WeightPublication) -> None:
        with self.lock:
            staged = self._current(publication)
            self.staged = None
        if staged.candidate is None:
            raise ValueError("Draft publication is incomplete")
        candidate = staged.candidate

        def install(current: Eagle3Draft) -> Eagle3Draft:
            if current is not staged.draft:
                raise ValueError("Draft changed after publication began")
            return candidate

        self.context.reload_draft(install, expected_version=publication.model_version)
