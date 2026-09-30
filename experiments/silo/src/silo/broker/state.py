# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Where the broker keeps what a restart must not lose.

A broker restart used to drop every snapshot (2026-09-29 05:26Z): snapshots
existed only in memory, and recipes built ``FROM silo.local/snapshots/...:built``
could not be rebuilt afterwards. The broker now writes its snapshot index and
host registry to one JSON document and reads it back on start.

The location is a URL. A plain path is written atomically on local disk (tests,
or a single-node deployment); anything with a scheme goes through fsspec, e.g.
``s3://marin-us-east-02a/...`` -- Iris task pods on CoreWeave carry object-storage
credentials (the ``iris-task-env`` secret), so no extra secret is needed there.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import threading
from pathlib import Path
from typing import Any, Protocol

logger = logging.getLogger(__name__)


class StateStore(Protocol):
    url: str

    def load(self) -> dict[str, Any] | None:
        """The saved state, or None if nothing was ever saved. Raises if unreadable."""
        ...

    def save(self, state: dict[str, Any]) -> None: ...


class FileStateStore:
    def __init__(self, path: str | os.PathLike[str]) -> None:
        self.path = Path(path)
        self.url = str(self.path)

    def load(self) -> dict[str, Any] | None:
        if not self.path.exists():
            return None
        return json.loads(self.path.read_text())

    def save(self, state: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=self.path.parent, prefix=f".{self.path.name}.")
        try:
            with os.fdopen(fd, "w") as handle:
                json.dump(state, handle)
            os.replace(tmp, self.path)
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise


class FsspecStateStore:
    """Object storage. Written whole: object stores replace a key atomically."""

    def __init__(self, url: str) -> None:
        self.url = url
        if url.startswith(("s3://", "s3a://")):
            try:
                # Outside a CoreWeave pod (a dev box) this configures s3:// from
                # CW_KEY_*; inside one it is a no-op and the pod's config wins.
                from rigging.filesystem.s3_compat import configure_coreweave_s3  # noqa: PLC0415 - optional dep

                configure_coreweave_s3()
            except Exception:  # rigging absent or unconfigured: fsspec's own config applies
                logger.debug("rigging s3 configuration unavailable", exc_info=True)
        # Optional: marin-silo does not depend on fsspec; the broker's venv has it via marin-iris.
        import fsspec  # noqa: PLC0415 - optional dep

        self._fs, self._path = fsspec.core.url_to_fs(url)

    def load(self) -> dict[str, Any] | None:
        if not self._fs.exists(self._path):
            return None
        with self._fs.open(self._path, "r") as handle:
            return json.load(handle)

    def save(self, state: dict[str, Any]) -> None:
        with self._fs.open(self._path, "w") as handle:
            json.dump(state, handle)


def open_state_store(url: str | None) -> StateStore | None:
    if not url:
        return None
    if "://" not in url or url.startswith("file://"):
        return FileStateStore(url.removeprefix("file://"))
    return FsspecStateStore(url)


class StatePersister:
    """Saves the broker's state whenever it changed, at most every ``interval``.

    If the saved state could not be READ at startup, saving is refused for the
    whole run: writing an empty index over one we merely failed to read would
    destroy exactly what this exists to keep.
    """

    def __init__(self, broker: Any, store: StateStore, *, writable: bool = True) -> None:
        self._broker = broker
        self._store = store
        self._writable = writable
        self._lock = threading.Lock()

    def save_if_changed(self) -> bool:
        if not self._writable:
            return False
        with self._lock:
            exported = self._broker.export_state()
            if exported is None:
                return False
            generation, state = exported
            self._store.save(state)
            self._broker.mark_saved(generation)
            return True


def restore(broker: Any, store: StateStore) -> tuple[bool, dict[str, int] | None]:
    """Load saved state into ``broker``. Returns ``(writable, counts)``.

    ``writable`` is False when a state document exists (or may exist) but could
    not be read or applied -- the caller must then not overwrite it.
    """
    try:
        state = store.load()
    except Exception:
        logger.exception("could NOT read broker state from %s; serving without it and NOT saving over it", store.url)
        return False, None
    if state is None:
        logger.warning("no broker state at %s yet; starting empty (it will be written there)", store.url)
        return True, None
    try:
        counts = broker.restore_state(state)
    except Exception:
        logger.exception("could NOT apply broker state from %s; serving without it and NOT saving over it", store.url)
        return False, None
    logger.info("restored broker state from %s (saved_at=%s): %s", store.url, state.get("saved_at"), counts)
    return True, counts
