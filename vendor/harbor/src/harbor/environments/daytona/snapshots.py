"""Daytona snapshot lifecycle: resolve, create, wait, and per-name locking."""

from __future__ import annotations

import asyncio
import os
from enum import Enum
from logging import Logger
from pathlib import Path
from typing import TYPE_CHECKING, Any

from tenacity import retry

from harbor.environments.base import SandboxBuildFailedError
from harbor.environments.daytona.utils import (
    SNAPSHOT_CREATE_RETRY,
    SNAPSHOT_CREATE_WAIT,
    SNAPSHOT_GET_RETRY,
    SNAPSHOT_GET_WAIT,
    is_snapshot_quota_error,
)
from harbor.utils.container_cache import environment_dir_hash_truncated

if TYPE_CHECKING:
    from daytona import AsyncDaytona, CreateSnapshotParams, Image, Resources

try:
    from daytona import AsyncDaytona, CreateSnapshotParams, Image
    from daytona._async.snapshot import SnapshotState

    from harbor.environments.daytona.creation import create_in_subprocess

    _HAS_DAYTONA = True
except ImportError:
    _HAS_DAYTONA = False
    SnapshotState = Any  # type: ignore[misc, assignment]


# Agent runtime tooling baked into EVERY auto-snapshot at BUILD time so the
# per-episode terminus-2 `_attempt_tmux_installation` short-circuits ("already
# installed") instead of re-running apt (or the slow compile-from-source
# fallback) inside every sandbox. This is the fix for the ~401s agent_setup +
# AgentSetupTimeoutError seen across agentic harbor rollouts (RL + eval): the
# snapshot build runs on Daytona's build infra (which reaches apt mirrors),
# whereas the runtime sandbox may be `network_block_all` (allow_internet=False),
# which is what forced the runtime apt down the compile path.
#
# Package names + the non-interactive apt invocation MIRROR
# terminus_2/tmux_session.py::_get_combined_install_command so the baked binary
# satisfies its `tmux -V` probe and it returns early.
# Harmless no-op for non-terminus-2 agents (they never probe for these). Kept
# minimal + idempotent (`apt-get install -y` is a no-op if already present) and
# appended as a trailing layer so the task Dockerfile's own FROM/steps are
# preserved unchanged.
_AGENT_TOOLING_LAYER = (
    "RUN DEBIAN_FRONTEND=noninteractive apt-get update "
    "&& DEBIAN_FRONTEND=noninteractive apt-get install -y tmux "
    "&& rm -rf /var/lib/apt/lists/*"
)


def bake_agent_tooling(image: "Image") -> "Image":
    """Append the agent-tooling tmux install layer to ``image``.

    Applied to EVERY Daytona image built from a task Dockerfile — the auto-
    snapshot build (the warm path) as well as the declarative per-sandbox
    fall-throughs (cap-full fall-through / force_build / non-auto-snapshot) — so
    the terminus-2 tmux probe short-circuits regardless of which build
    path a trial takes. Idempotent (apt no-op if already present) and harmless
    for non-terminus-2 agents.

    Newline-safe append: the Daytona SDK's ``Image.from_dockerfile`` sets
    ``_dockerfile = path.read_text()`` WITHOUT forcing a trailing newline, and
    ``dockerfile_commands`` appends ``"\n".join(cmds) + "\n"`` with NO leading
    separator. So a task Dockerfile whose file lacks a trailing newline would
    concatenate its last line straight into our ``RUN`` layer (e.g.
    ``...numpy==2.2.0RUN DEBIAN_FRONTEND=...``) → a malformed instruction →
    deterministic ``SandboxBuildFailedError`` for that env on every org/rep.
    Passing a leading empty element makes the join emit a separating newline,
    which is byte-harmless (a blank line) when the base already ends in "\n".
    """
    return image.dockerfile_commands(["", _AGENT_TOOLING_LAYER])


class SnapshotPolicy(Enum):
    """How to handle a snapshot in ERROR state after a successful GET."""

    EXPLICIT = "explicit"  # Fail fast (user-provided template names)
    AUTO = "auto"  # Delete and recreate (content-hash auto snapshots)


class _SnapshotNeedsCreate(Exception):
    """Internal: snapshot was in ERROR (AUTO) and was deleted; caller should create."""


class SnapshotQuotaExceeded(Exception):
    """The org's HARD snapshot cap is full, so a new auto-snapshot cannot be minted.

    Raised by ``ensure_auto`` when the shared snapshot does not already exist AND
    creating it is refused for quota reasons. The caller (the DaytonaEnvironment
    start path) catches this and falls through to a DECLARATIVE Dockerfile build so
    the trial still runs and gets scored — instead of failing it unscored or
    minting yet another per-env snapshot against a cap that won't clear.
    """


class _SnapshotLockRegistry:
    """Per-snapshot asyncio locks, lazily created in the running event loop."""

    _guard: asyncio.Lock | None = None
    _locks: dict[str, asyncio.Lock] = {}

    @classmethod
    async def acquire(cls, snapshot_name: str) -> asyncio.Lock:
        if cls._guard is None:
            cls._guard = asyncio.Lock()
        async with cls._guard:
            if snapshot_name not in cls._locks:
                cls._locks[snapshot_name] = asyncio.Lock()
            return cls._locks[snapshot_name]

    @classmethod
    async def evict_if_idle(cls, snapshot_name: str) -> None:
        if cls._guard is None:
            return
        async with cls._guard:
            lock = cls._locks.get(snapshot_name)
            if lock is not None and not lock.locked():
                cls._locks.pop(snapshot_name, None)


class _CapFullRegistry:
    """Process-wide memo of auto-snapshot names known to be un-mintable because
    the org snapshot cap is full.

    Once ANY trial of an env discovers the org snapshot cap is full (its
    ``snapshot.create`` is refused for quota), every *sibling / subsequent* trial
    of that SAME env (``n_rep_eval > 1`` or concurrent trials of the same task)
    would otherwise re-enter ``ensure_auto`` and re-hit the doomed create — and
    if that quota error surfaces from a path/phrasing that isn't wrapped into
    ``SnapshotQuotaExceeded``, the trial dies UNSCORED. Recording the name here
    lets ``ensure_auto`` short-circuit those siblings straight to the declarative
    build (it raises ``SnapshotQuotaExceeded`` up front), so the fall-through
    covers EVERY trial of a cap-full env, not just the lock-winner — and we stop
    hammering a cap that won't clear within the run.

    Keyed by snapshot name (== per-env), so a cap-full env in one org does not
    poison a different env. Plain ``set`` (idempotent add); no eviction — a cap
    that fills mid-run does not un-fill within the same eval, and the set is at
    most one entry per distinct env.
    """

    _names: set[str] = set()

    @classmethod
    def mark(cls, snapshot_name: str) -> None:
        cls._names.add(snapshot_name)

    @classmethod
    def is_full(cls, snapshot_name: str) -> bool:
        return snapshot_name in cls._names


def mark_snapshot_cap_full(snapshot_name: str) -> None:
    """Record that ``snapshot_name``'s org snapshot cap is full (un-mintable).

    Public entry point so the environment start path can memoize a cap-full env
    even when the quota error surfaced as a RAW Daytona error (not wrapped into
    ``SnapshotQuotaExceeded``), so sibling trials short-circuit to the declarative
    build instead of re-hitting the doomed create.
    """
    _CapFullRegistry.mark(snapshot_name)


class DaytonaSnapshotService:
    """Resolve, create, and wait for Daytona snapshots."""

    def __init__(
        self,
        *,
        logger: Logger,
        environment_dir: Path,
        dockerfile_path: Path,
        api_key: str | None = None,
    ) -> None:
        self._logger = logger
        self._environment_dir = environment_dir
        self._dockerfile_path = dockerfile_path
        self._api_key = api_key

    def environment_hash(self, *, truncate: int = 12) -> str:
        return environment_dir_hash_truncated(self._environment_dir, truncate=truncate)

    def auto_snapshot_name(self) -> str:
        env_hash = self.environment_hash()
        target = os.environ.get("DAYTONA_TARGET")
        if target:
            return f"harbor__{env_hash}__{target}__snapshot"
        return f"harbor__{env_hash}__snapshot"

    async def resolve_template(
        self,
        daytona: AsyncDaytona,
        snapshot_name: str,
        *,
        assume_global_on_missing: bool,
    ) -> str | None:
        """Return snapshot name to use, or None to fall back to a Dockerfile build.

        When GET fails and ``assume_global_on_missing`` is True, returns the name
        optimistically (global snapshots are invisible to GET but work at create).
        """
        try:
            return await self._resolve_existing(
                daytona,
                snapshot_name,
                policy=SnapshotPolicy.EXPLICIT,
            )
        except (SandboxBuildFailedError, TimeoutError):
            raise
        except Exception:
            if assume_global_on_missing:
                self._logger.debug(
                    "snapshot.get(%s) failed; assuming global snapshot",
                    snapshot_name,
                )
                return snapshot_name
            self._logger.warning(
                "snapshot.get(%s) failed and assume_global_snapshot=False; "
                "falling back to declarative build",
                snapshot_name,
            )
            return None

    async def ensure_auto(
        self,
        daytona: AsyncDaytona,
        resources: Resources,
    ) -> str:
        """Ensure an auto-snapshot exists and is active; return its name."""
        snapshot_name = self.auto_snapshot_name()

        # If a prior trial of this same env already proved the org snapshot cap is
        # full, do not re-run the doomed GET/create — short-circuit straight to the
        # quota signal so the caller falls through to a declarative build. This is
        # what makes the fall-through cover EVERY trial of a cap-full env (siblings
        # / n_rep_eval>1), not just the first lock-winner.
        if _CapFullRegistry.is_full(snapshot_name):
            self._logger.warning(
                "Snapshot %s skipped: org snapshot cap was already found full for "
                "this env; falling through to a declarative build (no create "
                "attempted).",
                snapshot_name,
            )
            raise SnapshotQuotaExceeded(
                f"org snapshot cap full (memoized) for {snapshot_name}"
            )

        name = await self._try_resolve_auto(daytona, snapshot_name)
        if name is not None:
            self._logger.debug("Using existing snapshot: %s", snapshot_name)
            return name

        lock = await _SnapshotLockRegistry.acquire(snapshot_name)
        self._logger.debug("Waiting for snapshot creation lock for %s", snapshot_name)

        async with lock:
            try:
                self._logger.debug(
                    "Acquired snapshot creation lock for %s", snapshot_name
                )
                return await self._ensure_auto_under_lock(
                    daytona, snapshot_name, resources
                )
            finally:
                await _SnapshotLockRegistry.evict_if_idle(snapshot_name)

    async def _try_resolve_auto(
        self, daytona: AsyncDaytona, snapshot_name: str
    ) -> str | None:
        """Return snapshot name when ready; None if locked create path is required."""
        try:
            return await self._resolve_existing(
                daytona,
                snapshot_name,
                policy=SnapshotPolicy.AUTO,
            )
        except _SnapshotNeedsCreate:
            return None
        except (SandboxBuildFailedError, TimeoutError):
            raise
        except Exception as e:
            # GET failed. The common, benign case is a genuine 404 (snapshot does
            # not exist yet) -> proceed to the locked create path. But this used to
            # swallow EVERY error silently (throttle / auth / transient 5xx / org
            # mismatch), so a snapshot that actually EXISTS could be missed and a
            # NEW per-env snapshot minted instead of reused. Log it so that brittle
            # fall-through is no longer invisible; the create path still de-dupes
            # via the "already exists" branch + the per-name lock.
            self._logger.debug(
                "snapshot.get(%s) did not resolve (%s: %s); will attempt the "
                "locked create/reuse path",
                snapshot_name,
                type(e).__name__,
                e,
            )
            return None

    async def _ensure_auto_under_lock(
        self,
        daytona: AsyncDaytona,
        snapshot_name: str,
        resources: Resources,
    ) -> str:
        name = await self._try_resolve_auto(daytona, snapshot_name)
        if name is not None:
            return name
        return await self._create_with_retry(daytona, snapshot_name, resources)

    async def _resolve_existing(
        self,
        daytona: AsyncDaytona,
        snapshot_name: str,
        *,
        policy: SnapshotPolicy,
    ) -> str:
        """Handle a snapshot returned by GET (ACTIVE, PENDING, or ERROR)."""
        snapshot = await self._get_with_retry(daytona, snapshot_name)

        if snapshot.state == SnapshotState.ACTIVE:
            return snapshot_name

        if snapshot.state == SnapshotState.PENDING:
            self._logger.debug(
                "Snapshot %s being created by another trial, waiting...",
                snapshot_name,
            )
            return await self._wait_for_active(daytona, snapshot_name)

        if snapshot.state == SnapshotState.ERROR:
            if policy is SnapshotPolicy.EXPLICIT:
                raise SandboxBuildFailedError(
                    f"Snapshot {snapshot_name} is in ERROR state"
                )
            self._logger.warning(
                "Deleting ERROR-state snapshot %s before retry", snapshot_name
            )
            await self._delete_snapshot(daytona, snapshot, snapshot_name)
            raise _SnapshotNeedsCreate

        raise RuntimeError(f"Unexpected snapshot state: {snapshot.state}")

    async def _delete_snapshot(
        self,
        daytona: AsyncDaytona,
        snapshot: Any,
        snapshot_name: str,
    ) -> None:
        try:
            await daytona.snapshot.delete(snapshot)
        except Exception as e:
            self._logger.warning(
                "Failed to delete ERROR snapshot %s: %s", snapshot_name, e
            )

    @retry(retry=SNAPSHOT_GET_RETRY, wait=SNAPSHOT_GET_WAIT, reraise=True)
    async def _get_with_retry(self, daytona: AsyncDaytona, snapshot_name: str) -> Any:
        return await daytona.snapshot.get(snapshot_name)

    async def _get_optional(
        self, daytona: AsyncDaytona, snapshot_name: str
    ) -> Any | None:
        """Best-effort GET without retries (snapshot may not exist)."""
        try:
            return await daytona.snapshot.get(snapshot_name)
        except Exception:
            return None

    @retry(
        retry=SNAPSHOT_CREATE_RETRY,
        wait=SNAPSHOT_CREATE_WAIT,
        reraise=True,
    )
    async def _create_with_retry(
        self,
        daytona: AsyncDaytona,
        snapshot_name: str,
        resources: Resources,
    ) -> str:
        existing = await self._get_optional(daytona, snapshot_name)
        if existing is not None:
            if existing.state == SnapshotState.ACTIVE:
                self._logger.debug(
                    "Snapshot %s already active before create", snapshot_name
                )
                return snapshot_name
            if existing.state == SnapshotState.PENDING:
                self._logger.debug(
                    "Snapshot %s pending before create, waiting...", snapshot_name
                )
                return await self._wait_for_active(daytona, snapshot_name)
            if existing.state == SnapshotState.ERROR:
                self._logger.warning(
                    "Removing stale ERROR snapshot %s before retry", snapshot_name
                )
                await daytona.snapshot.delete(existing)

        self._logger.debug(
            "Creating snapshot: %s (this may take a few minutes)", snapshot_name
        )

        target = os.environ.get("DAYTONA_TARGET")
        try:
            await create_in_subprocess(
                CreateSnapshotParams(
                    name=snapshot_name,
                    image=bake_agent_tooling(
                        Image.from_dockerfile(str(self._dockerfile_path))
                    ),
                    resources=resources,
                ),
                timeout=0,
                target="us" if target and target.upper() == "RL" else None,
                api_key=self._api_key,
            )
        except Exception as e:
            error_msg = str(e).lower()
            if "already exists" in error_msg or "conflict" in error_msg:
                self._logger.debug(
                    "Snapshot %s already exists (global), waiting for active",
                    snapshot_name,
                )
                return await self._wait_for_active(daytona, snapshot_name)
            if is_snapshot_quota_error(e):
                # Org snapshot cap is full and this env's snapshot does not yet
                # exist -> minting is impossible. Surface a distinct, NON-retryable
                # signal so ensure_auto's caller can fall through to a declarative
                # build (run the trial from the Dockerfile) instead of failing it
                # unscored. Retrying here would just hammer a cap that won't clear.
                self._logger.warning(
                    "Snapshot %s could not be created: org snapshot cap is full "
                    "(%s). Will fall through to a declarative build.",
                    snapshot_name,
                    e,
                )
                # Memoize so sibling / subsequent trials of this same env skip the
                # doomed create and fall through to the declarative build too.
                _CapFullRegistry.mark(snapshot_name)
                raise SnapshotQuotaExceeded(str(e)) from e
            raise

        return await self._wait_for_active(daytona, snapshot_name)

    async def _wait_for_active(
        self,
        daytona: AsyncDaytona,
        snapshot_name: str,
        timeout: int = 600,
    ) -> str:
        for _ in range(timeout // 5):
            await asyncio.sleep(5)
            try:
                snapshot = await self._get_with_retry(daytona, snapshot_name)
                if snapshot.state == SnapshotState.ACTIVE:
                    self._logger.debug("Snapshot ready: %s", snapshot_name)
                    return snapshot_name
                if snapshot.state == SnapshotState.ERROR:
                    self._logger.error("Snapshot creation failed: %s", snapshot_name)
                    raise SandboxBuildFailedError(
                        f"Snapshot entered ERROR state: {snapshot_name}"
                    )
            except SandboxBuildFailedError:
                raise
            except Exception as e:
                self._logger.warning("Error checking snapshot: %s", e)

        self._logger.error("Snapshot creation timed out: %s", snapshot_name)
        raise TimeoutError(
            f"Snapshot creation timed out after {timeout}s: {snapshot_name}"
        )
