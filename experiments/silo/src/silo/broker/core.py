# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Broker: the snapshot index, host registry, placement, and the named ceiling.

The broker is control plane only. It decides where a sandbox runs and hands the
caller that host's address plus a per-sandbox capability; exec, sessions and files
then go straight to the host. So broker load scales with creates and deletes, not
with the poll loops of hundreds of running builds.

It is also where the ceiling becomes visible. Daytona's 40-snapshot quota was
discovered by stalling (49 polls, 2026-09-22). Here snapshots are metadata with
no quota at all, sandbox capacity is a number ``/capacity`` reports, and a create
that must wait for capacity says so in the log while it waits.

Host liveness has three states (``HostHealth``). A host that misses heartbeats is
SUSPECT: nothing new is placed on it, and a request for one of its sandboxes is a
retryable 503 -- never a not-found, because the sandbox is most likely still
running. Only a long silence, or a refused connection, makes a host DEAD.

A host that keeps failing creates, or whose report is provably wrong (negative
allocation: the accounting drift of 2026-09-29), is logged. A failing host is
QUARANTINED for placement; a drifted one stays placeable against a conservative
estimate of its allocation. Existing sandboxes stay routable either way.
"""

from __future__ import annotations

import dataclasses
import logging
import math
import os
import threading
import time
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, ClassVar, Protocol

from silo.auth import sandbox_capability
from silo.errors import (
    NETWORK_REFUSAL,
    SiloConflictError,
    SiloError,
    SiloNotFoundError,
    SiloRateLimitError,
    SiloRecipeError,
)
from silo.http import TransportError
from silo.model import (
    CANDIDATE_DEFAULT,
    SNAPSHOT_ACTIVE,
    SNAPSHOT_ERROR,
    VERIFIER_DEFAULT,
    ImagePlan,
    ResourceProfile,
    SnapshotRecord,
    build_dockerfile,
    new_sandbox_id,
    parse_recipe,
    utcnow,
)

logger = logging.getLogger(__name__)

PLACEMENT_POLL_SECONDS = 2.0
# Bounded well under the 600 s the pipeline passes to create(); past this a
# create fails with a rate-limit error the pipeline already knows how to handle.
PLACEMENT_WAIT_SECONDS = 480.0
# While creates wait for room, hosts are asked directly -- at most this often --
# instead of waiting up to a heartbeat interval for capacity news.
REFRESH_MIN_INTERVAL_SECONDS = 1.0
# A placement the broker made but no host report has confirmed yet. Past this age
# any report would have included it, so an entry this old is dropped rather than
# leaking capacity out of the broker's view forever.
UNCONFIRMED_TTL_SECONDS = 60.0
# How long a 503 tells the caller to wait before retrying.
RETRY_AFTER_SECONDS = 5
# Built snapshot images live only in the containerd of the host that built them.
LOCAL_IMAGE_PREFIX = "silo.local/"
STATE_VERSION = 1


class PlaceAgainLater(Exception):
    """A host failed a create the way a sick host does. Pause, then place again.

    Raised by ``attempt_create``; the create loops catch it. Distinct from a
    ``None`` result (host full or gone: place again at once) so that a host
    failing fast cannot turn a waiting create into a hot loop.
    """

    def __init__(self, host_id: str, reason: str) -> None:
        super().__init__(f"host {host_id}: {reason}")
        self.host_id = host_id
        self.reason = reason


class HostHealth:
    LIVE = "live"
    SUSPECT = "suspect"
    DEAD = "dead"


def _env_float(environ: Mapping[str, str], name: str, default: float) -> float:
    raw = environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    value = float(raw)  # a malformed value fails startup loudly
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name}={raw!r} must be a positive number")
    return value


@dataclass(frozen=True)
class BrokerConfig:
    """Liveness and placement-safety knobs, each overridable by environment.

    ``heartbeat_seconds`` / ``heartbeat_timeout_seconds`` mirror the host's own
    settings (``host/main.py`` reads the same variables) so the broker can check
    that its windows comfortably exceed one heartbeat cycle. Each host also sends
    its actual values, and a mismatch is logged per host.
    """

    heartbeat_seconds: float = 10.0
    heartbeat_timeout_seconds: float = 15.0
    # No placement after this much silence; sandboxes answer 503, not not-found.
    suspect_after_seconds: float = 60.0
    # Only after this much silence are the host's sandboxes presumed gone.
    dead_after_seconds: float = 600.0
    # A sick host can pin at most this many create calls (and broker threads).
    max_inflight_creates_per_host: int = 8
    quarantine_after_failures: int = 3
    quarantine_seconds: float = 300.0
    quarantine_max_seconds: float = 3600.0
    # Never quarantine more than this share of live hosts for failures: past
    # it the problem is not a host, and idle capacity is the worse error.
    quarantine_max_fraction: float = 0.25

    ENV: ClassVar[dict[str, str]] = {
        "heartbeat_seconds": "SILO_HEARTBEAT_SECONDS",
        "heartbeat_timeout_seconds": "SILO_HEARTBEAT_TIMEOUT_SECONDS",
        "suspect_after_seconds": "SILO_HOST_SUSPECT_AFTER_SECONDS",
        "dead_after_seconds": "SILO_HOST_DEAD_AFTER_SECONDS",
        "max_inflight_creates_per_host": "SILO_HOST_MAX_INFLIGHT_CREATES",
        "quarantine_after_failures": "SILO_HOST_QUARANTINE_AFTER_FAILURES",
        "quarantine_seconds": "SILO_HOST_QUARANTINE_SECONDS",
        "quarantine_max_seconds": "SILO_HOST_QUARANTINE_MAX_SECONDS",
        "quarantine_max_fraction": "SILO_HOST_QUARANTINE_MAX_FRACTION",
    }

    @property
    def heartbeat_cycle_seconds(self) -> float:
        """Worst-case gap between two heartbeats that both reach the broker."""
        return self.heartbeat_seconds + self.heartbeat_timeout_seconds

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> tuple[BrokerConfig, list[str]]:
        """The config, plus WARNING lines for any value that had to be raised.

        A window shorter than two heartbeat cycles would make healthy hosts flap,
        so it is raised to that floor rather than obeyed -- loudly, with the
        declared value next to the resolved one.
        """
        environ = os.environ if environ is None else environ
        defaults = cls()
        values: dict[str, Any] = {}
        for attr, name in cls.ENV.items():
            default = getattr(defaults, attr)
            value = _env_float(environ, name, float(default))
            values[attr] = int(value) if isinstance(default, int) else value
        config = cls(**values)
        return config.sanitized()

    def sanitized(self) -> tuple[BrokerConfig, list[str]]:
        warnings: list[str] = []
        floor = 2 * self.heartbeat_cycle_seconds
        suspect = self.suspect_after_seconds
        if suspect < floor:
            warnings.append(
                f"SILO_HOST_SUSPECT_AFTER_SECONDS declared={suspect:g} is under 2x the heartbeat cycle "
                f"({self.heartbeat_seconds:g}s period + {self.heartbeat_timeout_seconds:g}s timeout); "
                f"resolved={floor:g}"
            )
            suspect = floor
        dead = self.dead_after_seconds
        if dead < 2 * suspect:
            warnings.append(
                f"SILO_HOST_DEAD_AFTER_SECONDS declared={dead:g} is under 2x the suspect window; "
                f"resolved={2 * suspect:g}"
            )
            dead = 2 * suspect
        fraction = min(max(self.quarantine_max_fraction, 0.0), 1.0)
        return (
            dataclasses.replace(
                self,
                suspect_after_seconds=suspect,
                dead_after_seconds=dead,
                max_inflight_creates_per_host=max(1, self.max_inflight_creates_per_host),
                quarantine_after_failures=max(1, self.quarantine_after_failures),
                quarantine_max_fraction=fraction,
            ),
            warnings,
        )

    def describe(self) -> str:
        return " ".join(f"{name}={getattr(self, attr):g}" for attr, name in self.ENV.items())


class HostClient(Protocol):
    """What the broker needs from a host. Implemented over HTTP in production."""

    def create_sandbox(self, body: Mapping[str, Any]) -> dict[str, Any]: ...

    def get_sandbox(self, sandbox_id: str) -> dict[str, Any]: ...

    def list_sandboxes(self) -> list[dict[str, Any]]: ...

    def delete_sandbox(self, sandbox_id: str) -> None: ...

    def ensure_image(self, ref: str) -> None: ...

    def build_image(self, tag: str, dockerfile: str) -> None: ...

    def capacity(self) -> dict[str, Any]: ...


def report_inconsistency(capacity: Mapping[str, Any]) -> str:
    """Why a host's capacity report cannot be true, or "" if it can.

    A correct host never reports negative allocation, and every live sandbox
    holds at least one core. Hosts with the double-release bug fail both: their
    reports claim far more free room than exists, which made max-free-slots
    placement send them nearly every create (2026-09-29).
    """
    problems = []
    for key in ("cpu", "memory_bytes", "disk_bytes"):
        allocated = (capacity.get(key) or {}).get("allocated")
        if allocated is not None and float(allocated) < 0:
            problems.append(f"{key}.allocated={allocated}")
    live = capacity.get("sandboxes_live")
    cpu_allocated = (capacity.get("cpu") or {}).get("allocated")
    if not problems and live and cpu_allocated is not None and float(cpu_allocated) < int(live):
        problems.append(f"cpu.allocated={cpu_allocated} < sandboxes_live={live}")
    return ", ".join(problems)


def _is_local(ref: str) -> bool:
    return ref.startswith(LOCAL_IMAGE_PREFIX)


def _is_host_fault(error: SiloError) -> bool:
    """A failure that says the host is sick, as opposed to the request being bad.

    Deliberately narrow: a bad image or a failing RUN step must never quarantine
    hosts. A timeout (nerdctl run hanging to its 600 s limit arrives as a 500
    whose message says "timed out") or an explicit 503 does count.
    """
    if isinstance(error, TransportError):
        return not error.refused
    status = error.status_code or 500
    return status == 503 or (status >= 500 and "timed out" in str(error).lower())


@dataclass(eq=False)
class HostState:
    """The broker's view of one host: its last report, plus what it placed since.

    A host report (heartbeat or direct refresh) is the truth as of when it was
    taken. Placements the broker has made that no report reflects yet are held in
    ``unconfirmed`` by sandbox id and counted on top, so concurrent creates cannot
    all see the same free slot -- and, unlike a counter overwritten on the next
    heartbeat, they stop counting exactly when a report includes them, when the
    create fails, or when the sandbox is deleted.
    """

    host_id: str
    url: str
    capacity: dict[str, Any]
    last_seen: float
    client: HostClient
    images: set[str] = field(default_factory=set)
    sandbox_ids: set[str] = field(default_factory=set)
    unconfirmed: dict[str, tuple[ResourceProfile, float]] = field(default_factory=dict)
    # Creates the broker has sent to this host and not yet heard back from.
    inflight: dict[str, float] = field(default_factory=dict)
    builds_inflight: int = 0
    # A refused connection: nothing listens at ``url`` any more.
    gone: bool = False
    consecutive_failures: int = 0
    last_failure: str = ""
    quarantined_until: float = float("-inf")
    quarantine_count: int = 0
    quarantine_reason: str = ""
    # After a quarantine ends: one create at a time until one succeeds.
    probation: bool = False
    # Non-empty while the host's latest report is impossible (see report_inconsistency).
    drift: str = ""
    last_health: str = HostHealth.LIVE
    cycle_warned: bool = False

    def apply_report(self, capacity: dict[str, Any], sandbox_ids: list[str], now: float) -> None:
        self.capacity = capacity
        self.sandbox_ids = set(sandbox_ids)
        self.last_seen = now
        for sandbox_id, (_, placed_at) in list(self.unconfirmed.items()):
            if sandbox_id in self.sandbox_ids or now - placed_at > UNCONFIRMED_TTL_SECONDS:
                del self.unconfirmed[sandbox_id]

    def _pending(self, attr: str) -> float:
        return float(sum(getattr(profile, attr) for profile, _ in self.unconfirmed.values()))

    def free(self, key: str) -> float:
        section = self.capacity.get(key, {})
        budget = float(section.get("budget", 0))
        if key == "cpu":
            budget *= float(section.get("oversubscribe", 1.0))
        # A negative allocation is drift, never free room: it must not make a host
        # look bigger than its budget.
        allocated = max(0.0, float(section.get("allocated", 0)))
        if self.drift:
            # The host's own number is untrustworthy, so count every live sandbox as a
            # full candidate-sized one. Fencing drifted hosts off instead stranded
            # 39 of 42 hosts within 30 minutes (2026-09-30): the double release
            # eventually hits every host that deletes, so drift is the norm.
            live = max(int(self.capacity.get("sandboxes_live") or 0), len(self.sandbox_ids))
            allocated = max(allocated, float(live * getattr(CANDIDATE_DEFAULT, key)))
        return budget - allocated - self._pending(key)

    def slots_for(self, profile: ResourceProfile) -> int:
        return max(
            0,
            int(
                min(
                    self.free("cpu") // profile.cpu,
                    self.free("memory_bytes") // profile.memory_bytes,
                    self.free("disk_bytes") // profile.disk_bytes,
                )
            ),
        )

    def reserve(self, sandbox_id: str, profile: ResourceProfile, now: float) -> None:
        self.unconfirmed[sandbox_id] = (profile, now)

    def release(self, sandbox_id: str) -> None:
        self.unconfirmed.pop(sandbox_id, None)


@dataclass
class _Build:
    logs: list[str] = field(default_factory=list)
    done: threading.Event = field(default_factory=threading.Event)


@dataclass(frozen=True)
class CreatePlan:
    record: SnapshotRecord
    image: str
    labels: dict[str, str]
    ttl_minutes: int
    runtime: str | None


def local_image_tag(snapshot_name: str) -> str:
    """Where a built snapshot lives in a host's containerd. Never pushed anywhere."""
    return f"silo.local/snapshots/{snapshot_name}:built"


class _Deferred:
    """Log records collected under the lock and emitted after it is released.

    A slow stdout must never hold up the broker's lock.
    """

    def __init__(self) -> None:
        self._records: list[tuple[int, str, tuple[Any, ...]]] = []

    def log(self, level: int, message: str, *args: Any) -> None:
        self._records.append((level, message, args))

    def info(self, message: str, *args: Any) -> None:
        self.log(logging.INFO, message, *args)

    def warning(self, message: str, *args: Any) -> None:
        self.log(logging.WARNING, message, *args)

    def emit(self) -> None:
        for level, message, args in self._records:
            logger.log(level, message, *args)


class Broker:
    def __init__(
        self,
        *,
        host_secret: str,
        host_client_factory: Callable[[str], HostClient],
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
        placement_wait_seconds: float = PLACEMENT_WAIT_SECONDS,
        config: BrokerConfig | None = None,
    ) -> None:
        self._host_secret = host_secret
        self._host_client_factory = host_client_factory
        self._clock = clock
        self._sleep = sleep
        self._placement_wait = placement_wait_seconds
        self._config = config or BrokerConfig()
        self._lock = threading.RLock()
        self._hosts: dict[str, HostState] = {}
        self._snapshots: dict[str, SnapshotRecord] = {}
        self._builds: dict[str, _Build] = {}
        # sandbox id -> host id. Retired ids stay here (host None) forever, so a
        # lookup of a deleted sandbox is a not-found, never a resurrection.
        self._routes: dict[str, str | None] = {}
        self._waiting = 0
        self._refresh_lock = threading.Lock()
        self._last_refresh = float("-inf")
        self._last_capacity_log = float("-inf")
        # Until then, an unknown sandbox id answers 503 (see _route and begin_warmup).
        self._warm_until = float("-inf")
        self._warmup_seconds = 0.0
        self._heard_from_hosts = False
        self._started_at = clock()
        # Bumped on every change worth persisting (snapshots, host registry, images).
        self._generation = 0
        self._saved_generation = 0

    @property
    def config(self) -> BrokerConfig:
        return self._config

    def begin_warmup(self, seconds: float) -> None:
        """Answer 503, not not-found, for unknown sandbox ids for ``seconds``.

        Called at process start: any start may be a restart, and until every
        host has heartbeated once the routing table is incomplete. A brand-new
        deployment has no sandboxes to ask about, so it costs nothing there.

        The window is re-anchored at the first heartbeat, because a replacement
        broker started alongside the old one (the overlap cutover) hears from no
        host until the old one is cancelled.
        """
        self._warmup_seconds = seconds
        self._warm_until = self._clock() + seconds

    # ------------------------------------------------------------------ #
    # Hosts
    # ------------------------------------------------------------------ #

    def heartbeat(
        self,
        host_id: str,
        url: str,
        capacity: dict[str, Any],
        sandbox_ids: list[str],
        *,
        images: list[str] | None = None,
        heartbeat_seconds: float | None = None,
        heartbeat_timeout_seconds: float | None = None,
    ) -> None:
        """Apply one host report. In-memory only and O(sandboxes on that host).

        It must stay that way: this runs on its own thread pool precisely so that
        no amount of slow host I/O elsewhere can delay it (2026-09-29, when it
        queued behind hundreds of 600 s creates and every host looked dead).
        """
        log = _Deferred()
        with self._lock:
            now = self._clock()
            if not self._heard_from_hosts:
                self._heard_from_hosts = True
                self._warm_until = max(self._warm_until, now + self._warmup_seconds)
            host = self._hosts.get(host_id)
            if host is None or host.url != url:
                log.info("host registered id=%s url=%s", host_id, url)
                host = HostState(
                    host_id=host_id,
                    url=url,
                    capacity=capacity,
                    last_seen=now,
                    client=self._host_client_factory(url),
                )
                self._hosts[host_id] = host
                self._generation += 1
            elif host.last_health != HostHealth.LIVE or host.gone:
                log.warning(
                    "host %s is live again (was %s; silent %.0fs)",
                    host_id,
                    "gone" if host.gone else host.last_health,
                    now - host.last_seen,
                )
            host.gone = False
            host.last_health = HostHealth.LIVE
            self._apply_report(host, capacity, sandbox_ids, now, log)
            if images:
                before = len(host.images)
                host.images.update(images)
                if len(host.images) != before:
                    self._generation += 1
            if heartbeat_seconds and heartbeat_timeout_seconds and not host.cycle_warned:
                cycle = float(heartbeat_seconds) + float(heartbeat_timeout_seconds)
                if self._config.suspect_after_seconds < 2 * cycle:
                    host.cycle_warned = True
                    log.warning(
                        "host %s heartbeats every %gs with a %gs timeout; the broker's suspect window "
                        "(%gs) is under 2x that cycle, so it will flap. Raise SILO_HOST_SUSPECT_AFTER_SECONDS.",
                        host_id,
                        heartbeat_seconds,
                        heartbeat_timeout_seconds,
                        self._config.suspect_after_seconds,
                    )
            # A broker restart loses its routing table; hosts rebuild it.
            for sandbox_id in sandbox_ids:
                self._routes.setdefault(sandbox_id, host_id)
        log.emit()

    def _apply_report(
        self, host: HostState, capacity: dict[str, Any], sandbox_ids: list[str], now: float, log: _Deferred
    ) -> None:
        host.apply_report(capacity, sandbox_ids, now)
        drift = report_inconsistency(capacity)
        if drift and not host.drift:
            log.warning(
                "host %s DRIFTED: its report is impossible (%s) -- its capacity accounting has drifted. "
                "Placing there against a conservative estimate (every live sandbox counted as candidate-sized; "
                "sandboxes_live=%s). Restart it once it drains.",
                host.host_id,
                drift,
                capacity.get("sandboxes_live"),
            )
        elif host.drift and not drift:
            log.info("host %s reports a consistent allocation again; placing there", host.host_id)
        host.drift = drift

    def _health(self, host: HostState, now: float) -> str:
        if host.gone:
            return HostHealth.DEAD
        age = now - host.last_seen
        if age <= self._config.suspect_after_seconds:
            return HostHealth.LIVE
        if age <= self._config.dead_after_seconds:
            return HostHealth.SUSPECT
        return HostHealth.DEAD

    def _quarantined(self, host: HostState, now: float) -> bool:
        # Drift is NOT quarantine: a drifted host stays placeable against a
        # conservative allocation estimate (HostState.free).
        return now < host.quarantined_until

    def _placeable(self, host: HostState, now: float) -> bool:
        return self._health(host, now) == HostHealth.LIVE and not self._quarantined(host, now)

    def _inflight_cap(self, host: HostState) -> int:
        return 1 if host.probation else self._config.max_inflight_creates_per_host

    def _live_hosts(self) -> list[HostState]:
        now = self._clock()
        return [h for h in self._hosts.values() if self._health(h, now) == HostHealth.LIVE]

    def _placeable_hosts(self) -> list[HostState]:
        now = self._clock()
        return [h for h in self._hosts.values() if self._placeable(h, now)]

    def _expire_quarantines(self, now: float, log: _Deferred) -> None:
        for host in self._hosts.values():
            if host.quarantined_until != float("-inf") and now >= host.quarantined_until:
                host.quarantined_until = float("-inf")
                host.probation = True
                host.consecutive_failures = 0
                log.warning(
                    "host %s quarantine ended; on probation (one create at a time until one succeeds)", host.host_id
                )

    def _record_outcome(self, host: HostState, failure: str | None, log: _Deferred) -> None:
        """Track create outcomes per host; quarantine a host that keeps failing."""
        now = self._clock()
        if failure is None:
            if host.probation or host.consecutive_failures:
                log.info(
                    "host %s create succeeded; clearing %d failure(s)%s",
                    host.host_id,
                    host.consecutive_failures,
                    " and probation" if host.probation else "",
                )
            host.consecutive_failures = 0
            host.probation = False
            host.quarantine_count = 0
            return
        host.consecutive_failures += 1
        host.last_failure = failure[:300]
        threshold = 1 if host.probation else self._config.quarantine_after_failures
        if host.consecutive_failures < threshold or now < host.quarantined_until:
            return
        live = self._live_hosts()
        already = sum(1 for h in live if now < h.quarantined_until)
        limit = max(1, math.floor(len(live) * self._config.quarantine_max_fraction))
        if already >= limit:
            log.warning(
                "host %s failed %d create(s) in a row (last: %s) but %d of %d live hosts are already "
                "quarantined (cap %d); deprioritizing it instead",
                host.host_id,
                host.consecutive_failures,
                host.last_failure,
                already,
                len(live),
                limit,
            )
            return
        duration = min(self._config.quarantine_max_seconds, self._config.quarantine_seconds * (2**host.quarantine_count))
        host.quarantine_count += 1
        host.quarantined_until = now + duration
        host.probation = False
        host.quarantine_reason = f"{host.consecutive_failures} consecutive create failures (last: {host.last_failure})"
        log.warning(
            "host %s QUARANTINED for placement for %.0fs: %s. Existing sandboxes stay routable.",
            host.host_id,
            duration,
            host.quarantine_reason,
        )

    def sweep(self) -> None:
        """Log liveness transitions, end quarantines, forget long-dead hosts.

        Called periodically by the broker's maintenance thread. Health itself is
        computed from report age wherever it is needed; this only makes the
        transitions visible.
        """
        log = _Deferred()
        with self._lock:
            now = self._clock()
            self._expire_quarantines(now, log)
            for host_id, host in list(self._hosts.items()):
                health = self._health(host, now)
                if health != host.last_health:
                    level = logging.WARNING if health != HostHealth.LIVE else logging.INFO
                    log.log(
                        level,
                        "host %s is now %s (last report %.0fs ago)%s",
                        host_id,
                        health,
                        now - host.last_seen,
                        (
                            "; its sandboxes answer 503 until it reports"
                            if health == HostHealth.SUSPECT
                            else "; its sandboxes are presumed gone" if health == HostHealth.DEAD else ""
                        ),
                    )
                    host.last_health = health
                if health == HostHealth.DEAD and now - host.last_seen > 2 * self._config.dead_after_seconds:
                    del self._hosts[host_id]
                    self._generation += 1
                    log.info("host %s forgotten after %.0fs dead", host_id, now - host.last_seen)
        log.emit()

    def _counts(self, now: float) -> dict[str, int]:
        counts = {"live": 0, "suspect": 0, "dead": 0, "quarantined": 0, "placeable": 0}
        for host in self._hosts.values():
            health = self._health(host, now)
            counts[health] += 1
            if health == HostHealth.LIVE:
                if self._quarantined(host, now):
                    counts["quarantined"] += 1
                else:
                    counts["placeable"] += 1
        return counts

    def capacity(self) -> dict[str, Any]:
        """The named ceiling. Totals across live hosts, plus slots per profile."""
        with self._lock:
            now = self._clock()
            counts = self._counts(now)
            live = [h for h in self._hosts.values() if self._health(h, now) == HostHealth.LIVE]
            placeable = [h for h in live if not self._quarantined(h, now)]
            live_sandboxes = sum(int(h.capacity.get("sandboxes_live", 0)) for h in live)
            return {
                "hosts_live": counts["live"],
                "hosts_suspect": counts["suspect"],
                "hosts_dead": counts["dead"],
                "hosts_quarantined": counts["quarantined"],
                "hosts_placeable": counts["placeable"],
                "hosts_known": len(self._hosts),
                "sandboxes_live": live_sandboxes,
                "creates_waiting_for_capacity": self._waiting,
                "creates_inflight": sum(len(h.inflight) for h in self._hosts.values()),
                "slots_free": {
                    "candidate_4cpu_8gb": sum(h.slots_for(CANDIDATE_DEFAULT) for h in placeable),
                    "verifier_2cpu_1gb": sum(h.slots_for(VERIFIER_DEFAULT) for h in placeable),
                },
                # Snapshots are metadata plus, for built recipes, a cached image
                # on a host. There is no snapshot quota.
                "snapshots": len(self._snapshots),
                "snapshot_quota": None,
                "liveness": {
                    "suspect_after_s": self._config.suspect_after_seconds,
                    "dead_after_s": self._config.dead_after_seconds,
                    "max_inflight_creates_per_host": self._config.max_inflight_creates_per_host,
                },
                "hosts": [
                    {
                        "host_id": h.host_id,
                        "url": h.url,
                        "age_s": round(now - h.last_seen, 1),
                        "health": self._health(h, now),
                        "quarantined": self._quarantined(h, now),
                        "quarantine_reason": h.drift or (h.quarantine_reason if now < h.quarantined_until else ""),
                        "probation": h.probation,
                        "creates_inflight": len(h.inflight),
                        "consecutive_failures": h.consecutive_failures,
                        "images": len(h.images),
                        **h.capacity,
                    }
                    for h in self._hosts.values()
                    if self._health(h, now) != HostHealth.DEAD
                ],
            }

    # ------------------------------------------------------------------ #
    # Snapshots
    # ------------------------------------------------------------------ #

    def create_snapshot(self, name: str, dockerfile_content: str, profile: ResourceProfile) -> SnapshotRecord:
        plan = parse_recipe(dockerfile_content)  # SiloRecipeError -> 422, not retried
        with self._lock:
            if name in self._snapshots:
                # Same name, whatever the content: the caller resolves the
                # existing record and validate_snapshot_recipe checks its bytes.
                # That is how name reuse with different content fails closed.
                raise SiloConflictError("snapshot", name)
            record = SnapshotRecord(
                name=name,
                id=f"snap-{uuid.uuid4().hex[:16]}",
                state="building" if plan.requires_build else "pending",
                dockerfile_content=dockerfile_content,
                plan=plan,
                profile=profile,
                created_at=utcnow(),
            )
            self._snapshots[name] = record
            self._builds[name] = _Build()
            self._generation += 1
        threading.Thread(target=self._materialize, args=(name,), name=f"snap-{name}", daemon=True).start()
        return record

    def _materialize(self, name: str) -> None:
        """Make a snapshot usable: pre-pull its base, and build it if it has RUN steps.

        A FROM-only recipe needs no build. Its readiness is just "a host can pull
        this digest", which is why it goes active in seconds and why the 300 s
        readiness bound stops being interesting for the common case.
        """
        with self._lock:
            record = self._snapshots[name]
            build = self._builds[name]
        host: HostState | None = None
        try:
            host = self._wait_for_any_host(timeout=240, base_ref=record.plan.base_ref)
            build.logs.append(f"{utcnow()} pulling {record.plan.base_ref} on {host.host_id}")
            host.client.ensure_image(record.plan.base_ref)
            with self._lock:
                host.images.add(record.plan.base_ref)
            resolved = record.plan.base_ref
            if record.plan.requires_build:
                tag = local_image_tag(name)
                build.logs.append(f"{utcnow()} building {tag} on {host.host_id}")
                host.client.build_image(tag, build_dockerfile(record.plan))
                with self._lock:
                    host.images.add(tag)
                resolved = tag
            with self._lock:
                self._snapshots[name] = dataclasses.replace(record, state=SNAPSHOT_ACTIVE, resolved_ref=resolved)
                self._generation += 1
            build.logs.append(f"{utcnow()} active resolved={resolved}")
        except Exception as error:
            logger.exception("snapshot %s failed to materialize", name)
            build.logs.append(f"{utcnow()} error {type(error).__name__}: {error}")
            with self._lock:
                self._snapshots[name] = dataclasses.replace(
                    record, state=SNAPSHOT_ERROR, error_message=f"{type(error).__name__}: {error}"[:1000]
                )
                self._generation += 1
        finally:
            if host is not None:
                with self._lock:
                    host.builds_inflight -= 1
            build.done.set()

    def _wait_for_any_host(self, timeout: float, base_ref: str = "") -> HostState:
        """A placeable host for a pull/build, claimed (``builds_inflight``) for the caller.

        A host that already holds the base image wins -- for a ``FROM silo.local/...``
        recipe it is the only kind that can build at all -- then the least busy
        builder, then the roomiest.
        """
        deadline = self._clock() + timeout
        while True:
            with self._lock:
                hosts = self._placeable_hosts()
                if hosts:
                    host = max(hosts, key=lambda h: (base_ref in h.images, -h.builds_inflight, h.free("memory_bytes")))
                    host.builds_inflight += 1
                    return host
            if self._clock() >= deadline:
                raise SiloError("no live sandbox hosts registered with the broker", status_code=503)
            self._sleep(PLACEMENT_POLL_SECONDS)

    def get_snapshot(self, name: str) -> SnapshotRecord:
        with self._lock:
            record = self._snapshots.get(name)
        if record is None:
            raise SiloNotFoundError("snapshot", name)
        return record

    def list_snapshots(self, page: int = 1, limit: int = 100) -> dict[str, Any]:
        with self._lock:
            names = sorted(self._snapshots)
            start = max(0, (page - 1) * limit)
            records = [self._snapshots[n] for n in names[start : start + limit]]
        items = [record.to_dict() for record in records]
        return {"items": items, "total": len(names), "page": page, "total_pages": max(1, -(-len(names) // limit))}

    def delete_snapshot(self, name: str) -> None:
        with self._lock:
            if self._snapshots.pop(name, None) is None:
                raise SiloNotFoundError("snapshot", name)
            self._builds.pop(name, None)
            self._generation += 1

    def snapshot_build_logs(self, name: str) -> str:
        self.get_snapshot(name)
        with self._lock:
            build = self._builds.get(name)
        return "\n".join(build.logs) if build else ""

    # ------------------------------------------------------------------ #
    # Sandboxes
    # ------------------------------------------------------------------ #

    # -- the create path, in steps that never block while holding a thread --
    #
    # A create that must wait for room used to sleep inside a request thread.
    # Under a burst, waiting creates filled the server's pool and queued the very
    # deletes that would have freed room: the broker jammed itself at exactly the
    # moment it was built to handle visibly (width test 01, 2026-09-22). So the
    # wait now lives in the caller -- an async handler, or the sync loop below --
    # and each step here is short.

    def plan_create(
        self,
        *,
        snapshot: str,
        labels: Mapping[str, str] | None = None,
        ttl_minutes: int = 180,
        network_block_all: bool = True,
        runtime: str | None = None,
    ) -> CreatePlan:
        if not network_block_all:
            # Stated rather than silently honoured. Nothing in the pipeline asks
            # for a networked sandbox, and the provider does not have one to give.
            raise SiloError(NETWORK_REFUSAL, status_code=400)
        record = self.get_snapshot(snapshot)  # "snapshot 'x' not found" -> recovery path upstream
        if record.state != SNAPSHOT_ACTIVE:
            raise SiloError(f"snapshot {snapshot!r} is not active (state={record.state})", status_code=409)
        return CreatePlan(
            record=record,
            image=record.resolved_ref or record.plan.base_ref,
            labels=dict(labels or {}),
            ttl_minutes=ttl_minutes,
            runtime=runtime,
        )

    def try_place(self, plan: CreatePlan) -> tuple[HostState, str] | None:
        """Reserve a slot on the best host, or return None if none has room. Non-blocking.

        Only LIVE, unquarantined hosts with fewer than their in-flight cap of
        creates qualify. Among them: any host without a current failure streak
        beats any host with one; then a host holding the image, then one that
        can build it (a ``silo.local`` base must already be there), then the
        least busy, then the roomiest.
        """
        profile = plan.record.profile
        base = plan.record.plan.base_ref
        needs_local_base = plan.record.plan.requires_build and _is_local(base)
        log = _Deferred()
        placed: tuple[HostState, str] | None = None
        with self._lock:
            now = self._clock()
            self._expire_quarantines(now, log)
            candidates = [
                h
                for h in self._hosts.values()
                if self._placeable(h, now) and len(h.inflight) < self._inflight_cap(h) and h.slots_for(profile) >= 1
            ]
            if candidates:
                host = max(
                    candidates,
                    key=lambda h: (
                        -h.consecutive_failures,
                        plan.image in h.images,
                        not needs_local_base or base in h.images,
                        -len(h.inflight),
                        h.slots_for(profile),
                    ),
                )
                sandbox_id = new_sandbox_id()
                host.reserve(sandbox_id, profile, now)
                host.inflight[sandbox_id] = now
                self._routes[sandbox_id] = host.host_id
                placed = (host, sandbox_id)
        log.emit()
        return placed

    def _finish_attempt(
        self,
        host: HostState,
        sandbox_id: str,
        *,
        created: bool,
        failure: str | None = None,
        gone: bool = False,
        image: str | None = None,
    ) -> None:
        log = _Deferred()
        with self._lock:
            host.inflight.pop(sandbox_id, None)
            if not created:
                host.release(sandbox_id)
                self._routes[sandbox_id] = None
            if gone:
                host.gone = True
            if image is not None and image not in host.images:
                host.images.add(image)
                self._generation += 1
            if created or failure is not None:
                self._record_outcome(host, failure, log)
        log.emit()

    def attempt_create(self, plan: CreatePlan, host: HostState, sandbox_id: str) -> dict[str, Any] | None:
        """Create on the chosen host. None means: place it again now (elsewhere).

        The host is the authority on its own capacity. If it refuses with a 429
        the broker's view was stale: refresh that host and let the caller place
        again, rather than failing a create another host could take. A host that
        refuses the connection is gone: place again at once. A host that fails
        like a sick host (a timeout) raises ``PlaceAgainLater`` and the failure
        counts against it. Any other error is the request's and is raised.
        """
        record = plan.record
        try:
            if record.plan.requires_build and plan.image not in host.images:
                # Built images are host-local. Rebuild here rather than ship them:
                # recipes are content-identified and their inputs pinned.
                host.client.build_image(plan.image, build_dockerfile(record.plan))
                with self._lock:
                    host.images.add(plan.image)
            created = host.client.create_sandbox(
                {
                    "sandbox_id": sandbox_id,
                    "snapshot_name": record.name,
                    "image_ref": plan.image,
                    "plan": _plan_body(record.plan),
                    "profile": {
                        "cpu": record.profile.cpu,
                        "memory_gb": record.profile.memory_gb,
                        "disk_gb": record.profile.disk_gb,
                    },
                    "runtime": plan.runtime,
                    "labels": plan.labels,
                    "ttl_minutes": plan.ttl_minutes,
                }
            )
        except SiloRateLimitError:
            self._finish_attempt(host, sandbox_id, created=False)
            self._refresh_host(host)
            return None
        except TransportError as error:
            if error.refused:
                # Nothing listens there: the host was cancelled, preempted or
                # restarted elsewhere. Stop placing there now.
                logger.warning("host %s refused the connection during create (%s); marking it dead", host.host_id, error)
                self._finish_attempt(host, sandbox_id, created=False, gone=True)
                return None
            logger.warning("host %s did not answer a create (%s); placing again elsewhere", host.host_id, error)
            self._finish_attempt(host, sandbox_id, created=False, failure=f"TransportError: {error}")
            raise PlaceAgainLater(host.host_id, f"TransportError: {error}") from error
        except SiloError as error:
            if _is_host_fault(error):
                logger.warning(
                    "host %s failed a create like a sick host (%s: %s); placing again elsewhere",
                    host.host_id,
                    type(error).__name__,
                    str(error)[:300],
                )
                self._finish_attempt(host, sandbox_id, created=False, failure=f"{error.status_code}: {error}")
                raise PlaceAgainLater(host.host_id, f"{error.status_code}: {str(error)[:300]}") from error
            self._finish_attempt(host, sandbox_id, created=False)
            raise
        except BaseException:
            self._finish_attempt(host, sandbox_id, created=False)
            raise
        self._finish_attempt(host, sandbox_id, created=True, image=plan.image)
        return self._with_access(created, host)

    def refresh_hosts(self) -> None:
        """Ask every placeable host for its capacity now. Rate-limited across callers."""
        if not self._refresh_lock.acquire(blocking=False):
            return  # another caller is refreshing; its result serves everyone
        try:
            self._refresh_due_hosts()
        finally:
            self._refresh_lock.release()

    def kick_refresh(self) -> None:
        """Start a refresh in the background if one is due. Never blocks the caller.

        The async create loop calls this every poll; a waiting create used to
        spend a pool thread per poll just to find the refresh lock taken.
        """
        if self._clock() - self._last_refresh < REFRESH_MIN_INTERVAL_SECONDS:
            return
        if not self._refresh_lock.acquire(blocking=False):
            return

        def run() -> None:
            try:
                self._refresh_due_hosts()
            finally:
                self._refresh_lock.release()

        threading.Thread(target=run, name="silo-refresh", daemon=True).start()

    def _refresh_due_hosts(self) -> None:
        if self._clock() - self._last_refresh < REFRESH_MIN_INTERVAL_SECONDS:
            return
        with self._lock:
            hosts = self._placeable_hosts()
        for host in hosts:
            self._refresh_host(host)
        self._last_refresh = self._clock()

    def _refresh_host(self, host: HostState) -> None:
        try:
            report = host.client.capacity()
        except Exception:
            logger.warning("capacity refresh failed for host %s", host.host_id, exc_info=True)
            return
        log = _Deferred()
        with self._lock:
            self._apply_report(host, report, report.get("sandbox_ids") or [], self._clock(), log)
        log.emit()

    def note_waiting(self, delta: int, profile: ResourceProfile | None = None) -> None:
        message = None
        with self._lock:
            self._waiting += delta
            now = self._clock()
            if delta > 0 and profile is not None and now - self._last_capacity_log >= 10.0:
                self._last_capacity_log = now
                message = (
                    "AT CAPACITY: %d create(s) waiting for a free %dcpu/%dGB slot. %s",
                    (self._waiting, profile.cpu, profile.memory_gb, self._ceiling_summary()),
                )
        if message is not None:
            logger.warning(message[0], *message[1])

    def create_sandbox(
        self,
        *,
        snapshot: str,
        labels: Mapping[str, str] | None = None,
        ttl_minutes: int = 180,
        network_block_all: bool = True,
        runtime: str | None = None,
        timeout: float = 600.0,
    ) -> dict[str, Any]:
        """Synchronous create, waiting for room with this broker's sleep.

        The HTTP server does not use this: it runs the same steps with an async
        wait so that waiting never holds a request thread.
        """
        plan = self.plan_create(
            snapshot=snapshot,
            labels=labels,
            ttl_minutes=ttl_minutes,
            network_block_all=network_block_all,
            runtime=runtime,
        )
        waited = min(timeout, self._placement_wait)
        deadline = self._clock() + waited
        waiting = False
        last_failure: PlaceAgainLater | None = None
        try:
            while True:
                placed = self.try_place(plan)
                if placed is not None:
                    try:
                        created = self.attempt_create(plan, *placed)
                    except PlaceAgainLater as failure:
                        last_failure = failure
                        if self._clock() >= deadline:
                            raise self.capacity_exhausted(waited, last_failure) from None
                        self._sleep(PLACEMENT_POLL_SECONDS)
                        continue
                    if created is not None:
                        return created
                    if self._clock() >= deadline:
                        raise self.capacity_exhausted(waited, last_failure)
                    continue
                if not waiting:
                    waiting = True
                    self.note_waiting(+1, plan.record.profile)
                if self._clock() >= deadline:
                    raise self.capacity_exhausted(waited, last_failure)
                self.refresh_hosts()
                self._sleep(PLACEMENT_POLL_SECONDS)
        finally:
            if waiting:
                self.note_waiting(-1)

    def _ceiling_summary(self) -> str:
        now = self._clock()
        counts = self._counts(now)
        live = [h for h in self._hosts.values() if self._health(h, now) == HostHealth.LIVE]
        sandboxes = sum(int(h.capacity.get("sandboxes_live", 0)) for h in live)
        inflight = sum(len(h.inflight) for h in self._hosts.values())
        return (
            f"hosts_live={counts['live']} hosts_suspect={counts['suspect']} hosts_dead={counts['dead']} "
            f"hosts_quarantined={counts['quarantined']} hosts_placeable={counts['placeable']} "
            f"sandboxes_live={sandboxes} creates_inflight={inflight}"
        )

    def capacity_exhausted(self, waited: float, last_failure: PlaceAgainLater | None = None) -> SiloRateLimitError:
        with self._lock:
            summary = self._ceiling_summary()
        suffix = f"; last host failure: {last_failure}" if last_failure is not None else ""
        return SiloRateLimitError(
            f"sandbox capacity exhausted ({summary}); waited {waited:.0f}s{suffix}",
            retry_after_seconds=30,
        )

    def placement_deadline(self, timeout: float) -> float:
        return self._clock() + min(timeout, self._placement_wait)

    def now(self) -> float:
        return self._clock()

    def _with_access(self, record: dict[str, Any], host: HostState) -> dict[str, Any]:
        return {**record, "host_url": host.url, "token": sandbox_capability(self._host_secret, record["id"])}

    def _route(self, sandbox_id: str) -> HostState:
        """The host a sandbox lives on, if it can be reached right now.

        LIVE host: route. SUSPECT host (silent, not yet presumed dead): a
        retryable 503 -- the sandbox is most likely still running, and a
        not-found here would tell the pipeline it is gone. Unknown id shortly
        after a broker restart: also 503, while hosts re-register their
        sandboxes. Otherwise (retired id, dead or forgotten host): not-found.
        Neither 503 message contains the words the pipeline reads as not-found.
        """
        with self._lock:
            now = self._clock()
            known = sandbox_id in self._routes
            host_id = self._routes.get(sandbox_id)
            host = self._hosts.get(host_id) if host_id else None
            health = self._health(host, now) if host is not None else None
            age = now - host.last_seen if host is not None else 0.0
            uptime = now - self._started_at
            warming = now < self._warm_until
        if host is not None and health == HostHealth.LIVE:
            return host
        if host is not None and health == HostHealth.SUSPECT:
            raise SiloError(
                f"sandbox {sandbox_id!r} is on host {host.host_id}, which has not reported for {age:.0f}s "
                f"(suspect, not presumed dead); retry in {RETRY_AFTER_SECONDS}s",
                status_code=503,
            )
        if warming and (not known or (host_id and host is None)):
            raise SiloError(
                f"sandbox {sandbox_id!r} has no route yet: the broker started {uptime:.0f}s ago and hosts are "
                f"still re-registering their sandboxes; retry in {RETRY_AFTER_SECONDS}s",
                status_code=503,
            )
        raise SiloNotFoundError("sandbox", sandbox_id)

    def get_sandbox(self, sandbox_id: str) -> dict[str, Any]:
        host = self._route(sandbox_id)
        try:
            record = host.client.get_sandbox(sandbox_id)
        except SiloNotFoundError:
            with self._lock:
                self._routes[sandbox_id] = None
            raise
        return self._with_access(record, host)

    def delete_sandbox(self, sandbox_id: str) -> None:
        host = self._route(sandbox_id)
        host.client.delete_sandbox(sandbox_id)
        with self._lock:
            # Created and deleted between two reports: never confirmed, and
            # never will be. Stop counting it now.
            host.release(sandbox_id)

    def list_sandboxes(self) -> list[dict[str, Any]]:
        """Every sandbox on every live host, like Daytona's org-wide list.

        ``dt sandbox list`` / ``delete-mine`` filter this by label, so it has to
        be complete. A host that cannot be listed is logged and skipped rather
        than failing the whole call; its sandboxes are simply absent this time.
        """
        with self._lock:
            hosts = list(self._live_hosts())
        records: list[dict[str, Any]] = []
        for host in hosts:
            try:
                items = host.client.list_sandboxes()
            except Exception:
                logger.warning("could not list sandboxes on host %s", host.host_id, exc_info=True)
                continue
            records.extend(self._with_access(item, host) for item in items)
        return records

    # ------------------------------------------------------------------ #
    # Persistence: what a restart must not lose
    # ------------------------------------------------------------------ #
    #
    # Snapshots exist only here, so they are persisted. Hosts are persisted with
    # the images they hold (built snapshots are host-local, and a FROM
    # silo.local/... recipe can only build where its parent image is). Routes are
    # not: every host reports its sandbox ids each heartbeat, and until they have,
    # an unknown id answers 503 rather than not-found (see _route).

    def export_state(self) -> tuple[int, dict[str, Any]] | None:
        """``(generation, state)`` if anything changed since the last save, else None."""
        with self._lock:
            if self._generation == self._saved_generation:
                return None
            now = self._clock()
            state = {
                "version": STATE_VERSION,
                "saved_at": utcnow(),
                "snapshots": [_snapshot_state(record) for record in self._snapshots.values()],
                "hosts": [
                    {"host_id": h.host_id, "url": h.url, "images": sorted(h.images)}
                    for h in self._hosts.values()
                    if self._health(h, now) != HostHealth.DEAD
                ],
            }
            return self._generation, state

    def mark_saved(self, generation: int) -> None:
        with self._lock:
            self._saved_generation = max(self._saved_generation, generation)

    def restore_state(self, state: Mapping[str, Any]) -> dict[str, int]:
        """Load persisted snapshots and hosts. Call before serving.

        Restored hosts start SUSPECT: routable as 503, never placed on, until
        they heartbeat. A snapshot that was still building is built again.
        """
        if int(state.get("version", 0)) != STATE_VERSION:
            raise ValueError(f"unsupported broker state version {state.get('version')!r}")
        counts = {"snapshots": 0, "rebuilding": 0, "skipped": 0, "hosts": 0}
        rebuild: list[str] = []
        with self._lock:
            now = self._clock()
            for item in state.get("snapshots") or []:
                name = item["name"]
                if name in self._snapshots:
                    continue
                try:
                    plan = parse_recipe(item["dockerfile_content"])
                    profile = ResourceProfile(
                        cpu=int(item["cpu"]), memory_gb=int(item["memory_gb"]), disk_gb=int(item["disk_gb"])
                    )
                except (SiloRecipeError, KeyError, ValueError, TypeError) as error:
                    logger.error("broker state: cannot restore snapshot %r (%s)", name, error)
                    counts["skipped"] += 1
                    continue
                record = SnapshotRecord(
                    name=name,
                    id=str(item.get("id") or f"snap-{uuid.uuid4().hex[:16]}"),
                    state=str(item.get("state") or "pending"),
                    dockerfile_content=item["dockerfile_content"],
                    plan=plan,
                    profile=profile,
                    created_at=str(item.get("created_at") or utcnow()),
                    resolved_ref=str(item.get("resolved_ref") or ""),
                    error_message=str(item.get("error_message") or ""),
                )
                build = _Build(logs=[f"{utcnow()} restored from broker state (state={record.state})"])
                if record.state not in (SNAPSHOT_ACTIVE, SNAPSHOT_ERROR):
                    record = dataclasses.replace(
                        record, state="building" if plan.requires_build else "pending", resolved_ref=""
                    )
                    rebuild.append(name)
                else:
                    build.done.set()
                self._snapshots[name] = record
                self._builds[name] = build
                counts["snapshots"] += 1
            for item in state.get("hosts") or []:
                host_id = item["host_id"]
                if host_id in self._hosts:
                    continue
                host = HostState(
                    host_id=host_id,
                    url=item["url"],
                    capacity={},
                    last_seen=now - self._config.suspect_after_seconds - 1e-3,
                    client=self._host_client_factory(item["url"]),
                    images=set(item.get("images") or []),
                )
                host.last_health = HostHealth.SUSPECT
                self._hosts[host_id] = host
                counts["hosts"] += 1
            self._saved_generation = self._generation
        for name in rebuild:
            threading.Thread(target=self._materialize, args=(name,), name=f"snap-{name}", daemon=True).start()
        counts["rebuilding"] = len(rebuild)
        return counts


def _snapshot_state(record: SnapshotRecord) -> dict[str, Any]:
    return {
        "name": record.name,
        "id": record.id,
        "state": record.state,
        "dockerfile_content": record.dockerfile_content,
        "cpu": record.profile.cpu,
        "memory_gb": record.profile.memory_gb,
        "disk_gb": record.profile.disk_gb,
        "created_at": record.created_at,
        "resolved_ref": record.resolved_ref,
        "error_message": record.error_message,
    }


def snapshot_state_from_api(item: Mapping[str, Any]) -> dict[str, Any]:
    """A persisted-state snapshot entry from a ``GET /snapshots`` item.

    Lets a running broker's snapshots be exported through its public API and
    restored into a replacement (``silo.broker.export``).
    """
    state = str(item.get("state") or "")
    return {
        "name": item["name"],
        "id": item.get("id"),
        "state": state,
        "dockerfile_content": (item.get("build_info") or {})["dockerfile_content"],
        "cpu": item["cpu"],
        "memory_gb": item["mem"],
        "disk_gb": item["disk"],
        "created_at": item.get("created_at"),
        "resolved_ref": item.get("ref") if state == SNAPSHOT_ACTIVE else "",
        "error_message": item.get("error_message") or "",
    }


def _plan_body(plan: ImagePlan) -> dict[str, Any]:
    return {
        "base_ref": plan.base_ref,
        "entrypoint": list(plan.entrypoint) if plan.entrypoint else None,
        "cmd": list(plan.cmd) if plan.cmd else None,
        "user": plan.user,
        "workdir": plan.workdir,
        "env": dict(plan.env),
        "run_steps": list(plan.run_steps),
    }
