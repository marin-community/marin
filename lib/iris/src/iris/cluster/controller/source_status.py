# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Canonical resource operations over controller state and backend observations."""

from collections.abc import Mapping

from rigging.timing import Timestamp

from iris.backends.protocol import ProviderError
from iris.cluster.controller.dependencies import ResourceDependencies
from iris.cluster.federation.protocol import FederationPeerObservation
from iris.resources.source import (
    MAX_SOURCE_ERROR_MESSAGE,
    Freshness,
    ResourceSourceStatus,
    SourceState,
)

_BACKEND_UNAVAILABLE = "backend_unavailable"
_PEER_UNAVAILABLE = "peer_unavailable"
_FINELOG_UNAVAILABLE = "finelog_unavailable"
_SOURCE_UNSUPPORTED = "unsupported"


def _available_source(source_id: str, *, backend_id: str = "") -> ResourceSourceStatus:
    return ResourceSourceStatus(
        source_id=source_id,
        backend_id=backend_id,
        state=SourceState.AVAILABLE,
        freshness=Freshness.CURRENT,
        observed_at=Timestamp.now(),
        error_code="",
        error_message="",
    )


def _unavailable_backend_source(backend_id: str, error: Exception) -> ResourceSourceStatus:
    return ResourceSourceStatus(
        source_id=f"backend:{backend_id}",
        backend_id=backend_id,
        state=SourceState.UNAVAILABLE,
        freshness=Freshness.UNKNOWN,
        observed_at=None,
        error_code=_BACKEND_UNAVAILABLE,
        error_message=str(error)[:MAX_SOURCE_ERROR_MESSAGE],
    )


def _unavailable_finelog_source(cluster_id: str, error: Exception) -> ResourceSourceStatus:
    return ResourceSourceStatus(
        source_id=f"finelog:{cluster_id}",
        backend_id="",
        state=SourceState.UNAVAILABLE,
        freshness=Freshness.UNKNOWN,
        observed_at=None,
        error_code=_FINELOG_UNAVAILABLE,
        error_message=str(error)[:MAX_SOURCE_ERROR_MESSAGE],
    )


def _unsupported_source(source_id: str, *, backend_id: str = "") -> ResourceSourceStatus:
    return ResourceSourceStatus(
        source_id=source_id,
        backend_id=backend_id,
        state=SourceState.UNSUPPORTED,
        freshness=Freshness.UNKNOWN,
        observed_at=None,
        error_code=_SOURCE_UNSUPPORTED,
        error_message="",
    )


def resource_source_statuses(dependencies: ResourceDependencies) -> tuple[ResourceSourceStatus, ...]:
    statuses = [_available_source(f"controller:{dependencies.cluster_id}")]
    for backend_id, backend in sorted(dependencies.backends.items()):
        try:
            backend.status()
        except (ConnectionError, ProviderError) as exc:
            statuses.append(_unavailable_backend_source(backend_id, exc))
        else:
            statuses.append(_available_source(f"backend:{backend_id}", backend_id=backend_id))
    peer_observations = {peer.peer_id: peer for peer in dependencies.runtime.federation.peer_observations()}
    statuses.extend(peer_source_statuses(dependencies, set(peer_observations), observations=peer_observations))
    return tuple(statuses)


def peer_source_statuses(
    dependencies: ResourceDependencies,
    peer_ids: set[str],
    *,
    observations: Mapping[str, FederationPeerObservation] | None = None,
) -> tuple[ResourceSourceStatus, ...]:
    if not peer_ids:
        return ()
    if observations is None:
        observations = {peer.peer_id: peer for peer in dependencies.runtime.federation.peer_observations()}
    statuses = []
    for peer_id in sorted(peer_ids):
        peer = observations.get(peer_id)
        if peer is None:
            statuses.append(
                ResourceSourceStatus(
                    source_id=f"federation:{peer_id}",
                    backend_id="",
                    state=SourceState.UNAVAILABLE,
                    freshness=Freshness.UNKNOWN,
                    observed_at=None,
                    error_code=_PEER_UNAVAILABLE,
                    error_message=f"Federation peer {peer_id} is not configured",
                )
            )
            continue
        observed_at = Timestamp.from_ms(peer.last_contact_ms) if peer.last_contact_ms else None
        if peer.reachable:
            statuses.append(
                ResourceSourceStatus(
                    source_id=f"federation:{peer.peer_id}",
                    backend_id="",
                    state=SourceState.AVAILABLE,
                    freshness=Freshness.CURRENT,
                    observed_at=observed_at,
                    error_code="",
                    error_message="",
                )
            )
        else:
            statuses.append(
                ResourceSourceStatus(
                    source_id=f"federation:{peer.peer_id}",
                    backend_id="",
                    state=SourceState.UNAVAILABLE,
                    freshness=Freshness.STALE if observed_at is not None else Freshness.UNKNOWN,
                    observed_at=observed_at,
                    error_code=_PEER_UNAVAILABLE,
                    error_message=f"Federation peer {peer.peer_id} is unreachable",
                )
            )
    return tuple(statuses)
