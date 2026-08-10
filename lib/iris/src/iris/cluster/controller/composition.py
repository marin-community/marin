# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Composition roots for the controller runtime and hosted process."""

import secrets
from collections.abc import Sequence

from iris.backends.protocol import TaskBackend
from iris.cluster.config import BackendConfig
from iris.cluster.controller.auth import (
    ControllerAuth,
    FederationTokenProvider,
    native_proxy_auth_policy,
    request_auth_policy,
)
from iris.cluster.controller.controller import Controller
from iris.cluster.controller.dependencies import CapabilityUrlConfig
from iris.cluster.controller.endpoint_registry import EndpointRegistry
from iris.cluster.controller.federation_proxy import FederatedEndpointHandoff
from iris.cluster.controller.log_stack import LogStack
from iris.cluster.controller.operations import OperationalServices
from iris.cluster.controller.persistence.database import ControllerDB
from iris.cluster.controller.process import ControllerProcess
from iris.cluster.controller.runtime import ControllerConfig, ControllerRuntime
from iris.cluster.federation.peer import FederationPeer, build_peers
from iris.managed_thread import ThreadContainer, get_thread_container
from iris.rpc.dashboard import ControllerDashboard
from iris.rpc.endpoint_service import EndpointServiceImpl
from iris.rpc.federation_client import peer_connection_factory
from iris.rpc.legacy.controller_service import LegacyControllerService
from iris.rpc.log_reader import FinelogLogReader
from iris.rpc.resource_service import ResourceServiceImpl


def _federation_token_provider(config: ControllerConfig) -> FederationTokenProvider | None:
    if not config.peers or config.auth is None or config.auth.jwt_manager is None:
        return None
    return FederationTokenProvider(config.cluster_id, config.auth.jwt_manager)


def _federation_peers(
    config: ControllerConfig,
    peers: Sequence[FederationPeer] | None,
) -> Sequence[FederationPeer]:
    if peers is not None:
        return peers
    return build_peers(
        config.peers,
        connect=peer_connection_factory(_federation_token_provider(config)),
    )


def compose_controller_runtime(
    *,
    config: ControllerConfig,
    backends: dict[str, TaskBackend],
    log_stack: LogStack,
    threads: ThreadContainer | None = None,
    db: ControllerDB | None = None,
    backend_configs: dict[str, BackendConfig] | None = None,
    federation_peers: Sequence[FederationPeer] | None = None,
) -> ControllerRuntime:
    """Build the durable control runtime without network or presentation surfaces."""
    return ControllerRuntime(
        config=config,
        backends=backends,
        log_stack=log_stack,
        threads=threads,
        db=db,
        backend_configs=backend_configs,
        federation_peers=federation_peers,
    )


def compose_controller_process(
    *,
    config: ControllerConfig,
    backends: dict[str, TaskBackend],
    log_stack: LogStack,
    threads: ThreadContainer | None = None,
    db: ControllerDB | None = None,
    backend_configs: dict[str, BackendConfig] | None = None,
    federation_peers: Sequence[FederationPeer] | None = None,
) -> ControllerProcess:
    """Build a hosted controller process around a pure control runtime."""
    process_threads = threads if threads is not None else get_thread_container()
    runtime = compose_controller_runtime(
        config=config,
        backends=backends,
        log_stack=log_stack,
        threads=process_threads,
        db=db,
        backend_configs=backend_configs,
        federation_peers=_federation_peers(config, federation_peers),
    )
    endpoint_registry = EndpointRegistry(db=runtime.database)
    endpoint_service = EndpointServiceImpl(endpoint_registry)
    auth = config.auth or ControllerAuth()
    controller = Controller(
        cluster_id=config.cluster_id,
        db=runtime.database,
        runtime=runtime,
        bundle_store=runtime.bundle_store,
        endpoint_registry=endpoint_registry,
        auth=auth,
        user_budget_defaults=config.user_budget_defaults,
        capability_url_config=CapabilityUrlConfig(
            cluster_name=config.cluster_id,
            local_origin=config.dashboard_url,
            parent_origin=config.federation_public_parent,
        ),
        backends=runtime.backends,
        backend_configs=runtime.backend_configs,
        log_reader=FinelogLogReader(runtime.log_client),
    )
    controller_service = LegacyControllerService(
        runtime=runtime,
        bundle_store=runtime.bundle_store,
        log_client=runtime.log_client,
        operations=OperationalServices.from_database(runtime.database),
        endpoint_service=endpoint_service,
        controller=controller,
        auth=config.auth,
        user_budget_defaults=config.user_budget_defaults,
    )
    resource_service = ResourceServiceImpl(controller)
    federation_token_provider = _federation_token_provider(config)
    federated_handoff = (
        FederatedEndpointHandoff(runtime.federation.peer_controller_address, federation_token_provider.get_token)
        if federation_token_provider is not None
        else None
    )
    external_auth_policy = request_auth_policy(config.auth)
    dashboard = ControllerDashboard(
        controller_service,
        resource_service=resource_service,
        endpoint_service=endpoint_service,
        auth_provider=config.auth_provider,
        auth_policy=native_proxy_auth_policy(),
        jwt_manager=config.auth.jwt_manager if config.auth else None,
        federated_handoff=federated_handoff,
        federation_owner_check=controller.received_job_from_peer,
        proxy_decision_secret=secrets.token_urlsafe(32),
    )
    return ControllerProcess(
        config=config,
        runtime=runtime,
        controller=controller,
        controller_service=controller_service,
        resource_service=resource_service,
        endpoint_registry=endpoint_registry,
        endpoint_service=endpoint_service,
        dashboard=dashboard,
        external_auth_allows_anonymous=external_auth_policy.allows_anonymous,
        threads=process_threads,
    )
