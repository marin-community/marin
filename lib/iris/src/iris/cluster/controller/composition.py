# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Controller process composition root."""

import secrets
from collections.abc import Sequence

from iris.backends.protocol import TaskBackend
from iris.cluster.config import BackendConfig
from iris.cluster.controller.admin import ControllerAdmin
from iris.cluster.controller.application import ControllerApplication
from iris.cluster.controller.auth import (
    ControllerAuth,
    FederationTokenProvider,
    native_proxy_auth_policy,
    request_auth_policy,
)
from iris.cluster.controller.controller import CapabilityUrlConfig, Controller
from iris.cluster.controller.dashboard import ControllerDashboard
from iris.cluster.controller.endpoint_service import EndpointServiceImpl
from iris.cluster.controller.federation_proxy import FederatedEndpointHandoff
from iris.cluster.controller.log_stack import LogStack
from iris.cluster.controller.persistence import reads
from iris.cluster.controller.persistence.database import ControllerDB
from iris.cluster.controller.runtime import ControllerConfig, ControllerRuntime
from iris.cluster.federation.peer import FederationPeer
from iris.cluster.types import JobName
from iris.managed_thread import ThreadContainer
from iris.rpc.controller_service import ControllerServiceImpl
from iris.rpc.resource_service import ResourceServiceImpl


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
    """Build the control runtime and attach its resource/RPC/HTTP application."""
    runtime = ControllerRuntime(
        config=config,
        backends=backends,
        log_stack=log_stack,
        threads=threads,
        db=db,
        backend_configs=backend_configs,
        federation_peers=federation_peers,
    )
    endpoint_service = EndpointServiceImpl(db=runtime.database, system_endpoints={})
    controller = Controller(
        cluster_id=config.cluster_id,
        db=runtime.database,
        runtime=runtime,
        bundle_store=runtime.bundle_store,
        endpoint_service=endpoint_service,
        auth=config.auth or ControllerAuth(),
        user_budget_defaults=config.user_budget_defaults,
        capability_url_config=CapabilityUrlConfig(
            cluster_name=config.cluster_id,
            local_origin=config.dashboard_url,
            parent_origin=config.federation_public_parent,
        ),
        backends=runtime.backends,
        backend_configs=runtime.backend_configs,
        log_client=runtime.log_client,
    )
    controller_service = ControllerServiceImpl(
        runtime=runtime,
        bundle_store=runtime.bundle_store,
        log_client=runtime.log_client,
        admin=ControllerAdmin(runtime.database),
        endpoint_service=endpoint_service,
        controller=controller,
        auth=config.auth,
        user_budget_defaults=config.user_budget_defaults,
    )
    resource_service = ResourceServiceImpl(controller)
    federation_token_provider = (
        FederationTokenProvider(config.cluster_id, config.auth.jwt_manager)
        if config.peers and config.auth and config.auth.jwt_manager
        else None
    )
    federated_handoff = (
        FederatedEndpointHandoff(runtime.federation.peer_controller_address, federation_token_provider.get_token)
        if federation_token_provider is not None
        else None
    )

    def federation_owner_check(root_job: JobName, peer_id: str) -> bool:
        with runtime.database.read_snapshot() as query:
            return reads.has_received_job_from_peer(query, peer_id, root_job)

    external_auth_policy = request_auth_policy(config.auth)
    auth_policy = native_proxy_auth_policy(external_auth_policy)
    dashboard = ControllerDashboard(
        controller_service,
        resource_service=resource_service,
        endpoint_service=endpoint_service,
        auth_provider=config.auth_provider,
        auth_policy=auth_policy,
        reported_auth_policy=external_auth_policy,
        jwt_manager=config.auth.jwt_manager if config.auth else None,
        federated_handoff=federated_handoff,
        federation_owner_check=federation_owner_check,
        proxy_decision_secret=secrets.token_urlsafe(32),
    )
    runtime.attach_application(
        ControllerApplication(
            controller=controller,
            controller_service=controller_service,
            resource_service=resource_service,
            endpoint_service=endpoint_service,
            dashboard=dashboard,
            external_auth_allows_anonymous=external_auth_policy.allows_anonymous,
        )
    )
    return runtime
