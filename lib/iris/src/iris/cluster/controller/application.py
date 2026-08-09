# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Composed controller application surfaces hosted by the process runtime."""

from dataclasses import dataclass

from iris.cluster.controller.controller import Controller
from iris.cluster.controller.dashboard import ControllerDashboard
from iris.cluster.controller.endpoint_service import EndpointServiceImpl
from iris.rpc.controller_service import ControllerServiceImpl
from iris.rpc.resource_service import ResourceServiceImpl


@dataclass(frozen=True, slots=True)
class ControllerApplication:
    """Resource application plus its RPC and HTTP boundary adapters."""

    controller: Controller
    controller_service: ControllerServiceImpl
    resource_service: ResourceServiceImpl
    endpoint_service: EndpointServiceImpl
    dashboard: ControllerDashboard
    external_auth_allows_anonymous: bool
