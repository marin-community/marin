# Copyright 2025 The Marin Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Generic RPC CLI infrastructure for Iris services.

Provides a registry of services and methods discovered from the generated
Connect RPC clients, allowing arbitrary RPC calls via CLI.
"""

import json
import re
from dataclasses import dataclass
from typing import Any

import click
from google.protobuf import json_format
from google.protobuf.descriptor import FieldDescriptor
from google.protobuf.message import Message

from iris.cli.main import require_controller_url
from iris.rpc import actor_connect, cluster_connect

PROTO_TYPE_TO_CLICK: dict[int, click.ParamType] = {
    FieldDescriptor.TYPE_STRING: click.STRING,
    FieldDescriptor.TYPE_INT32: click.INT,
    FieldDescriptor.TYPE_INT64: click.INT,
    FieldDescriptor.TYPE_UINT32: click.INT,
    FieldDescriptor.TYPE_UINT64: click.INT,
    FieldDescriptor.TYPE_SINT32: click.INT,
    FieldDescriptor.TYPE_SINT64: click.INT,
    FieldDescriptor.TYPE_FIXED32: click.INT,
    FieldDescriptor.TYPE_FIXED64: click.INT,
    FieldDescriptor.TYPE_SFIXED32: click.INT,
    FieldDescriptor.TYPE_SFIXED64: click.INT,
    FieldDescriptor.TYPE_BOOL: click.BOOL,
    FieldDescriptor.TYPE_FLOAT: click.FLOAT,
    FieldDescriptor.TYPE_DOUBLE: click.FLOAT,
}


@dataclass
class MethodInfo:
    """Information about an RPC method."""

    name: str  # PascalCase name (e.g., "ListJobs")
    method_fn_name: str  # snake_case method name on client (e.g., "list_jobs")
    input_type: type[Message]
    output_type: type[Message]


@dataclass
class ServiceInfo:
    """Information about an RPC service."""

    name: str  # Short name (e.g., "controller")
    full_name: str  # Full service name (e.g., "iris.cluster.ControllerService")
    client_class: type
    methods: dict[str, MethodInfo]


# Global registry of services
SERVICES: dict[str, ServiceInfo] = {}


def _discover_methods_from_client(client_class: type) -> dict[str, MethodInfo]:
    """Discover RPC methods by introspecting a sync client class."""
    methods: dict[str, MethodInfo] = {}

    for attr_name in dir(client_class):
        if attr_name.startswith("_"):
            continue

        method = getattr(client_class, attr_name)
        if not callable(method):
            continue

        annotations = getattr(method, "__annotations__", {})
        if not annotations:
            continue

        if "request" not in annotations:
            continue
        if "return" not in annotations:
            continue

        input_type = annotations["request"]
        output_type = annotations["return"]

        if not (isinstance(input_type, type) and issubclass(input_type, Message)):
            continue
        if not (isinstance(output_type, type) and issubclass(output_type, Message)):
            continue

        pascal_name = "".join(word.capitalize() for word in attr_name.split("_"))

        methods[pascal_name] = MethodInfo(
            name=pascal_name,
            method_fn_name=attr_name,
            input_type=input_type,
            output_type=output_type,
        )

    return methods


def _register_service(
    name: str,
    full_name: str,
    client_class: type,
) -> None:
    """Register a service in the global registry."""
    methods = _discover_methods_from_client(client_class)
    SERVICES[name] = ServiceInfo(
        name=name,
        full_name=full_name,
        client_class=client_class,
        methods=methods,
    )


def register_services() -> None:
    """Populate the global SERVICES registry."""
    if SERVICES:
        return  # Already registered

    _register_service(
        name="controller",
        full_name="iris.cluster.ControllerService",
        client_class=cluster_connect.ControllerServiceClientSync,
    )

    _register_service(
        name="worker",
        full_name="iris.cluster.WorkerService",
        client_class=cluster_connect.WorkerServiceClientSync,
    )

    _register_service(
        name="actor",
        full_name="iris.actor.ActorService",
        client_class=actor_connect.ActorServiceClientSync,
    )


def get_service(name: str) -> ServiceInfo | None:
    """Get a service by name."""
    register_services()
    return SERVICES.get(name)


def list_services() -> list[ServiceInfo]:
    """List all registered services."""
    register_services()
    return list(SERVICES.values())


def build_request(method_info: MethodInfo, json_str: str | None, kwargs: dict[str, Any]) -> Message:
    """Build a protobuf request message from JSON or keyword arguments."""
    if json_str:
        data = json.loads(json_str)
    else:
        data = kwargs

    return json_format.ParseDict(data, method_info.input_type())


def call_rpc(service_name: str, method_name: str, url: str, request: Message) -> Message:
    """Execute an RPC call and return the response."""
    register_services()

    service = SERVICES.get(service_name)
    if not service:
        available = ", ".join(SERVICES.keys())
        raise ValueError(f"Unknown service '{service_name}'. Available: {available}")

    method = service.methods.get(method_name)
    if not method:
        available = ", ".join(service.methods.keys())
        raise ValueError(f"Unknown method '{method_name}' on service '{service_name}'. Available: {available}")

    client = service.client_class(url)
    method_fn = getattr(client, method.method_fn_name)
    return method_fn(request)


def format_response(response: Message) -> str:
    """Format a protobuf response message as JSON."""
    return json_format.MessageToJson(
        response,
        preserving_proto_field_name=True,
        indent=2,
    )


def get_method_signature(method: MethodInfo) -> str:
    """Get a human-readable signature for a method."""
    input_name = method.input_type.DESCRIPTOR.name
    output_name = method.output_type.DESCRIPTOR.name
    return f"{input_name} -> {output_name}"


# =============================================================================
# Dynamic Click Command Generation
# =============================================================================


def to_kebab_case(name: str) -> str:
    """Convert PascalCase to kebab-case."""
    return re.sub(r"(?<!^)(?=[A-Z])", "-", name).lower()


def kebab_to_pascal(name: str) -> str:
    """Convert kebab-case to PascalCase."""
    return "".join(word.capitalize() for word in name.split("-"))


def _is_simple_field(field: FieldDescriptor) -> bool:
    """Check if a protobuf field is a simple scalar type that maps to Click."""
    if field.label == FieldDescriptor.LABEL_REPEATED:
        return False
    if field.message_type is not None:
        return False
    return field.type in PROTO_TYPE_TO_CLICK


def _build_options_from_proto(input_type: type[Message]) -> list[click.Option]:
    """Build Click options from a protobuf message's simple fields."""
    options: list[click.Option] = []

    for field in input_type.DESCRIPTOR.fields:
        if not _is_simple_field(field):
            continue

        click_type = PROTO_TYPE_TO_CLICK[field.type]
        kebab_name = field.name.replace("_", "-")

        if field.type == FieldDescriptor.TYPE_BOOL:
            options.append(
                click.Option(
                    param_decls=[f"--{kebab_name}/--no-{kebab_name}"],
                    default=None,
                )
            )
        else:
            options.append(click.Option(param_decls=[f"--{kebab_name}"], type=click_type, default=None))

    return options


class ServiceCommands(click.MultiCommand):
    """Dynamic Click group for RPC service methods.

    Lazily generates Click commands from protobuf service definitions.
    """

    def __init__(self, service_name: str, **attrs):
        super().__init__(**attrs)
        self.service_name = service_name

    def list_commands(self, _ctx: click.Context) -> list[str]:
        svc = get_service(self.service_name)
        if not svc:
            return []
        return [to_kebab_case(m) for m in sorted(svc.methods.keys())]

    def get_command(self, ctx: click.Context, name: str) -> click.Command | None:
        svc = get_service(self.service_name)
        if not svc:
            return None
        pascal_name = kebab_to_pascal(name)
        method = svc.methods.get(pascal_name)
        if not method:
            return None

        return self._build_command_for_method(method)

    def _build_command_for_method(self, method: MethodInfo) -> click.Command:
        """Build a Click command for an RPC method."""
        options: list[click.Parameter] = [
            click.Option(["--json", "json_str"], default=None, help="Full JSON request body"),
        ]
        options.extend(_build_options_from_proto(method.input_type))

        service_name = self.service_name

        @click.pass_context
        def callback(ctx: click.Context, json_str: str | None, **kwargs):
            controller_url = require_controller_url(ctx)
            field_values = {k: v for k, v in kwargs.items() if v is not None}
            request = build_request(method, json_str, field_values)
            response = call_rpc(service_name, method.name, controller_url, request)
            click.echo(format_response(response))

        return click.Command(
            name=to_kebab_case(method.name),
            callback=callback,
            params=options,
            help=f"RPC: {get_method_signature(method)}",
        )


@click.group()
def rpc():
    """Execute RPC calls against Iris services."""
    pass


rpc.add_command(ServiceCommands("controller", name="controller", help="Controller service RPC methods"))
rpc.add_command(ServiceCommands("worker", name="worker", help="Worker service RPC methods"))
rpc.add_command(ServiceCommands("actor", name="actor", help="Actor service RPC methods"))


def register_rpc_commands(iris_group: click.Group) -> None:
    """Register the rpc command group on the top-level iris group."""
    iris_group.add_command(rpc)
