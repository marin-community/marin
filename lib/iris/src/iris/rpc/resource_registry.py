# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Composition-time bindings for the generic resource verbs."""

from collections import defaultdict
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType

from connectrpc.code import Code
from connectrpc.errors import ConnectError
from google.protobuf.message import Message

from iris.rpc import resource_pb2


class ResourceVerb(StrEnum):
    GET = "get"
    UPDATE = "update"


ResourceHandler = Callable[..., Message]


@dataclass(frozen=True, slots=True)
class ResourceBinding:
    handler: ResourceHandler
    payload_type: type[Message] | None

    @property
    def payload_type_url(self) -> str | None:
        if self.payload_type is None:
            return None
        return f"type.googleapis.com/{self.payload_type.DESCRIPTOR.full_name}"


class ResourceRegistryBuilder:
    """Collect controller-owned noun bindings, then freeze them for serving."""

    def __init__(self) -> None:
        self._bindings: dict[tuple[str, ResourceVerb], ResourceBinding] = {}

    def bind(
        self,
        path: str,
        handler: ResourceHandler,
        *,
        payload: type[Message] | None = None,
    ) -> None:
        resource_type, verb = _parse_path(path)
        if verb is ResourceVerb.UPDATE and payload is None:
            raise ValueError("update bindings require a protobuf payload")
        if verb is ResourceVerb.GET and payload is not None:
            raise ValueError("get bindings do not accept a payload")
        key = (resource_type, verb)
        if key in self._bindings:
            raise ValueError(f"duplicate resource binding: {path}")
        self._bindings[key] = ResourceBinding(handler, payload)

    def freeze(self) -> "ResourceRegistry":
        return ResourceRegistry(self._bindings)


class ResourceRegistry:
    """Immutable verb bindings indexed by resource type."""

    def __init__(self, bindings: Mapping[tuple[str, ResourceVerb], ResourceBinding]) -> None:
        self._bindings = MappingProxyType(dict(bindings))
        self._types = frozenset(resource_type for resource_type, _ in bindings)

    def require(self, resource_type: str, verb: ResourceVerb) -> ResourceBinding:
        binding = self._bindings.get((resource_type, verb))
        if binding is not None:
            return binding
        if resource_type not in self._types:
            raise ConnectError(Code.NOT_FOUND, f"unknown resource type: {resource_type!r}")
        raise ConnectError(Code.UNIMPLEMENTED, f"resource type {resource_type!r} does not support {verb.value}")

    @property
    def capabilities(self) -> tuple[resource_pb2.ResourceCapability, ...]:
        by_type: dict[str, list[tuple[ResourceVerb, ResourceBinding]]] = defaultdict(list)
        for (resource_type, verb), binding in self._bindings.items():
            by_type[resource_type].append((verb, binding))

        result = []
        for resource_type, bindings in sorted(by_type.items()):
            result.append(
                resource_pb2.ResourceCapability(
                    type=resource_type,
                    verbs=sorted(verb.value for verb, _ in bindings),
                    update_type_urls=sorted(
                        binding.payload_type_url
                        for verb, binding in bindings
                        if verb is ResourceVerb.UPDATE and binding.payload_type_url is not None
                    ),
                )
            )
        return tuple(result)


def _parse_path(path: str) -> tuple[str, ResourceVerb]:
    parts = path.strip("/").split("/")
    if len(parts) != 2 or not all(parts):
        raise ValueError("resource binding path must be /noun/verb")
    noun, verb_name = parts
    try:
        verb = ResourceVerb(verb_name)
    except ValueError as error:
        raise ValueError(f"unknown resource verb: {verb_name!r}") from error
    return f"iris/{noun}", verb
