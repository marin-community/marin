# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import pytest
from connectrpc.code import Code
from connectrpc.errors import ConnectError
from connectrpc.request import RequestContext
from google.protobuf import any_pb2, wrappers_pb2
from google.protobuf.message import Message
from iris.rpc import resource_pb2
from iris.rpc.resource_registry import ResourceRouteRegistryBuilder, ResourceWireContract
from iris.rpc.resource_service import ResourceServiceImpl


def _pack(value: Message) -> any_pb2.Any:
    result = any_pb2.Any()
    result.Pack(value)
    return result


class _GetSynthetic:
    contract = ResourceWireContract(
        views=(resource_pb2.RESOURCE_VIEW_BASIC,),
        body_types=(wrappers_pb2.StringValue,),
        features=("synthetic-v1",),
    )

    def run(
        self,
        request: resource_pb2.GetResourceRequest,
        _context: RequestContext,
    ) -> resource_pb2.GetResourceResponse:
        return resource_pb2.GetResourceResponse(
            resource=resource_pb2.Resource(
                ref=request.ref,
                body=_pack(wrappers_pb2.StringValue(value=f"read:{request.ref.id}")),
            )
        )


class _CreateSynthetic:
    contract = ResourceWireContract(
        body_types=(wrappers_pb2.StringValue,),
        input_type=wrappers_pb2.StringValue,
    )

    def run(
        self,
        request: resource_pb2.CreateResourceRequest,
        body: wrappers_pb2.StringValue,
        _context: RequestContext,
    ) -> resource_pb2.Operation:
        return resource_pb2.Operation(
            ref=resource_pb2.ResourceRef(
                authority_cluster_id="test",
                type=request.type,
                id=request.id,
                uid="created",
            ),
            verb="create",
            result=_pack(wrappers_pb2.StringValue(value=f"created:{body.value}")),
        )


def test_composition_can_install_a_new_noun_without_a_generic_service_change() -> None:
    builder = ResourceRouteRegistryBuilder()
    builder.bind("/synthetic/get", _GetSynthetic())
    service = ResourceServiceImpl(builder.freeze())

    response = service.get_resource(
        resource_pb2.GetResourceRequest(
            ref=resource_pb2.ResourceRef(authority_cluster_id="test", type="iris/synthetic", id="one"),
            view=resource_pb2.RESOURCE_VIEW_BASIC,
        ),
        None,
    )
    body = wrappers_pb2.StringValue()
    assert response.resource.body.Unpack(body)
    assert body.value == "read:one"

    (capability,) = service.get_service_info(resource_pb2.GetServiceInfoRequest(), None).resources
    assert capability.type == "iris/synthetic"
    assert list(capability.verbs) == ["get"]
    assert list(capability.body_type_urls) == ["type.googleapis.com/google.protobuf.StringValue"]
    assert list(capability.features) == ["synthetic-v1"]


def test_registry_decodes_a_registered_create_payload_before_invocation() -> None:
    builder = ResourceRouteRegistryBuilder()
    builder.bind("/synthetic/create", _CreateSynthetic())
    service = ResourceServiceImpl(builder.freeze())

    operation = service.create_resource(
        resource_pb2.CreateResourceRequest(
            mutation=resource_pb2.MutationMetadata(request_id="create-one"),
            type="iris/synthetic",
            id="one",
            body=_pack(wrappers_pb2.StringValue(value="payload")),
        ),
        None,
    )
    result = wrappers_pb2.StringValue()
    assert operation.result.Unpack(result)
    assert result.value == "created:payload"

    (capability,) = service.get_service_info(resource_pb2.GetServiceInfoRequest(), None).resources
    assert list(capability.create_type_urls) == ["type.googleapis.com/google.protobuf.StringValue"]


def test_registry_rejects_an_unregistered_create_payload() -> None:
    builder = ResourceRouteRegistryBuilder()
    builder.bind("/synthetic/create", _CreateSynthetic())
    service = ResourceServiceImpl(builder.freeze())

    with pytest.raises(ConnectError) as exc_info:
        service.create_resource(
            resource_pb2.CreateResourceRequest(
                mutation=resource_pb2.MutationMetadata(request_id="create-one"),
                type="iris/synthetic",
                id="one",
                body=_pack(wrappers_pb2.Int32Value(value=1)),
            ),
            None,
        )
    assert exc_info.value.code is Code.INVALID_ARGUMENT


def test_duplicate_binding_fails_during_composition() -> None:
    builder = ResourceRouteRegistryBuilder()
    builder.bind("/synthetic/get", _GetSynthetic())

    with pytest.raises(ValueError, match="duplicate resource binding: /synthetic/get"):
        builder.bind("/synthetic/get", _GetSynthetic())


@pytest.mark.parametrize(
    ("resource_type", "expected_code"),
    [
        ("iris/synthetic", Code.UNIMPLEMENTED),
        ("iris/unknown", Code.NOT_FOUND),
    ],
)
def test_missing_routes_distinguish_an_unsupported_verb_from_an_unknown_noun(
    resource_type: str,
    expected_code: Code,
) -> None:
    builder = ResourceRouteRegistryBuilder()
    builder.bind("/synthetic/get", _GetSynthetic())
    service = ResourceServiceImpl(builder.freeze())

    with pytest.raises(ConnectError) as exc_info:
        service.delete_resource(
            resource_pb2.DeleteResourceRequest(
                mutation=resource_pb2.MutationMetadata(request_id="delete-one"),
                ref=resource_pb2.ResourceRef(authority_cluster_id="test", type=resource_type, id="one"),
            ),
            None,
        )
    assert exc_info.value.code is expected_code
