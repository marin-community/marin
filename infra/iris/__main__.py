# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Pulumi entry point for existing Iris controllers."""

from typing import Any

import pulumi
from marin_deploy.iris import IrisActivationSpec, activate_controller, activation_marker_path

ACTIVATION_DIGEST_KEY = "activation_digest"
ACTIVATION_SPEC_KEY = "activation_spec"
ADDRESS_KEY = "address"


def _activation_spec(properties: dict[str, Any]) -> IrisActivationSpec:
    spec = IrisActivationSpec.from_json(str(properties[ACTIVATION_SPEC_KEY]))
    if spec.digest() != properties[ACTIVATION_DIGEST_KEY]:
        raise ValueError("Iris activation inputs do not match iris:activation_digest")
    return spec


class IrisControllerProvider(pulumi.dynamic.ResourceProvider):
    """Apply in-place controller activations; deletion leaves the controller running."""

    def create(self, properties: dict[str, Any]) -> pulumi.dynamic.CreateResult:
        spec = _activation_spec(properties)
        address = activate_controller(spec, on_activation_start=lambda: _mark_activation_started(spec))
        return pulumi.dynamic.CreateResult(
            id_=spec.cluster,
            outs={**properties, ADDRESS_KEY: address},
        )

    def diff(
        self,
        _id: str,
        old_properties: dict[str, Any],
        new_properties: dict[str, Any],
    ) -> pulumi.dynamic.DiffResult:
        changes = old_properties.get(ACTIVATION_DIGEST_KEY) != new_properties.get(ACTIVATION_DIGEST_KEY)
        return pulumi.dynamic.DiffResult(changes=changes)

    def update(
        self,
        _id: str,
        _old_properties: dict[str, Any],
        new_properties: dict[str, Any],
    ) -> pulumi.dynamic.UpdateResult:
        spec = _activation_spec(new_properties)
        address = activate_controller(spec, on_activation_start=lambda: _mark_activation_started(spec))
        return pulumi.dynamic.UpdateResult(outs={**new_properties, ADDRESS_KEY: address})

    def delete(self, _id: str, _properties: dict[str, Any]) -> None:
        return None


def _mark_activation_started(spec: IrisActivationSpec) -> None:
    activation_marker_path(spec).write_text("started\n")


class IrisControllerActivation(pulumi.dynamic.Resource):
    """A restart-only controller resource; removal leaves the controller running."""

    def __init__(
        self,
        name: str,
        *,
        spec: IrisActivationSpec,
        activation_digest: str,
    ) -> None:
        super().__init__(
            IrisControllerProvider(),
            name,
            {
                ACTIVATION_SPEC_KEY: spec.to_json(),
                ACTIVATION_DIGEST_KEY: activation_digest,
            },
        )


def main() -> None:
    config = pulumi.Config("iris")
    spec = IrisActivationSpec(
        cluster=config.require("cluster"),
        controller_image=config.require("controller_image"),
        worker_image=config.require("worker_image"),
        task_image=config.require("task_image"),
        activation_id=config.require("activation_id"),
    )
    activation_digest = config.require(ACTIVATION_DIGEST_KEY)
    if spec.digest() != activation_digest:
        raise ValueError("Iris activation inputs do not match iris:activation_digest")
    activation = IrisControllerActivation(
        "controller",
        spec=spec,
        activation_digest=activation_digest,
    )
    pulumi.export("cluster", spec.cluster)
    pulumi.export("controllerImage", spec.controller_image)
    pulumi.export("activationDigest", activation_digest)
    pulumi.export("controllerResource", activation.id)


main()
