# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Pulumi entry point for one existing Iris controller."""

from typing import Any

import pulumi
from marin_deploy.iris import IrisActivationSpec, activate_controller, activation_marker_path


def _mark_activation_started(spec: IrisActivationSpec) -> None:
    activation_marker_path(spec).write_text("started\n")


def _activation_spec(properties: dict[str, Any]) -> IrisActivationSpec:
    spec = IrisActivationSpec(
        cluster=str(properties["cluster"]),
        controller_image=str(properties["controller_image"]),
        worker_image=str(properties["worker_image"]),
        task_image=str(properties["task_image"]),
        activation_id=str(properties["activation_id"]),
    )
    if spec.digest() != properties["activation_digest"]:
        raise ValueError("Iris activation inputs do not match iris:activation_digest")
    return spec


class IrisControllerProvider(pulumi.dynamic.ResourceProvider):
    """Apply controller activations without giving Pulumi delete ownership."""

    def create(self, properties: dict[str, Any]) -> pulumi.dynamic.CreateResult:
        spec = _activation_spec(properties)
        address = activate_controller(spec, on_activation_start=lambda: _mark_activation_started(spec))
        return pulumi.dynamic.CreateResult(
            id_=spec.cluster,
            outs={**properties, "address": address},
        )

    def diff(
        self,
        _id: str,
        old_properties: dict[str, Any],
        new_properties: dict[str, Any],
    ) -> pulumi.dynamic.DiffResult:
        compared_fields = ("cluster", "activation_digest")
        changes = any(old_properties.get(field) != new_properties.get(field) for field in compared_fields)
        return pulumi.dynamic.DiffResult(changes=changes)

    def update(
        self,
        _id: str,
        _old_properties: dict[str, Any],
        new_properties: dict[str, Any],
    ) -> pulumi.dynamic.UpdateResult:
        spec = _activation_spec(new_properties)
        address = activate_controller(spec, on_activation_start=lambda: _mark_activation_started(spec))
        return pulumi.dynamic.UpdateResult(outs={**new_properties, "address": address})

    def delete(self, _id: str, _properties: dict[str, Any]) -> None:
        return None


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
                "cluster": spec.cluster,
                "controller_image": spec.controller_image,
                "worker_image": spec.worker_image,
                "task_image": spec.task_image,
                "activation_id": spec.activation_id,
                "activation_digest": activation_digest,
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
    activation_digest = config.require("activation_digest")
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
