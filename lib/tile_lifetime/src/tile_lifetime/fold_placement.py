# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Place generic Map/Fold dataflow in an owner-compute tile lifetime."""

from dataclasses import dataclass
from enum import StrEnum

from tile_lifetime.tensor_program import FoldPrimitive, MapPrimitive, ProgramValue, TensorAxis


class FoldAttachmentSite(StrEnum):
    """Point in one owner traversal where a Fold executes."""

    OWNER_PREPARATION = "owner_preparation"


class FoldResultDisposition(StrEnum):
    """Lifetime of the Fold result after owner-local evaluation."""

    MATERIALIZE_FOR_CONSUMERS = "materialize_for_consumers"


@dataclass(frozen=True)
class OwnerTileAvailability:
    """A value present in an owner tile, with axes known to be complete."""

    value: ProgramValue
    complete_axes: tuple[TensorAxis, ...]

    def __post_init__(self) -> None:
        if len(set(self.complete_axes)) != len(self.complete_axes):
            raise ValueError("owner-tile completeness evidence repeats an axis")
        if not set(self.complete_axes) <= set(self.value.axes):
            raise ValueError("owner-tile complete axes must belong to the available value")


@dataclass(frozen=True)
class FoldAttachment:
    """A pointwise producer and Fold placed on a compatible owner domain."""

    producer: MapPrimitive
    fold: FoldPrimitive
    owner_axes: tuple[TensorAxis, ...]
    input_availability: tuple[OwnerTileAvailability, ...]
    site: FoldAttachmentSite
    result_disposition: FoldResultDisposition


def attach_fold_to_owner_preparation(
    producer: MapPrimitive,
    fold: FoldPrimitive,
    *,
    owner_axes: tuple[TensorAxis, ...],
    input_availability: tuple[OwnerTileAvailability, ...],
    result_disposition: FoldResultDisposition,
) -> FoldAttachment:
    """Attach a Fold when one owner tile has its complete output domain and inputs."""
    if producer.output != fold.input:
        raise ValueError("Fold attachment producer must define the Fold input")
    if fold.output.axes != owner_axes:
        raise ValueError("Fold attachment owner axes must equal the complete Fold output domain")
    availability_by_value = {available.value: available for available in input_availability}
    if len(availability_by_value) != len(input_availability):
        raise ValueError("Fold attachment owner preparation repeats an available value")
    missing = tuple(value.name for value in producer.inputs if value not in availability_by_value)
    if missing:
        raise ValueError(f"Fold attachment owner preparation lacks producer inputs {missing}")
    incomplete = tuple(
        value.name
        for value in producer.inputs
        if not (set(fold.reduction_axes) & set(value.axes)) <= set(availability_by_value[value].complete_axes)
    )
    if incomplete:
        raise ValueError(
            "Fold attachment requires every producer input to be complete along the Fold reduction axes; "
            f"incomplete inputs: {incomplete}"
        )
    return FoldAttachment(
        producer=producer,
        fold=fold,
        owner_axes=owner_axes,
        input_availability=tuple(availability_by_value[value] for value in producer.inputs),
        site=FoldAttachmentSite.OWNER_PREPARATION,
        result_disposition=result_disposition,
    )


def verify_owner_preparation_fold_attachment(attachment: FoldAttachment) -> None:
    """Recheck the data availability proof carried by a selected attachment."""
    expected = attach_fold_to_owner_preparation(
        attachment.producer,
        attachment.fold,
        owner_axes=attachment.owner_axes,
        input_availability=attachment.input_availability,
        result_disposition=attachment.result_disposition,
    )
    if attachment != expected:
        raise ValueError("Fold attachment is inconsistent with its owner-tile availability proof")
