# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The ``iris.provisioning`` finelog namespace: one row per slice provisioning outcome."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import ClassVar

# finelog namespace for ``IrisProvisioning`` rows.
PROVISIONING_NAMESPACE = "iris.provisioning"

# Raw cloud error messages can be long (the GCP stockout text is a paragraph);
# keep enough to disambiguate without bloating every row.
ERROR_MESSAGE_MAX_LEN = 200

# A capacity stockout — the dominant create-failure mode — says so in the reason;
# anything else at create time is a real fault.
STOCKOUT_MARKER = "no more capacity"


class ProvisioningOutcome(StrEnum):
    """How a slice's provisioning attempt ended.

    One flat outcome rather than an outcome+cause pair: the create failure modes
    (``STOCKOUT``/``ERROR``) and the runtime death (``PREEMPTED``) are the
    distinctions consumers actually split on. Success rate is
    ``READY / (READY + STOCKOUT + ERROR)``; ``PREEMPTED`` is a post-ready death,
    excluded from it.
    """

    READY = "ready"  # bootstrap succeeded
    STOCKOUT = "stockout"  # create failed: no capacity in the zone
    ERROR = "error"  # create failed: a fault other than stockout
    PREEMPTED = "preempted"  # reached ready, then lost at runtime


@dataclass
class IrisProvisioning:
    """One slice provisioning outcome. Doubles as the finelog table schema.

    ``outcome`` is stored as a string (finelog columns are primitive) but always
    holds a :class:`ProvisioningOutcome` value.
    """

    # Pool-level queries (per scale group over time) dominate; clustering parquet
    # by scale_group lets row-group min/max prune scans to a few groups.
    key_column: ClassVar[str] = "scale_group"

    ts: datetime
    resource_type: str  # "tpu" | "gpu" | "cpu"
    scale_group: str  # full authoritative pool name, e.g. tpu_v6e-preemptible_8-europe-west4-a
    zone: str
    accelerator_variant: str  # e.g. "v6e" ("" for cpu)
    outcome: str  # ProvisioningOutcome value
    error_message: str
    worker_count: int
    provision_latency_ms: int  # create→ready wall time; 0 for non-ready outcomes


def classify_create_failure(error_message: str) -> ProvisioningOutcome:
    """Classify a create/bootstrap failure as ``STOCKOUT`` (no capacity) or ``ERROR``."""
    if STOCKOUT_MARKER in error_message.lower():
        return ProvisioningOutcome.STOCKOUT
    return ProvisioningOutcome.ERROR
