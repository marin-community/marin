# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Provisioning-outcome classification for ``iris.provisioning`` rows.

The row schema lives in ``iris.cluster.stats.tables``; this module holds the
producer-side classification the autoscaler applies when it writes rows.
"""

from iris.cluster.stats.tables import ProvisioningOutcome

# Raw cloud error messages can be long (the GCP stockout text is a paragraph);
# keep enough to disambiguate without bloating every row.
ERROR_MESSAGE_MAX_LEN = 200

# A capacity stockout — the dominant create-failure mode — says so in the reason;
# anything else at create time is a real fault.
STOCKOUT_MARKER = "no more capacity"


def classify_create_failure(error_message: str) -> ProvisioningOutcome:
    """Classify a create/bootstrap failure as ``STOCKOUT`` (no capacity) or ``ERROR``."""
    if STOCKOUT_MARKER in error_message.lower():
        return ProvisioningOutcome.STOCKOUT
    return ProvisioningOutcome.ERROR
