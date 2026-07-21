# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import pytest
from iac.gcp.cloud_run import normalize_private_invoker


@pytest.mark.parametrize("member", ["allUsers", "allAuthenticatedUsers"])
def test_private_invoker_rejects_public_principals(member: str):
    with pytest.raises(ValueError, match="private invoker"):
        normalize_private_invoker(member)
