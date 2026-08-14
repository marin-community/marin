# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import pytest
from iac.github.stack_outputs import federation_profile


def test_federation_profile_selects_named_loom_mapping() -> None:
    assert federation_profile({"fork-ferry": "fork-ferry", "review": "github-comment"}, "fork-ferry") == "fork-ferry"


@pytest.mark.parametrize("value", [None, [], {"fork-ferry": ""}, {"other": "ops"}])
def test_federation_profile_requires_named_nonempty_profile(value: object) -> None:
    with pytest.raises(ValueError, match=r"githubFederationProfiles|must export"):
        federation_profile(value, "fork-ferry")
