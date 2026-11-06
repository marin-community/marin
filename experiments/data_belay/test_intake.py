# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import pytest

from experiments.data_belay.intake import LicenseVerdict, license_verdict
from experiments.data_belay.prepare import cache_epochs


@pytest.mark.parametrize(
    ("licenses", "expected"),
    [
        (("odc-by",), LicenseVerdict.ALLOWED),
        (("apache-2.0",), LicenseVerdict.ALLOWED),
        (("cc-by-4.0",), LicenseVerdict.ALLOWED),
        (("cc-by-sa-3.0",), LicenseVerdict.ALLOWED),
        (("CC0-1.0",), LicenseVerdict.ALLOWED),
        (("openmdw",), LicenseVerdict.ALLOWED),
        (("cc-by-nc-4.0",), LicenseVerdict.REJECTED),
        (("cc-by-nc-sa-4.0",), LicenseVerdict.REJECTED),
        (("cc-by-nd-4.0",), LicenseVerdict.REJECTED),
        (("other",), LicenseVerdict.NEEDS_REVIEW),
        (("llama3",), LicenseVerdict.NEEDS_REVIEW),
        ((), LicenseVerdict.NEEDS_REVIEW),
        # Every id on the card must be allowed: an unknown second license sends the dataset to a person.
        (("cc-by-sa-3.0", "gfdl"), LicenseVerdict.NEEDS_REVIEW),
        (("odc-by", "cc-by-nc-4.0"), LicenseVerdict.REJECTED),
    ],
)
def test_license_verdict(licenses, expected):
    assert license_verdict(licenses) is expected


@pytest.mark.parametrize(
    ("need", "cache", "dataset", "ok"),
    [
        (600, 900, 10_000, True),
        # A dataset that holds the need, sampled into a cache that does not, is an intake estimate that was too high.
        (600, 400, 10_000, False),
        # A small dataset may repeat, up to MAX_EPOCHS times.
        (600, 200, 200, True),
        (600, 100, 100, False),
    ],
)
def test_cache_epochs(need, cache, dataset, ok):
    if ok:
        assert cache_epochs(need, cache, dataset) == need / cache
    else:
        with pytest.raises(ValueError):
            cache_epochs(need, cache, dataset)
