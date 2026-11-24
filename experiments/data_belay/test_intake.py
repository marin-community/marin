# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import pytest
from finestore.rl.licenses import LicenseVerdict, license_verdict

from experiments.data_belay.prepare import cache_epochs


@pytest.mark.parametrize(
    ("licenses", "expected"),
    [
        # A Blue Oak id, of any rating, is allowed whatever its case, as is a version of OpenMDW.
        (("BSD-3-Clause",), LicenseVerdict.ALLOWED),
        (("wtfpl",), LicenseVerdict.ALLOWED),
        (("openmdw-1.1",), LicenseVerdict.ALLOWED),
        # The attribution and share-alike data licenses are allowed too.
        (("odc-by",), LicenseVerdict.ALLOWED),
        (("cc-by-sa-4.0",), LicenseVerdict.ALLOWED),
        # An NC or ND fragment rejects the id.
        (("cc-by-nc-sa-4.0",), LicenseVerdict.REJECTED),
        # An id on no list, or a card with no license, goes to a person.
        (("llama3",), LicenseVerdict.NEEDS_REVIEW),
        ((), LicenseVerdict.NEEDS_REVIEW),
        # Every id on the card must be allowed: a second license off the list sends the dataset to a person.
        (("mit", "gfdl"), LicenseVerdict.NEEDS_REVIEW),
        # The shared table admits the ODC-By 1.0 and PDDL/ODC-PDDL dedications.
        (("odc-by-1.0",), LicenseVerdict.ALLOWED),
        (("pddl-1.0",), LicenseVerdict.ALLOWED),
        (("odc-pddl",), LicenseVerdict.ALLOWED),
        (("apache-2.0", "cc-by-nc-4.0"), LicenseVerdict.REJECTED),
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
        # A dataset smaller than the need may repeat.
        (600, 200, 200, True),
    ],
)
def test_cache_epochs(need, cache, dataset, ok):
    if ok:
        assert cache_epochs(need, cache, dataset) == need / cache
    else:
        with pytest.raises(ValueError):
            cache_epochs(need, cache, dataset)
