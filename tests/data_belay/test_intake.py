# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import pytest

from experiments.data_belay.intake import LicenseVerdict, license_verdict


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
