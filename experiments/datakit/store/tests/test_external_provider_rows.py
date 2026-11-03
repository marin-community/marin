# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import pytest

from experiments.datakit.store.external_provider_rows import software_heritage_rows


def test_software_heritage_verification_preserves_provider_bytes_and_rejects_corruption():
    # Independently known content identities for the six bytes b"hello\n".
    raw = {
        "content": b"hello\n",
        "present": True,
        "sha1": "f572d396fae9206628714fb2ce00f72e94f2258f",
        "sha1_git": "ce013625030ba8dba906f756967f9e9ca394464a",
    }
    assert software_heritage_rows(raw) == (raw,)
    with pytest.raises(ValueError, match="plain SHA1"):
        software_heritage_rows({**raw, "content": b"edited\n"})
    with pytest.raises(ValueError, match="Git blob SHA1"):
        software_heritage_rows({**raw, "sha1_git": "0" * 40})
    assert software_heritage_rows({**raw, "present": False, "content": b""}) == ()
