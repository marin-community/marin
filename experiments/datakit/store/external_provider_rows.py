# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Verify cached original Software Heritage bytes by their provider content identities."""

import hashlib


def software_heritage_rows(row: dict) -> tuple[dict, ...]:
    """Return an original content row only when both SWH content hashes match.

    The pinned downloader stores the decompressed provider payload unchanged as
    binary ``content``. Verify before UTF-8 decoding or normalization. A matching
    Git blob identity establishes the original bytes; it does not establish which
    repository occurrence supplied those bytes.
    """
    if row["present"] is not True:
        if row["present"] is not False or row["content"] != b"":
            raise ValueError("Software Heritage absent content must have present=False and an empty byte payload")
        return ()
    payload = row["content"]
    if not isinstance(payload, bytes):
        raise ValueError("Software Heritage content must retain the original binary payload")
    if hashlib.sha1(payload).hexdigest() != row["sha1"]:
        raise ValueError("Software Heritage content differs from its plain SHA1 identity")
    git_blob = b"blob " + str(len(payload)).encode("ascii") + b"\0" + payload
    if hashlib.sha1(git_blob).hexdigest() != row["sha1_git"]:
        raise ValueError("Software Heritage content differs from its Git blob SHA1 identity")
    return (row,)
