# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""One-shot migrations over the eval archive fleet.

finestore is accumulative: a writer appends, and the reader keeps one row per primary key. Nothing
in the library deletes. A migration is the exception — it is the only code that removes rows, and it
snapshots what it removes to region-local 30-day storage first.
"""
