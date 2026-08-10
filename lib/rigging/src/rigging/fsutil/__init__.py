# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""``fsutil`` — one browser and CLI for every Marin bucket.

Listing, reading, and copying across GCS, CoreWeave AI Object Storage, and R2 from a
single process, routed per bucket by :func:`rigging.filesystem.filesystem_for`.
``cli`` is the ``fsutil`` console script; ``tui`` is its interactive browser.
"""
