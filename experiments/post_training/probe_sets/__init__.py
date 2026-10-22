# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Probe sets (WP8): pinned prompt sources, screening and manifests, and token-level
KL buckets over recorded conversations.

- ``sources`` loads every probe source at a pinned commit.
- ``builder`` selects, screens and writes the six sets and their manifest.
- ``buckets`` tags each token of a recorded conversation for bucketed KL.
- ``build`` is the command line.
"""
