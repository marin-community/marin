# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Evaluation run records, result readers, and the statistics computed from them.

Everything here describes results that already exist. Consumers such as the Marina
dashboards install this package without the code that runs evaluations, so it must not
import ``marin``, ``rolloutengine``, ``levanter``, or another training dependency.
"""
