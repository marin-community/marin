# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Autoscaler/VM failure chaos tests.

This file previously contained tests that validated VM lifecycle failure modes
using the old cloud-query-based status system (slice_handle_status).

Those tests have been removed as part of the migration to the new lifecycle
tracking system. New chaos tests should be added that work with the new
architecture.
"""
