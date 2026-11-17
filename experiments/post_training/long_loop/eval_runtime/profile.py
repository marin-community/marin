# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Frozen inputs shared by Micro210 preparation, budgeting and execution."""

DATASET = "open-athena/Marin-Micro-Eval-210"
SOURCE_REVISION = "4564ed081c316bc3231ae5c3a53a46b39248c819"
REVISION = SOURCE_REVISION
DESTINATION = "s3://marin-us-east-02a/marin/experiments/mini-eval-policy/e2e-20261031"
POLICY_INDEX = "open-athena/Marin-Mini-Eval-Policy-Index"
POLICY_INDEX_REVISION = "d5d2f92ce4866c431063bd123ea31a3865530809"


# Temporary user-authorized smoke exclusion; preserve the full release unchanged.
SKIPPED_BENCHMARKS = {"tau3-pi": "Unsupported Compose/network contract; marin issue #11083"}
SMOKE_PATH_COUNT = 21 - len(SKIPPED_BENCHMARKS)
