# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Frozen source pins and regional inputs for single-expert CruxEval transfer."""

BATCH_SIZE = 16
GENERATION_TOKENS = 30720
COLLECTION_SIZE = 12800

STUDENT_MODEL = "open-athena/Snowball-67B-A2B-10T-Mixed-RLVR-Sync-Step92"
STUDENT_REVISION = "fdc3c3c5fff489d3fb04dcbc2387c179b1bf3fab"
# The collection fetched this dataset revision; evaluation records used the later runtime below.
DATA_REVISION = "c131e5ab84d3014490549ec06d45deb13b5673c2"
EVALCHEMY_REVISION = "ae362881cf53f869a6ad52c8a59fa17343c0b5f8"
DATA_SHA256 = "fa0fad60cd6b3d083975b81943770cdc7d15482d01305a3d80c4e51405c2a978"
PARENT_DESTINATION = "marin-us-east-02a/marin/teacher-traces/cruxeval-test-exposed/step92-fdc3c3c5"
DATA_DESTINATION = "marin-us-east-02a/marin/teacher-traces/cruxeval-test-exposed/task133-20261007-prepared-v2"
