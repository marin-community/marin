# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Frozen MMLU-Pro source identities shared by preparation and collection."""

PROMPT_COUNT = 1606
SAMPLES_PER_PROMPT = 8
COLLECTION_SIZE = PROMPT_COUNT * SAMPLES_PER_PROMPT

DATASET_REVISION = "b189ec765aa7ed75c8acfea42df31fdae71f97be"
CANONICAL_SHA256 = "ac5acdc984fc2698947e6c818c3595bc5d81d6eb494532a557d3734e787c424f"
SUBSET_SHA256 = "c12af58d8b7d75acdc421c281f9f0df7c438b9293415761fed07d8a63812f27b"
CAPTURE = "s3://marin-us-east-02a/marin/scratch/karan/glm-bench-traces/glm/MMLUPro"
CANONICAL = "s3://marin-us-east-02a/marin/scratch/karan/mopd2/mmlupro-canonical-v2"
