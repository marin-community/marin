# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Offline sanity checks for the testbed-wide settings constants."""

from experiments.datakit_testbed.settings import (
    RAW_TARGET_TOTAL_TOKENS_B,
    TESTBED_SEQ_LEN,
    TESTBED_STAGING_REGION,
    TESTBED_TOKENIZER,
)


def test_tokenizer_pinned():
    assert isinstance(TESTBED_TOKENIZER, str) and TESTBED_TOKENIZER
    assert "/" in TESTBED_TOKENIZER, "expected a HuggingFace org/repo form"


def test_staging_region_is_us_central1():
    assert TESTBED_STAGING_REGION == "us-central1"


def test_seq_len_matches_grug_default():
    assert TESTBED_SEQ_LEN == 4096


def test_raw_target_total_tokens_is_1t():
    assert RAW_TARGET_TOTAL_TOKENS_B == 1000.0
