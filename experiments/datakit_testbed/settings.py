# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Testbed-wide settings used by the ferry and training harness.

These are the knobs that are specific to *this* experiment (tokenizer, target
token budget, region pin, training seq len) as opposed to the general
registry in ``marin.datakit.sources``.
"""

from experiments.llama import llama3_tokenizer

TESTBED_TOKENIZER: str = llama3_tokenizer
"""Tokenizer shared across every tokenize step in the testbed. Matches Grug MoE."""

TESTBED_SEQ_LEN: int = 4096
"""Grug MoE default; reused across ranking + IsoFLOP confirmation runs."""

TESTBED_STAGING_REGION: str = "us-central1"
"""All raw dumps must be reachable without cross-region reads.

The ferry's ``MARIN_PREFIX`` pins to ``gs://marin-us-central1/...``; every
source in the registry must either be pre-staged there or downloadable into it.
"""

# TODO(rav): update this to 1T
RAW_TARGET_TOTAL_TOKENS_B: float = 10.0
"""Target size (billions of tokens) for the pre-normalize by-provenance sample.

Drives per-source sampling fractions via
:func:`experiments.datakit_testbed.sampler.proportional_sample_fractions` so
every experiment downstream pays O(sampled) rather than O(full upstream).
Matches the "~1T token testbed" headline.
"""
