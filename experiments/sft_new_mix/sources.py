# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Candidate SFT sources using the sealed, accepted Nemotron science subset.

This defines inputs for future materialization; historical caches and training
launches are unchanged. The private HF dataset requires authorized credentials.
"""

from marin.datakit.download.huggingface import download_hf_step
from marin.datakit.normalize import DedupMode, normalize_step
from marin.datakit.sft_sources import NEMOTRON_SFT_V3_TOKEN_COUNTS_B, DatakitChatSource, all_sft_sources
from marin.datakit.sources import DatakitSource


def accepted_science_sources() -> dict[str, DatakitSource]:
    """Build exact-text sources without rendering or dropping duplicate rows."""
    sources: dict[str, DatakitSource] = {}
    science = download_hf_step(
        "raw/science-accepted-20261105",
        hf_dataset_id="open-athena/Nemotron-SFT-Science-v2-GLM-Accepted-2026.11.05",
        revision="1583049c1df02b418e07d14047eb37db9df46eba",
        hf_urls_glob=["data/*/*.jsonl.gz"],
    )
    for component in ("rqa", "so", "syn_mcq", "vendor"):
        name = f"nemotron_sft_v3/science_v2/{component}"
        normalized = normalize_step(
            name=f"normalized/sft/science-accepted-20261105/{component}",
            download=science,
            relative_input_path=f"data/{component}",
            file_extensions=(".jsonl.gz",),
            dedup_mode=DedupMode.NONE,
            # The seal verifies every entire text is shorter than this bound.
            max_whitespace_run_chars=16_777_216,
            bare=True,
        )
        sources[name] = DatakitSource(
            name=name,
            normalize_steps=(science, normalized),
            rough_token_count_b=NEMOTRON_SFT_V3_TOKEN_COUNTS_B[name],
        )
    return sources


def candidate_sft_sources_v1() -> dict[str, DatakitChatSource | DatakitSource]:
    """Replace only the four science components in the candidate SFT source map."""
    sources: dict[str, DatakitChatSource | DatakitSource] = dict(all_sft_sources())
    sources.update(accepted_science_sources())
    return sources
