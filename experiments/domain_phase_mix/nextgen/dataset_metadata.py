# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Dataset-spec metadata helpers for nextgen loops."""

from __future__ import annotations

import numpy as np

from experiments.domain_phase_mix.nextgen.contracts import LoopConfig
from experiments.domain_phase_mix.starcoder_metadata import infer_starcoder_metadata


def _default_small_domains(domains: list[str]) -> list[int] | None:
    for idx, name in enumerate(domains):
        lower = name.lower()
        if "starcoder" in lower or "rare" in lower or "small" in lower:
            return [idx]
    return None


def _loop_epoch_metadata(
    loop: LoopConfig | None,
    phase_names: list[str],
    domain_names: list[str],
) -> tuple[np.ndarray, list[int] | None] | None:
    if loop is None or loop.target_budget is None or loop.phase_fractions is None or loop.domain_token_counts is None:
        return None

    if len(loop.phase_fractions) != len(phase_names):
        raise ValueError(
            f"Loop metadata has {len(loop.phase_fractions)} phase fractions but dataframe has phases {phase_names}"
        )

    missing_domains = [domain_name for domain_name in domain_names if domain_name not in loop.domain_token_counts]
    if missing_domains:
        raise ValueError(f"Loop metadata missing domain token counts for {missing_domains}")

    epoch_multipliers = np.array(
        [
            [phase_fraction * loop.target_budget / loop.domain_token_counts[domain_name] for domain_name in domain_names]
            for phase_fraction in loop.phase_fractions
        ],
        dtype=float,
    )
    small_domains = [
        domain_idx for domain_idx in range(len(domain_names)) if float(np.max(epoch_multipliers[:, domain_idx])) > 1.0
    ]
    return epoch_multipliers, small_domains


def resolve_dataset_epoch_metadata(
    *,
    loop: LoopConfig | None,
    phase_names: list[str],
    domain_names: list[str],
) -> tuple[np.ndarray, list[int] | None]:
    """Resolve epoch multipliers and small-domain indices for a merged run table."""
    loop_metadata = _loop_epoch_metadata(loop, phase_names, domain_names)
    if loop_metadata is not None:
        return loop_metadata

    starcoder_metadata = infer_starcoder_metadata(phase_names, domain_names)
    if starcoder_metadata is not None:
        return starcoder_metadata

    return np.ones((len(phase_names), len(domain_names)), dtype=float), _default_small_domains(domain_names)
