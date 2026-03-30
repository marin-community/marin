# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Two-phase Dolma3/Dolmino topology with CC high/low pairs collapsed into topic domains."""

from __future__ import annotations

import re
from fray.cluster import ResourceConfig
from levanter.main.train_lm import LmConfig
from levanter.optim import MuonHConfig

from experiments.domain_phase_mix.config import Domain, PhaseSchedule
from experiments.domain_phase_mix.experiment import MixtureExperiment
from experiments.domain_phase_mix.two_phase_dolma3_dolmino_top_level import (
    BATCH_SIZE,
    EVAL_DATASETS_CACHE_PATH,
    EVAL_TASKS,
    EXPERIMENT_BUDGET,
    NAME,
    PHASE_BOUNDARIES,
    PHASE_NAMES,
    SAMPLING_PARAMS,
    SEQ_LEN,
    TARGET_BUDGET,
    build_top_level_domains,
    create_two_phase_dolma3_dolmino_top_level_optimizer_config,
)
from experiments.domain_phase_mix.proxy_sweep import regmix_60m_proxy

COLLAPSED_NAME = f"{NAME}_topic_collapsed"
_CC_DOMAIN_PATTERN = re.compile(r"^(dolma3_cc/.+)_(high|low)$")


def build_top_level_domains_with_collapsed_cc_topics() -> list[Domain]:
    """Return the top-level domain list with CC high/low pairs collapsed by topic."""
    base_domains = build_top_level_domains()
    collapsed_domains: list[Domain] = []
    pending_cc_domains: dict[str, list[Domain]] = {}

    for domain in base_domains:
        match = _CC_DOMAIN_PATTERN.match(domain.name)
        if match is None:
            collapsed_domains.append(domain)
            continue

        topic_name, _quality = match.groups()
        topic_domains = pending_cc_domains.setdefault(topic_name, [])
        topic_domains.append(domain)
        if len(topic_domains) != 2:
            continue

        ordered_components = []
        for topic_domain in sorted(topic_domains, key=lambda item: item.name):
            ordered_components.extend(topic_domain.components)
        collapsed_domains.append(
            Domain(
                name=topic_name,
                components=ordered_components,
                description=f"Collapsed CC topic domain combining {topic_domains[0].name} and {topic_domains[1].name}.",
            )
        )

    if len(collapsed_domains) != 26:
        raise ValueError(f"Expected 26 topic-collapsed domains, found {len(collapsed_domains)}")
    return collapsed_domains


def create_two_phase_dolma3_dolmino_top_level_topic_collapsed_experiment(
    *,
    name: str = COLLAPSED_NAME,
    experiment_budget: int = EXPERIMENT_BUDGET,
    target_budget: int = TARGET_BUDGET,
    batch_size: int = BATCH_SIZE,
    seq_len: int = SEQ_LEN,
    eval_datasets_cache_path: str | None = EVAL_DATASETS_CACHE_PATH,
    model_config: LmConfig | None = None,
    optimizer_config: MuonHConfig | None = None,
    resources: ResourceConfig | None = None,
) -> MixtureExperiment:
    """Create the topic-collapsed variant of the two-phase Dolma3/Dolmino experiment."""
    phase_schedule = PhaseSchedule.from_boundaries(
        boundaries=PHASE_BOUNDARIES,
        names=list(PHASE_NAMES),
    )
    resolved_optimizer_config = create_two_phase_dolma3_dolmino_top_level_optimizer_config(
        experiment_budget=experiment_budget,
        batch_size=batch_size,
        seq_len=seq_len,
        phase_schedule=phase_schedule,
        optimizer_config=optimizer_config,
    )
    return MixtureExperiment(
        name=name,
        domains=build_top_level_domains_with_collapsed_cc_topics(),
        phase_schedule=phase_schedule,
        model_config=model_config or regmix_60m_proxy,
        batch_size=batch_size,
        seq_len=seq_len,
        num_train_steps=experiment_budget // (batch_size * seq_len),
        target_budget=target_budget,
        resources=resources or ResourceConfig.with_tpu("v5p-8"),
        eval_harness_tasks=EVAL_TASKS,
        sampling_params=SAMPLING_PARAMS,
        optimizer_config=resolved_optimizer_config,
        eval_datasets_cache_path=eval_datasets_cache_path,
        initial_fixed_weight_configs=(),
        hierarchical_runtime_domains=True,
    )
