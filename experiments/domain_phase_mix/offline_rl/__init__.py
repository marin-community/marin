# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Offline RL tooling for domain/phase mixture experiments."""

from experiments.domain_phase_mix.offline_rl.contracts import (
    DEFAULT_ACTION_BINS,
    DEFAULT_ACTION_BOUNDS,
    DEFAULT_HISTORY_FEATURE_KEYS,
    DEFAULT_OBJECTIVE_METRIC,
    DEFAULT_PHASE_END_STEPS,
    DEFAULT_POLICY_FEATURES_V2,
    DEFAULT_STARCODER_FAMILIES,
    DEFAULT_TOTAL_STEPS,
    ActionGridConfig,
    ExperimentFamilyConfig,
    PooledDatasetConfig,
    RLFeatureConfig,
    TransitionRow,
    default_feature_config,
    default_pooled_dataset_config,
)
from experiments.domain_phase_mix.offline_rl.policy_artifact import (
    PolicyArtifactV1,
    PolicyArtifactV2,
    clip_action,
    discretize_action,
    load_policy_artifact,
    save_policy_artifact,
)

__all__ = [
    "DEFAULT_ACTION_BINS",
    "DEFAULT_ACTION_BOUNDS",
    "DEFAULT_HISTORY_FEATURE_KEYS",
    "DEFAULT_OBJECTIVE_METRIC",
    "DEFAULT_PHASE_END_STEPS",
    "DEFAULT_POLICY_FEATURES_V2",
    "DEFAULT_STARCODER_FAMILIES",
    "DEFAULT_TOTAL_STEPS",
    "ActionGridConfig",
    "ExperimentFamilyConfig",
    "OfflineRLPolicyValidationAdapter",
    "PolicyArtifactV1",
    "PolicyArtifactV2",
    "PooledDatasetConfig",
    "RLFeatureConfig",
    "TransitionRow",
    "clip_action",
    "default_feature_config",
    "default_pooled_dataset_config",
    "discretize_action",
    "load_policy_artifact",
    "register_offline_rl_validation_adapter",
    "save_policy_artifact",
]


def __getattr__(name: str):
    if name in {"OfflineRLPolicyValidationAdapter", "register_offline_rl_validation_adapter"}:
        from experiments.domain_phase_mix.offline_rl.nextgen_policy_validation_adapter import (
            OfflineRLPolicyValidationAdapter,
            register_offline_rl_validation_adapter,
        )

        exports = {
            "OfflineRLPolicyValidationAdapter": OfflineRLPolicyValidationAdapter,
            "register_offline_rl_validation_adapter": register_offline_rl_validation_adapter,
        }
        return exports[name]

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
