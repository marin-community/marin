# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Midtraining launcher.

Two modes via concrete classes (``CptMode`` / ``CooldownMode``). A
:class:`MidtrainSpec` glues a base, identity, data manifest, compute, and
mode together. ``resolve_midtrain_spec`` materializes the manifest and
resolves the CPT budget; ``validate_midtrain_spec`` runs every guard.

There is no CLI. Operators write a Python launcher script that imports
these primitives — matches the ``downstream_scaling/evals`` pattern.
Downstream analysis tools should depend on :mod:`marin.midtraining.schema`
(stable JSON manifest schema) rather than on the dataclasses here.
"""

from marin.midtraining.budget import BudgetKind, BudgetPolicy, ResolvedBudget, resolve_cpt_budget
from marin.midtraining.data_manifest import (
    DataCacheComponent,
    DataCacheManifest,
    DataManifestPointer,
    load_data_manifest,
    load_data_manifest_pointer,
)
from marin.midtraining.identity import RunIdentity, attempt_group_manifest_uri, build_run_identity, output_region
from marin.midtraining.launch import (
    LaunchRequest,
    LaunchResult,
    append_to_attempt_group,
    build_launch_request,
    build_manifest_row,
    submit_launch,
    write_manifest,
    write_train_config,
)
from marin.midtraining.levanter_config import render_train_lm_config, render_train_lm_yaml
from marin.midtraining.modes import (
    CheckpointOverride,
    CheckpointSourceKind,
    CooldownMode,
    CooldownResume,
    CptInit,
    CptMode,
    TrainingMode,
)
from marin.midtraining.preflight import (
    CooldownStageRecord,
    CrossRegionCopyPolicy,
    PreflightReport,
    default_gcs_exists,
    default_gcs_list,
    fake_gcs,
    preflight,
    stage_cooldown_checkpoint,
)
from marin.midtraining.schema import (
    SCHEMA_VERSION,
    RunManifestRow,
    is_run_manifest,
    read_run_manifest,
    write_run_manifest,
)
from marin.midtraining.spec import (
    BaseModelRef,
    ComputeProfile,
    MidtrainSpec,
    ResolvedMidtrainSpec,
    replace_run_identity,
    resolve_midtrain_spec,
    validate_midtrain_spec,
)
from marin.midtraining.tokenizers import LLAMA3_TOKENIZER, QWEN3_TOKENIZER, TokenizerRef, get_tokenizer
from marin.midtraining.watch import StartupProof, evaluate_startup

__all__ = [
    "LLAMA3_TOKENIZER",
    "QWEN3_TOKENIZER",
    "SCHEMA_VERSION",
    "BaseModelRef",
    "BudgetKind",
    "BudgetPolicy",
    "CheckpointOverride",
    "CheckpointSourceKind",
    "ComputeProfile",
    "CooldownMode",
    "CooldownResume",
    "CooldownStageRecord",
    "CptInit",
    "CptMode",
    "CrossRegionCopyPolicy",
    "DataCacheComponent",
    "DataCacheManifest",
    "DataManifestPointer",
    "LaunchRequest",
    "LaunchResult",
    "MidtrainSpec",
    "PreflightReport",
    "ResolvedBudget",
    "ResolvedMidtrainSpec",
    "RunIdentity",
    "RunManifestRow",
    "StartupProof",
    "TokenizerRef",
    "TrainingMode",
    "append_to_attempt_group",
    "attempt_group_manifest_uri",
    "build_launch_request",
    "build_manifest_row",
    "build_run_identity",
    "default_gcs_exists",
    "default_gcs_list",
    "evaluate_startup",
    "fake_gcs",
    "get_tokenizer",
    "is_run_manifest",
    "load_data_manifest",
    "load_data_manifest_pointer",
    "output_region",
    "preflight",
    "read_run_manifest",
    "render_train_lm_config",
    "render_train_lm_yaml",
    "replace_run_identity",
    "resolve_cpt_budget",
    "resolve_midtrain_spec",
    "stage_cooldown_checkpoint",
    "submit_launch",
    "validate_midtrain_spec",
    "write_manifest",
    "write_run_manifest",
    "write_train_config",
]
