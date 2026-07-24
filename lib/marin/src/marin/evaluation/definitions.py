# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Reusable evaluation definitions and resolved worker-run contracts."""

from dataclasses import dataclass

from marin.evaluation.evalchemy import EvalchemyRunConfig
from marin.evaluation.evaluation_config import EvalTaskConfig
from marin.evaluation.harbor_runner import HarborRunConfig
from marin.evaluation.records import EvalRef


@dataclass(frozen=True)
class EvalchemyDefinition:
    """A named group of Evalchemy tasks."""

    name: str
    tasks: tuple[EvalTaskConfig, ...]
    max_gen_toks: int = 2048
    max_eval_instances: int | None = None


@dataclass(frozen=True)
class HarborDefinition:
    """A named Harbor benchmark."""

    name: str
    config: HarborRunConfig
    max_eval_instances: int | None = None


type EvalDefinition = EvalchemyDefinition | HarborDefinition


@dataclass(frozen=True)
class EvalRunIdentity:
    """Stable record identity shared by every resolved mechanism."""

    run_id: str
    created_at: str
    output_dir: str
    eval_ref: EvalRef


@dataclass(frozen=True)
class ResolvedEvalchemyRun:
    """One fully configured Evalchemy run in a worker payload."""

    identity: EvalRunIdentity
    config: EvalchemyRunConfig


@dataclass(frozen=True)
class ResolvedHarborRun:
    """One fully configured Harbor run in a worker payload."""

    identity: EvalRunIdentity
    config: HarborRunConfig


type ResolvedEvalRun = ResolvedEvalchemyRun | ResolvedHarborRun
