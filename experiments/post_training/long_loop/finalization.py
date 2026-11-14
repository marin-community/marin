# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Continue the existing postprocessing graph after a recorded RL terminal outcome."""

import json
from dataclasses import dataclass, replace
from typing import Any

from marin.execution.lazy import ArtifactStep, StepContext
from marin.rl.skyrl import SkyRLRun
from rigging.filesystem.storage_path import StoragePath


def terminal_outcome(trained: str) -> dict:
    run = SkyRLRun.raw_load(trained)
    return json.loads(StoragePath(run.terminal_manifest_uri).read_text())["result"]


def record_finalization(*, output: str, trained: str) -> None:
    run = SkyRLRun.raw_load(trained)
    document = {
        "rl_terminal": terminal_outcome(trained),
        "selected_step": run.global_step,
        "export_uri": run.hf_model_uri,
        "status": "queued" if run.hf_model_uri is not None else "skipped_no_checkpoint",
    }
    (StoragePath(output) / "finalization.json").write_text(json.dumps(document, indent=2))


@dataclass(frozen=True)
class SkippedCheckpointStage:
    output: str
    trained: str


def after_checkpoint(step: ArtifactStep, trained: ArtifactStep[SkyRLRun]) -> ArtifactStep:
    """Keep one graph, skipping postprocessing before dispatch when no checkpoint was committed."""
    original_builder = step.build_config
    original_run = step.run

    def build_config(context: StepContext) -> Any:
        if not context.is_fingerprint and context.resolved(trained).hf_model_uri is None:
            return SkippedCheckpointStage(context.output_path, context.artifact_path(trained))
        return original_builder(context)

    def execute(config: Any):
        if not isinstance(config, SkippedCheckpointStage):
            return original_run(config)
        root = StoragePath(config.output)
        outcome = terminal_outcome(config.trained)
        (root / "finalization-skipped.json").write_text(
            json.dumps({"reason": "no_committed_best_checkpoint", "rl_terminal": outcome}, indent=2)
        )
        (root / "timing.md").write_text(
            "RL terminal outcome: "
            + json.dumps(outcome, sort_keys=True)
            + "\nNo committed best checkpoint exists. Export, evaluation and evidence archive were skipped.\n"
        )
        if step.artifact_type is SkyRLRun:
            return SkyRLRun.raw_load(config.trained).model_copy(update={"path": config.output})
        return None

    return replace(
        step,
        run=execute,
        build_config=build_config,
        deps=step.deps if trained in step.deps else (*step.deps, trained),
    )
