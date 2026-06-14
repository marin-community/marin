# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""IID completion algorithm that runs generation in-process (no Fray dispatch).

Identical to :class:`IIDCompletionAlgorithm` except the completion step uses a
plain ``fn`` with no ``remote()`` wrapper and leaves ``ExecutorStep.resources``
unset. Per ``ExecutorStep.resources`` semantics, that makes the step run
in-thread in the task that runs the executor — on that task's own TPU — instead
of being dispatched as a separate Fray job. There is therefore no child TPU job
for the executor to region-pin.

This is the shape used to run the executor directly from N replica TPU tasks
(``iris job run --tpu … --replicas N`` with no ``--region``), so each replica's
work lands wherever iris schedules it, with ``--prefix mirror://`` handling
cross-replica dedup.

``worker_resources`` carried in the step config is inert in this mode:
``run_iid_completion_chunks`` never reads it, and the TPU comes from the replica
task itself rather than from a dispatched job's resources.
"""

from __future__ import annotations

from dataclasses import dataclass

from marin.execution.executor import ExecutorStep, InputName, MirroredValue
from marin.execution.types import this_output_path, versioned

from experiments.downstream_scaling.evals.algorithms.iid import (
    IIDCompletionStepConfig,
    IIDConfig,
    run_iid_completion_chunks,
)
from experiments.downstream_scaling.evals.utils import version_path


@dataclass(frozen=True)
class InlineIIDCompletionAlgorithm:
    config: IIDConfig

    def make_completions_step(
        self,
        *,
        name: str,
        model_path: str | InputName | MirroredValue,
        prompts_path: str | InputName | MirroredValue,
    ) -> ExecutorStep:
        return ExecutorStep(
            name=name,
            fn=run_iid_completion_chunks,
            config=IIDCompletionStepConfig(
                output_path=this_output_path(),
                model_path=version_path(model_path),  # type: ignore[arg-type]
                prompts_path=version_path(prompts_path),  # type: ignore[arg-type]
                sampling=versioned(self.config.sampling),  # type: ignore[arg-type]
                num_workers=self.config.execution.num_workers,
                chunk_size=versioned(self.config.execution.chunk_size),  # type: ignore[arg-type]
                worker_resources=self.config.execution.worker_resources,
            ),
        )
