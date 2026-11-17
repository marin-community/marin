# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run ScienceAgentBench's isolated grading stages behind Harbor's verifier hook.

The prepared Harbor task keeps its TaskSpec in grading.json on the evaluator
host. This verifier captures only solution.py from the actor and does not upload
the tests directory there. Harbor retains the Pi agent loop and result reporting.
"""

import asyncio
import json
import os
from dataclasses import asdict, replace

from harbor.environments.daytona import DaytonaEnvironment
from harbor.models.verifier.result import VerifierResult
from harbor.verifier.base import BaseVerifier
from harbor.verifier.verifier import VerifierRuntimeError
from shellbox.image import RegistryImage
from taskcompendium.grading_result import Outcome
from taskcompendium.importers.scienceagentbench import SUBMISSION_PATH
from taskcompendium.models import ConversationTrace, GradingAttempt, ScriptGrader, TaskSpec, TextMessage
from taskcompendium.runtime.output_capture import MAX_OUTPUT_BYTES, captured_output_files
from taskcompendium.runtime.task_grading import sandbox_grade
from verifyit.modes.grade_judge import JudgeConnection

from experiments.evaluation.scienceagentbench_tasks import TASK_SPEC_FILENAME

CAPTURE_TIMEOUT = 60.0


class ScienceAgentBenchVerifier(BaseVerifier):
    """Use the actor's Daytona factory for separate replay and grading machines."""

    def __init__(self, *, judge_base_url: str | None = None, judge_api_key_env: str | None = None, **kwargs):
        super().__init__(**kwargs)
        self.judge_base_url = judge_base_url
        self.judge_api_key_env = judge_api_key_env

    async def verify(self) -> VerifierResult:
        environment = self.environment
        if not isinstance(environment, DaytonaEnvironment) or environment.machine is None:
            raise ValueError("ScienceAgentBench's initial Harbor integration requires a running Daytona actor")
        if environment.model_endpoint is not None:
            raise ValueError("Bind the visual judge to the verifier, not an actor-accessible Harbor relay")
        if self.override_env or self.verifier_env:
            raise ValueError(
                "Pass judge connection declarations through verifier kwargs, not guest environment overrides"
            )
        task = TaskSpec.model_validate_json((self.task.paths.task_dir / TASK_SPEC_FILENAME).read_text())
        if not isinstance(task.grader, ScriptGrader) or task.grader.replay is None:
            raise ValueError("ScienceAgentBench requires a replay-enabled script grader")
        if task.grader.environment is None or task.grader.environment.docker_image is None:
            raise ValueError("ScienceAgentBench requires a prepared grading image")
        if task.output_paths != (SUBMISSION_PATH,):
            raise ValueError("ScienceAgentBench must capture only its submitted Python program")
        if task.environment_requirements.docker_image != self.task.config.environment.docker_image:
            raise ValueError("Harbor and TaskSpec actor images must match")
        connection = None
        if task.grader.judge is not None:
            if self.judge_base_url is None or self.judge_api_key_env is None:
                raise ValueError("A visual grader requires an explicit host judge endpoint and credential variable")
            connection = JudgeConnection(self.judge_base_url, os.environ[self.judge_api_key_env])
        files = await captured_output_files(
            environment.machine,
            task.output_paths,
            timeout=CAPTURE_TIMEOUT,
            limit_bytes=MAX_OUTPUT_BYTES,
        )
        if SUBMISSION_PATH in files:
            await asyncio.to_thread((self.trial_paths.artifacts_dir / "solution.py").write_bytes, files[SUBMISSION_PATH])
        # File grading consumes only the captured program. Harbor keeps the Pi trajectory.
        attempt = GradingAttempt(
            ConversationTrace(events=(*task.context.events, TextMessage(role="assistant", content=""))), files
        )
        result = await sandbox_grade(
            task,
            attempt,
            environment.factory,
            replace(environment.machine.spec, source=RegistryImage(task.grader.environment.docker_image), env={}),
            judge_connection=connection,
        )
        await asyncio.to_thread(
            (self.trial_paths.verifier_dir / "taskcompendium-result.json").write_text,
            json.dumps(asdict(result), allow_nan=False) + "\n",
        )
        if result.status not in (Outcome.GRADED, Outcome.SUBMISSION_FAILURE) or result.reward is None:
            raise VerifierRuntimeError(result.error or f"ScienceAgentBench grading ended with {result.status}")
        rewards = {"reward": result.reward}
        await asyncio.to_thread(self.trial_paths.reward_json_path.write_text, json.dumps(rewards) + "\n")
        return VerifierResult(rewards=rewards)
