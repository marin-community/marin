# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run ScienceAgentBench's isolated grading stages behind Harbor's verifier hook.

The prepared Harbor task keeps its TaskSpec in grading.json on the evaluator
host. This verifier captures only solution.py from the actor and does not upload
the tests directory there. Harbor retains the Pi agent loop and result reporting.
"""

import asyncio
import hashlib
import json
import os
from dataclasses import asdict, replace
from pathlib import Path, PurePosixPath
from tempfile import TemporaryDirectory

from harbor.environments.daytona import DaytonaEnvironment
from harbor.models.verifier.result import VerifierResult
from harbor.verifier.base import BaseVerifier
from harbor.verifier.verifier import VerifierRuntimeError
from shellbox.image import DockerfileSource, RegistryImage, image_source_key
from taskcompendium.grading_result import Outcome
from taskcompendium.importers.scienceagentbench import SUBMISSION_PATH
from taskcompendium.models import ConversationTrace, GraderMachine, GradingAttempt, ScriptGrader, TaskSpec, TextMessage
from taskcompendium.runtime.output_capture import MAX_OUTPUT_BYTES, captured_output_files
from taskcompendium.runtime.resources import materialize_build_context
from taskcompendium.runtime.task_grading import sandbox_grade
from verifyit.modes.grade_judge import JudgeConnection

from experiments.evaluation.scienceagentbench_tasks import ACTOR_BUILD_KEY, TASK_SPEC_FILENAME, TASK_SPEC_HASH_KEY

CAPTURE_TIMEOUT = 60.0


class ScienceAgentBenchVerifier(BaseVerifier):
    """Capture the submitted program and publish isolated replay/grading outcomes."""

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
        specification = (self.task.paths.task_dir / TASK_SPEC_FILENAME).read_bytes()
        if hashlib.sha256(specification).hexdigest() != self.task.config.metadata.get(TASK_SPEC_HASH_KEY):
            raise ValueError("Private grading specification does not match the pinned Harbor task")
        task = TaskSpec.model_validate_json(specification)
        if not isinstance(task.grader, ScriptGrader) or task.grader.replay is None:
            raise ValueError("ScienceAgentBench requires a replay-enabled script grader")
        if not isinstance(task.grader.placement, GraderMachine):
            raise ValueError("ScienceAgentBench requires a prepared grading image")
        if task.output_paths != (SUBMISSION_PATH,):
            raise ValueError("ScienceAgentBench must capture only its submitted Python program")
        if task.environment_requirements.docker_image != self.task.config.environment.docker_image:
            raise ValueError("Harbor and TaskSpec actor images must match")
        if task.environment_requirements != task.grader.replay.environment:
            raise ValueError("The actor and replay environments must match")
        if task.environment_requirements.docker_build is not None:
            actor_source = environment.machine.spec.source
            if not isinstance(actor_source, DockerfileSource) or (
                await asyncio.to_thread(image_source_key, actor_source)
            ) != self.task.config.metadata.get(ACTOR_BUILD_KEY):
                raise ValueError("Actor build context differs from the exported Harbor task")
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
            await asyncio.to_thread(
                (self.trial_paths.artifacts_dir / PurePosixPath(SUBMISSION_PATH).name).write_bytes,
                files[SUBMISSION_PATH],
            )
        # File grading consumes only the captured program. Harbor keeps the Pi trajectory.
        attempt = GradingAttempt(
            ConversationTrace(events=(*task.context.events, TextMessage(role="assistant", content=""))), files
        )
        requirements = task.grader.placement.environment
        with TemporaryDirectory(prefix="scienceagentbench-grader-") as temporary:
            if requirements.docker_build is not None:
                source = await asyncio.to_thread(
                    materialize_build_context, requirements.docker_build, Path(temporary) / "build"
                )
            elif requirements.docker_image is not None:
                source = RegistryImage(requirements.docker_image)
            else:
                raise ValueError("ScienceAgentBench requires a grader image or Docker build recipe")
            result = await sandbox_grade(
                task,
                attempt,
                environment.factory,
                replace(environment.machine.spec, source=source, env={}),
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
