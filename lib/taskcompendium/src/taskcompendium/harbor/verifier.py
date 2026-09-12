# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Harbor verifier that keeps semantic grading outside the agent's world."""

import asyncio
import json
import os
import shutil
import tempfile
from pathlib import Path, PurePosixPath

import msgspec
from harbor.models.verifier.result import VerifierResult
from harbor.verifier.base import BaseVerifier

from taskcompendium.grading import grade_attempt
from taskcompendium.grading_paths import EXTERNAL_DIRECTORY, submission_relative
from taskcompendium.harbor.container import grade_in_container
from taskcompendium.judging import OpenAIJudgeClient
from taskcompendium.models import ContainerRuntime, FileSubmission, FinalState, NoEnvironment, Outcome
from taskcompendium.serialization import from_json, protocol_from_json


class ExtractionError(RuntimeError):
    """The agent's submission cannot be decoded under its declared protocol."""


class GradingInfrastructureError(RuntimeError):
    """No semantic reward was produced because grading failed."""


class SemanticVerifier(BaseVerifier):
    """Download declared evidence and pass it to the canonical grading API."""

    def __init__(self, *args, judge_api_key_env: str | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.judge_client = OpenAIJudgeClient(os.environ[judge_api_key_env]) if judge_api_key_env else None

    async def _download_evidence(self, source: str, target: Path) -> None:
        if await self.environment.is_dir(source):
            target.mkdir(parents=True, exist_ok=True)
            await self.environment.download_dir(source, target)
        elif await self.environment.is_file(source):
            target.parent.mkdir(parents=True, exist_ok=True)
            await self.environment.download_file(source, target)

    async def verify(self) -> VerifierResult:
        root = self.task.paths.task_dir
        spec = from_json((root / "specification.json").read_bytes())
        protocol = protocol_from_json((root / "protocol.json").read_bytes())
        response_path = self.trial_paths.agent_dir / "response.txt"
        transcript_path = self.trial_paths.agent_dir / "transcript.json"
        response = response_path.read_text() if response_path.exists() else None
        transcript = tuple(json.loads(transcript_path.read_text())) if transcript_path.exists() else ()
        workdir = self.task.config.environment.workdir or "/app"
        paths: set[str] = set()
        if isinstance(protocol.submission, FileSubmission):
            paths.add(protocol.submission.path)
        elif isinstance(protocol.submission, FinalState):
            paths.update(protocol.submission.paths)
        if spec.verifier.judge is not None:
            paths.update(spec.verifier.judge.view.files)
        with tempfile.TemporaryDirectory(prefix="taskcompendium-evidence-") as temporary:
            workspace = Path(temporary)
            directories = (
                spec.environment.additional_directories if not isinstance(spec.environment, NoEnvironment) else ()
            )
            locations = [(path, submission_relative(path, workdir, directories)) for path in sorted(paths)]
            for path, relative in locations:
                if not relative.startswith(EXTERNAL_DIRECTORY + "/"):
                    await self._download_evidence(str(PurePosixPath(workdir) / path), workspace / relative)
            external = workspace / EXTERNAL_DIRECTORY
            if external.is_symlink() or external.is_file():
                external.unlink()
            elif external.is_dir():
                shutil.rmtree(external)
            for path, relative in locations:
                if relative.startswith(EXTERNAL_DIRECTORY + "/"):
                    await self._download_evidence(path, workspace / relative)
            if isinstance(spec.verifier_runtime, ContainerRuntime):
                result = await asyncio.to_thread(grade_in_container, spec, protocol, response, workspace, transcript)
            else:
                result = await asyncio.to_thread(
                    grade_attempt, spec, protocol, response, workspace, transcript, self.judge_client
                )

        self.trial_paths.verifier_dir.mkdir(parents=True, exist_ok=True)
        (self.trial_paths.verifier_dir / "taskcompendium-result.json").write_bytes(msgspec.json.encode(result))
        if result.status == Outcome.EXTRACTION_ERROR:
            raise ExtractionError(json.dumps(result.detail))
        if result.status != Outcome.GRADED or result.reward is None:
            raise GradingInfrastructureError(json.dumps(result.detail))
        return VerifierResult(rewards={"reward": result.reward}, stdout=json.dumps(result.detail))
