"""Pinned native semantic verifier with pre-extraction delivery attestations.

This adapter copies the small private ``SemanticVerifier._verify`` orchestration
because TaskCompendium exposes no callback between evidence download and answer
extraction.  It deliberately does not replace ``grade_attempt`` or mutate an
upstream module-global, so concurrent Harbor trials retain their native grader.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import shutil
import tempfile
from pathlib import Path, PurePosixPath

import msgspec
import taskcompendium.harbor.verifier as native_verifier
from harbor.models.verifier.result import VerifierResult
from taskcompendium.grading import grade_attempt
from taskcompendium.grading_paths import EXTERNAL_DIRECTORY, submission_relative
from taskcompendium.harbor.verifier import (
    ExtractionError,
    GradingInfrastructureError,
    SemanticVerifier,
)
from taskcompendium.models import (
    Embedded,
    FileSubmission,
    FinalState,
    GradingResult,
    Outcome,
    ResourceRef,
    ResourceRole,
    TaskTroveVerifier,
    verifier_runtime,
)
from taskcompendium.resources import resource_bytes
from taskcompendium.serialization import from_json, renderings_from_json, to_json
from tasktrove_verify.spec import Mode

from .composite_extension import BASE_VERIFIER_SHA256, PATCHED_VERIFIER_SHA256
from .grading_input import grading_input_fingerprint

RECEIPT_SCHEMA = "capability-native-verifier-receipt-v1"
NATIVE_MODES = frozenset(
    {
        Mode.MCQ,
        Mode.MATH,
        Mode.NUMERIC,
        Mode.EXACT,
        Mode.JSON_SCHEMA,
        Mode.XML_ELEMENTS,
        Mode.CSV_COLUMNS,
        Mode.IFEVAL,
    }
)
_COMPOSITE_PATHS = ("composite-specification.json", "composite-verifier.json")


def _sha256(value: bytes | Path) -> str:
    return hashlib.sha256(
        value if isinstance(value, bytes) else value.read_bytes()
    ).hexdigest()


def _adapter_sha256() -> str:
    return _sha256(Path(__file__))


def _embedded_specification(specification, step_index: int):
    """Freeze verifier resources once, before both receipt and native grading."""

    def embed(resource):
        if ResourceRole.VERIFIER in resource.roles and isinstance(
            resource.content, ResourceRef
        ):
            return msgspec.structs.replace(
                resource, content=Embedded(resource_bytes(resource))
            )
        return resource

    return msgspec.structs.replace(
        specification,
        resources=tuple(map(embed, specification.resources)),
        steps=tuple(
            msgspec.structs.replace(step, resources=tuple(map(embed, step.resources)))
            if index == step_index
            else step
            for index, step in enumerate(specification.steps)
        ),
    )


def _verified_native_source(root: Path) -> str:
    """Return the exact supported upstream source hash or fail before grading."""
    source_sha256 = _sha256(Path(native_verifier.__file__))
    if source_sha256 not in {BASE_VERIFIER_SHA256, PATCHED_VERIFIER_SHA256}:
        raise RuntimeError(
            "native diagnostic verifier requires the pinned TaskCompendium verifier source"
        )
    manifest = json.loads((root / "manifest.json").read_text())
    extensions = manifest.get("required_extensions", [])
    if extensions or any((root / path).exists() for path in _COMPOSITE_PATHS):
        raise RuntimeError(
            "native diagnostic verifier does not support composite or mandatory task extensions"
        )
    return source_sha256


def _receipt(
    result: GradingResult, fingerprint: dict, source_sha256: str
) -> GradingResult:
    detail = dict(result.detail)
    detail["grading_input_fingerprint"] = fingerprint
    detail["native_verifier_receipt"] = {
        "schema_version": RECEIPT_SCHEMA,
        "adapter_sha256": _adapter_sha256(),
        "semantic_verifier_sha256": source_sha256,
    }
    return GradingResult(result.status, result.reward, detail)


class NativeDiagnosticSemanticVerifier(SemanticVerifier):
    """Stock native semantic grading, with a receipt made before extraction."""

    async def _verify(self) -> VerifierResult:
        root = self.task.paths.task_dir
        source_sha256 = _verified_native_source(root)
        specification_bytes = (root / "specification.json").read_bytes()
        renderings_bytes = (root / "renderings.json").read_bytes()
        specification = from_json(specification_bytes)
        renderings = renderings_from_json(renderings_bytes)
        names = json.loads((root / "manifest.json").read_text())["step_names"]
        step_index = names.index(self.step_name) if self.step_name is not None else 0
        specification = _embedded_specification(specification, step_index)
        protocol = renderings[step_index]
        source_verifier = specification.steps[step_index].verifier
        if not isinstance(source_verifier, TaskTroveVerifier):
            raise TypeError(
                "native diagnostic verifier requires a direct TaskTrove verifier"
            )
        if (
            source_verifier.mode not in NATIVE_MODES
            or source_verifier.judge is not None
            or verifier_runtime(source_verifier) is not None
        ):
            raise RuntimeError(
                "native diagnostic verifier requires an allowed non-judge runtime-null mode"
            )

        response_path = self.trial_paths.agent_dir / "response.txt"
        transcript_path = self.trial_paths.agent_dir / "transcript.json"
        response = response_path.read_text() if response_path.exists() else None
        transcript = (
            tuple(json.loads(transcript_path.read_text()))
            if transcript_path.exists()
            else ()
        )
        workdir = self.task.config.environment.workdir or "/app"
        paths: set[str] = set()
        if isinstance(protocol.submission, FileSubmission):
            paths.add(protocol.submission.path)
        elif isinstance(protocol.submission, FinalState):
            paths.update(protocol.submission.paths)
        with tempfile.TemporaryDirectory(
            prefix="native-diagnostic-evidence-"
        ) as temporary:
            workspace = Path(temporary)
            directories = specification.requirements.state.additional_directories
            locations = [
                (path, submission_relative(path, workdir, directories))
                for path in sorted(paths)
            ]
            for path, relative in locations:
                if not relative.startswith(EXTERNAL_DIRECTORY + "/"):
                    await self._download_evidence(
                        str(PurePosixPath(workdir) / path), workspace / relative
                    )
            external = workspace / EXTERNAL_DIRECTORY
            if external.is_symlink() or external.is_file():
                external.unlink()
            elif external.is_dir():
                shutil.rmtree(external)
            for path, relative in locations:
                if relative.startswith(EXTERNAL_DIRECTORY + "/"):
                    await self._download_evidence(path, workspace / relative)
            fingerprint = grading_input_fingerprint(
                to_json(specification),
                msgspec.json.encode(protocol),
                response,
                workspace,
                transcript,
                step_index=step_index,
                payload=json.dumps(
                    {
                        "specification_file_sha256": _sha256(specification_bytes),
                        "renderings_file_sha256": _sha256(renderings_bytes),
                        "step_index": step_index,
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode(),
            )
            result = await asyncio.to_thread(
                grade_attempt,
                specification,
                protocol,
                response,
                workspace,
                transcript,
                self.judge_client,
                step_index,
            )
        result = _receipt(result, fingerprint, source_sha256)
        self._write_result(result)
        if result.status == Outcome.EXTRACTION_ERROR:
            raise ExtractionError(json.dumps(result.detail))
        if result.status != Outcome.GRADED or result.reward is None:
            raise GradingInfrastructureError(json.dumps(result.detail))
        return VerifierResult(
            rewards={"reward": result.reward}, stdout=json.dumps(result.detail)
        )
