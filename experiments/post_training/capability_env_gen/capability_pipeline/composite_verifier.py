"""Harbor verifier adapter for deterministic machine checks plus native judging."""

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
    AssistantFinal,
    CodeAnswerVerifier,
    ContainerRuntime,
    Embedded,
    FileSubmission,
    FinalState,
    GradingResult,
    Outcome,
    Resource,
    ResourceRole,
    TaskTroveVerifier,
)
from taskcompendium.resources import materialize
from taskcompendium.serialization import from_json, renderings_from_json
from tasktrove_verify.spec import Mode

from capability_pipeline.composite_extension import (
    COMPOSITE_SPECIFICATION,
    PATCHED_VERIFIER_SHA256,
    validate_extension_marker,
)
from capability_pipeline.composite_policy import (
    aggregate_composite,
    composite_evidence_detail,
    consensus_disagreements,
    validate_composite_config,
)
from capability_pipeline.daytona_verifier import grade_in_daytona
from capability_pipeline.native_judge_protocol import RetryingTerminalScoreClient


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _adapter_sha256() -> str:
    return _sha256(Path(__file__))


def _policy_sha256() -> str:
    return _sha256(Path(__file__).with_name("composite_policy.py"))


def _with_composite_evidence(
    result: GradingResult,
    machine_results: list[dict],
    config_path: Path,
) -> GradingResult:
    """Retain completed private checks when the later judge is ungraded."""
    detail = composite_evidence_detail(
        dict(result.detail),
        machine_results,
        adapter_sha256=_adapter_sha256(),
        policy_sha256=_policy_sha256(),
        config_sha256=_sha256(config_path),
    )
    return GradingResult(result.status, result.reward, detail)


async def _run_judge_consensus(grade_once, policy, initial_samples):
    """Run the declared initial pass and exactly one adjudicator pass if needed."""
    initial = await grade_once(initial_samples)
    if initial.status != Outcome.GRADED or initial.reward is None:
        return initial, None
    if policy["judge"].get("consensus") is None:
        return initial, None
    disagreements = consensus_disagreements(
        {
            "status": initial.status.value,
            "reward": initial.reward,
            "detail": initial.detail,
        },
        policy,
    )
    adjudicator = await grade_once(1) if disagreements else None
    return initial, adjudicator


class CompositeSemanticVerifier(SemanticVerifier):
    """Execute private scripts in Daytona, then apply the pinned native judge."""

    async def _verify(self) -> VerifierResult:
        self._machine_captures = []
        root = self.task.paths.task_dir
        if _sha256(Path(native_verifier.__file__)) != PATCHED_VERIFIER_SHA256:
            raise RuntimeError(
                "composite verifier requires the pinned fail-closed TaskCompendium guard"
            )
        config_path = root / "composite-verifier.json"
        if not config_path.is_file():
            raise RuntimeError("composite verifier configuration is absent")
        specification_path = root / COMPOSITE_SPECIFICATION
        specification = from_json(specification_path.read_bytes())
        renderings = renderings_from_json((root / "renderings.json").read_bytes())
        names = json.loads((root / "manifest.json").read_text())["step_names"]
        step_index = names.index(self.step_name) if self.step_name is not None else 0
        policies = validate_composite_config(
            json.loads(config_path.read_text()),
            specification_sha256=_sha256(specification_path),
            adapter_sha256=_adapter_sha256(),
            policy_sha256=_policy_sha256(),
            step_count=len(specification.steps),
        )
        validate_extension_marker(
            root,
            adapter_sha256=_adapter_sha256(),
            policy_sha256=_policy_sha256(),
            config_sha256=_sha256(config_path),
            supported=True,
        )
        policy = policies[step_index]
        protocol = renderings[step_index]
        source_verifier = specification.steps[step_index].verifier
        consensus = policy["judge"].get("consensus")
        if (
            not isinstance(source_verifier, TaskTroveVerifier)
            or source_verifier.mode != Mode.JUDGE
            or source_verifier.judge is None
            or source_verifier.parameters.get("rubric") != "checklist"
            or source_verifier.parameters.get("exact_gate", False)
            or source_verifier.parameters.get("constraints")
            or len(source_verifier.parameters.get("criteria", ()))
            != len(policy["judge"]["criterion_weights"])
            or (
                consensus is not None
                and source_verifier.judge.policy.samples != consensus["initial_samples"]
            )
        ):
            return self._finish(
                GradingResult(
                    Outcome.INVALID_TASK,
                    None,
                    {
                        "error": "composite verifier requires checklist-only native judge with response evidence"
                    },
                )
            )
        response_path = self.trial_paths.agent_dir / "response.txt"
        transcript_path = self.trial_paths.agent_dir / "transcript.json"
        response = response_path.read_text() if response_path.exists() else None
        transcript = (
            tuple(json.loads(transcript_path.read_text()))
            if transcript_path.exists()
            else ()
        )
        paths = set(source_verifier.judge.view.files)
        if isinstance(protocol.submission, FileSubmission):
            paths.add(protocol.submission.path)
        elif isinstance(protocol.submission, FinalState):
            paths.update(protocol.submission.paths)
        machine_results = []
        with tempfile.TemporaryDirectory(prefix="composite-verifier-") as temporary:
            workspace = Path(temporary)
            workdir = self.task.config.environment.workdir or "/app"
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
            for check_index, check in enumerate(policy["machine_checks"]):
                machine = TaskTroveVerifier(
                    Mode.SCRIPT,
                    {"path": check["script_path"], "args": check.get("args", [])},
                    runtime=ContainerRuntime(
                        check["image"],
                        timeout=check["timeout"],
                        supervisor_python=check.get("supervisor_python", "python3"),
                    ),
                )
                machine_contract = machine
                machine_protocol = protocol
                if isinstance(protocol.submission, AssistantFinal):
                    machine_contract = CodeAnswerVerifier(
                        machine, ".composite/candidate.txt"
                    )
                elif isinstance(protocol.submission, FileSubmission):
                    machine_protocol = msgspec.structs.replace(
                        protocol,
                        submission=FinalState(tuple(sorted(paths))),
                    )
                composite_specification = msgspec.structs.replace(
                    specification,
                    steps=(
                        *specification.steps[:step_index],
                        msgspec.structs.replace(
                            specification.steps[step_index],
                            verifier=machine_contract,
                        ),
                        *specification.steps[step_index + 1 :],
                    ),
                )
                capture_relative = f"composite-machine-captures/check-{check_index:03d}"
                capture_root = self.trial_paths.verifier_dir / capture_relative
                result = await asyncio.to_thread(
                    grade_in_daytona,
                    composite_specification,
                    machine_protocol,
                    response,
                    workspace,
                    transcript,
                    step_index,
                    capture_root=capture_root,
                )
                self._machine_captures.append(
                    {
                        "check_id": check["id"],
                        "check_index": check_index,
                        "step_index": step_index,
                        "path": capture_relative,
                        "manifest_sha256": _sha256(capture_root / "manifest.json"),
                        "source_specification_sha256": _sha256(specification_path),
                        "source_renderings_sha256": _sha256(root / "renderings.json"),
                        "config_sha256": _sha256(config_path),
                    }
                )
                machine_results.append(
                    {
                        "id": check["id"],
                        "status": result.status.value,
                        "reward": result.reward,
                        "detail": result.detail,
                    }
                )
                if result.status != Outcome.GRADED or result.reward is None:
                    return self._finish(
                        _with_composite_evidence(result, machine_results, config_path)
                    )
            if any(
                check["role"] == "gate" and result["reward"] < 1.0
                for check, result in zip(
                    policy["machine_checks"], machine_results, strict=True
                )
            ):
                aggregated = aggregate_composite(machine_results, {}, policy)
                aggregated["detail"].update(
                    {
                        "composite_adapter_sha256": _adapter_sha256(),
                        "composite_policy_sha256": _policy_sha256(),
                        "composite_config_sha256": _sha256(config_path),
                    }
                )
                return self._finish(
                    GradingResult(
                        Outcome(aggregated["status"]),
                        aggregated["reward"],
                        aggregated["detail"],
                    )
                )
            context_path = "__composite_machine_context.json"
            if any(
                resource.path == context_path for resource in specification.resources
            ):
                return self._finish(
                    _with_composite_evidence(
                        GradingResult(
                            Outcome.INVALID_TASK,
                            None,
                            {
                                "error": "reserved composite judge context path is already used"
                            },
                        ),
                        machine_results,
                        config_path,
                    )
                )
            original_contexts = {}
            context_root = workspace / "original-context"
            materialize(specification, ResourceRole.VERIFIER, context_root, step_index)
            for path in source_verifier.judge.view.reference_context:
                context_file = context_root / path
                if not context_file.is_file():
                    return self._finish(
                        _with_composite_evidence(
                            GradingResult(
                                Outcome.INVALID_TASK,
                                None,
                                {"error": f"judge context is absent: {path}"},
                            ),
                            machine_results,
                            config_path,
                        )
                    )
                original_contexts[path] = context_file.read_text(errors="replace")
            context = json.dumps(
                {
                    "machine_results": machine_results,
                    "declared_reference_context": original_contexts,
                },
                sort_keys=True,
            ).encode()
            judge_parameters = dict(source_verifier.parameters)
            judge_parameters["context"] = context_path
            judge_config = msgspec.structs.replace(
                source_verifier.judge,
                view=msgspec.structs.replace(
                    source_verifier.judge.view,
                    reference_context=(context_path,),
                ),
            )
            judge_specification = msgspec.structs.replace(
                specification,
                resources=(
                    *specification.resources,
                    Resource(
                        context_path,
                        (ResourceRole.VERIFIER,),
                        Embedded(context),
                    ),
                ),
                steps=(
                    *specification.steps[:step_index],
                    msgspec.structs.replace(
                        specification.steps[step_index],
                        verifier=msgspec.structs.replace(
                            source_verifier,
                            parameters=judge_parameters,
                            judge=judge_config,
                        ),
                    ),
                    *specification.steps[step_index + 1 :],
                ),
            )
            tests = workspace / "tests"
            materialize(judge_specification, ResourceRole.VERIFIER, tests, step_index)

            async def grade_once(samples):
                configured = judge_specification
                if judge_config.policy.samples != samples:
                    configured_policy = msgspec.structs.replace(
                        judge_config.policy, samples=samples
                    )
                    configured_judge = msgspec.structs.replace(
                        judge_config, policy=configured_policy
                    )
                    configured = msgspec.structs.replace(
                        judge_specification,
                        steps=(
                            *judge_specification.steps[:step_index],
                            msgspec.structs.replace(
                                judge_specification.steps[step_index],
                                verifier=msgspec.structs.replace(
                                    judge_specification.steps[step_index].verifier,
                                    judge=configured_judge,
                                ),
                            ),
                            *judge_specification.steps[step_index + 1 :],
                        ),
                    )
                judge_client = (
                    RetryingTerminalScoreClient(self.judge_client)
                    if self.judge_client is not None
                    else None
                )
                result = await asyncio.to_thread(
                    grade_attempt,
                    configured,
                    protocol,
                    response,
                    workspace,
                    transcript,
                    judge_client,
                    step_index,
                )
                if judge_client is not None:
                    result = GradingResult(
                        result.status,
                        result.reward,
                        {
                            **dict(result.detail),
                            "judge_protocol": judge_client.evidence(),
                        },
                    )
                return result

            try:
                judge, adjudicator = await _run_judge_consensus(
                    grade_once, policy, judge_config.policy.samples
                )
            except (TypeError, ValueError) as error:
                return self._finish(
                    _with_composite_evidence(
                        GradingResult(
                            Outcome.INVALID_TASK,
                            None,
                            {"error": f"invalid composite consensus: {error}"},
                        ),
                        machine_results,
                        config_path,
                    )
                )
        if judge.status != Outcome.GRADED or judge.reward is None:
            return self._finish(
                _with_composite_evidence(judge, machine_results, config_path)
            )
        if adjudicator is not None and (
            adjudicator.status != Outcome.GRADED or adjudicator.reward is None
        ):
            return self._finish(
                _with_composite_evidence(adjudicator, machine_results, config_path)
            )
        try:
            aggregated = aggregate_composite(
                machine_results,
                {
                    "status": judge.status.value,
                    "reward": judge.reward,
                    "detail": judge.detail,
                },
                policy,
                None
                if adjudicator is None
                else {
                    "status": adjudicator.status.value,
                    "reward": adjudicator.reward,
                    "detail": adjudicator.detail,
                },
            )
        except (TypeError, ValueError) as error:
            return self._finish(
                _with_composite_evidence(
                    GradingResult(
                        Outcome.INVALID_TASK,
                        None,
                        {"error": f"invalid composite aggregation: {error}"},
                    ),
                    machine_results,
                    config_path,
                )
            )
        aggregated["detail"].update(
            {
                "composite_adapter_sha256": _adapter_sha256(),
                "composite_policy_sha256": _policy_sha256(),
                "composite_config_sha256": _sha256(config_path),
            }
        )
        return self._finish(
            GradingResult(
                Outcome(aggregated["status"]),
                aggregated["reward"],
                aggregated["detail"],
            )
        )

    def _finish(self, result: GradingResult) -> VerifierResult:
        # Attach replay metadata only after judging. It must not alter the
        # machine-result context presented to the native rubric judge.
        result.detail["composite_machine_captures"] = self._machine_captures
        self._write_result(result)
        if result.status == Outcome.EXTRACTION_ERROR:
            raise ExtractionError(json.dumps(result.detail))
        if result.status != Outcome.GRADED or result.reward is None:
            raise GradingInfrastructureError(json.dumps(result.detail))
        return VerifierResult(
            rewards={"reward": result.reward}, stdout=json.dumps(result.detail)
        )
