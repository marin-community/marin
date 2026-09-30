"""Validate captured private grades against their original Harbor evidence."""

from __future__ import annotations

import json
from pathlib import Path

from .fixed_grading_capture import load_capture
from .runtime import provider_isolation_record, sha256, verifier_isolation_record


def _read(path: Path) -> dict:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"missing or linked artifact: {path.name}")
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise TypeError("artifact must be a JSON object")
    return value


def _bound(path: Path, digest: object) -> dict:
    value = _read(path)
    if not isinstance(digest, str) or sha256(path) != digest:
        raise ValueError(f"artifact hash differs: {path.name}")
    return value


def artifact_issues(
    plan: dict, report: dict, output: Path, controls: dict
) -> list[str]:
    """Require one real capture per case and independently isolated replay cells."""
    try:
        return _validate(plan, report, output, controls)
    except (
        OSError,
        ValueError,
        TypeError,
        KeyError,
        AttributeError,
        RuntimeError,
    ) as error:
        return [f"captured grading evidence: {error}"]


def _validate(plan: dict, report: dict, output: Path, controls: dict) -> list[str]:
    if (
        plan.get("binding_kind") not in {"docker", "shellsim"}
        or plan.get("verifier_surface") == "native_deterministic"
    ):
        raise ValueError("capture strategy requires Docker or ShellSim executable grading")
    if output.is_symlink() or any(p.is_symlink() for p in output.rglob("*")):
        raise ValueError("linked captured evidence")
    expected_cases = {case["id"] for case in controls["cases"]}
    captures = report.get("captures")
    if (
        not isinstance(captures, list)
        or len(captures) != len(expected_cases)
        or any(not isinstance(row, dict) for row in captures)
        or {row.get("case_id") for row in captures} != expected_cases
    ):
        raise ValueError("capture inventory differs from controls")
    inputs = output.parent / "plan-bundle" / "input" / "bundle"
    source_spec = sha256(inputs / "specification.json")
    source_renderings = sha256(inputs / "renderings.json")
    trials = output / "runtime-trials"
    candidate_ids: set[str] = set()
    shellsim_sessions: set[str] = set()
    shellsim_snapshots: set[str] = set()
    verifier_ids: set[str] = set()
    records = {}
    # Collect every candidate before checking private sandbox disjointness.
    for row in captures:
        case = row["case_id"]
        name = f"capture-{case}"
        if row.get("trial_name") != name:
            raise ValueError("capture trial name differs")
        root = trials / name
        if plan["binding_kind"] == "shellsim":
            from .non_docker_reset import shellsim_candidate_record

            candidate = shellsim_candidate_record(
                root, shellsim_sessions, report.get("shellsim_bridge_sha256"),
            )
            shellsim_snapshots.add(candidate["initial_snapshot_sha256"])
        else:
            candidate = provider_isolation_record(
                root / "daytona-environment.json", trials, candidate_ids
            )
        if candidate != row.get("candidate_environment"):
            raise ValueError("capture candidate receipt differs")
        records[case] = (row, root)
    if plan["binding_kind"] == "shellsim" and len(shellsim_snapshots) != 1:
        raise ValueError("ShellSim initial VFS differs across captured cases")
    frozen = {}
    for case, (row, root) in records.items():
        relative = f"runtime-trials/capture-{case}/verifier/fixed-grading-capture"
        if row.get("capture_path") != relative:
            raise ValueError("capture path differs")
        capture = load_capture(output / relative)
        manifest = capture["manifest"]
        response_path = root / "agent" / "response.txt"
        transcript_path = root / "agent" / "transcript.json"
        response = response_path.read_text() if response_path.exists() else None
        transcript = (
            json.loads(transcript_path.read_text()) if transcript_path.exists() else []
        )
        if capture["response"] != response or capture["transcript"] != transcript:
            raise ValueError(
                "capture response or transcript differs from original agent"
            )
        if (
            capture["manifest_sha256"] != row.get("capture_manifest_sha256")
            or manifest.get("source_specification_sha256") != source_spec
            or manifest.get("source_renderings_sha256") != source_renderings
        ):
            raise ValueError("capture source binding differs")
        trial = _bound(root / "result.json", row.get("trial_sha256"))
        grade = _bound(
            root / "verifier/taskcompendium-result.json", row.get("grading_sha256")
        )
        detail = grade.get("detail", {})
        if (
            trial.get("exception_info") is not None
            or "exception_info" not in trial
            or grade.get("status") != "graded"
            or not isinstance(grade.get("reward"), (int, float))
            or trial.get("verifier_result", {}).get("rewards", {}).get("reward")
            != grade["reward"]
            or row.get("status") != grade["status"]
            or row.get("reward") != grade["reward"]
            or row.get("exception") is not None
            or row.get("verifier_result_present") is not True
            or detail.get("grading_input_fingerprint") != manifest["fingerprint"]
            or row.get("grading_input_fingerprint") != manifest["fingerprint"]
        ):
            raise ValueError("original capture trial does not prove completed grading")
        original_private = verifier_isolation_record(
            grade,
            verifier_ids,
            candidate_ids,
            runtime_image=plan["runtime"]["image"],
            supervisor_python=plan["runtime"]["supervisor_python"],
        )
        if row.get("private_verifier") != original_private:
            raise ValueError("original private verifier receipt differs")
        frozen[case] = capture
    cells = report.get("cells")
    planned = {cell["trial_name"]: cell for cell in plan["cells"]}
    if (
        not isinstance(cells, list)
        or len(cells) != len(planned)
        or any(not isinstance(row, dict) for row in cells)
        or {row.get("trial_name") for row in cells} != set(planned)
    ):
        raise ValueError("captured grade matrix differs")
    for row in cells:
        cell = planned[row["trial_name"]]
        root = trials / cell["trial_name"]
        capture = frozen[cell["case_id"]]
        grade = _bound(root / "private-grade.json", row.get("grading_sha256"))
        if (root / "result.json").exists():
            raise ValueError("direct private grade must not fabricate a Harbor trial")
        detail = grade.get("detail", {})
        if (
            grade.get("schema_version") != "capability-captured-private-grade-v1"
            or grade.get("cell") != cell
            or any(row.get(key) != value for key, value in cell.items())
            or row.get("transport") != "captured_private_grade"
            or row.get("trial_sha256") is not None
            or row.get("verifier_result_present") is not None
            or row.get("exception") is not None
            or grade.get("status") != "graded"
            or row.get("outcome_class") != "graded"
            or row.get("status") != grade.get("status")
            or row.get("reward") != grade.get("reward")
            or grade.get("capture_manifest_sha256") != capture["manifest_sha256"]
            or row.get("capture_manifest_sha256") != capture["manifest_sha256"]
            or detail.get("fixed_grading_capture_manifest_sha256")
            != capture["manifest_sha256"]
            or detail.get("grading_input_fingerprint")
            != capture["manifest"]["fingerprint"]
            or row.get("grading_input_fingerprint")
            != capture["manifest"]["fingerprint"]
        ):
            raise ValueError("private grade does not bind frozen cell and capture")
        private = verifier_isolation_record(
            grade,
            verifier_ids,
            candidate_ids,
            runtime_image=plan["runtime"]["image"],
            supervisor_python=plan["runtime"]["supervisor_python"],
        )
        if private != row.get("private_verifier"):
            raise ValueError("private grade isolation differs")
    return []
