"""TaskCompendium executable-verifier isolation backed by Daytona 0.200.2."""

from __future__ import annotations

import contextvars
import hashlib
import json
import re
import shlex
import tempfile
from pathlib import Path
from urllib.parse import urlsplit

import httpx
import msgspec
import taskcompendium
import taskcompendium.harbor.verifier as _upstream_verifier
import tasktrove_verify
from daytona import (
    CreateSandboxFromSnapshotParams,
    CreateSnapshotParams,
    Image,
    Resources,
)
from taskcompendium.grading_paths import EXTERNAL_DIRECTORY
from taskcompendium.lowering import validate_workspace_submission
from taskcompendium.models import (
    ContainerRuntime,
    Embedded,
    GradingResult,
    ImageOverlay,
    Outcome,
    ResourceRef,
    ResourceRole,
    TaskSpec,
    verifier_runtime,
)
from taskcompendium.resources import resource_bytes
from taskcompendium.serialization import from_json, rendering_from_json, to_json

from . import sandbox_provider
from .composite_extension import BASE_VERIFIER_SHA256, PATCHED_VERIFIER_SHA256
from .daytona_environment import _dt, _run_portable
from .daytona_policy import (
    verifier_bootstrap_sha256,
    verifier_snapshot_recipe,
)
from .daytona_resources import VERIFIER_DEFAULT, resolve_profile, snapshot_name
from .daytona_snapshot import (
    snapshot_conflict,
    snapshot_not_found,
    wait_for_sandbox_deletion,
    wait_for_snapshot_active,
)
from .fixed_grading_capture import load_capture, materialize_workspace, write_capture
from .grading_input import grading_input_fingerprint
from .provider_retry import provision_with_rate_limit_retry


class VerifierImageCompatibilityError(RuntimeError):
    """Provider evidence proves the authored supervisor cannot host the pinned verifier."""

    def __init__(self, *, snapshot_name: str, snapshot_id: str, log_sha256: str, excerpt: str):
        self.reason = "supervisor_python_too_old"
        self.snapshot_name = snapshot_name
        self.snapshot_id = snapshot_id
        self.log_sha256 = log_sha256
        self.excerpt = excerpt
        self.requirement = "numpy==2.5.3"
        super().__init__(
            "verifier image supervisor Python is incompatible with pinned "
            "numpy==2.5.3 (requires Python >=3.12); "
            f"snapshot={snapshot_name}; provider_build_log_sha256={log_sha256}"
        )


def _compatibility_failure(client, name: str, definition: str):
    """Classify only the exact pinned dependency/Python mismatch from provider logs."""
    try:
        snapshot = client.snapshot.get(name)
        state = getattr(snapshot.state, "value", snapshot.state)
        info = getattr(snapshot, "build_info", None)
        if (
            snapshot.name != name
            or str(state).lower().removeprefix("snapshotstate.") != "error"
            or not snapshot.id
            or getattr(info, "dockerfile_content", None) != definition
        ):
            return None
        public_logs = getattr(client.snapshot, "build_logs", None)
        if callable(public_logs):
            # silo exposes build logs publicly.  The Daytona path below reaches
            # into a private SDK attribute, which on silo does not exist, so the
            # except below turned every lookup into None and this classifier
            # silently stopped classifying.
            text = public_logs(name)
            log = text.encode() if isinstance(text, str) else bytes(text)
        else:
            api = client.snapshot._SnapshotService__snapshots_api
            url = api.get_snapshot_build_logs_url(snapshot.id).url
            parsed = urlsplit(url)
            if (
                parsed.scheme != "https"
                or parsed.netloc != "daytonaproxy01.net"
                or parsed.username is not None
                or parsed.password is not None
            ):
                return None
            authorization = api.api_client.default_headers.get("Authorization")
            if not authorization:
                return None
            response = httpx.get(
                url, headers={"Authorization": authorization},
                timeout=20, follow_redirects=False,
            )
            response.raise_for_status()
            log = response.content
        if len(log) > 1_000_000:
            return None
    except Exception:  # noqa: BLE001 - classification must never mask provider failure
        return None
    if (
        b"No matching distribution found for numpy==2.5.3" not in log
        or re.search(rb"2\.5\.3 Requires-Python >=3\.12(?:[ ;,\r\n]|$)", log)
        is None
        or b"cp311" not in log
    ):
        return None
    return VerifierImageCompatibilityError(
        snapshot_name=name,
        snapshot_id=str(snapshot.id),
        log_sha256=hashlib.sha256(log).hexdigest(),
        excerpt=(
            "pip selected a cp311 wheel; numpy 2.5.3 requires Python >=3.12; "
            "pip found no matching distribution for numpy==2.5.3"
        ),
    )


def _adapter_sha256() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _cleanup_private_sandbox(sandbox, client, sandbox_id: str) -> dict:
    """Bounded deletion confirmation; never rerun the private grader."""
    attempts: list[dict] = []
    observations: list[dict] = []
    for attempt in (1, 2):
        try:
            sandbox.delete()
        except Exception as error:  # noqa: BLE001 - preserve completed grade
            attempts.append({
                "attempt": attempt,
                "state": "delete_error",
                "error_type": type(error).__name__,
            })
        else:
            attempts.append({"attempt": attempt, "state": "delete_requested"})
        try:
            state, checked = wait_for_sandbox_deletion(
                client, sandbox_id,
                delays=(0, 1, 2, 3, 4) if attempt == 1 else (0, 2, 4, 8, 12),
            )
        except Exception as error:  # noqa: BLE001 - preserve completed grade
            state = "lookup_error"
            checked = [{"state": state, "error_type": type(error).__name__}]
        observations.extend({"attempt": attempt, **item} for item in checked)
        if state == "not_found":
            return {
                "state": "deleted", "attempts": attempts,
                "observations": observations,
            }
    return {
        "state": "unconfirmed", "attempts": attempts,
        "observations": observations,
    }


def _ensure_snapshot(
    client,
    image: str,
    supervisor_python: str = "python3",
    resource_profile=None,
) -> str:
    definition = verifier_snapshot_recipe(image, supervisor_python)
    profile = resolve_profile(resource_profile, default=VERIFIER_DEFAULT)
    name = snapshot_name("cap-verifier", definition, profile)
    try:
        snapshot = client.snapshot.get(name)
    except Exception as error:
        if not snapshot_not_found(error):
            raise
    else:
        try:
            wait_for_snapshot_active(
                client.snapshot.get, name, definition, initial_snapshot=snapshot
            )
        except RuntimeError:
            compatibility = _compatibility_failure(client, name, definition)
            if compatibility is not None:
                raise compatibility from None
            raise
        return name
    with tempfile.NamedTemporaryFile(
        "w", suffix=".Dockerfile", delete=False
    ) as dockerfile:
        dockerfile.write(definition)
        dockerfile_path = Path(dockerfile.name)
    try:
        try:
            client.snapshot.create(
                CreateSnapshotParams(
                    name=name,
                    image=Image.from_dockerfile(str(dockerfile_path)),
                    resources=Resources(
                        cpu=profile.cpu,
                        memory=profile.memory_gb,
                        disk=profile.disk_gb,
                    ),
                ),
                timeout=3600,
            )
        except Exception as error:
            if not snapshot_conflict(error):
                raise
        try:
            wait_for_snapshot_active(client.snapshot.get, name, definition)
        except RuntimeError:
            compatibility = _compatibility_failure(client, name, definition)
            if compatibility is not None:
                raise compatibility from None
            raise
    finally:
        dockerfile_path.unlink(missing_ok=True)
    return name


def _embedded_specification(specification: TaskSpec, step_index: int) -> TaskSpec:
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
        steps=tuple(
            msgspec.structs.replace(step, resources=tuple(map(embed, step.resources)))
            if index == step_index
            else step
            for index, step in enumerate(specification.steps)
        ),
        resources=tuple(map(embed, specification.resources)),
    )


_CAPTURE_ROOT: contextvars.ContextVar[Path | None] = contextvars.ContextVar(
    "fixed_grading_capture_root", default=None
)

_CAPTURE_SOURCE: contextvars.ContextVar[tuple[str, str] | None] = (
    contextvars.ContextVar("fixed_grading_capture_source", default=None)
)


def grade_in_daytona(
    specification: TaskSpec,
    protocol,
    response: str | None,
    workspace: Path,
    transcript: tuple[dict, ...] = (),
    step_index: int = 0,
    resource_profile=None,
    *,
    capture_root: Path | None = None,
    embedded: bool = False,
    payload: bytes | None = None,
) -> GradingResult:
    """Run the pinned container supervisor in a fresh network-blocked sandbox."""

    validate_workspace_submission(specification, protocol, step_index)
    runtime = verifier_runtime(specification.steps[step_index].verifier)
    if not isinstance(runtime, ContainerRuntime):
        raise TypeError("Daytona container grading requires ContainerRuntime")
    if not embedded:
        specification = _embedded_specification(specification, step_index)
    embedded_specification = to_json(specification)
    protocol_value = msgspec.to_builtins(protocol)
    generated_payload = json.dumps(
        {
            "specification": json.loads(embedded_specification),
            "protocol": protocol_value,
            "step_index": step_index,
            "attempt": {"response": response, "transcript": transcript},
        }
    ).encode()
    if payload is None:
        payload = generated_payload
    elif not isinstance(payload, bytes):
        raise TypeError("private verifier payload override must be raw bytes")
    else:
        try:
            supplied = json.loads(payload)
        except (TypeError, ValueError) as error:
            raise ValueError("private verifier payload override is not JSON") from error
        if supplied != json.loads(generated_payload):
            raise ValueError("private verifier payload override does not bind this delivery")
    external_sources = []
    for directory in specification.requirements.state.additional_directories:
        source = workspace / EXTERNAL_DIRECTORY / directory.lstrip("/")
        source.mkdir(parents=True, exist_ok=True)
        external_sources.append((source, directory))
    input_fingerprint = grading_input_fingerprint(
        embedded_specification,
        protocol_value,
        response,
        workspace,
        transcript,
        step_index=step_index,
        payload=payload,
    )
    capture_root = capture_root or _CAPTURE_ROOT.get()
    if capture_root is not None:
        source_hashes = _CAPTURE_SOURCE.get()
        if source_hashes is None:
            source_hashes = (
                hashlib.sha256(embedded_specification).hexdigest(),
                hashlib.sha256(msgspec.json.encode(protocol)).hexdigest(),
            )
        write_capture(
            capture_root,
            specification=embedded_specification,
            protocol=msgspec.json.encode(protocol),
            response=response,
            transcript=transcript,
            payload=payload,
            workspace=workspace,
            fingerprint=input_fingerprint,
            source_specification_sha256=source_hashes[0],
            source_renderings_sha256=source_hashes[1],
        )
    dt = _dt()
    client = dt.client()
    snapshot_recipe = verifier_snapshot_recipe(runtime.image, runtime.supervisor_python)
    snapshot_recipe_sha256 = hashlib.sha256(snapshot_recipe.encode()).hexdigest()
    profile = resolve_profile(resource_profile, default=VERIFIER_DEFAULT)
    try:
        snapshot = _ensure_snapshot(
            client, runtime.image, runtime.supervisor_python, profile
        )
    except VerifierImageCompatibilityError as error:
        return GradingResult(
            Outcome.INFRA_ERROR,
            None,
            {
                "error": "authored verifier image has an incompatible supervisor Python",
                "verifier_image_compatibility": {
                    "schema_version": "capability-verifier-image-compatibility-v1",
                    "reason": error.reason,
                    "snapshot_name": error.snapshot_name,
                    "snapshot_id": error.snapshot_id,
                    "build_log_sha256": error.log_sha256,
                    "image": runtime.image,
                    "supervisor_python": runtime.supervisor_python,
                    "snapshot_recipe_sha256": snapshot_recipe_sha256,
                    "pinned_requirement": error.requirement,
                    "required_python": ">=3.12",
                    "observed_python": "3.11",
                    "provider_log_excerpt": error.excerpt,
                    "adapter_sha256": _adapter_sha256(),
                },
                "grading_input_fingerprint": input_fingerprint,
            },
        )
    parameters = CreateSandboxFromSnapshotParams(
        snapshot=snapshot,
        labels={"envgen": "1", "envgen_purpose": "private-verifier"},
        ephemeral=True,
        auto_stop_interval=0,
        ttl_minutes=max(30, int(runtime.timeout / 60) + 15),
        network_block_all=True,
    )
    sandbox, provisioning_attempts = provision_with_rate_limit_retry(
        lambda: client.create(parameters, timeout=600)
    )
    sandbox_id = sandbox.id
    grade_result: GradingResult | None = None
    try:
        workdir = specification.requirements.state.workdir
        preserved = (
            runtime.workspace.preserved_directories
            if isinstance(runtime.workspace, ImageOverlay)
            else ()
        )
        prepare = [
            "mkdir -p /input /result /tests /snapshot /opt/runtime",
            "chmod 700 /input /result /tests",
            (
                "find / -xdev -type f \\( -perm -4000 -o -perm -2000 \\) "
                "-exec chmod a-s {} +"
            ),
            "rm -rf /opt/runtime/taskcompendium /opt/runtime/tasktrove_verify",
        ]
        if not preserved:
            prepare.append(
                f"rm -rf {shlex.quote(workdir)} && mkdir -p {shlex.quote(workdir)}"
            )
        result = _run_portable(
            sandbox, " && ".join(prepare), "/", None, min(600, int(runtime.timeout))
        )
        if result["exit"] != 0:
            raise RuntimeError(f"private verifier preparation failed: {result}")
        uploads = (
            (workspace, "/snapshot"),
            (
                Path(taskcompendium.__file__).resolve().parent,
                "/opt/runtime/taskcompendium",
            ),
            (
                Path(tasktrove_verify.__file__).resolve().parent,
                "/opt/runtime/tasktrove_verify",
            ),
        )
        for source, target in uploads:
            uploaded = dt.upload_path(sandbox, source, target)
            if uploaded["exit"] != 0:
                raise RuntimeError(f"private verifier upload failed: {uploaded}")
        for source, directory in external_sources:
            uploaded = dt.upload_path(sandbox, source, directory)
            if uploaded["exit"] != 0:
                raise RuntimeError(
                    f"private external-directory upload failed: {uploaded}"
                )
            protected = _run_portable(
                sandbox,
                f"chmod -R a-w {shlex.quote(directory)}",
                "/",
                None,
                min(600, int(runtime.timeout)),
            )
            if protected["exit"] != 0:
                raise RuntimeError(
                    f"private external-directory protection failed: {protected}"
                )
        sandbox.fs.upload_file(payload, "/input/payload.json")
        command = (
            "chmod -R a-w /snapshot /opt/runtime && "
            f"{shlex.quote(runtime.supervisor_python)} -I "
            "/opt/runtime/taskcompendium/harbor/container_entry.py < /input/payload.json"
        )
        executed = _run_portable(
            sandbox, command, "/", None, max(1, int(runtime.timeout))
        )
        if executed["exit"] != 0:
            grade_result = GradingResult(
                Outcome.INFRA_ERROR,
                None,
                {
                    "error": "Daytona verifier did not produce a result",
                    "exit_code": executed["exit"],
                    "output": (executed["stderr"] or executed["stdout"])[-4000:],
                    "sandbox_id": sandbox_id,
                    "grading_input_fingerprint": input_fingerprint,
                },
            )
            return grade_result
        graded = msgspec.json.decode(executed["stdout"], type=GradingResult)
        graded.detail.update(
            {
                "verifier_isolation": sandbox_provider.isolation_id(),
                "verifier_sandbox_id": sandbox_id,
                "verifier_snapshot": snapshot,
                "verifier_adapter_sha256": _adapter_sha256(),
                # Keep the legacy field stable for default runtimes while
                # binding the actual interpreter-specific command separately.
                "verifier_bootstrap_sha256": verifier_bootstrap_sha256(),
                "verifier_bootstrap_command_sha256": verifier_bootstrap_sha256(
                    runtime.supervisor_python
                ),
                "verifier_snapshot_recipe_sha256": snapshot_recipe_sha256,
                "verifier_requested_resource_profile": profile.receipt(),
                "verifier_supervisor_python": runtime.supervisor_python,
                "verifier_provisioning_attempts": provisioning_attempts,
                "grading_input_fingerprint": input_fingerprint,
            }
        )
        grade_result = graded
        return grade_result
    finally:
        cleanup = _cleanup_private_sandbox(sandbox, client, sandbox_id)
        if grade_result is not None:
            grade_result.detail["verifier_cleanup"] = cleanup


def grade_captured_in_daytona(
    capture_root: Path,
    resource_profile=None,
    *,
    expected_manifest_sha256: str | None = None,
) -> GradingResult:
    """Grade one verified frozen delivery in a fresh private Daytona sandbox."""
    captured = load_capture(
        capture_root, expected_manifest_sha256=expected_manifest_sha256
    )
    specification = from_json(captured["specification"])
    protocol = rendering_from_json(captured["protocol"])
    payload = json.loads(captured["payload"])
    step_index = payload.get("step_index")
    if type(step_index) is not int:
        raise ValueError("captured grading payload has an invalid step index")
    with tempfile.TemporaryDirectory(prefix="fixed-grading-replay-") as temporary:
        workspace = materialize_workspace(captured, Path(temporary) / "workspace")
        result = grade_in_daytona(
            specification,
            protocol,
            captured["response"],
            workspace,
            tuple(captured["transcript"]),
            step_index,
            resource_profile,
            embedded=True,
            payload=captured["payload"],
        )
    expected = captured["manifest"].get("fingerprint")
    actual = result.detail.get("grading_input_fingerprint")
    if actual != expected:
        raise RuntimeError(
            "captured grading input fingerprint changed before private grading"
        )
    result.detail.update(
        {"fixed_grading_capture_manifest_sha256": captured["manifest_sha256"]}
    )
    return result


class CapturingDaytonaSemanticVerifier(_upstream_verifier.SemanticVerifier):
    """Pinned upstream verifier that records its downloaded input once per trial."""

    async def _verify(self):
        source_sha256 = hashlib.sha256(
            Path(_upstream_verifier.__file__).read_bytes()
        ).hexdigest()
        if source_sha256 not in {BASE_VERIFIER_SHA256, PATCHED_VERIFIER_SHA256}:
            raise RuntimeError(
                "capture verifier requires pinned TaskCompendium SemanticVerifier source"
            )
        capture_root = self.trial_paths.verifier_dir / "fixed-grading-capture"
        root = self.task.paths.task_dir
        source = (
            hashlib.sha256((root / "specification.json").read_bytes()).hexdigest(),
            hashlib.sha256((root / "renderings.json").read_bytes()).hexdigest(),
        )
        capture_token = _CAPTURE_ROOT.set(capture_root)
        source_token = _CAPTURE_SOURCE.set(source)
        try:
            return await super()._verify()
        finally:
            _CAPTURE_SOURCE.reset(source_token)
            _CAPTURE_ROOT.reset(capture_token)


# SemanticVerifier resolves this module-global at call time. This trusted import
# replaces only the container backend; parsing, evidence selection, semantic
# outcome handling, and all non-container grading remain pinned upstream.
_upstream_verifier.grade_in_container = grade_in_daytona
DaytonaSemanticVerifier = _upstream_verifier.SemanticVerifier
