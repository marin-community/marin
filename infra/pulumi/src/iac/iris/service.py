# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Pulumi component for an always-on Iris service job.

Wraps the :mod:`iac.iris.deploy` CLI in a ``command.local.Command``: Command state
holds only strings (a dynamic provider would pickle provider code into the state
file, where it goes stale), and the delete verb runs the current checkout's CLI.
The spec rides in the Command's environment, so ``pulumi destroy`` replays the spec
the resource was created with and nothing stack-controlled is interpolated into a
shell line.

Update semantics: any change to the serialized spec or to the code hash runs ``up``
again (an in-place RECREATE of the job); the resource is never replaced, so a spec
change cannot pass through ``down``.
"""

import hashlib
import json
import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import pulumi
import pulumi_command as command
from google.protobuf import json_format
from iris.cluster.types import ResourceSpec

from iac.iris.spec import ALWAYS_ON_RETRIES, DEFAULT_READY_WAIT, SPEC_ENV_VAR, ServiceSpec

CODE_HASH_ENV_VAR = "IRIS_SVC_CODE_HASH"
DEPLOY_COMMAND = ".venv/bin/python -m iac.iris.deploy"


@dataclass(frozen=True)
class IrisServiceArgs:
    """Deploy-shape of one Iris service. Field semantics are documented on
    :class:`iac.iris.spec.ServiceSpec`; the extras here are component-side only."""

    cluster: str
    name: str
    user: str
    entrypoint: tuple[str, ...]
    resources: ResourceSpec
    regions: tuple[str, ...]
    port: str
    endpoint: str
    health_path: str
    env: dict[str, str] = field(default_factory=dict)
    secret_env: dict[str, str] = field(default_factory=dict)
    pip_packages: tuple[str, ...] = ()
    sync_packages: tuple[str, ...] = ()
    wait: int = DEFAULT_READY_WAIT
    deploy_generation: int = 0
    code_paths: tuple[str, ...] = ()
    """Redeploy-trigger scope: git-tracked files under these paths are content-hashed.
    Changes outside them (shared libraries) do not trigger a redeploy — the same
    scoping CI path filters apply today. Widen per service to change that."""
    build_commands: tuple[str, ...] = ()
    extra_bundle_includes: tuple[str, ...] = ()
    max_retries_preemption: int = ALWAYS_ON_RETRIES
    max_retries_failure: int = ALWAYS_ON_RETRIES
    max_task_failures: int = ALWAYS_ON_RETRIES


def workspace_root() -> Path:
    """Toplevel of the git checkout the Pulumi program runs in (the Iris workspace)."""
    result = subprocess.run(["git", "rev-parse", "--show-toplevel"], check=True, capture_output=True, text=True)
    return Path(result.stdout.strip())


def code_hash(root: Path, code_paths: tuple[str, ...]) -> str:
    """Deterministic content hash of the service's git-tracked sources under ``code_paths``.

    Hashes (path, content) pairs, so it is stable across checkouts and mtimes — the
    workspace-zip sha256 the controller stores is not, because the zip embeds file
    metadata. Generated build outputs (``extra_bundle_includes``) are deliberately
    absent: ``up`` rebuilds them on every deploy and their sources are tracked under
    ``code_paths``, so hashing built bytes would only make the trigger depend on
    which machine built last.
    """
    files: set[Path] = set()
    if code_paths:
        result = subprocess.run(
            ["git", "ls-files", "-z", "--", *code_paths],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        )
        files.update(root / name for name in result.stdout.split("\0") if name)
    digest = hashlib.sha256()
    for path in sorted(files):
        digest.update(str(path.relative_to(root)).encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def wire_spec(args: IrisServiceArgs) -> ServiceSpec:
    """Project the component args onto the JSON-native wire spec."""
    return ServiceSpec(
        cluster=args.cluster,
        name=args.name,
        user=args.user,
        entrypoint=args.entrypoint,
        resources=json_format.MessageToDict(args.resources.to_proto()),
        regions=args.regions,
        port=args.port,
        endpoint=args.endpoint,
        health_path=args.health_path,
        env=dict(args.env),
        secret_env=dict(args.secret_env),
        pip_packages=args.pip_packages,
        sync_packages=args.sync_packages,
        wait=args.wait,
        deploy_generation=args.deploy_generation,
        build_commands=args.build_commands,
        extra_bundle_includes=args.extra_bundle_includes,
        max_retries_preemption=args.max_retries_preemption,
        max_retries_failure=args.max_retries_failure,
        max_task_failures=args.max_task_failures,
    )


def _parse_outputs(stdout: str) -> dict:
    try:
        outputs = json.loads(stdout)
    except json.JSONDecodeError as exc:
        raise ValueError(f"iac.iris.deploy stdout is not a JSON document: {stdout!r}") from exc
    if not isinstance(outputs, dict) or "job_id" not in outputs or "url" not in outputs:
        raise ValueError(f"iac.iris.deploy outputs missing job_id/url: {stdout!r}")
    return outputs


class IrisService(pulumi.ComponentResource):
    """An always-on Iris service job, deployed through the ``iac.iris.deploy`` CLI."""

    job_id: pulumi.Output[str]
    url: pulumi.Output[str]

    def __init__(
        self,
        name: str,
        args: IrisServiceArgs,
        opts: pulumi.ResourceOptions | None = None,
    ) -> None:
        super().__init__("marin:iris:Service", name, None, opts)
        spec = wire_spec(args)
        spec.validate()

        root = workspace_root()
        digest = code_hash(root, args.code_paths)
        # Relative dir keeps state portable across checkouts (an absolute path
        # recorded at create time would replay wrong on another machine).
        rel_root = os.path.relpath(root, Path.cwd())

        job = command.local.Command(
            f"{name}-job",
            create=f"{DEPLOY_COMMAND} up",
            update=f"{DEPLOY_COMMAND} up",
            delete=f"{DEPLOY_COMMAND} down",
            environment={SPEC_ENV_VAR: spec.to_json(), CODE_HASH_ENV_VAR: digest},
            dir=rel_root,
            opts=pulumi.ResourceOptions(parent=self),
        )
        outputs = job.stdout.apply(_parse_outputs)
        self.job_id = outputs.apply(lambda o: o["job_id"])
        self.url = outputs.apply(lambda o: o["url"])
        self.register_outputs({"job_id": self.job_id, "url": self.url})
