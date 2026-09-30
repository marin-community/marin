#!/usr/bin/env python3
"""Cold reconstruct an approved OCI image and check two isolated database boots.

This is a trusted controller. All image code runs in owned, network-blocked
Daytona sandboxes. Full TaskCompendium task validation is a subsequent gate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shlex
import sys
import tempfile
import time
from pathlib import Path

from capability_pipeline import image_runtime_metadata
from capability_pipeline.daytona_snapshot import (
    snapshot_not_found,
    validate_snapshot_recipe,
    wait_for_sandbox_deletion,
)
from capability_pipeline.generic_image_publication import published_registry_host
from capability_pipeline.provider_retry import provision_with_rate_limit_retry

PROBES = Path("/Users/k3sc0re/openathena/build_envs/envgen/probes")
SQL = "PGPASSWORD=storm psql -h 127.0.0.1 -U storm -d storm -X -v ON_ERROR_STOP=1 -tAc "


class ReadinessError(RuntimeError):
    def __init__(self, diagnostics: dict):
        super().__init__("cold image did not become SQL-ready")
        self.diagnostics = diagnostics


def checked_publication(
    plan_bytes: bytes, publication_bytes: bytes
) -> tuple[dict, str]:
    plan, publication = json.loads(plan_bytes), json.loads(publication_bytes)
    if plan.get("state") != "approved_for_capture" or plan.get("review_blockers") != []:
        raise ValueError("capture plan has not passed review")
    if publication.get("plan_sha256") != hashlib.sha256(plan_bytes).hexdigest():
        raise ValueError("publication belongs to a different capture plan")
    if publication.get("state") != "published_pending_cold_pull":
        raise ValueError("image publication is incomplete")
    matches = [
        image for image in plan["images"] if image["role"] == publication.get("role")
    ]
    if len(matches) != 1:
        raise ValueError("publication role is ambiguous")
    image = matches[0]
    review, transport = (
        publication.get("rootfs_review", {}),
        publication.get("publication", {}),
    )
    if (
        review.get("state") != "passed"
        or transport.get("state") != "integrity_verified"
    ):
        raise ValueError("image lacks content review or registry readback")
    if review.get("required_file_hashes") != image.get("required_ready_hashes"):
        raise ValueError("published image content is not the reviewed task content")
    reference = transport.get("image", "")
    try:
        host = published_registry_host(plan, publication)
    except (TypeError, ValueError) as error:
        raise ValueError("published image reference differs from reviewed repository") from error
    if (
        not isinstance(reference, str)
        or re.fullmatch(
            re.escape(host + "/" + image["repository"])
            + r"@sha256:[0-9a-f]{64}",
            reference,
        )
        is None
    ):
        raise ValueError("published image reference differs from reviewed repository")
    if reference.rsplit("@", 1)[1] != transport.get("manifest_digest"):
        raise ValueError("published manifest identity disagrees")
    return image, reference


def command(dtx, sandbox, text: str, timeout=60) -> str:
    result = dtx.sh(sandbox, text, timeout=timeout)
    if result.get("exit") != 0:
        raise RuntimeError("cold-image probe command failed")
    return result.get("stdout", "").strip()


def ready(dtx, sandbox) -> dict:
    for attempt in range(60):
        result = dtx.sh(
            sandbox,
            "test -f /tmp/task-ready && " + SQL + shlex.quote("SELECT 1"),
            timeout=30,
        )
        if result.get("exit") == 0 and result.get("stdout", "").strip() == "1":
            break
        time.sleep(2)
    else:
        diagnostics = {}
        for name, query in {
            "ready_marker": "ls -ld /tmp/task-ready",
            "postgres_ready": "pg_isready -h 127.0.0.1 -U storm -d storm",
            "postgres_startup": "tail -c 8192 /var/log/postgres-console.log",
            "entrypoint_processes": "ps -eo pid,ppid,comm | head -n 60",
        }.items():
            observed = dtx.sh(sandbox, query, timeout=30)
            diagnostics[name] = {
                "exit": observed.get("exit"),
                "stdout": observed.get("stdout", "")[:8192],
                "stderr": observed.get("stderr", "")[:2048],
            }
        raise ReadinessError(diagnostics)
    tables = command(
        dtx,
        sandbox,
        SQL
        + shlex.quote(
            "SELECT tablename FROM pg_tables WHERE schemaname='public' ORDER BY tablename"
        ),
    )
    return {
        "ready_attempts": attempt + 1,
        "public_tables": tables.splitlines(),
        "postgis_version": command(
            dtx, sandbox, SQL + shlex.quote("SELECT postgis_full_version()")
        ),
    }


def probe(args) -> dict:
    from daytona import CreateSnapshotParams, Image, Resources

    sys.path.insert(0, str(PROBES))
    import dtx  # type: ignore[import-not-found]

    plan_bytes, publication_bytes = (
        args.plan.read_bytes(),
        args.publication.read_bytes(),
    )
    image, reference = checked_publication(plan_bytes, publication_bytes)
    recipe = image_runtime_metadata.derive_daytona_recipe(reference)
    name = "cap-cold-" + hashlib.sha256(recipe.encode()).hexdigest()[:24]
    client = dtx.client()
    receipt = {
        "schema_version": "capability-image-cold-pull-v1",
        "state": "running",
        "role": image["role"],
        "image": reference,
        "plan_sha256": hashlib.sha256(plan_bytes).hexdigest(),
        "publication_sha256": hashlib.sha256(publication_bytes).hexdigest(),
        "probe_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "image_metadata_policy_sha256": hashlib.sha256(
            Path(image_runtime_metadata.__file__).read_bytes()
        ).hexdigest(),
        "image_metadata_catalog_sha256": hashlib.sha256(
            image_runtime_metadata.DEFAULT_CATALOG.read_bytes()
        ).hexdigest(),
        "reconstruction_recipe": recipe,
        "sandboxes": [],
        "cleanup": [],
        "task_gates": "pending",
    }
    owned = []
    stage = "snapshot_resolution"
    try:
        try:
            snapshot = client.snapshot.get(name)
            receipt["snapshot_created"] = False
        except Exception as error:
            if not snapshot_not_found(error):
                raise
            stage = "snapshot_creation"
            with tempfile.TemporaryDirectory(prefix="cold-image-recipe-") as directory:
                path = Path(directory) / "Dockerfile"
                path.write_text(recipe)
                client.snapshot.create(
                    CreateSnapshotParams(
                        name=name,
                        image=Image.from_dockerfile(str(path)),
                        resources=Resources(cpu=4, memory=8, disk=10),
                    ),
                    timeout=3600,
                )
            snapshot = client.snapshot.get(name)
            receipt["snapshot_created"] = True
        stage = "snapshot_binding"
        binding = validate_snapshot_recipe(
            snapshot, expected_name=name, expected_dockerfile=recipe
        )
        receipt["snapshot"] = {**binding, "id": snapshot.id, "ref": snapshot.ref}
        baselines = []
        for index in range(2):
            stage = f"sandbox_{index + 1}_provisioning"
            created, attempts = provision_with_rate_limit_retry(
                lambda: dtx.create(
                    client,
                    name,
                    purpose="capability-cold-image",
                    ttl_min=60,
                    block_all=True,
                )
            )
            sandbox, seconds = created
            owned.append(sandbox)
            observed = client.get(sandbox.id)
            record = {
                "sandbox_id": sandbox.id,
                "create_seconds": seconds,
                "provisioning_attempts": attempts,
                "network_block_all": getattr(observed, "network_block_all", None),
                "resources": {
                    key: getattr(observed, key, None)
                    for key in ("cpu", "memory", "disk")
                },
            }
            receipt["sandboxes"].append(record)
            if record["network_block_all"] is not True:
                raise RuntimeError(
                    "provider did not confirm blocked sandbox networking"
                )
            stage = f"sandbox_{index + 1}_readiness"
            baseline = ready(dtx, sandbox)
            baselines.append(baseline)
            record["database"] = baseline
            stage = f"sandbox_{index + 1}_content"
            matched = {}
            for path, expected in image["required_ready_hashes"].items():
                output = command(dtx, sandbox, "sha256sum -- " + shlex.quote(path))
                checksum = output.split()[0] if output else None
                if checksum != expected:
                    raise RuntimeError("cold image content differs from reviewed task")
                matched[path] = checksum
            record["required_file_hashes"] = matched
            stage = f"sandbox_{index + 1}_reset"
            command(
                dtx,
                sandbox,
                "test ! -e /workspace/out/cold-probe-state && "
                + SQL
                + shlex.quote(
                    "SELECT CASE WHEN to_regclass('public.cap_cold_probe') IS NULL THEN 1 ELSE 0 END"
                ),
            )
            if (
                command(
                    dtx,
                    sandbox,
                    SQL
                    + shlex.quote(
                        "SELECT to_regclass('public.cap_cold_probe') IS NULL"
                    ),
                )
                != "t"
            ):
                raise RuntimeError("database mutation survived into fresh sandbox")
            if image["role"] == "candidate":
                command(
                    dtx,
                    sandbox,
                    'test -d /workspace/out && test -z "$(ls -A /workspace/out)"',
                )
            record["database_and_workspace_pristine"] = True
            stage = f"sandbox_{index + 1}_network"
            # Reachability, not HTTP authentication success: even a registry 401
            # would prove that a fully blocked trial can reach the registry.
            host = reference.split("/", 1)[0]
            network_probe = (
                "import json,urllib.request,urllib.error\n"
                "try:\n"
                f" urllib.request.urlopen({('https://' + host + '/v2/')!r},timeout=8)\n"
                "except urllib.error.HTTPError:\n print(json.dumps({'reachable':True}))\n"
                "except (urllib.error.URLError,TimeoutError,OSError):\n print(json.dumps({'reachable':False}))\n"
                "else:\n print(json.dumps({'reachable':True}))\n"
            )
            network = json.loads(
                command(
                    dtx, sandbox, "python3 -c " + shlex.quote(network_probe), timeout=30
                )
            )
            record["registry_reachability"] = network
            if network.get("reachable") is not False:
                raise RuntimeError("registry is reachable from blocked trial")
            if index == 0:
                stage = "sandbox_1_mutation"
                command(
                    dtx,
                    sandbox,
                    SQL
                    + shlex.quote(
                        "CREATE TABLE cap_cold_probe(value integer); INSERT INTO cap_cold_probe VALUES (1)"
                    ),
                )
                command(
                    dtx,
                    sandbox,
                    "mkdir -p /workspace/out && printf probe > /workspace/out/cold-probe-state",
                )
                record["database_and_workspace_mutated"] = True
        stage = "baseline_comparison"
        if (
            baselines[0]["public_tables"] != baselines[1]["public_tables"]
            or baselines[0]["postgis_version"] != baselines[1]["postgis_version"]
        ):
            raise RuntimeError("fresh image boots disagree")
        receipt["state"] = "passed"
    except Exception as error:  # noqa: BLE001 - persist stage, never provider secrets
        receipt.update(
            state="failed", error_type=type(error).__name__, failed_stage=stage
        )
        if isinstance(error, ReadinessError):
            receipt["readiness_diagnostics"] = error.diagnostics
    finally:
        for sandbox in owned:
            record = {"sandbox_id": sandbox.id, "verified_absent": False}
            try:
                sandbox.delete()
                state, observations = wait_for_sandbox_deletion(client, sandbox.id)
                record.update(
                    verified_absent=state == "not_found", observations=observations
                )
            except Exception as error:  # noqa: BLE001 - preserve all cleanup observations
                record["error_type"] = type(error).__name__
            receipt["cleanup"].append(record)
        if any(not row["verified_absent"] for row in receipt["cleanup"]):
            receipt["state"] = "cleanup_unverified"
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    return receipt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--publication", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("cold-pull receipt already exists")
    receipt = probe(args)
    print(json.dumps({"state": receipt["state"], "receipt": str(args.output)}))
    return 0 if receipt["state"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
