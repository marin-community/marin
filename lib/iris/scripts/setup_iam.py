#!/usr/bin/env python3
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Initialize Iris GCP service accounts and IAM bindings.

Creates a controller service account and a worker service account if missing,
grants their project roles, and wires impersonation / act-as bindings for the
configured CI and operator principals.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import tempfile

import click

logger = logging.getLogger("setup-iam")

# Marin-specific default CI principal used by the repo's Iris GCP workflows.
DEFAULT_CI_PRINCIPAL = "serviceAccount:iris-ci-smoke@hai-gcp-models.iam.gserviceaccount.com"
DEFAULT_CONTROLLER_SA_ID = "iris-controller"
DEFAULT_WORKER_SA_ID = "iris-worker"
REQUIRED_APIS = (
    "compute.googleapis.com",
    "iam.googleapis.com",
    "oslogin.googleapis.com",
    "tpu.googleapis.com",
)
CONTROLLER_PROJECT_ROLES = (
    "roles/artifactregistry.reader",
    "roles/compute.instanceAdmin.v1",
    "roles/compute.osAdminLogin",
    "roles/logging.logWriter",
    "roles/monitoring.metricWriter",
    "roles/storage.objectAdmin",
    "roles/tpu.admin",
)
WORKER_PROJECT_ROLES = (
    "roles/artifactregistry.reader",
    "roles/compute.osAdminLogin",
    "roles/logging.logWriter",
    "roles/monitoring.metricWriter",
    "roles/storage.objectAdmin",
    "roles/storage.bucketViewer",
    "roles/tpu.viewer",
)
IMPERSONATION_ROLES = (
    "roles/iam.serviceAccountTokenCreator",
    "roles/iam.serviceAccountUser",
)
PROJECT_USER_IMPERSONATION_ROLES = (
    "roles/iam.serviceAccountTokenCreator",
    "roles/iam.serviceAccountUser",
)
# Project-level roles granted to human users so they can SSH to instances
# via metadata-style keys (gcloud compute ssh).
USER_PROJECT_ROLES = ("roles/compute.instanceAdmin.v1",)


def _principal_member(principal: str) -> str:
    if ":" in principal:
        return principal
    if principal.endswith(".iam.gserviceaccount.com"):
        return f"serviceAccount:{principal}"
    return f"user:{principal}"


def _run(cmd: list[str], *, dry_run: bool = False, capture_output: bool = False) -> subprocess.CompletedProcess[str]:
    logger.info("$ %s", " ".join(cmd))
    if dry_run:
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
    return subprocess.run(cmd, check=True, text=True, capture_output=capture_output)


def _enabled_apis(project: str) -> set[str]:
    result = _run(
        [
            "gcloud",
            "services",
            "list",
            "--enabled",
            f"--project={project}",
            "--format=value(config.name)",
        ],
        capture_output=True,
    )
    return {line.strip() for line in result.stdout.splitlines() if line.strip()}


def _require_apis(project: str) -> None:
    enabled = _enabled_apis(project)
    missing = [api for api in REQUIRED_APIS if api not in enabled]
    if missing:
        raise RuntimeError(
            f"Project {project} is missing required APIs: {', '.join(missing)}. "
            "Enable them before initializing service accounts."
        )


def _service_account_email(project: str, sa_id: str) -> str:
    return f"{sa_id}@{project}.iam.gserviceaccount.com"


def _service_account_exists(project: str, email: str) -> bool:
    result = subprocess.run(
        [
            "gcloud",
            "iam",
            "service-accounts",
            "describe",
            email,
            f"--project={project}",
            "--format=json",
        ],
        text=True,
        capture_output=True,
    )
    return result.returncode == 0


def _project_user_members(project: str) -> list[str]:
    result = _run(
        [
            "gcloud",
            "projects",
            "get-iam-policy",
            project,
            "--format=json",
        ],
        capture_output=True,
    )
    policy = json.loads(result.stdout)
    users = {
        member
        for binding in policy.get("bindings", [])
        for member in binding.get("members", [])
        if member.startswith("user:")
    }
    return sorted(users)


def _project_member_roles(project: str, member: str) -> set[str]:
    """Return the set of project-level IAM roles bound to *member*."""
    result = _run(
        ["gcloud", "projects", "get-iam-policy", project, "--format=json"],
        capture_output=True,
    )
    policy = json.loads(result.stdout)
    return {binding["role"] for binding in policy.get("bindings", []) if member in binding.get("members", [])}


def _service_account_policy(service_account: str) -> dict[str, set[str]]:
    result = _run(
        [
            "gcloud",
            "iam",
            "service-accounts",
            "get-iam-policy",
            service_account,
            "--format=json",
        ],
        capture_output=True,
    )
    policy = json.loads(result.stdout)
    return {
        binding["role"]: set(binding.get("members", [])) for binding in policy.get("bindings", []) if "role" in binding
    }


def _ensure_service_account(project: str, sa_id: str, display_name: str, *, dry_run: bool) -> str:
    email = _service_account_email(project, sa_id)
    if _service_account_exists(project, email):
        logger.info("Service account already exists: %s", email)
        return email

    _run(
        [
            "gcloud",
            "iam",
            "service-accounts",
            "create",
            sa_id,
            f"--project={project}",
            f"--display-name={display_name}",
        ],
        dry_run=dry_run,
    )
    return email


def _add_policy_binding(policy: dict, role: str, member: str) -> bool:
    """Add *member* to *role* in a raw IAM policy dict. Returns True if the binding was new."""
    for binding in policy.get("bindings", []):
        if binding["role"] == role:
            members = set(binding.get("members", []))
            if member in members:
                return False
            members.add(member)
            binding["members"] = sorted(members)
            return True
    policy.setdefault("bindings", []).append({"role": role, "members": [member]})
    return True


def _set_project_iam_policy(project: str, policy: dict, *, dry_run: bool) -> None:
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(policy, f)
        policy_path = f.name
    try:
        _run(
            ["gcloud", "projects", "set-iam-policy", project, policy_path, "--format=json"],
            dry_run=dry_run,
            capture_output=True,
        )
    finally:
        os.unlink(policy_path)


def _set_sa_iam_policy(service_account: str, policy: dict, *, dry_run: bool) -> None:
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(policy, f)
        policy_path = f.name
    try:
        _run(
            ["gcloud", "iam", "service-accounts", "set-iam-policy", service_account, policy_path, "--format=json"],
            dry_run=dry_run,
            capture_output=True,
        )
    finally:
        os.unlink(policy_path)


def _get_raw_policy(cmd: list[str]) -> dict:
    result = subprocess.run(cmd, check=True, text=True, capture_output=True)
    return json.loads(result.stdout)


def _print_summary(summary: dict[str, object]) -> None:
    print(json.dumps(summary, indent=2, sort_keys=True))


def _active_gcloud_account() -> str | None:
    result = subprocess.run(
        ["gcloud", "auth", "list", "--filter=status:ACTIVE", "--format=value(account)"],
        text=True,
        capture_output=True,
    )
    if result.returncode == 0 and result.stdout.strip():
        return result.stdout.strip()
    return None


def _gcloud_probe(cmd: list[str]) -> tuple[bool, str]:
    """Run a gcloud command and return (success, stderr_or_summary)."""
    result = subprocess.run(cmd, text=True, capture_output=True)
    if result.returncode == 0:
        return True, ""
    return False, result.stderr.strip()


def _normalize_gcs_probe_path(path: str) -> str:
    path = path.strip()
    if not path:
        raise ValueError("GCS probe path cannot be empty")
    if not path.startswith("gs://"):
        path = f"gs://{path.lstrip('/')}"
    return path.rstrip("/")


def _gcs_bucket_from_path(path: str) -> str:
    path = _normalize_gcs_probe_path(path)
    bucket = path.removeprefix("gs://").split("/", 1)[0]
    if not bucket:
        raise ValueError(f"Invalid GCS probe path: {path}")
    return bucket


def _gcloud_storage_bucket_probe(bucket: str, service_account: str) -> tuple[bool, str]:
    return _gcloud_probe(
        [
            "gcloud",
            "storage",
            "buckets",
            "describe",
            f"gs://{bucket}",
            "--format=value(location)",
            f"--impersonate-service-account={service_account}",
        ]
    )


@click.group(help=__doc__)
@click.option("-v", "--verbose", is_flag=True, help="Enable verbose logging")
@click.pass_context
def cli(ctx: click.Context, verbose: bool) -> None:
    ctx.ensure_object(dict)
    ctx.obj["verbose"] = verbose
    level = logging.INFO if verbose else logging.WARNING
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(message)s", stream=sys.stdout)


@cli.command()
@click.option("--project", required=True, help="GCP project id")
@click.option(
    "--operator-principal",
    required=True,
    help="Operator principal email or full IAM member string (e.g. russell.power@gmail.com)",
)
@click.option(
    "--ci-principal", default=DEFAULT_CI_PRINCIPAL, show_default=True, help="CI principal email or IAM member string"
)
@click.option("--controller-sa-id", default=DEFAULT_CONTROLLER_SA_ID, show_default=True)
@click.option("--worker-sa-id", default=DEFAULT_WORKER_SA_ID, show_default=True)
@click.option("--dry-run", is_flag=True, help="Print planned actions without changing IAM")
def init(
    project: str,
    operator_principal: str,
    ci_principal: str,
    controller_sa_id: str,
    worker_sa_id: str,
    dry_run: bool,
) -> None:
    """Create service accounts and wire IAM bindings.

    Fetches each IAM policy once, computes all missing bindings in memory,
    then writes each policy back in a single set-iam-policy call.
    """
    logging.getLogger().setLevel(logging.INFO)
    operator_member = _principal_member(operator_principal)
    ci_member = _principal_member(ci_principal)

    _require_apis(project)

    controller_sa = _ensure_service_account(
        project, controller_sa_id, "Iris controller service account", dry_run=dry_run
    )
    worker_sa = _ensure_service_account(project, worker_sa_id, "Iris worker service account", dry_run=dry_run)

    # Fetch raw policies (with etags) so we can batch-update them.
    if dry_run:
        proj_policy = {"bindings": []}
        ctrl_policy = {"bindings": []}
        wrkr_policy = {"bindings": []}
    else:
        proj_policy = _get_raw_policy(["gcloud", "projects", "get-iam-policy", project, "--format=json"])
        ctrl_policy = _get_raw_policy(
            ["gcloud", "iam", "service-accounts", "get-iam-policy", controller_sa, "--format=json"]
        )
        wrkr_policy = _get_raw_policy(
            ["gcloud", "iam", "service-accounts", "get-iam-policy", worker_sa, "--format=json"]
        )

    project_user_members = sorted(
        {m for b in proj_policy.get("bindings", []) for m in b.get("members", []) if m.startswith("user:")}
    )

    controller_sa_member = f"serviceAccount:{controller_sa}"
    worker_sa_member = f"serviceAccount:{worker_sa}"

    summary = {
        "project": project,
        "dry_run": dry_run,
        "controller_service_account": controller_sa,
        "worker_service_account": worker_sa,
        "operator_principal": operator_member,
        "ci_principal": ci_member,
        "project_user_count": len(project_user_members),
        "project_user_members": project_user_members,
    }
    _print_summary(summary)

    # -- Project-level bindings (single set-iam-policy call) -------------------
    proj_additions: list[str] = []
    for role in CONTROLLER_PROJECT_ROLES:
        if _add_policy_binding(proj_policy, role, controller_sa_member):
            proj_additions.append(f"  {controller_sa_member} -> {role}")
    for role in WORKER_PROJECT_ROLES:
        if _add_policy_binding(proj_policy, role, worker_sa_member):
            proj_additions.append(f"  {worker_sa_member} -> {role}")
    for principal in {operator_member, *project_user_members}:
        for role in USER_PROJECT_ROLES:
            if _add_policy_binding(proj_policy, role, principal):
                proj_additions.append(f"  {principal} -> {role}")

    if proj_additions:
        logger.info("Adding %d project-level bindings:\n%s", len(proj_additions), "\n".join(proj_additions))
        _set_project_iam_policy(project, proj_policy, dry_run=dry_run)
    else:
        logger.info("Project IAM policy already up to date")

    # -- Controller SA bindings (single set-iam-policy call) -------------------
    ctrl_additions: list[str] = []
    for principal in (operator_member, ci_member):
        for role in IMPERSONATION_ROLES:
            if _add_policy_binding(ctrl_policy, role, principal):
                ctrl_additions.append(f"  {principal} -> {role}")
    for principal in project_user_members:
        for role in PROJECT_USER_IMPERSONATION_ROLES:
            if _add_policy_binding(ctrl_policy, role, principal):
                ctrl_additions.append(f"  {principal} -> {role}")
    for role in IMPERSONATION_ROLES:
        if _add_policy_binding(ctrl_policy, role, controller_sa_member):
            ctrl_additions.append(f"  {controller_sa_member} -> {role} (self)")

    if ctrl_additions:
        logger.info("Adding %d controller SA bindings:\n%s", len(ctrl_additions), "\n".join(ctrl_additions))
        _set_sa_iam_policy(controller_sa, ctrl_policy, dry_run=dry_run)
    else:
        logger.info("Controller SA IAM policy already up to date")

    # -- Worker SA bindings (single set-iam-policy call) -----------------------
    wrkr_additions: list[str] = []
    for principal in (operator_member, ci_member):
        for role in IMPERSONATION_ROLES:
            if _add_policy_binding(wrkr_policy, role, principal):
                wrkr_additions.append(f"  {principal} -> {role}")
    for principal in project_user_members:
        for role in PROJECT_USER_IMPERSONATION_ROLES:
            if _add_policy_binding(wrkr_policy, role, principal):
                wrkr_additions.append(f"  {principal} -> {role}")
    for role in IMPERSONATION_ROLES:
        if _add_policy_binding(wrkr_policy, role, worker_sa_member):
            wrkr_additions.append(f"  {worker_sa_member} -> {role} (self)")
        if _add_policy_binding(wrkr_policy, role, controller_sa_member):
            wrkr_additions.append(f"  {controller_sa_member} -> {role}")

    if wrkr_additions:
        logger.info("Adding %d worker SA bindings:\n%s", len(wrkr_additions), "\n".join(wrkr_additions))
        _set_sa_iam_policy(worker_sa, wrkr_policy, dry_run=dry_run)
    else:
        logger.info("Worker SA IAM policy already up to date")


def _print_results_table(results: list[tuple[str, bool]]) -> None:
    if not results:
        return
    click.echo()
    label_width = max(len(label) for label, _ in results)
    click.echo(f"  {'Check':<{label_width}}  Status")
    click.echo(f"  {'─' * label_width}  ──────")
    for label, passed in results:
        status = "OK" if passed else "FAIL"
        click.echo(f"  {label:<{label_width}}  {status}")
    passed_count = sum(1 for _, p in results if p)
    click.echo()
    click.echo(f"  {passed_count}/{len(results)} checks passed.")


def _check_result(
    label: str,
    passed: bool,
    detail: str = "",
    *,
    results: list[tuple[str, bool]] | None = None,
) -> bool:
    tag = "[OK]  " if passed else "[FAIL]"
    click.echo(f"  {tag} {label}")
    if not passed and detail:
        for line in detail.splitlines():
            click.echo(f"         {line}")
    if results is not None:
        results.append((label, passed))
    return passed


@cli.command()
@click.option("--project", required=True, help="GCP project id")
@click.option("--controller-sa-id", default=DEFAULT_CONTROLLER_SA_ID, show_default=True)
@click.option("--worker-sa-id", default=DEFAULT_WORKER_SA_ID, show_default=True)
@click.option(
    "--probe-gcs-path",
    multiple=True,
    help=(
        "Optional gs:// bucket or path used to verify bucket metadata access "
        "with the worker SA, e.g. gs://marin-us-east5 or "
        "gs://marin-tmp-us-east5/ttl=1d"
    ),
)
@click.argument("email")
def check(project: str, controller_sa_id: str, worker_sa_id: str, probe_gcs_path: tuple[str, ...], email: str) -> None:
    """Check whether a user has the IAM bindings and live credentials to use Iris.

    Checks IAM policy bindings, then performs live gcloud probes for each
    capability the Iris CLI needs: SA impersonation, OS Login SSH key
    registration, compute instance listing, OS Login profile resolution,
    metadata-style SSH key setup, and optional worker-SA GCS bucket
    metadata probes.

    \b
    Example:
        python setup_iam.py check --project=hai-gcp-models tim@openathena.ai
    """
    member = _principal_member(email)
    checked_email = member.split(":", 1)[1] if ":" in member else member
    controller_sa = _service_account_email(project, controller_sa_id)
    worker_sa = _service_account_email(project, worker_sa_id)

    active_account = _active_gcloud_account()
    click.echo(f"Active gcloud account: {active_account or '(none)'}")
    click.echo(f"Checking:              {member}")
    click.echo(f"Project:               {project}")
    click.echo(f"Controller SA:         {controller_sa}")
    click.echo(f"Worker SA:             {worker_sa}")
    if probe_gcs_path:
        click.echo(f"Probe GCS paths:       {', '.join(_normalize_gcs_probe_path(path) for path in probe_gcs_path)}")
    click.echo()

    ok = True
    results: list[tuple[str, bool]] = []
    can_do_live = active_account is not None and active_account == checked_email

    # -- 1. Project membership -------------------------------------------------
    click.echo("1. Project membership")
    project_users = _project_user_members(project)
    if member in project_users:
        _check_result(f"{member} is a direct user: member on project {project}", True, results=results)
    else:
        _check_result(
            f"{member} is NOT a direct user: member on project {project}",
            False,
            "Access may come via group/domain/org — the init command won't discover this user.",
            results=results,
        )
        # Not fatal — they might still have SA bindings from a manual grant.

    # -- 2. Project-level compute roles (metadata-style SSH) ------------------
    click.echo()
    click.echo("2. Project-level IAM roles (metadata-style SSH)")
    user_roles = _project_member_roles(project, member)
    for role in USER_PROJECT_ROLES:
        ok &= _check_result(f"project role — {role}", role in user_roles, results=results)

    # -- 3. IAM bindings on service accounts -----------------------------------
    click.echo()
    click.echo("3. IAM policy bindings on service accounts")
    for sa_label, sa_email in [("controller", controller_sa), ("worker", worker_sa)]:
        policy = _service_account_policy(sa_email)
        for role in IMPERSONATION_ROLES:
            has_role = member in policy.get(role, set())
            ok &= _check_result(f"{sa_label} SA — {role}", has_role, results=results)

    if not can_do_live:
        click.echo()
        if active_account:
            click.echo(
                f"Skipping live probes: active gcloud account ({active_account}) " f"!= checked email ({checked_email})"
            )
            click.echo(f"To run live probes: gcloud auth login {checked_email}")
        else:
            click.echo("Skipping live probes: no active gcloud account. Run: gcloud auth login")

        _print_results_table(results)
        raise SystemExit(0 if ok else 1)

    # -- 4. Live: impersonate SA (generate access token) -----------------------
    click.echo()
    click.echo("4. Live: impersonate service account (generate access token)")
    for sa_label, sa_email in [("controller", controller_sa), ("worker", worker_sa)]:
        passed, detail = _gcloud_probe(
            ["gcloud", "auth", "print-access-token", f"--impersonate-service-account={sa_email}"]
        )
        ok &= _check_result(f"impersonate {sa_label} SA ({sa_email})", passed, detail, results=results)

    # -- 5. Live: OS Login SSH key registration --------------------------------
    click.echo()
    click.echo("5. Live: OS Login SSH key registration (controller SA)")
    with tempfile.TemporaryDirectory() as tmpdir:
        test_key = os.path.join(tmpdir, "test_key")
        subprocess.run(
            ["ssh-keygen", "-t", "rsa", "-b", "2048", "-f", test_key, "-N", "", "-q"],
            check=True,
        )
        passed, detail = _gcloud_probe(
            [
                "gcloud",
                "compute",
                "os-login",
                "ssh-keys",
                "add",
                f"--key-file={test_key}.pub",
                "--ttl=60s",
                f"--impersonate-service-account={controller_sa}",
            ]
        )
        ok &= _check_result("register SSH key via OS Login", passed, detail, results=results)

        # Clean up the test key from OS Login
        if passed:
            subprocess.run(
                [
                    "gcloud",
                    "compute",
                    "os-login",
                    "ssh-keys",
                    "remove",
                    f"--key-file={test_key}.pub",
                    f"--impersonate-service-account={controller_sa}",
                ],
                capture_output=True,
            )

    # -- 6. Live: OS Login profile resolution ----------------------------------
    click.echo()
    click.echo("6. Live: OS Login profile resolution (controller SA)")
    passed, detail = _gcloud_probe(
        [
            "gcloud",
            "compute",
            "os-login",
            "describe-profile",
            f"--impersonate-service-account={controller_sa}",
        ]
    )
    ok &= _check_result("resolve OS Login profile", passed, detail, results=results)

    # -- 7. Live: compute instance listing -------------------------------------
    click.echo()
    click.echo("7. Live: compute instance listing (controller SA)")
    passed, detail = _gcloud_probe(
        [
            "gcloud",
            "compute",
            "instances",
            "list",
            f"--project={project}",
            "--limit=1",
            "--format=value(name)",
            f"--impersonate-service-account={controller_sa}",
        ]
    )
    ok &= _check_result("list compute instances", passed, detail, results=results)

    # -- 8. Live: set instance metadata (old-style SSH) ------------------------
    click.echo()
    click.echo("8. Live: project metadata read (needed for metadata-style SSH)")
    passed, detail = _gcloud_probe(
        [
            "gcloud",
            "compute",
            "project-info",
            "describe",
            f"--project={project}",
            "--format=value(commonInstanceMetadata.items)",
            f"--impersonate-service-account={controller_sa}",
        ]
    )
    ok &= _check_result("read project metadata", passed, detail, results=results)

    # -- 9. Live: storage bucket metadata (worker SA) --------------------------
    click.echo()
    click.echo("9. Live: GCS bucket metadata probes (worker SA)")
    normalized_probe_paths = [_normalize_gcs_probe_path(path) for path in probe_gcs_path]
    if not normalized_probe_paths:
        click.echo("  Skipping bucket probes: no --probe-gcs-path provided.")
    else:
        seen_buckets: set[str] = set()
        for path in normalized_probe_paths:
            bucket = _gcs_bucket_from_path(path)
            if bucket in seen_buckets:
                continue
            passed, detail = _gcloud_storage_bucket_probe(bucket, worker_sa)
            ok &= _check_result(
                f"read bucket metadata for gs://{bucket}",
                passed,
                detail,
                results=results,
            )
            seen_buckets.add(bucket)

    # -- Summary table ---------------------------------------------------------
    _print_results_table(results)
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    cli()
