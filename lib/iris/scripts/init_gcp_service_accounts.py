#!/usr/bin/env python3
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Initialize Iris GCP service accounts and IAM bindings.

Creates a controller service account and a worker service account if missing,
grants their project roles, and wires impersonation / act-as bindings for the
configured CI and operator principals.
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys

logger = logging.getLogger("init-gcp-service-accounts")

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
    "roles/compute.instanceAdmin.v1",
    "roles/compute.osAdminLogin",
    "roles/logging.logWriter",
    "roles/monitoring.metricWriter",
    "roles/storage.objectAdmin",
    "roles/tpu.admin",
)
WORKER_PROJECT_ROLES = (
    "roles/compute.osAdminLogin",
    "roles/logging.logWriter",
    "roles/monitoring.metricWriter",
    "roles/storage.objectAdmin",
    "roles/tpu.viewer",
)
IMPERSONATION_ROLES = (
    "roles/iam.serviceAccountTokenCreator",
    "roles/iam.serviceAccountUser",
)


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


def _bind_project_role(project: str, *, member: str, role: str, dry_run: bool) -> None:
    _run(
        [
            "gcloud",
            "projects",
            "add-iam-policy-binding",
            project,
            f"--member={member}",
            f"--role={role}",
            "--condition=None",
        ],
        dry_run=dry_run,
    )


def _bind_service_account_role(
    service_account: str,
    *,
    member: str,
    role: str,
    dry_run: bool,
) -> None:
    _run(
        [
            "gcloud",
            "iam",
            "service-accounts",
            "add-iam-policy-binding",
            service_account,
            f"--member={member}",
            f"--role={role}",
            "--condition=None",
        ],
        dry_run=dry_run,
    )


def _print_summary(summary: dict[str, object]) -> None:
    print(json.dumps(summary, indent=2, sort_keys=True))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", required=True, help="GCP project id")
    parser.add_argument(
        "--operator-principal",
        required=True,
        help=(
            "Operator principal email or full IAM member string "
            "(for example russell.power@gmail.com or user:foo@example.com)"
        ),
    )
    parser.add_argument(
        "--ci-principal",
        default=DEFAULT_CI_PRINCIPAL,
        help=f"CI principal email or IAM member string (default: {DEFAULT_CI_PRINCIPAL})",
    )
    parser.add_argument("--controller-sa-id", default=DEFAULT_CONTROLLER_SA_ID)
    parser.add_argument("--worker-sa-id", default=DEFAULT_WORKER_SA_ID)
    parser.add_argument("--dry-run", action="store_true", help="Print planned actions without changing IAM")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", stream=sys.stdout)

    operator_member = _principal_member(args.operator_principal)
    ci_member = _principal_member(args.ci_principal)

    _require_apis(args.project)

    controller_sa = _ensure_service_account(
        args.project,
        args.controller_sa_id,
        "Iris controller service account",
        dry_run=args.dry_run,
    )
    worker_sa = _ensure_service_account(
        args.project,
        args.worker_sa_id,
        "Iris worker service account",
        dry_run=args.dry_run,
    )

    summary = {
        "project": args.project,
        "dry_run": args.dry_run,
        "controller_service_account": controller_sa,
        "worker_service_account": worker_sa,
        "operator_principal": operator_member,
        "ci_principal": ci_member,
        "controller_project_roles": list(CONTROLLER_PROJECT_ROLES),
        "worker_project_roles": list(WORKER_PROJECT_ROLES),
        "impersonation_roles": list(IMPERSONATION_ROLES),
    }
    _print_summary(summary)

    for role in CONTROLLER_PROJECT_ROLES:
        _bind_project_role(
            args.project,
            member=f"serviceAccount:{controller_sa}",
            role=role,
            dry_run=args.dry_run,
        )

    for role in WORKER_PROJECT_ROLES:
        _bind_project_role(
            args.project,
            member=f"serviceAccount:{worker_sa}",
            role=role,
            dry_run=args.dry_run,
        )

    for principal in (operator_member, ci_member):
        for role in IMPERSONATION_ROLES:
            _bind_service_account_role(controller_sa, member=principal, role=role, dry_run=args.dry_run)
            _bind_service_account_role(worker_sa, member=principal, role=role, dry_run=args.dry_run)

    for service_account in (controller_sa, worker_sa):
        _bind_service_account_role(
            service_account,
            member=f"serviceAccount:{service_account}",
            role="roles/iam.serviceAccountUser",
            dry_run=args.dry_run,
        )

    _bind_service_account_role(
        worker_sa,
        member=f"serviceAccount:{controller_sa}",
        role="roles/iam.serviceAccountUser",
        dry_run=args.dry_run,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
