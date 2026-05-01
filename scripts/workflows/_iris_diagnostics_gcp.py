# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""GCP-specific Iris diagnostics: gcloud SSH, controller log fetch.

Internal to scripts/workflows/. Call collect_gcp_diagnostics from collect_diagnostics
in iris_monitor.py; do not invoke directly.
"""

import subprocess
from collections.abc import Callable
from pathlib import Path

# SSH command run on each controller VM. Fetches docker state and full iris-controller logs.
_CONTROLLER_SSH_COMMAND = """\
set +e
echo '=== docker ps -a ==='
sudo docker ps -a
echo '=== docker logs iris-controller (last 5000 lines) ==='
sudo docker logs --timestamps --tail 5000 iris-controller 2>&1
"""


def list_controller_instances(
    project: str,
    controller_label: str,
    *,
    run: Callable[[list[str]], subprocess.CompletedProcess],
) -> list[tuple[str, str]]:
    """Return (name, zone) pairs for GCE instances whose controller label is true.

    Args:
        project: GCP project ID.
        controller_label: Instance label key (value must be "true").
        run: Callable matching subprocess.run signature, injected for testing.

    Returns:
        List of (name, zone) tuples. Empty list if gcloud fails or finds nothing.
    """
    cmd = [
        "gcloud",
        "compute",
        "instances",
        "list",
        f"--project={project}",
        f"--filter=labels.{controller_label}=true",
        "--format=csv[no-heading](name,zone)",
    ]
    result = run(cmd)
    if result.returncode != 0:
        return []

    instances = []
    for line in result.stdout.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split(",", 1)
        if len(parts) == 2:
            instances.append((parts[0].strip(), parts[1].strip()))
    return instances


def fetch_controller_log(
    name: str,
    zone: str,
    project: str,
    output_path: Path,
    *,
    service_account: str | None,
    ssh_key: Path | None,
    run: Callable[[list[str]], subprocess.CompletedProcess],
) -> str | None:
    """SSH into a GCE controller instance and write docker logs to output_path.

    Args:
        name: GCE instance name.
        zone: GCE zone.
        project: GCP project ID.
        output_path: File path where the log should be written.
        service_account: Optional service account to impersonate for gcloud SSH.
        ssh_key: Optional path to an SSH key file.
        run: Injectable subprocess run callable.

    Returns:
        None on success; an error string if the SSH command failed.
    """
    cmd = [
        "gcloud",
        "compute",
        "ssh",
        name,
        f"--project={project}",
        f"--zone={zone}",
        "--quiet",
        f"--command={_CONTROLLER_SSH_COMMAND}",
    ]
    if service_account:
        cmd.append(f"--impersonate-service-account={service_account}")
    if ssh_key:
        cmd.append(f"--ssh-key-file={ssh_key}")

    result = run(cmd)
    output_path.write_text(result.stdout + result.stderr)
    if result.returncode != 0:
        return f"SSH to {name} ({zone}) failed with exit {result.returncode}"
    return None


def collect_gcp_diagnostics(
    job_id: str,
    output_dir: Path,
    project: str,
    controller_label: str,
    *,
    service_account: str | None,
    ssh_key: Path | None,
    run: Callable[[list[str]], subprocess.CompletedProcess],
) -> tuple[list[str], list[str], list[str]]:
    """Collect GCP-specific diagnostics: SSH into each controller VM and fetch logs.

    Args:
        job_id: Iris job ID (informational only for this function).
        output_dir: Directory to write diagnostic files into.
        project: GCP project ID.
        controller_label: Instance label key identifying controller VMs.
        service_account: Optional service account to impersonate.
        ssh_key: Optional path to an SSH key file.
        run: Injectable subprocess run callable.

    Returns:
        Tuple of (files_written, required_files, errors). required_files contains the
        controller log filenames; errors contains human-readable descriptions of any
        failures.
    """
    files_written: list[str] = []
    required_files: list[str] = []
    errors: list[str] = []

    instances = list_controller_instances(project, controller_label, run=run)
    if not instances:
        errors.append(f"gcloud found no instances with label {controller_label}=true in project {project}")
        return files_written, required_files, errors

    for name, zone in instances:
        filename = f"controller-{name}.log"
        output_path = output_dir / filename
        required_files.append(filename)
        error = fetch_controller_log(
            name,
            zone,
            project,
            output_path,
            service_account=service_account,
            ssh_key=ssh_key,
            run=run,
        )
        if error:
            errors.append(error)
        else:
            files_written.append(filename)

    return files_written, required_files, errors
