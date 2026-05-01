# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""CoreWeave-specific Iris diagnostics: kubectl pod listing.

Internal to scripts/workflows/. Call collect_coreweave_diagnostics from
collect_diagnostics in iris_monitor.py; do not invoke directly.
"""

import subprocess
from collections.abc import Callable
from pathlib import Path

# Kubernetes label values are capped at 63 characters.
_K8S_LABEL_MAX_LEN = 63


def k8s_job_label(job_id: str) -> str:
    """Convert an Iris job ID to a Kubernetes-safe label value.

    Kubernetes label values must be no longer than 63 characters and may not
    contain underscores. This matches the transformation marin-canary-ferry-cw.yaml
    applies: strip a leading '/', replace '_' with '-', and truncate.

    Args:
        job_id: Raw Iris job ID, e.g. "my_long_job_id/subtask".

    Returns:
        Sanitized label value safe for use as a Kubernetes label selector.
    """
    label = job_id.lstrip("/").replace("_", "-")
    return label[:_K8S_LABEL_MAX_LEN]


def collect_coreweave_diagnostics(
    job_id: str,
    output_dir: Path,
    namespace: str,
    *,
    kubeconfig: Path | None,
    run: Callable[[list[str]], subprocess.CompletedProcess],
) -> tuple[list[str], list[str], list[str]]:
    """List task pods for the job and write kubernetes-pods.json.

    Args:
        job_id: Iris job ID used to derive the Kubernetes label selector.
        output_dir: Directory to write diagnostic files into.
        namespace: Kubernetes namespace to query.
        kubeconfig: Optional path to a kubeconfig file.
        run: Injectable subprocess run callable.

    Returns:
        Tuple of (files_written, required_files, errors). required_files always
        contains "kubernetes-pods.json"; errors are populated on failure.
    """
    required_files = ["kubernetes-pods.json"]
    files_written: list[str] = []
    errors: list[str] = []

    label_value = k8s_job_label(job_id)
    cmd = ["kubectl"]
    if kubeconfig:
        cmd += [f"--kubeconfig={kubeconfig}"]
    cmd += [
        "-n",
        namespace,
        "get",
        "pods",
        f"-l=iris.job_id={label_value}",
        "-o",
        "json",
    ]

    result = run(cmd)
    pods_path = output_dir / "kubernetes-pods.json"
    if result.returncode != 0:
        errors.append(f"kubectl get pods failed (exit {result.returncode}): {result.stderr.strip()}")
        # Write whatever output we got so the artifact upload has something.
        pods_path.write_text(result.stdout or result.stderr or "")
    else:
        pods_path.write_text(result.stdout)
        files_written.append("kubernetes-pods.json")

    return files_written, required_files, errors
