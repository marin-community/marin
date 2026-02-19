# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Kubectl CLI wrapper for Kubernetes API operations.

Encapsulates all subprocess interactions with kubectl into a single class.
Handles command building, JSON serialization/deserialization, and the
namespaced vs cluster-scoped resource distinction.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# Default timeout for kubectl commands (seconds)
DEFAULT_TIMEOUT: float = 60.0


class KubectlError(RuntimeError):
    """Error raised for kubectl command failures."""


@dataclass
class Kubectl:
    """Wrapper around the kubectl CLI for Kubernetes API operations.

    Encapsulates command prefix construction (including optional --kubeconfig),
    namespace injection, JSON parsing, and error handling. All operations use
    subprocess with a configurable timeout.

    The `namespace` field is the default namespace for namespaced operations.
    Cluster-scoped resources (e.g. NodePools) use `cluster_scoped=True` to
    skip namespace injection.
    """

    namespace: str
    kubeconfig_path: str | None = None
    timeout: float = DEFAULT_TIMEOUT
    _prefix: list[str] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.kubeconfig_path:
            self.kubeconfig_path = os.path.expanduser(self.kubeconfig_path)
        cmd = ["kubectl"]
        if self.kubeconfig_path:
            cmd.extend(["--kubeconfig", self.kubeconfig_path])
        self._prefix = cmd

    @property
    def prefix(self) -> list[str]:
        """Base kubectl command prefix (includes --kubeconfig if configured)."""
        return list(self._prefix)

    def run(
        self,
        args: list[str],
        *,
        namespaced: bool = False,
        timeout: float | None = None,
        stdin: str | None = None,
    ) -> subprocess.CompletedProcess[str]:
        """Run a kubectl command with consistent timeout and error capture.

        Args:
            args: Arguments to pass after the kubectl prefix.
            namespaced: If True, inject ``-n {namespace}`` before the args.
            timeout: Override the default timeout for this call.
            stdin: Optional stdin data piped to the subprocess.

        Returns:
            The completed subprocess result (stdout/stderr are captured as text).
        """
        effective_timeout = timeout if timeout is not None else self.timeout
        cmd = list(self._prefix)
        if namespaced:
            cmd.extend(["-n", self.namespace])
        cmd.extend(args)
        if stdin:
            logger.info("kubectl: %s\n  stdin=%s", " ".join(cmd), stdin[:2000])
        else:
            logger.info("kubectl: %s", " ".join(cmd))
        result = subprocess.run(cmd, input=stdin, capture_output=True, text=True, timeout=effective_timeout)
        if result.returncode != 0:
            logger.info("kubectl exit %d: stderr=%s", result.returncode, result.stderr.strip()[:500])
        return result

    def apply_json(self, manifest: dict) -> None:
        """Apply a Kubernetes manifest dict via ``kubectl apply -f -``.

        Raises PlatformError on non-zero exit.
        """
        result = self.run(["apply", "-f", "-"], stdin=json.dumps(manifest))
        if result.returncode != 0:
            raise KubectlError(f"kubectl apply failed: {result.stderr.strip()}")

    def get_json(self, resource: str, name: str, *, cluster_scoped: bool = False) -> dict | None:
        """Get a Kubernetes resource as a parsed dict.

        Returns None if the resource is not found. Raises PlatformError on
        other kubectl failures.

        Args:
            resource: The resource type (e.g. "pod", "nodepool", "deployment").
            name: The resource name.
            cluster_scoped: If True, omit the namespace flag (for resources
                like NodePools that are not namespaced).
        """
        result = self.run(
            ["get", resource, name, "-o", "json"],
            namespaced=not cluster_scoped,
        )
        if result.returncode != 0:
            if "not found" in result.stderr.lower() or "NotFound" in result.stderr:
                return None
            raise KubectlError(f"kubectl get {resource}/{name} failed: {result.stderr.strip()}")
        return json.loads(result.stdout)

    def list_json(
        self,
        resource: str,
        *,
        labels: dict[str, str] | None = None,
        cluster_scoped: bool = False,
    ) -> list[dict]:
        """List Kubernetes resources, optionally filtered by labels. Returns the items list.

        Args:
            resource: The resource type (e.g. "pods", "nodepools").
            labels: Label selector key=value pairs.
            cluster_scoped: If True, omit the namespace flag.
        """
        args = ["get", resource, "-o", "json"]
        if labels:
            selector = ",".join(f"{k}={v}" for k, v in labels.items())
            args.extend(["-l", selector])
        result = self.run(args, namespaced=not cluster_scoped)
        if result.returncode != 0:
            raise KubectlError(f"kubectl get {resource} failed: {result.stderr.strip()}")
        data = json.loads(result.stdout)
        return data.get("items", [])

    def delete(
        self, resource: str, name: str, *, cluster_scoped: bool = False, force: bool = False, wait: bool = True
    ) -> None:
        """Delete a Kubernetes resource, ignoring NotFound errors.

        Uses ``--ignore-not-found`` so this is always idempotent.

        Args:
            resource: The resource type.
            name: The resource name.
            cluster_scoped: If True, omit the namespace flag.
            force: If True, add --grace-period=0 --force for immediate deletion.
            wait: If False, pass --wait=false so kubectl returns immediately
                after issuing the delete (useful for slow resources like NodePools).
        """
        args = ["delete", resource, name, "--ignore-not-found"]
        if force:
            args.extend(["--grace-period=0", "--force"])
        if not wait:
            args.append("--wait=false")
        result = self.run(
            args,
            namespaced=not cluster_scoped,
        )
        if result.returncode != 0:
            raise KubectlError(f"kubectl delete {resource}/{name} failed: {result.stderr.strip()}")

    def set_image(self, resource: str, name: str, container: str, image: str, *, namespaced: bool = False) -> None:
        """Set the container image on a resource via ``kubectl set image``."""
        args = ["set", "image", f"{resource}/{name}", f"{container}={image}"]
        result = self.run(args, namespaced=namespaced)
        if result.returncode != 0:
            raise KubectlError(f"kubectl set image failed: {result.stderr.strip()}")

    def rollout_restart(self, resource: str, name: str, *, namespaced: bool = False) -> None:
        """Restart a rollout via ``kubectl rollout restart``."""
        args = ["rollout", "restart", f"{resource}/{name}"]
        result = self.run(args, namespaced=namespaced)
        if result.returncode != 0:
            raise KubectlError(f"kubectl rollout restart failed: {result.stderr.strip()}")

    def rollout_status(self, resource: str, name: str, *, timeout: float = 600.0, namespaced: bool = False) -> None:
        """Wait for a rollout to complete via ``kubectl rollout status``."""
        args = ["rollout", "status", f"{resource}/{name}", f"--timeout={int(timeout)}s"]
        result = self.run(args, namespaced=namespaced, timeout=timeout + 30)
        if result.returncode != 0:
            raise KubectlError(f"kubectl rollout status failed: {result.stderr.strip()}")

    def get_events(
        self,
        field_selector: str | None = None,
    ) -> list[dict]:
        """Get Kubernetes events, optionally filtered by field selector.

        Returns the items list from ``kubectl get events -o json``.
        """
        args = ["get", "events", "-o", "json"]
        if field_selector:
            args.extend(["--field-selector", field_selector])
        result = self.run(args, namespaced=True)
        if result.returncode != 0:
            logger.warning("kubectl get events failed: %s", result.stderr.strip()[:200])
            return []
        data = json.loads(result.stdout)
        return data.get("items", [])

    def logs(self, pod_name: str, *, container: str | None = None, tail: int = 50, previous: bool = False) -> str:
        """Fetch logs from a Pod container.

        Returns the log text, or an empty string if the Pod/container is not
        available (e.g. not yet started).
        """
        args = ["logs", pod_name]
        if container:
            args.extend(["-c", container])
        if previous:
            args.append("--previous")
        args.extend([f"--tail={tail}"])
        result = self.run(args, namespaced=True)
        if result.returncode != 0:
            return ""
        return result.stdout

    def exec(
        self,
        pod_name: str,
        cmd: list[str],
        *,
        container: str | None = None,
        timeout: float | None = None,
    ) -> subprocess.CompletedProcess[str]:
        """Run a command inside a Pod container via ``kubectl exec``.

        Args:
            pod_name: The Pod to exec into.
            cmd: The command and arguments to run.
            container: Target container name (omit for single-container Pods).
            timeout: Override the default timeout for this call.
        """
        args = ["exec", pod_name]
        if container:
            args.extend(["-c", container])
        args.extend(["--", *cmd])
        return self.run(args, namespaced=True, timeout=timeout)

    def read_file(
        self,
        pod_name: str,
        path: str,
        *,
        container: str | None = None,
    ) -> bytes:
        """Read a file from inside a Pod container.

        Returns the raw file contents. Raises RuntimeError if the read fails.
        """
        result = self.exec(pod_name, ["cat", path], container=container, timeout=10)
        if result.returncode != 0:
            raise RuntimeError(f"Failed to read {path}: {result.stderr}")
        return result.stdout.encode("utf-8")

    def rm_files(
        self,
        pod_name: str,
        paths: list[str],
        *,
        container: str | None = None,
    ) -> None:
        """Remove files inside a Pod container. Ignores missing files."""
        self.exec(pod_name, ["rm", "-f", *paths], container=container, timeout=10)

    def popen(
        self,
        args: list[str],
        *,
        namespaced: bool = False,
        **kwargs,
    ) -> subprocess.Popen:
        """Start a kubectl subprocess without waiting for completion.

        This is the escape hatch for streaming operations (exec with on_line,
        port-forward) that need a live process handle.

        Args:
            args: Arguments to pass after the kubectl prefix.
            namespaced: If True, inject ``-n {namespace}`` before the args.
            **kwargs: Additional keyword arguments passed to subprocess.Popen.
        """
        cmd = list(self._prefix)
        if namespaced:
            cmd.extend(["-n", self.namespace])
        cmd.extend(args)
        return subprocess.Popen(cmd, **kwargs)
