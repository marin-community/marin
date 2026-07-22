#!/usr/bin/env -S uv run --script
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.11"
# dependencies = ["click>=8.1"]
# ///
"""Verify and activate a loom VM after ``pulumi up``.

Pulumi owns durable resources. This small imperative tail deliberately owns the
checks that are observations rather than resources: public DNS and SSH,
startup-script completion, HTTPS readiness, session-container preservation,
supervisor preservation, and ACP reattachment.
"""

from __future__ import annotations

import json
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NoReturn

import click

PULUMI_DIR = Path(__file__).resolve().parent
STARTUP_UNIT = "google-startup-scripts"
STARTUP_BANNER = "loom startup-script:"
STARTUP_DONE = "loom startup-script done"
STARTUP_FAILURES = ("failed with error", "Failed with result")


def log(message: str) -> None:
    click.echo(f"▶ {message}", err=True)


def fail(message: str) -> NoReturn:
    raise click.ClickException(message)


def stack_outputs(stack: str | None) -> dict[str, Any]:
    command = ["pulumi", "stack", "output", "--json", "--cwd", str(PULUMI_DIR)]
    if stack:
        command.extend(["--stack", stack])
    result = subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
        stdin=subprocess.DEVNULL,
    )
    return json.loads(result.stdout)


def string_output(outputs: dict[str, Any], name: str) -> str:
    value = outputs.get(name)
    if not isinstance(value, str) or not value:
        fail(f"the stack does not export a valid {name}")
    return value


def resolved_ipv4(domain: str) -> set[str]:
    try:
        return {str(item[4][0]) for item in socket.getaddrinfo(domain, 443, socket.AF_INET)}
    except OSError:
        return set()


def wait_for_dns(domain: str, address: str, timeout: int) -> None:
    log(f"waiting for {domain} to resolve publicly to {address}")
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if address in resolved_ipv4(domain):
            return
        time.sleep(15)
    fail(f"{domain} does not resolve to {address}; verify the Cloudflare A record before activating loom")


@dataclass(frozen=True)
class RemoteHost:
    project: str
    zone: str
    instance: str

    def run(self, command: str, timeout: int = 120) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [
                "gcloud",
                f"--project={self.project}",
                "compute",
                "ssh",
                self.instance,
                f"--zone={self.zone}",
                "--quiet",
                f"--command={command}",
            ],
            capture_output=True,
            text=True,
            stdin=subprocess.DEVNULL,
            timeout=timeout,
        )

    def lines(self, command: str) -> set[str]:
        result = self.run(command)
        if result.returncode:
            fail(f"remote inventory command failed: {result.stderr.strip()}")
        return {line.strip() for line in result.stdout.splitlines() if line.strip()}

    def journal(self) -> list[str]:
        result = self.run(f"sudo journalctl -u {STARTUP_UNIT} --no-pager -o cat 2>/dev/null")
        return result.stdout.splitlines()


def wait_for_ssh(host: RemoteHost, timeout: int) -> None:
    log("waiting for SSH")
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            if host.run("true", timeout=60).returncode == 0:
                return
        except subprocess.TimeoutExpired:
            pass
        time.sleep(10)
    fail("SSH never became reachable; check the operator CIDR firewall rule")


def supervisor_inventory(host: RemoteHost) -> set[str]:
    return host.lines(
        "cd /opt/loom/deploy/standalone && sudo docker compose exec -T loom tapestry ls",
    )


def runner_session_inventory(host: RemoteHost) -> set[str]:
    return host.lines(
        "sudo docker ps --filter label=dev.loom.runner=session --format '{{.Label \"dev.loom.session\"}}'",
    )


def runner_container_inventory(host: RemoteHost) -> set[str]:
    return host.lines(
        "sudo docker ps --filter label=dev.loom.runner=session --format '{{.ID}}'",
    )


def activate_and_monitor(host: RemoteHost, timeout: int) -> None:
    # Anchor monitoring to a newly-triggered generation. Old failure text in the
    # persistent journal must never make this deployment look failed.
    prior = sum(STARTUP_BANNER in line for line in host.journal())
    log("triggering the current startup script")
    result = host.run(f"sudo systemctl restart --no-block {STARTUP_UNIT}.service")
    if result.returncode:
        fail(f"could not trigger startup script: {result.stderr.strip()}")

    log("streaming the current startup-script generation")
    deadline = time.monotonic() + timeout
    printed = 0
    while time.monotonic() < deadline:
        lines = host.journal()
        banners = [index for index, line in enumerate(lines) if STARTUP_BANNER in line]
        if len(banners) <= prior:
            time.sleep(15)
            continue
        boundary = banners[prior]
        start = max(boundary, printed)
        for line in lines[start:]:
            click.echo(f"  {line}", err=True)
        printed = len(lines)
        window = "\n".join(lines[boundary:])
        if STARTUP_DONE in window:
            return
        if any(marker in window for marker in STARTUP_FAILURES):
            fail("startup script failed; inspect the journal output above")
        time.sleep(15)
    fail("startup script did not finish before the timeout")


def health_is_ready(domain: str) -> bool:
    """Return whether loom's exact public liveness contract is healthy once."""
    try:
        with urllib.request.urlopen(f"https://{domain}/api/health", timeout=10) as response:
            return response.status == 200 and response.read().strip() == b"ok"
    except (urllib.error.HTTPError, urllib.error.URLError, OSError):
        # Caddy may already answer while its loom upstream is still starting;
        # 401/404/5xx are not success for this exact public endpoint.
        return False


def readiness_is_ready(domain: str) -> bool:
    """Return whether Loom reports its database and migrations ready once."""
    try:
        with urllib.request.urlopen(f"https://{domain}/api/ready", timeout=10) as response:
            payload = json.loads(response.read())
            return (
                response.status == 200
                and isinstance(payload, dict)
                and payload.get("status") == "ready"
                and payload.get("database") is True
            )
    except (json.JSONDecodeError, urllib.error.HTTPError, urllib.error.URLError, OSError):
        return False


def wait_for_health(domain: str, timeout: int) -> None:
    log(f"checking https://{domain}/api/health")
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if health_is_ready(domain) and readiness_is_ready(domain):
            log("loom health and readiness checks passed")
            return
        time.sleep(15)
    fail("the startup script succeeded, but Loom health or readiness is not ready")


def reattachment_failed(host: RemoteHost) -> bool:
    result = host.run(
        "cd /opt/loom/deploy/standalone && "
        "sudo docker compose logs --no-color loom 2>/dev/null | "
        "grep -Fq 'failed to reattach ACP session'",
    )
    if result.returncode not in (0, 1):
        fail(f"could not inspect Loom reattachment logs: {result.stderr.strip()}")
    return result.returncode == 0


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option("--stack", help="Pulumi stack name. Default: current stack.")
@click.option("--dns-timeout", default=1200, show_default=True)
@click.option("--ssh-timeout", default=300, show_default=True)
@click.option("--startup-timeout", default=2400, show_default=True)
@click.option("--https-timeout", default=300, show_default=True)
@click.option(
    "--allow-legacy-cutover",
    is_flag=True,
    help=(
        "Allow the first DockerRunner rollout to stop legacy in-container supervisors; "
        "the command rejects this flag once any DockerRunner session exists."
    ),
)
def main(
    stack: str | None,
    dns_timeout: int,
    ssh_timeout: int,
    startup_timeout: int,
    https_timeout: int,
    allow_legacy_cutover: bool,
) -> None:
    outputs = stack_outputs(stack)
    try:
        project = subprocess.run(
            [
                "pulumi",
                "config",
                "get",
                "gcp:project",
                "--cwd",
                str(PULUMI_DIR),
                *(["--stack", stack] if stack else []),
            ],
            check=True,
            capture_output=True,
            text=True,
            stdin=subprocess.DEVNULL,
        ).stdout.strip()
        address = string_output(outputs, "address")
        instance = string_output(outputs, "instanceName")
        zone = string_output(outputs, "zone")
        domain = string_output(outputs, "url").removeprefix("https://").rstrip("/")
        manage_runtime = outputs["manageRuntime"]
    except (KeyError, subprocess.CalledProcessError) as error:
        fail(f"could not read the Pulumi stack outputs: {error}")

    if manage_runtime is not True:
        fail("manageRuntime is false; refusing to trigger the startup script")
    host = RemoteHost(project, zone, instance)
    wait_for_dns(domain, address, dns_timeout)
    wait_for_ssh(host, ssh_timeout)
    supervisors_before = supervisor_inventory(host)
    runner_sessions_before = runner_session_inventory(host)
    legacy_sessions = supervisors_before - runner_sessions_before
    if allow_legacy_cutover and (runner_sessions_before or not legacy_sessions):
        fail("--allow-legacy-cutover is valid only before the first DockerRunner session exists")
    if legacy_sessions and not allow_legacy_cutover:
        fail(
            "legacy in-container supervisors would stop during this rollout: "
            f"{', '.join(sorted(legacy_sessions))}; rerun with --allow-legacy-cutover after inventorying them"
        )
    runner_containers_before = runner_container_inventory(host)
    activate_and_monitor(host, startup_timeout)
    wait_for_health(domain, https_timeout)
    runner_containers_after = runner_container_inventory(host)
    missing_containers = runner_containers_before - runner_containers_after
    if missing_containers:
        fail(f"session containers disappeared during control restart: {', '.join(sorted(missing_containers))}")
    if not allow_legacy_cutover:
        missing_supervisors = supervisors_before - supervisor_inventory(host)
        if missing_supervisors:
            fail(f"session supervisors disappeared during control restart: {', '.join(sorted(missing_supervisors))}")
    if not allow_legacy_cutover and reattachment_failed(host):
        fail("Loom logged an ACP reattachment failure after restart")
    log(f"loom is live: https://{domain}/")


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as error:
        click.echo(error.stderr or str(error), err=True)
        sys.exit(error.returncode)
