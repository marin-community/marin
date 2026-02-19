#!/usr/bin/env python3
# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Automated debugging loop for CoreWeave integration with Iris.

Iteratively:
1. Wipes the K8s cluster to a clean state
2. Runs the smoke test to capture failures
3. Hands failures to Claude for diagnosis and code fixes
4. Repeats until the smoke test passes or max iterations reached

Usage:
    uv run python lib/iris/scripts/debug-coreweave-loop.py

    # Resume a previous session
    uv run python lib/iris/scripts/debug-coreweave-loop.py \
        --resume logs/debug-coreweave-20260219-143000

    # Custom config and timeout
    uv run python lib/iris/scripts/debug-coreweave-loop.py \
        --config lib/iris/examples/coreweave.yaml \
        --timeout 2400 --max-iterations 5
"""

import json
import logging
import os
import re
import shutil
import signal
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import click

SCRIPT_DIR = Path(__file__).resolve().parent
IRIS_ROOT = SCRIPT_DIR.parent
REPO_ROOT = IRIS_ROOT.parent.parent

K8S_MANIFESTS_DIR = REPO_ROOT / "infra" / "coreweave" / "k8s"
DEBUG_LOG_PATH = REPO_ROOT / "docs" / "debug-log-coreweave.md"

SMOKE_BOOT_TIMEOUT = 3600
SMOKE_JOB_TIMEOUT = 120
CLAUDE_OUTPUT_TAIL_LINES = 300

logger = logging.getLogger(__name__)


@dataclass
class IterationResult:
    status: str  # "PASSED" or "FAILED"
    hypothesis: str
    changes: list[str]
    tests_run: str
    next_steps: str


def configure_logging(log_dir: Path | None = None):
    """Set up logging to stdout and optionally to a file in log_dir."""
    fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if log_dir is not None:
        log_dir.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_dir / "debug-loop.log"))
    logging.basicConfig(level=logging.INFO, format=fmt, handlers=handlers, force=True)


def check_preflight(kubeconfig: Path, config_path: Path):
    """Fail fast with actionable messages if prerequisites are missing."""
    kubeconfig = kubeconfig.expanduser()
    if not kubeconfig.exists():
        raise SystemExit(f"Kubeconfig not found: {kubeconfig}\nCreate it or pass --kubeconfig")

    result = subprocess.run(
        ["kubectl", "cluster-info"],
        env={**os.environ, "KUBECONFIG": str(kubeconfig)},
        capture_output=True,
        text=True,
        timeout=30,
    )
    if result.returncode != 0:
        raise SystemExit(
            f"kubectl cluster-info failed (exit {result.returncode}):\n{result.stderr}\n"
            f"Check KUBECONFIG={kubeconfig} and cluster connectivity."
        )

    result = subprocess.run(["claude", "--version"], capture_output=True, text=True, timeout=10)
    if result.returncode != 0:
        raise SystemExit("'claude --version' failed. Is Claude CLI installed and on PATH?")

    if not config_path.exists():
        raise SystemExit(f"Config file not found: {config_path}")

    if not K8S_MANIFESTS_DIR.exists():
        raise SystemExit(f"K8s manifests directory not found: {K8S_MANIFESTS_DIR}")

    for name in ["namespace.yaml", "service-account.yaml", "cluster-role.yaml", "cluster-role-binding.yaml"]:
        manifest = K8S_MANIFESTS_DIR / name
        if not manifest.exists():
            raise SystemExit(f"Required K8s manifest not found: {manifest}")

    logger.info("Pre-flight checks passed")


def _run_and_tee(cmd: list[str], env: dict[str, str], timeout: int = 120) -> subprocess.CompletedProcess[str]:
    """Run a command, streaming stdout/stderr to the terminal in real time."""
    result = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=timeout)
    if result.stdout.strip():
        sys.stdout.write(result.stdout)
        sys.stdout.flush()
    if result.stderr.strip():
        sys.stdout.write(result.stderr)
        sys.stdout.flush()
    return result


def cleanup_kubernetes(kubeconfig: Path) -> tuple[bool, str]:
    """Wipe the K8s cluster to a clean state for Iris.

    Deletion errors are non-fatal since resources may not exist yet.
    Returns (success, combined_output).
    """
    kubeconfig = kubeconfig.expanduser()
    env = {**os.environ, "KUBECONFIG": str(kubeconfig)}
    output_lines: list[str] = []
    had_error = False

    delete_commands = [
        ["kubectl", "delete", "pods", "-n", "iris", "-l", "iris-managed=true", "--ignore-not-found"],
        ["kubectl", "delete", "namespace", "iris", "--ignore-not-found"],
    ]

    for cmd in delete_commands:
        logger.info("Running: %s", " ".join(cmd))
        result = _run_and_tee(cmd, env, timeout=120)
        output_lines.append(f"$ {' '.join(cmd)}")
        if result.stdout.strip():
            output_lines.append(result.stdout.strip())
        if result.stderr.strip():
            output_lines.append(result.stderr.strip())
        if result.returncode != 0:
            logger.warning("Non-fatal: %s exited %d", " ".join(cmd[:3]), result.returncode)
            had_error = True

    # Wait for namespace deletion to complete
    wait_cmd = ["kubectl", "wait", "--for=delete", "namespace/iris", "--timeout=120s"]
    logger.info("Running: %s", " ".join(wait_cmd))
    result = _run_and_tee(wait_cmd, env, timeout=150)
    output_lines.append(f"$ {' '.join(wait_cmd)}")
    if result.stdout.strip():
        output_lines.append(result.stdout.strip())
    # Timeout or "not found" are both acceptable here
    if result.returncode != 0:
        logger.warning("Namespace wait returned %d (may already be gone)", result.returncode)

    apply_commands = [
        ["kubectl", "apply", "-f", str(K8S_MANIFESTS_DIR / "namespace.yaml")],
        ["kubectl", "apply", "-f", str(K8S_MANIFESTS_DIR / "service-account.yaml")],
        ["kubectl", "apply", "-f", str(K8S_MANIFESTS_DIR / "cluster-role.yaml")],
        ["kubectl", "apply", "-f", str(K8S_MANIFESTS_DIR / "cluster-role-binding.yaml")],
    ]

    for cmd in apply_commands:
        logger.info("Running: %s", " ".join(cmd))
        result = _run_and_tee(cmd, env, timeout=60)
        output_lines.append(f"$ {' '.join(cmd)}")
        if result.stdout.strip():
            output_lines.append(result.stdout.strip())
        if result.stderr.strip():
            output_lines.append(result.stderr.strip())
        if result.returncode != 0:
            logger.error("kubectl apply failed: %s", result.stderr)
            had_error = True

    combined = "\n".join(output_lines)
    return (not had_error, combined)


def run_smoke_test(config_path: str, step_dir: Path, kubeconfig: Path) -> tuple[int, str]:
    """Run the Iris smoke test, streaming output to stdout and capturing it.

    Returns (exit_code, combined_output).
    """
    cmd = [
        "uv",
        "run",
        "python",
        "lib/iris/scripts/smoke-test.py",
        "--config",
        str(config_path),
        "--boot-timeout",
        str(SMOKE_BOOT_TIMEOUT),
        "--job-timeout",
        str(SMOKE_JOB_TIMEOUT),
    ]

    logger.info("Running smoke test: %s", " ".join(cmd))
    output_file = step_dir / "smoke-test-output.txt"
    lines: list[str] = []

    proc = subprocess.Popen(
        cmd,
        env={**os.environ, "KUBECONFIG": str(kubeconfig.expanduser())},
        cwd=str(REPO_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    assert proc.stdout is not None

    with open(output_file, "w") as log:
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            log.write(line)
            lines.append(line)

    proc.wait(timeout=SMOKE_JOB_TIMEOUT * 5 + SMOKE_BOOT_TIMEOUT + 60)
    combined = "".join(lines)

    logger.info("Smoke test exited with code %d", proc.returncode)
    return proc.returncode, combined


def tail_output(text: str, max_lines: int = CLAUDE_OUTPUT_TAIL_LINES) -> str:
    """Return the last max_lines of text, noting truncation."""
    lines = text.splitlines()
    if len(lines) <= max_lines:
        return text
    return f"[... truncated {len(lines) - max_lines} lines ...]\n" + "\n".join(lines[-max_lines:])


def get_current_commit_sha() -> str:
    """Get the current HEAD commit SHA."""
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=10,
    )
    return result.stdout.strip() if result.returncode == 0 else ""


def pin_image_tags_to_sha(config_path: Path, sha: str) -> list[str]:
    """Rewrite :latest image tags in the config YAML to use the given git SHA.

    This forces Kubernetes to pull fresh images instead of using cached :latest.
    Returns the list of images that were rewritten.
    """
    text = config_path.read_text()
    rewritten: list[str] = []

    def _replace(m: re.Match) -> str:
        image = m.group(1)
        rewritten.append(f"{image}:{sha[:12]}")
        return f"{image}:{sha[:12]}"

    updated = re.sub(r"(ghcr\.io/[^:\s]+):[^\s]+", _replace, text)
    config_path.write_text(updated)

    for img in rewritten:
        logger.info("Pinned image tag: %s", img)
    return rewritten


def save_base_commit(log_dir: Path) -> str:
    """Save the current commit SHA as the baseline for this debug session.

    On resume, returns the previously saved SHA instead of overwriting it.
    """
    base_file = log_dir / "base-commit.txt"
    if base_file.exists():
        sha = base_file.read_text().strip()
        logger.info("Resuming from base commit %s", sha[:12])
        return sha

    sha = get_current_commit_sha()
    base_file.write_text(sha + "\n")
    logger.info("Saved base commit %s", sha[:12])
    return sha


def get_cumulative_log(base_sha: str) -> str:
    """Get the git log of all debug iterations since the base commit."""
    result = subprocess.run(
        ["git", "log", "--oneline", f"{base_sha}..HEAD"],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=10,
    )
    return result.stdout.strip() if result.returncode == 0 else ""


def has_uncommitted_changes() -> bool:
    """Check if there are any uncommitted changes in the working tree."""
    result = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=30,
    )
    return bool(result.stdout.strip())


def commit_iteration_changes(step: int, result: IterationResult | None) -> bool:
    """Stage and commit all changes from this debug iteration.

    Returns True if a commit was created, False if there was nothing to commit.
    """
    if not has_uncommitted_changes():
        logger.info("No uncommitted changes after step %d, skipping commit", step)
        return False

    status = result.status if result else "INCOMPLETE"
    hypothesis = result.hypothesis if result else "no status produced"
    message = f"debug-coreweave step {step}: {status}\n\n{hypothesis}"

    subprocess.run(
        ["git", "add", "-A"],
        cwd=str(REPO_ROOT),
        check=True,
        timeout=30,
    )
    commit_result = subprocess.run(
        ["git", "commit", "-m", message],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=30,
    )
    if commit_result.returncode != 0:
        logger.warning("git commit failed: %s", commit_result.stderr)
        return False

    logger.info("Committed changes for step %d: %s", step, status)
    return True


def collect_kubectl_diagnostics(kubeconfig: Path) -> str:
    """Collect Kubernetes cluster state for debugging.

    Runs a set of kubectl commands to capture Pod status, events, and
    Deployment/NodePool state. Returns a formatted string for inclusion
    in the Claude prompt.
    """
    kubeconfig = kubeconfig.expanduser()
    env = {**os.environ, "KUBECONFIG": str(kubeconfig)}
    sections: list[str] = []

    commands = [
        ("Pods in iris namespace", ["kubectl", "get", "pods", "-n", "iris", "-o", "wide"]),
        ("Events in iris namespace", ["kubectl", "events", "-n", "iris"]),
        ("Deployments in iris namespace", ["kubectl", "get", "deployments", "-n", "iris", "-o", "wide"]),
        ("NodePools", ["kubectl", "get", "nodepools", "-o", "wide"]),
        ("Nodes", ["kubectl", "get", "nodes", "-o", "wide"]),
        (
            "Controller Pod describe",
            ["kubectl", "describe", "pods", "-n", "iris", "-l", "app=iris-controller"],
        ),
    ]

    for label, cmd in commands:
        try:
            result = _run_and_tee(cmd, env, timeout=30)
            output = result.stdout.strip() or result.stderr.strip() or "(no output)"
        except (subprocess.TimeoutExpired, FileNotFoundError) as e:
            output = f"(command failed: {e})"
        sections.append(f"### {label}\n```\n{output}\n```")

    return "\n\n".join(sections)


def build_claude_prompt(
    smoke_output: str,
    debug_history: str,
    config_path: str,
    step_dir: Path,
    kubectl_diagnostics: str = "",
    base_sha: str = "",
) -> str:
    """Build the prompt for Claude to diagnose and fix the smoke test failure."""
    cumulative_log = get_cumulative_log(base_sha) if base_sha else ""

    prompt_parts = [
        "You are debugging a CoreWeave integration failure in the Iris cluster autoscaler.",
        "",
        "Read these files for context:",
        f"- Design doc and conventions: {IRIS_ROOT / 'AGENTS.md'}",
        f"- CoreWeave config: {config_path}",
        f"- CoreWeave platform: {IRIS_ROOT / 'src' / 'iris' / 'cluster' / 'platform' / 'coreweave.py'}",
        f"- Kubectl wrapper: {IRIS_ROOT / 'src' / 'iris' / 'cluster' / 'platform' / 'kubectl.py'}",
        f"- Smoke test script: {IRIS_ROOT / 'scripts' / 'smoke-test.py'}",
        "",
        "## Smoke Test Failure Output",
        "",
        "```",
        tail_output(smoke_output),
        "```",
        "",
    ]

    if kubectl_diagnostics:
        prompt_parts.extend(
            [
                "## Kubernetes Cluster State (at time of failure)",
                "",
                kubectl_diagnostics,
                "",
            ]
        )

    if cumulative_log:
        prompt_parts.extend(
            [
                "## Git log (all debug iterations so far)",
                "",
                "```",
                cumulative_log,
                "```",
                "",
            ]
        )

    if debug_history:
        prompt_parts.extend(
            [
                "## Debug History (DO NOT re-try hypotheses that already failed)",
                "",
                debug_history,
                "",
            ]
        )

    prompt_parts.extend(
        [
            "## Instructions",
            "",
            "1. Diagnose the root cause of the failure by reading the relevant source files.",
            "2. Make code changes to fix it. Try ONE hypothesis at a time.",
            "3. Run local unit/integration tests only: uv run pytest lib/iris/tests/ -x -k coreweave",
            "4. Do NOT run the smoke test (lib/iris/scripts/smoke-test.py) yourself; the outer loop runs it.",
            "5. Do NOT commit your changes; the outer loop will commit for you.",
            f"6. As your LAST action, write {step_dir / 'status.json'} with this format:",
            "",
            "```json",
            "{",
            '  "hypothesis": "one-line description of what you tried",',
            '  "changes": ["file1.py", "file2.py"],',
            '  "tests_run": "description of tests you ran",',
            '  "next_steps": "what to try next if this did not work"',
            "}",
            "```",
            "",
            f"Also write a brief summary to {step_dir / 'summary.md'}.",
            "",
            "IMPORTANT: Write status.json and summary.md as your LAST action.",
        ]
    )

    return "\n".join(prompt_parts)


def run_claude(prompt: str, log_file: Path, timeout: int, model: str) -> int:
    """Run Claude CLI with the given prompt, streaming output to terminal and log file.

    Returns the process exit code, or -1 on timeout.
    """
    cmd = ["claude", "--dangerously-skip-permissions", "--verbose", "-p", prompt]
    if model:
        cmd.extend(["--model", model])

    # Strip ANTHROPIC_API_KEY so Claude uses OAuth instead of credit mode.
    env = {k: v for k, v in os.environ.items() if k != "ANTHROPIC_API_KEY"}

    with open(log_file, "w") as log:
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
            env=env,
            cwd=str(REPO_ROOT),
        )
        assert proc.stdout is not None

        try:
            for line in proc.stdout:
                sys.stdout.write(line)
                sys.stdout.flush()
                log.write(line)
            proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            logger.warning("Claude timed out after %ds, killing process group", timeout)
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()
            return -1
        except KeyboardInterrupt:
            logger.warning("Interrupted, killing Claude process group")
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()
            raise
        finally:
            if proc.poll() is None:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()

    return proc.returncode


def parse_iteration_result(step_dir: Path) -> IterationResult | None:
    """Read status.json from Claude's output directory. Returns None if not found."""
    status_file = step_dir / "status.json"
    if not status_file.exists():
        logger.warning("No status.json found in %s", step_dir)
        return None

    try:
        data = json.loads(status_file.read_text())
        return IterationResult(
            status=data.get("status", "FAILED"),
            hypothesis=data.get("hypothesis", "unknown"),
            changes=data.get("changes", []),
            tests_run=data.get("tests_run", "unknown"),
            next_steps=data.get("next_steps", ""),
        )
    except (json.JSONDecodeError, KeyError) as e:
        logger.warning("Failed to parse status.json: %s", e)
        return None


def load_debug_history(log_dir: Path) -> str:
    """Load the accumulated debug history from previous iterations."""
    history_file = log_dir / "debug-log.md"
    if history_file.exists():
        return history_file.read_text()
    return ""


def append_debug_log(log_dir: Path, step: int, result: IterationResult | None, smoke_output_summary: str):
    """Append an iteration entry to the debug log, following the debugger.md recipe format."""
    history_file = log_dir / "debug-log.md"
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    if not history_file.exists():
        header = (
            "# Debugging log for CoreWeave integration\n\n"
            "Automated debugging loop for Iris CoreWeave smoke test.\n\n"
            "## Initial status\n\n"
            "Smoke test failing on CoreWeave cluster.\n\n"
        )
        history_file.write_text(header)

    with open(history_file, "a") as f:
        f.write(f"## Iteration {step} ({timestamp})\n\n")
        if result:
            f.write(f"**Hypothesis:** {result.hypothesis}\n\n")
            f.write(f"**Status:** {result.status}\n\n")
            if result.changes:
                f.write("**Changes:**\n")
                for change in result.changes:
                    f.write(f"- `{change}`\n")
                f.write("\n")
            f.write(f"**Tests run:** {result.tests_run}\n\n")
            if result.next_steps:
                f.write(f"**Next steps:** {result.next_steps}\n\n")
        else:
            f.write("**Result:** No status.json produced (Claude may have crashed or timed out).\n\n")
            f.write(f"**Smoke output tail:**\n```\n{tail_output(smoke_output_summary, 30)}\n```\n\n")

    # Also update the repo-level debug log
    if DEBUG_LOG_PATH.parent.exists():
        shutil.copy2(history_file, DEBUG_LOG_PATH)


def find_start_step(log_dir: Path) -> int:
    """Find the next step number by scanning existing step directories."""
    max_step = 0
    if log_dir.exists():
        for child in log_dir.iterdir():
            if child.is_dir() and child.name.startswith("step-"):
                try:
                    step_num = int(child.name.split("-", 1)[1])
                    max_step = max(max_step, step_num)
                except ValueError:
                    pass
    return max_step + 1


@click.command()
@click.option(
    "--config",
    "config_path",
    default="lib/iris/examples/coreweave.yaml",
    show_default=True,
    help="Path to cluster config YAML (relative to repo root)",
)
@click.option("--max-iterations", default=10, show_default=True, help="Maximum debug iterations")
@click.option(
    "--log-dir",
    default=None,
    type=click.Path(path_type=Path),
    help="Log directory (default: logs/debug-coreweave-{timestamp})",
)
@click.option("--timeout", default=1800, show_default=True, help="Per-iteration timeout in seconds for Claude")
@click.option("--model", default="", help="Claude model override (e.g. claude-sonnet-4-5)")
@click.option("--resume", default=None, type=click.Path(path_type=Path), help="Resume from an existing log directory")
@click.option("--kubeconfig", default="~/.kube/coreweave-iris", show_default=True, help="Path to kubeconfig file")
def main(
    config_path: str,
    max_iterations: int,
    log_dir: Path | None,
    timeout: int,
    model: str,
    resume: Path | None,
    kubeconfig: str,
):
    """Automated debugging loop for CoreWeave integration.

    Repeatedly cleans the K8s cluster, runs the smoke test, and hands
    failures to Claude for diagnosis and code fixes until the test passes.
    """
    kubeconfig_path = Path(kubeconfig)
    full_config_path = REPO_ROOT / config_path

    if resume:
        log_dir = resume
    elif log_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        log_dir = Path("logs") / f"debug-coreweave-{timestamp}"

    log_dir.mkdir(parents=True, exist_ok=True)
    configure_logging(log_dir)

    check_preflight(kubeconfig_path, full_config_path)

    base_sha = save_base_commit(log_dir)

    sha = get_current_commit_sha()
    if sha:
        pin_image_tags_to_sha(full_config_path, sha)
    else:
        logger.warning("Could not determine git SHA; image tags left unchanged")

    if resume:
        logger.info("Resuming from %s", log_dir)
    start_step = find_start_step(log_dir)
    logger.info("Log directory: %s (starting at step %d)", log_dir, start_step)

    for step in range(start_step, start_step + max_iterations):
        step_dir = log_dir / f"step-{step}"
        step_dir.mkdir(parents=True, exist_ok=True)

        logger.info("=" * 60)
        logger.info("STEP %d", step)
        logger.info("=" * 60)

        # Phase 1: Clean up Kubernetes
        logger.info("Cleaning Kubernetes cluster...")
        cleanup_ok, cleanup_output = cleanup_kubernetes(kubeconfig_path)
        (step_dir / "cleanup-output.txt").write_text(cleanup_output)
        if not cleanup_ok:
            logger.warning("Kubernetes cleanup had errors (continuing anyway)")

        # Phase 2: Run smoke test
        logger.info("Running smoke test...")
        smoke_exit, smoke_output = run_smoke_test(str(full_config_path), step_dir, kubeconfig_path)

        if smoke_exit == 0:
            logger.info("Smoke test PASSED at step %d", step)
            append_debug_log(
                log_dir,
                step,
                IterationResult(
                    status="PASSED",
                    hypothesis="smoke test passed without code changes this iteration",
                    changes=[],
                    tests_run="smoke test",
                    next_steps="none - success",
                ),
                smoke_output,
            )
            sys.exit(0)

        logger.info("Smoke test failed (exit %d), collecting kubectl diagnostics...", smoke_exit)

        # Collect Kubernetes state for debugging context
        kubectl_diag = collect_kubectl_diagnostics(kubeconfig_path)
        (step_dir / "kubectl-diagnostics.txt").write_text(kubectl_diag)

        # Phase 3: Build prompt and run Claude
        logger.info("Handing off to Claude...")
        debug_history = load_debug_history(log_dir)
        prompt = build_claude_prompt(smoke_output, debug_history, config_path, step_dir, kubectl_diag, base_sha)

        prompt_file = step_dir / "prompt.txt"
        prompt_file.write_text(prompt)

        claude_log = step_dir / "claude-output.log"
        exit_code = run_claude(prompt, claude_log, timeout, model)

        if exit_code == -1:
            logger.warning("Claude timed out at step %d", step)
            append_debug_log(log_dir, step, None, smoke_output)
            commit_iteration_changes(step, None)
            continue
        if exit_code != 0:
            logger.warning("Claude exited with code %d at step %d", exit_code, step)

        # Phase 4: Parse result, commit changes, and update debug log.
        # Claude's self-reported status is informational only — the smoke test
        # exit code (checked at the top of the loop) is the sole source of truth.
        result = parse_iteration_result(step_dir)
        append_debug_log(log_dir, step, result, smoke_output)
        commit_iteration_changes(step, result)

        if result:
            logger.info("Hypothesis: %s", result.hypothesis)
            logger.info("Next steps: %s", result.next_steps)
        else:
            logger.warning("No status.json from Claude; continuing to next iteration")

    logger.error("Hit max iterations (%d). See %s for debug history.", max_iterations, log_dir)
    sys.exit(1)


if __name__ == "__main__":
    main()
