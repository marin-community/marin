#!/usr/bin/env python3
# Copyright 2026 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Gated DeltaNet TPU iteration utilities.

This CLI wraps the most repetitive commands used while optimizing
`levanter.layers.gated_deltanet` on TPU:

- correctness tests (Ray + dev_tpu)
- lightweight profile runs
- Ray job wait/log polling
- Hugging Face trace download
- unattended Codex hill-climb loops
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
RAY_RUN = ["uv", "run", "lib/marin/src/marin/run/ray_run.py"]
CLUSTER_CLI = ["uv", "run", "scripts/ray/cluster.py"]
DEV_TPU = ["uv", "run", "scripts/ray/dev_tpu.py"]

GDN_KERNEL_TEST = "tests/test_gdn_kernels.py"
GDN_LAYER_TEST = "tests/test_gdn_layer.py"
TORCH_CPU_INDEX = "https://download.pytorch.org/whl/cpu"
DEFAULT_HF_TRACE_PATTERN = r"(perfetto_trace\\.json\\.gz$|trace\\.json\\.gz$|profile\\.json$)"
DEFAULT_HF_TRACE_PATTERN_WITH_XPLANE = (
    r"(perfetto_trace\\.json\\.gz$|trace\\.json\\.gz$|profile\\.json$|xplane\\.pb$)"
)
DEFAULT_HILLCLIMB_LOG = REPO_ROOT / "lib/levanter/.agents/projects/gdn_pallas_tpu_hillclimb.md"

SESSION_DIRECTIVE_PRESETS = {
    "triangular-inversion": (
        "Prioritize strict lower-triangular inversion redesign on TPU. Favor numerically stable "
        "parallel formulations (for example block-recursive or hybrid doubling variants) that "
        "reduce sequential dependencies and improve MXU utilization without changing model semantics."
    ),
}

SUBMISSION_ID_RE = re.compile(r"Job submitted with ID:\s*([^\s]+)")
CODEX_SEARCH_FLAG_RE = re.compile(r"(^|\\s)--search(\\s|$)")
CODEX_REASONING_VARIANT_RE = re.compile(
    r"unknown variant `(?P<requested>[^`]+)`, expected one of (?P<expected>.+?)\s+in `model_reasoning_effort`",
    re.DOTALL,
)


def _echo_cmd(cmd: Sequence[str]) -> None:
    print(f"[gdnctl] $ {shlex.join(cmd)}", flush=True)


def _run(
    cmd: Sequence[str],
    *,
    cwd: Path | None = None,
    input_text: str | None = None,
    capture_output: bool = False,
    check: bool = True,
    extra_env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    _echo_cmd(cmd)
    env = None
    if extra_env:
        env = os.environ.copy()
        env.update(extra_env)
    return subprocess.run(
        list(cmd),
        cwd=str(cwd or REPO_ROOT),
        env=env,
        input=input_text,
        text=True,
        capture_output=capture_output,
        check=check,
    )


def _git_head(cwd: Path) -> str:
    proc = _run(["git", "rev-parse", "HEAD"], cwd=cwd, capture_output=True)
    return proc.stdout.strip()


def _git_dirty(cwd: Path) -> bool:
    proc = _run(["git", "status", "--porcelain"], cwd=cwd, capture_output=True)
    return bool(proc.stdout.strip())


def _stash_dirty_tree(cwd: Path, *, iteration: int) -> str:
    stash_message = f"gdnctl-codex-loop-iter-{iteration:03d}-{int(time.time())}"
    proc = _run(
        ["git", "stash", "push", "-u", "-m", stash_message],
        cwd=cwd,
        capture_output=True,
        check=True,
    )
    output = (proc.stdout + proc.stderr).strip()
    print(f"[gdnctl] stashed dirty tree: {stash_message}")
    if output:
        print(f"[gdnctl] {output}")
    return stash_message


def _test_targets(selection: str) -> list[str]:
    if selection == "kernels":
        return [GDN_KERNEL_TEST]
    if selection == "layer":
        return [GDN_LAYER_TEST]
    if selection == "both":
        return [GDN_KERNEL_TEST, GDN_LAYER_TEST]
    raise ValueError(f"Unknown test selection: {selection}")


def _build_remote_test_command(test_selection: str, extra_pytest_args: str | None) -> str:
    test_files = " ".join(_test_targets(test_selection))
    pytest_cmd = f"EQX_ON_ERROR=nan WANDB_MODE=offline uv run pytest {test_files} -v"
    if extra_pytest_args:
        pytest_cmd = f"{pytest_cmd} {extra_pytest_args.strip()}"

    pieces = [
        "cd lib/levanter",
        "uv sync --extra=tpu --group test",
        f"uv pip install torch --index-url {shlex.quote(TORCH_CPU_INDEX)}",
        pytest_cmd,
    ]
    return " && ".join(pieces)


def _extract_submission_id(text: str) -> str | None:
    match = SUBMISSION_ID_RE.search(text)
    return None if match is None else match.group(1)


def _load_session_directives(args: argparse.Namespace) -> list[str]:
    directives: list[str] = []

    for preset in args.directive_preset:
        directives.append(SESSION_DIRECTIVE_PRESETS[preset])

    for directive_file in args.directive_file:
        directive_text = Path(directive_file).read_text(encoding="utf-8").strip()
        if directive_text:
            directives.append(directive_text)

    for directive in args.directive:
        stripped = directive.strip()
        if stripped:
            directives.append(stripped)

    return directives


def _codex_exec_supports_search(codex_bin: str) -> bool:
    """Return True when `codex exec` advertises a `--search` flag."""
    proc = _run([codex_bin, "exec", "--help"], capture_output=True, check=False)
    if proc.returncode != 0:
        return False
    help_text = proc.stdout + proc.stderr
    return CODEX_SEARCH_FLAG_RE.search(help_text) is not None


def _extract_reasoning_error(text: str) -> tuple[str, list[str]] | None:
    """Parse codex config enum errors for model_reasoning_effort."""
    match = CODEX_REASONING_VARIANT_RE.search(text)
    if match is None:
        return None
    requested = match.group("requested")
    expected_raw = match.group("expected")
    expected = re.findall(r"`([^`]+)`", expected_raw)
    return requested, expected


def _candidate_codex_binaries(explicit_binary: str | None) -> list[str]:
    candidates: list[str] = []

    def add_candidate(candidate: str) -> None:
        resolved = shutil.which(candidate) if "/" not in candidate else candidate
        if resolved is None:
            return
        normalized = str(Path(resolved).expanduser().resolve(strict=False))
        if normalized not in candidates:
            candidates.append(normalized)

    if explicit_binary:
        add_candidate(explicit_binary)
    add_candidate("codex")
    add_candidate("/Applications/Codex.app/Contents/Resources/codex")
    add_candidate(str(Path.home() / "Applications/Codex.app/Contents/Resources/codex"))
    return candidates


def _codex_supported_reasoning_efforts(codex_bin: str) -> list[str] | None:
    """Return supported model_reasoning_effort values from local config validation."""
    probe_value = "__gdnctl_invalid_reasoning_effort_probe__"
    with tempfile.TemporaryDirectory(prefix="gdnctl-codex-probe-") as probe_home:
        proc = _run(
            [
                codex_bin,
                "exec",
                "-c",
                f"model_reasoning_effort={json.dumps(probe_value)}",
                "probe",
            ],
            capture_output=True,
            check=False,
            extra_env={"CODEX_HOME": probe_home, "HOME": probe_home},
        )
    parsed = _extract_reasoning_error(proc.stdout + proc.stderr)
    if parsed is None:
        return None
    _, expected = parsed
    return expected


def _resolve_codex_binary(
    *,
    explicit_binary: str | None,
    reasoning_effort: str | None,
) -> tuple[str, list[str] | None]:
    candidates = _candidate_codex_binaries(explicit_binary)
    if not candidates:
        raise SystemExit(
            "[gdnctl] Could not find a `codex` binary. Install Codex CLI or pass --codex-bin /path/to/codex."
        )

    if reasoning_effort is None:
        return candidates[0], None

    unknown_support: list[str] = []
    support_rows: list[tuple[str, list[str] | None]] = []
    for candidate in candidates:
        supported = _codex_supported_reasoning_efforts(candidate)
        support_rows.append((candidate, supported))
        if supported is None:
            unknown_support.append(candidate)
            continue
        if reasoning_effort in supported:
            return candidate, supported

    if unknown_support:
        chosen = unknown_support[0]
        print(
            "[gdnctl] WARNING: could not determine supported reasoning-effort values for "
            f"{chosen}; trying it for --reasoning-effort={reasoning_effort}.",
            file=sys.stderr,
        )
        return chosen, None

    detail_lines = []
    for candidate, supported in support_rows:
        if supported is None:
            detail = "unknown"
        else:
            detail = ", ".join(supported)
        detail_lines.append(f"  - {candidate}: {detail}")

    details = "\n".join(detail_lines)
    raise SystemExit(
        "[gdnctl] No discovered codex binary supports "
        f"--reasoning-effort={reasoning_effort!r}.\n"
        f"{details}\n"
        "Install a newer Codex CLI and/or pass --codex-bin to select it."
    )


def cmd_ray_test(args: argparse.Namespace) -> int:
    remote_cmd = _build_remote_test_command(args.tests, args.pytest_args)

    cmd = [
        *RAY_RUN,
        "--cluster",
        args.cluster,
        "--tpu",
        args.tpu,
        "-e",
        "EQX_ON_ERROR=nan",
        "-e",
        "WANDB_MODE=offline",
    ]
    if args.no_wait:
        cmd.append("--no_wait")
    cmd += ["--", "bash", "-lc", remote_cmd]

    proc = _run(cmd, capture_output=args.no_wait, check=False)
    if args.no_wait:
        sys.stdout.write(proc.stdout)
        sys.stderr.write(proc.stderr)
        submission_id = _extract_submission_id(proc.stdout + proc.stderr)
        if submission_id:
            print(f"[gdnctl] submission_id={submission_id}")
        else:
            print("[gdnctl] WARNING: could not parse submission id from output.", file=sys.stderr)
    return proc.returncode


def cmd_dev_tpu_allocate(args: argparse.Namespace) -> int:
    cmd = [
        *DEV_TPU,
        "--cluster",
        args.cluster,
        "--tpu-name",
        args.tpu_name,
        "allocate",
    ]
    if args.tpu_type:
        cmd += ["--tpu-type", args.tpu_type]
    return _run(cmd, check=False).returncode


def cmd_dev_tpu_test(args: argparse.Namespace) -> int:
    remote_cmd = _build_remote_test_command(args.tests, args.pytest_args)
    cmd = [
        *DEV_TPU,
        "--cluster",
        args.cluster,
        "--tpu-name",
        args.tpu_name,
        "execute",
        "-e",
        "EQX_ON_ERROR=nan",
        "-e",
        "WANDB_MODE=offline",
    ]
    if args.no_sync:
        cmd.append("--no-sync")
    cmd += ["--", remote_cmd]
    return _run(cmd, check=False).returncode


def _profile_command_lines(
    *,
    include_tpu_sync: bool,
    profile_args: Sequence[str],
) -> list[str]:
    lines = ["set -e"]
    if include_tpu_sync:
        lines.append("uv sync --all-packages --extra=tpu --python=3.11")
    else:
        lines.append("uv sync")
    lines.append(f"uv pip install --python .venv/bin/python --index-url {shlex.quote(TORCH_CPU_INDEX)} --force-reinstall torch")
    lines.append("(uv pip uninstall --python .venv/bin/python torchvision || true)")
    lines.append(f".venv/bin/python -m experiments.speedrun.hackable_transformer_gdn.tiny_profile {shlex.join(profile_args)}")
    return lines


def _append_profile_env(cmd: list[str], args: argparse.Namespace) -> None:
    cmd += ["-e", "EQX_ON_ERROR=nan"]
    cmd += ["-e", f"WANDB_MODE={args.wandb_mode}"]
    cmd += ["-e", f"GDN_PROFILE_SIZE={args.size}"]
    cmd += ["-e", f"GDN_PROFILE_NUM_STEPS={args.num_steps}"]
    cmd += ["-e", f"GDN_PROFILE_PROFILE_START_STEP={args.profile_start_step}"]
    cmd += ["-e", f"GDN_PROFILE_PROFILE_NUM_STEPS={args.profile_num_steps}"]
    cmd += ["-e", f"GDN_PROFILE_RUN_NAME_PREFIX={args.run_name_prefix}"]
    cmd += ["-e", f"GDN_PROFILE_TPU_VARIANT={args.tpu}"]
    if args.batch_size is not None:
        cmd += ["-e", f"GDN_PROFILE_BATCH_SIZE={args.batch_size}"]
    if args.chunk_size is not None:
        cmd += ["-e", f"GDN_PROFILE_CHUNK_SIZE={args.chunk_size}"]
    if args.segment_size is not None:
        cmd += ["-e", f"GDN_PROFILE_SEGMENT_SIZE={args.segment_size}"]


def cmd_ray_profile(args: argparse.Namespace) -> int:
    cmd = [
        *RAY_RUN,
        "--cluster",
        args.cluster,
        "--tpu",
        args.tpu,
    ]
    _append_profile_env(cmd, args)

    if args.no_wait:
        cmd.append("--no_wait")

    profile_args = ["--force_run_failed", "true"]
    if args.dry_run:
        profile_args += ["--dry_run", "true"]

    profile_cmd_lines = _profile_command_lines(include_tpu_sync=False, profile_args=profile_args)
    profile_cmd = " && ".join(profile_cmd_lines)

    cmd += ["--", "bash", "-lc", profile_cmd]

    proc = _run(cmd, capture_output=args.no_wait, check=False)
    if args.no_wait:
        sys.stdout.write(proc.stdout)
        sys.stderr.write(proc.stderr)
        submission_id = _extract_submission_id(proc.stdout + proc.stderr)
        if submission_id:
            print(f"[gdnctl] submission_id={submission_id}")
        else:
            print("[gdnctl] WARNING: could not parse submission id from output.", file=sys.stderr)
    return proc.returncode


def cmd_dev_tpu_profile(args: argparse.Namespace) -> int:
    cmd = [
        *DEV_TPU,
        "--cluster",
        args.cluster,
        "--tpu-name",
        args.tpu_name,
        "execute",
    ]
    _append_profile_env(cmd, args)
    if args.marin_prefix:
        marin_prefix = args.marin_prefix
    else:
        marin_prefix = f"gs://marin-{args.cluster}"
    cmd += ["-e", f"MARIN_PREFIX={marin_prefix}"]
    if args.no_sync:
        cmd.append("--no-sync")

    profile_args = ["--force_run_failed", "true"]
    if args.dry_run:
        profile_args += ["--dry_run", "true"]

    profile_cmd_lines = _profile_command_lines(include_tpu_sync=True, profile_args=profile_args)
    cmd += ["--", " && ".join(profile_cmd_lines)]
    return _run(cmd, check=False).returncode


def cmd_ray_wait(args: argparse.Namespace) -> int:
    cmd = [
        *CLUSTER_CLI,
        "--cluster",
        args.cluster,
        "wait-job",
        args.job_id,
        "--poll",
        str(args.poll),
    ]
    if args.timeout is not None:
        cmd += ["--timeout", str(args.timeout)]
    if args.show_logs:
        cmd.append("--show-logs")
    if args.tail is not None:
        cmd += ["--tail", str(args.tail)]
    if args.grep:
        cmd += ["--grep", args.grep]
    return _run(cmd, check=False).returncode


def cmd_ray_logs(args: argparse.Namespace) -> int:
    cmd = [
        *CLUSTER_CLI,
        "--cluster",
        args.cluster,
        "job-logs",
        "-n",
        str(args.tail),
    ]
    if args.grep:
        cmd += ["-g", args.grep]
    cmd += [args.job_id]
    return _run(cmd, check=False).returncode


def cmd_hf_download_trace(args: argparse.Namespace) -> int:
    try:
        from huggingface_hub import HfApi, hf_hub_download
    except ImportError as exc:
        print(
            "[gdnctl] huggingface_hub is required for hf-download-trace. Install with `uv pip install huggingface_hub`.",
            file=sys.stderr,
        )
        raise SystemExit(2) from exc

    token = args.token
    if token is None:
        token = None
        # huggingface_hub already picks up HF_TOKEN/HUGGINGFACE_HUB_TOKEN from env

    api = HfApi(token=token)
    files = api.list_repo_files(repo_id=args.repo_id, repo_type=args.repo_type, revision=args.revision)

    if args.path_prefix:
        normalized_prefix = args.path_prefix.strip("/")
        files = [f for f in files if f.startswith(normalized_prefix + "/") or f == normalized_prefix]

    pattern_text = args.pattern
    if args.include_xplane and pattern_text == DEFAULT_HF_TRACE_PATTERN:
        pattern_text = DEFAULT_HF_TRACE_PATTERN_WITH_XPLANE
    pattern = re.compile(pattern_text)
    selected = sorted([f for f in files if pattern.search(f)])

    if args.limit is not None:
        selected = selected[: args.limit]

    if not selected:
        print("[gdnctl] No matching files found.")
        return 0

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[gdnctl] downloading {len(selected)} files to {output_dir}")
    for file_path in selected:
        local_path = hf_hub_download(
            repo_id=args.repo_id,
            filename=file_path,
            repo_type=args.repo_type,
            revision=args.revision,
            token=token,
            local_dir=str(output_dir),
            local_dir_use_symlinks=False,
        )
        print(f"[gdnctl] {file_path} -> {local_path}")

    return 0


def _format_prompt(
    template: str,
    iteration: int,
    total: int,
    head_sha: str,
    session_directives: Sequence[str],
) -> str:
    prompt = (
        template.replace("{{ITERATION}}", str(iteration))
        .replace("{{TOTAL_ITERATIONS}}", str(total))
        .replace("{{HEAD_SHA}}", head_sha)
    )
    if not session_directives:
        return prompt

    blocks = []
    for idx, directive in enumerate(session_directives, start=1):
        blocks.append(f"[Directive {idx}]\n{directive}")
    joined_blocks = "\n".join(blocks)

    return (
        f"{prompt.rstrip()}\n\n"
        "Session directives for this codex-loop run (must follow unless they conflict with tests/correctness):\n\n"
        f"{joined_blocks}\n"
    )


def _run_post_checks(cwd: Path, commands: Sequence[str]) -> None:
    for command in commands:
        _run(["bash", "-lc", command], cwd=cwd)


def cmd_lint_log(args: argparse.Namespace) -> int:
    log_path = Path(args.log_file).resolve()
    if not log_path.exists():
        print(f"[gdnctl] log file not found: {log_path}", file=sys.stderr)
        return 2

    lines = log_path.read_text(encoding="utf-8").splitlines()
    if args.scope == "last-entry":
        last_entry_start = None
        for idx, line in enumerate(lines):
            if line.startswith("### Iteration "):
                last_entry_start = idx
        if last_entry_start is None:
            pending_lines = []
        else:
            pending_lines = [
                idx
                for idx, line in enumerate(lines[last_entry_start:], start=last_entry_start + 1)
                if line.strip() == "- Commit: (pending)"
            ]
    else:
        pending_lines = [idx for idx, line in enumerate(lines, start=1) if line.strip() == "- Commit: (pending)"]

    if pending_lines and not args.allow_pending:
        print(f"[gdnctl] unresolved `Commit: (pending)` entries in {log_path}:", file=sys.stderr)
        for line_no in pending_lines:
            print(f"[gdnctl]   line {line_no}", file=sys.stderr)
        return 1

    print(f"[gdnctl] log lint passed: {log_path}")
    return 0


def cmd_codex_loop(args: argparse.Namespace) -> int:
    workdir = Path(args.workdir).resolve()
    prompt_template = Path(args.prompt_file).read_text(encoding="utf-8")
    session_directives = _load_session_directives(args)
    log_dir = Path(args.log_dir).resolve()
    log_dir.mkdir(parents=True, exist_ok=True)
    codex_bin, supported_efforts = _resolve_codex_binary(
        explicit_binary=args.codex_bin,
        reasoning_effort=args.reasoning_effort,
    )
    selected_efforts = ", ".join(supported_efforts) if supported_efforts else "unknown"
    print(f"[gdnctl] using codex binary: {codex_bin}")
    print(f"[gdnctl] supported model_reasoning_effort values: {selected_efforts}")
    if session_directives:
        print(f"[gdnctl] session directives: {len(session_directives)} loaded")

    search_supported = _codex_exec_supports_search(codex_bin) if args.search else False
    if args.search and not search_supported:
        print(
            "[gdnctl] WARNING: installed `codex exec` does not support `--search`; continuing without it.",
            file=sys.stderr,
        )

    failures = 0
    for iteration in range(1, args.iterations + 1):
        print(f"\n[gdnctl] === codex iteration {iteration}/{args.iterations} ===")

        if args.require_clean and _git_dirty(workdir):
            if args.dirty_policy == "stash":
                _stash_dirty_tree(workdir, iteration=iteration)
            else:
                print(
                    "[gdnctl] Working tree is dirty before starting the iteration. "
                    "Commit/stash or rerun with --allow-dirty or --dirty-policy stash.",
                    file=sys.stderr,
                )
                return 2

        head_before = _git_head(workdir)
        prompt = _format_prompt(
            prompt_template,
            iteration,
            args.iterations,
            head_before,
            session_directives,
        )

        message_path = log_dir / f"iteration-{iteration:03d}-last-message.txt"
        prompt_path = log_dir / f"iteration-{iteration:03d}-prompt.txt"
        prompt_path.write_text(prompt, encoding="utf-8")

        cmd = [
            codex_bin,
            "exec",
            "-C",
            str(workdir),
            "--dangerously-bypass-approvals-and-sandbox",
            "-o",
            str(message_path),
        ]

        if args.model:
            cmd += ["-m", args.model]
        if args.reasoning_effort:
            cmd += ["-c", f"model_reasoning_effort={json.dumps(args.reasoning_effort)}"]
        if args.codex_profile:
            cmd += ["-p", args.codex_profile]
        if args.search and search_supported:
            cmd.append("--search")

        cmd.append("-")

        rc = _run(cmd, cwd=workdir, input_text=prompt, check=False).returncode

        if rc != 0:
            failures += 1
            print(f"[gdnctl] codex iteration failed with exit code {rc}", file=sys.stderr)
            if failures > args.max_failures:
                return rc
            continue

        head_after = _git_head(workdir)
        if args.require_new_commit and head_after == head_before:
            message = (
                "[gdnctl] Iteration did not produce a new commit. "
                "Use --allow-no-commit or set --no-commit-policy count-failure/continue."
            )
            if args.no_commit_policy == "fail":
                print(message, file=sys.stderr)
                return 3
            print(f"{message} Continuing due to policy={args.no_commit_policy}.", file=sys.stderr)
            if args.no_commit_policy == "count-failure":
                failures += 1
                if failures > args.max_failures:
                    return 3
            continue

        if args.post_check:
            try:
                _run_post_checks(workdir, args.post_check)
            except subprocess.CalledProcessError as exc:
                print(f"[gdnctl] post-check failed: {exc}", file=sys.stderr)
                return exc.returncode or 1

        if iteration < args.iterations and args.sleep_seconds > 0:
            time.sleep(args.sleep_seconds)

    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="GDN TPU optimization helper CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    ray_test = subparsers.add_parser("ray-test", help="Run GDN tests on TPU via ray_run")
    ray_test.add_argument("--cluster", default="us-central1", help="Cluster name for ray_run")
    ray_test.add_argument("--tpu", default="auto", help="TPU type for ray_run")
    ray_test.add_argument("--tests", choices=["kernels", "layer", "both"], default="both")
    ray_test.add_argument("--pytest-args", default=None, help="Extra args appended to pytest")
    ray_test.add_argument("--no-wait", action="store_true", help="Submit and return immediately")
    ray_test.set_defaults(func=cmd_ray_test)

    dev_alloc = subparsers.add_parser("dev-tpu-allocate", help="Allocate a development TPU")
    dev_alloc.add_argument("--cluster", default="us-central1", help="Cluster name")
    dev_alloc.add_argument("--tpu-name", required=True, help="Unique TPU name")
    dev_alloc.add_argument("--tpu-type", default=None, help="Optional TPU type (e.g., v5p-8)")
    dev_alloc.set_defaults(func=cmd_dev_tpu_allocate)

    dev_test = subparsers.add_parser("dev-tpu-test", help="Run GDN tests on an allocated dev TPU")
    dev_test.add_argument("--cluster", default="us-central1", help="Cluster name")
    dev_test.add_argument("--tpu-name", required=True, help="TPU name used during allocation")
    dev_test.add_argument("--tests", choices=["kernels", "layer", "both"], default="both")
    dev_test.add_argument("--pytest-args", default=None, help="Extra args appended to pytest")
    dev_test.add_argument("--no-sync", action="store_true", help="Skip rsync before execute")
    dev_test.set_defaults(func=cmd_dev_tpu_test)

    dev_profile = subparsers.add_parser("dev-tpu-profile", help="Run lightweight GDN profile on an allocated dev TPU")
    dev_profile.add_argument("--cluster", default="us-central1", help="Cluster name")
    dev_profile.add_argument("--tpu-name", required=True, help="TPU name used during allocation")
    dev_profile.add_argument("--tpu", default="v5p-8", help="TPU variant for train config")
    dev_profile.add_argument("--size", choices=["130m", "300m", "520m", "1_2b"], default="130m")
    dev_profile.add_argument("--num-steps", type=int, default=20, help="Training steps")
    dev_profile.add_argument("--profile-start-step", type=int, default=2)
    dev_profile.add_argument("--profile-num-steps", type=int, default=6)
    dev_profile.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Global batch override (defaults to a v5p-8-safe tiny profile batch).",
    )
    dev_profile.add_argument("--chunk-size", type=int, default=None, help="Optional GDN chunk override")
    dev_profile.add_argument("--segment-size", type=int, default=None, help="Optional GDN segment override")
    dev_profile.add_argument("--run-name-prefix", default="gdn_tinyprof", help="Run name prefix")
    dev_profile.add_argument("--wandb-mode", default="online", choices=["online", "offline", "disabled"])
    dev_profile.add_argument(
        "--marin-prefix",
        default=None,
        help="Optional MARIN_PREFIX override. Defaults to gs://marin-<cluster>.",
    )
    dev_profile.add_argument("--dry-run", action="store_true", help="Executor dry-run")
    dev_profile.add_argument("--no-sync", action="store_true", help="Skip rsync before execute")
    dev_profile.set_defaults(func=cmd_dev_tpu_profile)

    ray_profile = subparsers.add_parser("ray-profile", help="Submit a lightweight GDN profile run")
    ray_profile.add_argument("--cluster", default="us-central1", help="Cluster name")
    ray_profile.add_argument("--tpu", default="v5p-8", help="TPU variant for the run")
    ray_profile.add_argument("--size", choices=["130m", "300m", "520m", "1_2b"], default="130m")
    ray_profile.add_argument("--num-steps", type=int, default=20, help="Training steps")
    ray_profile.add_argument("--profile-start-step", type=int, default=2)
    ray_profile.add_argument("--profile-num-steps", type=int, default=6)
    ray_profile.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Global batch override (defaults to a v5p-8-safe tiny profile batch).",
    )
    ray_profile.add_argument("--chunk-size", type=int, default=None, help="Optional GDN chunk override")
    ray_profile.add_argument("--segment-size", type=int, default=None, help="Optional GDN segment override")
    ray_profile.add_argument("--run-name-prefix", default="gdn_tinyprof", help="Run name prefix")
    ray_profile.add_argument("--wandb-mode", default="online", choices=["online", "offline", "disabled"])
    ray_profile.add_argument("--dry-run", action="store_true", help="Executor dry-run")
    ray_profile.add_argument("--no-wait", action="store_true", help="Submit and return immediately")
    ray_profile.set_defaults(func=cmd_ray_profile)

    ray_wait = subparsers.add_parser("ray-wait", help="Wait for a Ray job to finish")
    ray_wait.add_argument("--cluster", default="us-central1")
    ray_wait.add_argument("job_id", help="Ray submission/job id")
    ray_wait.add_argument("--poll", type=float, default=5.0)
    ray_wait.add_argument("--timeout", type=float, default=None)
    ray_wait.add_argument("--show-logs", action="store_true")
    ray_wait.add_argument("--tail", type=int, default=200)
    ray_wait.add_argument("--grep", default=None)
    ray_wait.set_defaults(func=cmd_ray_wait)

    ray_logs = subparsers.add_parser("ray-logs", help="Tail Ray job logs")
    ray_logs.add_argument("--cluster", default="us-central1")
    ray_logs.add_argument("job_id", help="Ray submission/job id")
    ray_logs.add_argument("--tail", type=int, default=400)
    ray_logs.add_argument("--grep", default=None)
    ray_logs.set_defaults(func=cmd_ray_logs)

    hf_trace = subparsers.add_parser("hf-download-trace", help="Download trace files from Hugging Face")
    hf_trace.add_argument("--repo-id", required=True, help="HF repo id, e.g. marin-community/my-traces")
    hf_trace.add_argument("--repo-type", default="dataset", choices=["dataset", "model", "space"])
    hf_trace.add_argument("--revision", default="main")
    hf_trace.add_argument("--path-prefix", default=None, help="Only consider files under this prefix")
    hf_trace.add_argument(
        "--pattern",
        default=DEFAULT_HF_TRACE_PATTERN,
        help="Regex for files to download",
    )
    hf_trace.add_argument(
        "--include-xplane",
        action="store_true",
        help="Include .xplane.pb files when using the default --pattern.",
    )
    hf_trace.add_argument("--limit", type=int, default=None)
    hf_trace.add_argument("--output-dir", default=".profiles/hf")
    hf_trace.add_argument("--token", default=None, help="HF token (optional)")
    hf_trace.set_defaults(func=cmd_hf_download_trace)

    codex_loop = subparsers.add_parser("codex-loop", help="Run unattended Codex hill-climb iterations")
    codex_loop.add_argument("--iterations", type=int, required=True)
    codex_loop.add_argument(
        "--prompt-file",
        default=str(REPO_ROOT / "scripts/gdn/codex_iteration_prompt.md"),
        help="Prompt template file",
    )
    codex_loop.add_argument("--workdir", default=str(REPO_ROOT), help="Repository root for codex runs")
    codex_loop.add_argument("--model", default="gpt-5.3-codex", help="Codex model")
    codex_loop.add_argument(
        "--codex-bin",
        default=None,
        help="Optional path to codex binary. Defaults to an auto-discovered compatible binary.",
    )
    codex_loop.add_argument(
        "--reasoning-effort",
        default="xhigh",
        help="Reasoning effort passed as `model_reasoning_effort` config override.",
    )
    codex_loop.add_argument("--codex-profile", default=None, help="Codex CLI profile")
    codex_loop.add_argument("--search", action="store_true", help="Enable Codex web search tool")
    codex_loop.add_argument(
        "--directive",
        action="append",
        default=[],
        help="Extra per-session instruction appended to every loop iteration prompt (repeatable).",
    )
    codex_loop.add_argument(
        "--directive-file",
        action="append",
        default=[],
        help="Path to a file containing one extra per-session directive (repeatable).",
    )
    codex_loop.add_argument(
        "--directive-preset",
        choices=sorted(SESSION_DIRECTIVE_PRESETS),
        action="append",
        default=[],
        help="Named per-session directive preset (repeatable).",
    )
    codex_loop.add_argument("--sleep-seconds", type=float, default=5.0)
    codex_loop.add_argument(
        "--post-check",
        action="append",
        default=[],
        help="Shell command run after each successful iteration (repeatable)",
    )
    codex_loop.add_argument("--max-failures", type=int, default=0)
    codex_loop.add_argument("--log-dir", default=str(REPO_ROOT / ".agents/logs/gdn_codex_loop"))
    codex_loop.add_argument("--allow-dirty", action="store_true", help="Allow starting from a dirty tree")
    codex_loop.add_argument(
        "--dirty-policy",
        choices=["fail", "stash"],
        default="fail",
        help="Behavior when the tree is dirty at iteration start (ignored with --allow-dirty).",
    )
    codex_loop.add_argument("--allow-no-commit", action="store_true", help="Do not require a new commit")
    codex_loop.add_argument(
        "--no-commit-policy",
        choices=["fail", "count-failure", "continue"],
        default="fail",
        help="Behavior when an iteration exits cleanly but does not create a commit.",
    )
    codex_loop.set_defaults(func=cmd_codex_loop)

    lint_log = subparsers.add_parser("lint-log", help="Check hill-climb log for unresolved placeholders")
    lint_log.add_argument("--log-file", default=str(DEFAULT_HILLCLIMB_LOG))
    lint_log.add_argument(
        "--scope",
        choices=["last-entry", "all"],
        default="last-entry",
        help="Check placeholders only in the latest iteration entry or across the whole file.",
    )
    lint_log.add_argument(
        "--allow-pending",
        action="store_true",
        help="Do not fail when `Commit: (pending)` placeholders are present.",
    )
    lint_log.set_defaults(func=cmd_lint_log)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if getattr(args, "allow_dirty", False):
        args.require_clean = False
    else:
        args.require_clean = True

    if getattr(args, "allow_no_commit", False):
        args.require_new_commit = False
    else:
        args.require_new_commit = True

    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
