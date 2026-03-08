#!/usr/bin/env python3
# Copyright 2025 The Marin Authors
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
from collections.abc import Generator, Sequence
from contextlib import contextmanager
import json
import math
import os
import queue
import re
import signal
import shlex
import shutil
import statistics
import subprocess
import sys
import tempfile
import threading
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
DEFAULT_HF_TRACE_PATTERN_WITH_XPLANE = r"(perfetto_trace\\.json\\.gz$|trace\\.json\\.gz$|profile\\.json$|xplane\\.pb$)"
DEFAULT_HILLCLIMB_LOG = REPO_ROOT / "lib/levanter/.agents/projects/gdn_pallas_tpu_hillclimb.md"
DEFAULT_VALIDATION_CLUSTER_EXCLUDE_SUBSTRINGS = ("vllm", "big-run")
DEFAULT_MODERN_TPU_FAMILIES = ("v5p", "v5e", "v6e")

SESSION_DIRECTIVE_PRESET_FILES = {
    "training-chunk-kernel-focus": REPO_ROOT / "scripts/gdn/session_directives/training-chunk-kernel-focus.md",
    "triangular-inversion": REPO_ROOT / "scripts/gdn/session_directives/triangular-inversion.md",
    "tpu-layout-and-dtypes": REPO_ROOT / "scripts/gdn/session_directives/tpu-layout-and-dtypes.md",
    "emit-pipeline-fullseq": REPO_ROOT / "scripts/gdn/session_directives/emit-pipeline-fullseq.md",
    "fla-style-decomposition": REPO_ROOT / "scripts/gdn/session_directives/fla-style-decomposition.md",
    "expdiff-outer-product": REPO_ROOT / "scripts/gdn/session_directives/expdiff-outer-product.md",
    "matmul-batching": REPO_ROOT / "scripts/gdn/session_directives/matmul-batching.md",
    "macro-coverage-pivot": REPO_ROOT / "scripts/gdn/session_directives/macro-coverage-pivot.md",
}

SUBMISSION_ID_RE = re.compile(r"Job submitted with ID:\s*([^\s]+)")
CODEX_SEARCH_FLAG_RE = re.compile(r"(^|\\s)--search(\\s|$)")
CODEX_REASONING_VARIANT_RE = re.compile(
    r"unknown variant `(?P<requested>[^`]+)`, expected one of (?P<expected>.+?)\s+in `model_reasoning_effort`",
    re.DOTALL,
)
CODEX_FILE_UPDATE_HEADER_RE = re.compile(r"^file (update|create|add|delete|rename):")
CODEX_DIFF_LINE_RE = re.compile(
    r"^(diff --git |index [0-9a-f]+\.\.[0-9a-f]+|--- |\+\+\+ |@@ |\\ No newline at end of file$|"
    r"new file mode |deleted file mode |similarity index |rename from |rename to |old mode |"
    r"new mode |Binary files |[ +-].*)"
)
DEV_TPU_READY_MARKERS = (
    "TPU allocation is active. Press Ctrl-C to release...",
    "TPU allocated successfully!",
)
GIT_TRANSIENT_ERROR_RE = re.compile(
    r"(index\.lock|Another git process seems to be running|Unable to create '.+?\.lock')"
)
WANDB_RUN_URL_RE = re.compile(r"https://wandb\.ai/(?P<entity>[^/\s]+)/(?P<project>[^/\s]+)/runs/(?P<run_id>[^/\s?#]+)")
ITERATION_HEADING_RE = re.compile(r"^### Iteration (?P<num>\d+)(?P<suffix>[A-Za-z]*)\b")


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


def _run_streaming(
    cmd: Sequence[str],
    *,
    cwd: Path | None = None,
    input_text: str | None = None,
    check: bool = True,
    extra_env: dict[str, str] | None = None,
    hide_codex_file_updates: bool = False,
    max_runtime_seconds: float | None = None,
    idle_timeout_seconds: float | None = None,
) -> int:
    _echo_cmd(cmd)
    env = None
    if extra_env:
        env = os.environ.copy()
        env.update(extra_env)

    proc = subprocess.Popen(
        list(cmd),
        cwd=str(cwd or REPO_ROOT),
        env=env,
        stdin=subprocess.PIPE if input_text is not None else None,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    if input_text is not None and proc.stdin is not None:
        proc.stdin.write(input_text)
        proc.stdin.close()

    line_queue: queue.Queue[str | None] = queue.Queue()

    def _stdout_reader() -> None:
        stdout = proc.stdout
        if stdout is None:
            line_queue.put(None)
            return
        try:
            for line in stdout:
                line_queue.put(line)
        finally:
            line_queue.put(None)

    reader_thread = threading.Thread(target=_stdout_reader, daemon=True)
    reader_thread.start()

    suppressing_diff = False
    start_time = time.monotonic()
    last_output_time = start_time
    saw_eof = False
    timed_out = False
    while True:
        now = time.monotonic()
        elapsed = now - start_time
        idle_elapsed = now - last_output_time

        if max_runtime_seconds is not None and elapsed > max_runtime_seconds:
            print(
                "[gdnctl] streaming command exceeded max runtime; terminating "
                f"after {elapsed:.1f}s (limit={max_runtime_seconds:.1f}s).",
                file=sys.stderr,
            )
            timed_out = True
            break

        if idle_timeout_seconds is not None and idle_elapsed > idle_timeout_seconds:
            print(
                "[gdnctl] streaming command exceeded idle timeout; terminating "
                f"after {idle_elapsed:.1f}s without output (limit={idle_timeout_seconds:.1f}s).",
                file=sys.stderr,
            )
            timed_out = True
            break

        wait_timeout = 1.0
        if max_runtime_seconds is not None:
            wait_timeout = min(wait_timeout, max(0.05, max_runtime_seconds - elapsed))
        if idle_timeout_seconds is not None:
            wait_timeout = min(wait_timeout, max(0.05, idle_timeout_seconds - idle_elapsed))

        try:
            line = line_queue.get(timeout=wait_timeout)
        except queue.Empty:
            if proc.poll() is not None and saw_eof:
                break
            continue

        if line is None:
            saw_eof = True
            if proc.poll() is not None:
                break
            continue

        last_output_time = time.monotonic()
        line_text = line.rstrip("\n")
        stripped = line_text.strip()
        if hide_codex_file_updates:
            if CODEX_FILE_UPDATE_HEADER_RE.match(stripped):
                suppressing_diff = True
                continue
            if suppressing_diff:
                if stripped == "" or CODEX_DIFF_LINE_RE.match(line_text):
                    continue
                suppressing_diff = False
        sys.stdout.write(line)
        sys.stdout.flush()

    if timed_out:
        _terminate_process(proc, graceful_timeout=10.0)
        reader_thread.join(timeout=2.0)
        return_code = 124
        if check and return_code != 0:
            raise subprocess.CalledProcessError(return_code, list(cmd))
        return return_code

    return_code = proc.wait()
    reader_thread.join(timeout=2.0)
    if check and return_code != 0:
        raise subprocess.CalledProcessError(return_code, list(cmd))
    return return_code


def _run_git(
    git_args: Sequence[str],
    *,
    cwd: Path | None = None,
    capture_output: bool = True,
    check: bool = True,
    retries: int = 3,
    retry_sleep_seconds: float = 2.0,
) -> subprocess.CompletedProcess[str]:
    cmd = ["git", *git_args]
    for attempt in range(1, retries + 2):
        proc = _run(
            cmd,
            cwd=cwd,
            capture_output=capture_output,
            check=False,
        )
        if proc.returncode == 0:
            return proc

        text = (proc.stdout or "") + (proc.stderr or "")
        transient = GIT_TRANSIENT_ERROR_RE.search(text) is not None
        if transient and attempt <= retries:
            print(
                "[gdnctl] transient git error detected; retrying "
                f"{attempt}/{retries + 1} after {retry_sleep_seconds:.1f}s",
                file=sys.stderr,
            )
            if retry_sleep_seconds > 0:
                time.sleep(retry_sleep_seconds)
            continue

        if check:
            raise subprocess.CalledProcessError(
                proc.returncode,
                cmd,
                output=proc.stdout,
                stderr=proc.stderr,
            )
        return proc

    raise RuntimeError("[gdnctl] internal error: git retry loop exhausted unexpectedly.")


def _build_dev_tpu_allocate_cmd(args: argparse.Namespace, *, cluster: str) -> list[str]:
    cmd = [
        *DEV_TPU,
        "--cluster",
        cluster,
        "--tpu-name",
        args.dev_tpu_name,
    ]
    if args.dev_tpu_verbose:
        cmd.append("--verbose")
    cmd += ["allocate", "--sync-path", args.dev_tpu_sync_path]
    if args.dev_tpu_type:
        cmd += ["--tpu-type", args.dev_tpu_type]
    return cmd


def _tail_text(path: Path, *, max_lines: int = 80) -> str:
    if not path.exists():
        return ""
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return "\n".join(lines[-max_lines:])


def _terminate_process(
    proc: subprocess.Popen[str],
    *,
    graceful_timeout: float,
) -> None:
    if proc.poll() is not None:
        return
    try:
        proc.send_signal(signal.SIGINT)
    except Exception:
        proc.terminate()
    try:
        proc.wait(timeout=graceful_timeout)
    except subprocess.TimeoutExpired:
        proc.terminate()
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()


def _extract_last_wandb_run_url(text: str) -> str | None:
    matches = list(WANDB_RUN_URL_RE.finditer(text))
    if not matches:
        return None
    return matches[-1].group(0)


def _coerce_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _extract_metric_from_text(text: str, metric_key: str) -> float | None:
    pattern = re.compile(
        re.escape(metric_key) + r"[^0-9eE+\-]*(?P<value>[+\-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+\-]?\d+)?)"
    )
    matches = list(pattern.finditer(text))
    if not matches:
        return None
    try:
        return float(matches[-1].group("value"))
    except ValueError:
        return None


def _resolve_wandb_run_for_url(run_url: str) -> tuple[object | None, str | None]:
    match = WANDB_RUN_URL_RE.search(run_url)
    if match is None:
        return None, f"could not parse wandb run url: {run_url}"

    try:
        import wandb
    except ImportError:
        return None, "wandb package is not installed"

    entity = match.group("entity")
    project = match.group("project")
    run_ref = match.group("run_id")
    run_path = f"{entity}/{project}/{run_ref}"

    try:
        api = wandb.Api(timeout=30)
        run = api.run(run_path)
    except Exception as exc:  # pragma: no cover - external API behavior
        try:
            runs = api.runs(
                f"{entity}/{project}",
                order="-created_at",
                per_page=120,
            )
        except Exception:
            return None, f"wandb api fetch failed for {run_path}: {exc!r}"

        resolved_run = None
        for candidate in runs:
            candidate_id = str(getattr(candidate, "id", "") or "")
            candidate_name = str(getattr(candidate, "name", "") or "")
            candidate_display_name = str(getattr(candidate, "display_name", "") or "")
            if run_ref in {candidate_id, candidate_name, candidate_display_name}:
                resolved_run = candidate
                break
        if resolved_run is None:
            return None, f"wandb api fetch failed for {run_path}: {exc!r}"
        run = resolved_run

    return run, None


def _fetch_wandb_summary_for_run_url(
    run_url: str,
    *,
    metric_keys: Sequence[str],
) -> tuple[dict[str, float], str | None]:
    run, warning = _resolve_wandb_run_for_url(run_url)
    if run is None:
        return {}, warning

    summary = dict(getattr(run, "summary", {}))
    metrics: dict[str, float] = {}
    for key in metric_keys:
        numeric = _coerce_float(summary.get(key))
        if numeric is not None:
            metrics[key] = numeric
    return metrics, None


def _fetch_wandb_history_window_for_run_url(
    run_url: str,
    *,
    metric_keys: Sequence[str],
    step_start: int,
    step_end: int,
    aggregation: str,
    min_points: int,
) -> tuple[dict[str, float], dict[str, int], str | None]:
    run, warning = _resolve_wandb_run_for_url(run_url)
    if run is None:
        return {}, {}, warning

    history_keys = list(dict.fromkeys(["_step", *metric_keys]))
    try:
        history_rows = list(run.history(keys=history_keys, pandas=False))
    except Exception as exc:  # pragma: no cover - external API behavior
        return {}, {}, f"wandb history fetch failed for {run_url}: {exc!r}"

    aggregated: dict[str, float] = {}
    point_counts: dict[str, int] = {}
    insufficient: list[str] = []

    for key in metric_keys:
        values: list[float] = []
        for row in history_rows:
            if not isinstance(row, dict):
                continue
            step = _coerce_float(row.get("_step"))
            if step is None:
                continue
            if step < step_start or step > step_end:
                continue
            value = _coerce_float(row.get(key))
            if value is None or not math.isfinite(value):
                continue
            values.append(value)

        point_counts[key] = len(values)
        if len(values) < min_points:
            insufficient.append(f"{key}({len(values)}<{min_points})")
            continue
        if aggregation == "mean":
            aggregated[key] = sum(values) / len(values)
        else:
            aggregated[key] = statistics.median(values)

    if insufficient:
        warning = (
            "wandb history window had insufficient samples for: "
            + ", ".join(insufficient)
            + f" in steps [{step_start}, {step_end}]"
        )
    else:
        warning = None

    return aggregated, point_counts, warning


def _discover_wandb_run_for_prefix(
    *,
    entity: str,
    project: str,
    profile_prefix: str,
) -> tuple[str | None, str | None]:
    try:
        import wandb
    except ImportError:
        return None, "wandb package is not installed"

    try:
        api = wandb.Api(timeout=30)
        runs = api.runs(
            f"{entity}/{project}",
            order="-created_at",
            per_page=80,
        )
    except Exception as exc:  # pragma: no cover - external API behavior
        return None, f"wandb api listing failed for {entity}/{project}: {exc!r}"

    needle = f"{profile_prefix}_"
    for run in runs:
        run_id = str(getattr(run, "id", "") or "")
        if not run_id:
            continue

        candidates = [
            str(getattr(run, "name", "") or ""),
            str(getattr(run, "display_name", "") or ""),
            str(getattr(run, "path", "") or ""),
        ]
        if any(needle in value for value in candidates):
            return f"https://wandb.ai/{entity}/{project}/runs/{run_id}", None
    return None, f"no matching run found for prefix {profile_prefix!r} in {entity}/{project}"


def _collect_profile_metrics(
    args: argparse.Namespace,
    *,
    output_text: str,
    profile_prefix: str,
) -> dict[str, object]:
    metric_keys = [args.perf_metric, "throughput/mfu", "throughput/tokens_per_second", "throughput/duration"]
    metric_keys = [key for key in dict.fromkeys(metric_keys) if key]

    run_url = _extract_last_wandb_run_url(output_text)
    warnings: list[str] = []
    metrics: dict[str, float] = {}
    metric_source = "cli-output"
    metric_points: dict[str, int] = {}

    for key in metric_keys:
        parsed = _extract_metric_from_text(output_text, key)
        if parsed is not None:
            metrics[key] = parsed

    def _ingest_wandb_metrics(url: str) -> str | None:
        nonlocal metric_source, metric_points
        summary_metrics, summary_warning = _fetch_wandb_summary_for_run_url(
            url,
            metric_keys=metric_keys,
        )
        if summary_metrics:
            metrics.update(summary_metrics)
            metric_source = "wandb-summary"

        if args.perf_aggregation == "history-window":
            history_metrics, point_counts, history_warning = _fetch_wandb_history_window_for_run_url(
                url,
                metric_keys=metric_keys,
                step_start=args.perf_history_step_start,
                step_end=args.perf_history_step_end,
                aggregation=args.perf_history_aggregation,
                min_points=args.perf_history_min_points,
            )
            metric_points = point_counts
            if history_metrics:
                metrics.update(history_metrics)
                metric_source = (
                    f"wandb-history-{args.perf_history_aggregation}"
                    f"[{args.perf_history_step_start}:{args.perf_history_step_end}]"
                )
            elif summary_metrics:
                warnings.append(
                    "wandb history-window aggregation unavailable; "
                    "falling back to summary metrics for performance scoring."
                )
            if history_warning is not None:
                warnings.append(history_warning)

        return summary_warning

    summary_warning: str | None = None
    if run_url is not None and args.validation_profile_wandb_mode == "online":
        summary_warning = _ingest_wandb_metrics(run_url)

    needs_discovery = args.validation_profile_wandb_mode == "online" and (
        run_url is None or args.perf_metric not in metrics or summary_warning is not None
    )
    if needs_discovery:
        discovered_url, discover_warning = _discover_wandb_run_for_prefix(
            entity=args.perf_wandb_entity,
            project=args.perf_wandb_project,
            profile_prefix=profile_prefix,
        )
        if discovered_url is not None and discovered_url != run_url:
            run_url = discovered_url
            summary_warning = _ingest_wandb_metrics(run_url)
        elif discover_warning is not None:
            warnings.append(discover_warning)

    if summary_warning is not None and args.perf_metric not in metrics:
        warnings.append(summary_warning)

    return {
        "profile_prefix": profile_prefix,
        "run_url": run_url,
        "metrics": metrics,
        "metric_source": metric_source,
        "metric_points": metric_points,
        "warnings": warnings,
    }


def _load_perf_state(path: Path) -> dict[str, object]:
    if not path.exists():
        return {"version": 1, "champion": None, "history": []}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(
            f"[gdnctl] WARNING: could not parse perf state {path}: {exc!r}. Reinitializing.",
            file=sys.stderr,
        )
        return {"version": 1, "champion": None, "history": []}

    if not isinstance(data, dict):
        return {"version": 1, "champion": None, "history": []}
    if "history" not in data or not isinstance(data["history"], list):
        data["history"] = []
    if "champion" not in data:
        data["champion"] = None
    data["version"] = 1
    return data


def _save_perf_state(path: Path, state: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _percent_delta(candidate: float, baseline: float) -> float:
    if baseline == 0:
        if candidate == 0:
            return 0.0
        return float("inf") if candidate > 0 else float("-inf")
    return ((candidate - baseline) / abs(baseline)) * 100.0


def _revert_single_commit(workdir: Path, *, commit_sha: str) -> tuple[bool, str]:
    proc = _run_git(
        ["revert", "--no-edit", commit_sha],
        cwd=workdir,
        capture_output=True,
        check=False,
    )
    output = (proc.stdout or "") + (proc.stderr or "")
    if output.strip():
        sink = sys.stdout if proc.returncode == 0 else sys.stderr
        print(f"[gdnctl] {output.strip()}", file=sink)

    if proc.returncode == 0:
        return True, "revert commit created"

    _run_git(["revert", "--abort"], cwd=workdir, capture_output=True, check=False)
    return False, f"git revert failed for {commit_sha} (exit={proc.returncode})"


def _apply_performance_policy(
    args: argparse.Namespace,
    *,
    workdir: Path,
    perf_state_path: Path,
    iteration: int,
    commit_sha: str,
    validation_info: dict[str, object],
) -> tuple[bool, bool, int]:
    if args.perf_mode == "off":
        return True, False, 0

    metrics_obj = validation_info.get("metrics")
    metrics = metrics_obj if isinstance(metrics_obj, dict) else {}
    metric_value = _coerce_float(metrics.get(args.perf_metric))
    run_url = validation_info.get("run_url")
    metric_source_obj = validation_info.get("metric_source")
    metric_source = str(metric_source_obj) if isinstance(metric_source_obj, str) else "unknown"
    metric_points_obj = validation_info.get("metric_points")
    metric_points = metric_points_obj if isinstance(metric_points_obj, dict) else {}
    warnings_obj = validation_info.get("warnings")
    warnings = warnings_obj if isinstance(warnings_obj, list) else []

    for warning in warnings:
        print(f"[gdnctl] WARNING: {warning}", file=sys.stderr)

    if metric_value is None:
        print(
            f"[gdnctl] required performance metric is unavailable after validation profile: {args.perf_metric}",
            file=sys.stderr,
        )
        if args.perf_regression_policy.startswith("revert"):
            reverted, reason = _revert_single_commit(workdir, commit_sha=commit_sha)
            if not reverted:
                print(f"[gdnctl] {reason}", file=sys.stderr)
                return False, True, 1
            print(
                "[gdnctl] unscored iteration commit was reverted to avoid building on unknown-performance code.",
                file=sys.stderr,
            )
            if args.perf_regression_policy == "revert-continue":
                return True, False, 0
            return True, True, 0
        return False, True, 1

    state = _load_perf_state(perf_state_path)
    history_obj = state.get("history")
    history = history_obj if isinstance(history_obj, list) else []

    candidate_record: dict[str, object] = {
        "iteration": iteration,
        "commit": commit_sha,
        "metric": args.perf_metric,
        "metric_value": metric_value,
        "metric_source": metric_source,
        "metric_points": metric_points,
        "metrics": metrics,
        "run_url": run_url,
        "timestamp_unix": time.time(),
    }

    champion_obj = state.get("champion")
    champion = champion_obj if isinstance(champion_obj, dict) else None

    if champion is None:
        candidate_record["decision"] = "baseline"
        history.append(candidate_record)
        state["history"] = history
        state["champion"] = candidate_record
        _save_perf_state(perf_state_path, state)
        print(
            "[gdnctl] performance baseline initialized: "
            f"{args.perf_metric}={metric_value:.6g} @ {commit_sha[:12]} "
            f"(source={metric_source})"
        )
        return True, False, 0

    champion_value = _coerce_float(champion.get("metric_value"))
    if champion_value is None:
        candidate_record["decision"] = "promote_corrupt_champion"
        history.append(candidate_record)
        state["history"] = history
        state["champion"] = candidate_record
        _save_perf_state(perf_state_path, state)
        print("[gdnctl] champion record was invalid; replaced with current candidate.")
        return True, False, 0

    delta_pct = _percent_delta(metric_value, champion_value)
    candidate_record["delta_vs_champion_pct"] = delta_pct

    if delta_pct >= args.perf_min_improvement_pct:
        candidate_record["decision"] = "promoted"
        history.append(candidate_record)
        state["history"] = history
        state["champion"] = candidate_record
        _save_perf_state(perf_state_path, state)
        print(
            "[gdnctl] performance champion promoted: "
            f"{champion_value:.6g} -> {metric_value:.6g} ({delta_pct:+.2f}%, source={metric_source})"
        )
        return True, False, 0

    if delta_pct <= -args.perf_max_regression_pct:
        candidate_record["decision"] = "regression"
        history.append(candidate_record)
        state["history"] = history
        _save_perf_state(perf_state_path, state)

        print(
            "[gdnctl] performance regression detected vs champion: "
            f"{champion_value:.6g} -> {metric_value:.6g} ({delta_pct:+.2f}%, source={metric_source})",
            file=sys.stderr,
        )

        if args.perf_regression_policy.startswith("revert"):
            reverted, reason = _revert_single_commit(workdir, commit_sha=commit_sha)
            if not reverted:
                print(f"[gdnctl] {reason}", file=sys.stderr)
                return False, True, 1
            print(
                "[gdnctl] regression commit reverted successfully; champion remains "
                f"{str(champion.get('commit', 'unknown'))[:12]}."
            )
            if args.perf_regression_policy == "revert-continue":
                return True, False, 0
            return True, True, 0

        if args.perf_regression_policy == "continue":
            return True, False, 0
        if args.perf_regression_policy == "count-failure":
            return True, True, 0
        return False, True, 1

    candidate_record["decision"] = "within_threshold"
    history.append(candidate_record)
    state["history"] = history
    _save_perf_state(perf_state_path, state)
    print(
        "[gdnctl] performance within hold band vs champion: "
        f"{champion_value:.6g} -> {metric_value:.6g} ({delta_pct:+.2f}%, source={metric_source})"
    )
    return True, False, 0


def _stream_subprocess_output_to_file(
    proc: subprocess.Popen[str],
    *,
    output_path: Path,
    ready_markers: Sequence[str],
    ready_event: threading.Event,
    suppress_output_after_ready: bool = False,
) -> threading.Thread:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    def _worker() -> None:
        ready_seen = False
        with output_path.open("a", encoding="utf-8") as fout:
            stdout = proc.stdout
            if stdout is None:
                return
            for line in stdout:
                matched_ready = any(marker in line for marker in ready_markers)
                if matched_ready and not ready_seen:
                    ready_seen = True
                    ready_event.set()
                    if suppress_output_after_ready:
                        fout.write(line)
                        fout.write(
                            "[gdnctl] dev TPU became active; suppressing further allocation log output "
                            "while keeping the hold process drained.\n"
                        )
                        fout.flush()
                        continue

                if suppress_output_after_ready and ready_seen:
                    continue

                fout.write(line)
                fout.flush()

    thread = threading.Thread(target=_worker, daemon=True)
    thread.start()
    return thread


def _dev_tpu_cluster_candidates(args: argparse.Namespace) -> list[str]:
    cluster_candidates = [args.dev_tpu_cluster, *args.dev_tpu_fallback_cluster]
    deduped_clusters: list[str] = []
    seen_clusters: set[str] = set()
    for cluster in cluster_candidates:
        if cluster in seen_clusters:
            continue
        seen_clusters.add(cluster)
        deduped_clusters.append(cluster)
    return deduped_clusters


def _dev_tpu_allocation_log_path(args: argparse.Namespace) -> Path:
    log_path = getattr(args, "_managed_dev_tpu_allocation_log", None)
    if log_path is None:
        log_path = args.dev_tpu_allocate_log
    if log_path is None:
        log_path = REPO_ROOT / ".agents/logs/gdn_codex_loop/dev_tpu_allocate.log"
    return Path(log_path).resolve()


def _stop_managed_dev_tpu_process(args: argparse.Namespace, *, graceful_timeout: float) -> None:
    proc = getattr(args, "_managed_dev_tpu_proc", None)
    pump_thread = getattr(args, "_managed_dev_tpu_pump_thread", None)
    if isinstance(proc, subprocess.Popen):
        _terminate_process(proc, graceful_timeout=graceful_timeout)
    if isinstance(pump_thread, threading.Thread):
        pump_thread.join(timeout=2.0)
    if hasattr(args, "_managed_dev_tpu_proc"):
        delattr(args, "_managed_dev_tpu_proc")
    if hasattr(args, "_managed_dev_tpu_pump_thread"):
        delattr(args, "_managed_dev_tpu_pump_thread")


def _try_reacquire_managed_dev_tpu(args: argparse.Namespace, *, reason: str) -> bool:
    if not args.hold_dev_tpu or not args.dev_tpu_name:
        return False

    now = time.time()
    last_attempt = float(getattr(args, "_managed_dev_tpu_reacquire_last_attempt", 0.0))
    min_retry_interval = max(30.0, float(getattr(args, "validation_retry_sleep", 0.0)))
    if now - last_attempt < min_retry_interval:
        remaining = min_retry_interval - (now - last_attempt)
        print(
            f"[gdnctl] managed dev TPU re-acquire is rate-limited; skipping for {remaining:.1f}s (reason={reason}).",
            file=sys.stderr,
        )
        return False
    args._managed_dev_tpu_reacquire_last_attempt = now

    print(f"[gdnctl] attempting managed dev TPU re-acquire ({reason})", file=sys.stderr)
    _stop_managed_dev_tpu_process(args, graceful_timeout=min(args.dev_tpu_stop_timeout, 15.0))
    if hasattr(args, "active_dev_tpu_cluster"):
        delattr(args, "active_dev_tpu_cluster")

    allocation_log = _dev_tpu_allocation_log_path(args)
    allocation_log.parent.mkdir(parents=True, exist_ok=True)
    if allocation_log.exists():
        try:
            allocation_log.unlink()
        except OSError:
            pass

    clusters = _dev_tpu_cluster_candidates(args)
    last_error: str | None = None
    attempts = max(1, int(args.dev_tpu_allocate_attempts))
    for attempt in range(1, attempts + 1):
        for cluster in clusters:
            allocate_cmd = _build_dev_tpu_allocate_cmd(args, cluster=cluster)
            print(
                f"[gdnctl] managed dev TPU re-acquire attempt {attempt}/{attempts} on cluster {cluster}",
                file=sys.stderr,
            )
            _echo_cmd(allocate_cmd)
            proc = subprocess.Popen(
                allocate_cmd,
                cwd=str(REPO_ROOT),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            ready_event = threading.Event()
            pump_thread = _stream_subprocess_output_to_file(
                proc,
                output_path=allocation_log,
                ready_markers=DEV_TPU_READY_MARKERS,
                ready_event=ready_event,
                suppress_output_after_ready=True,
            )

            deadline = time.time() + args.dev_tpu_ready_timeout
            timed_out = False
            while time.time() < deadline:
                if ready_event.is_set():
                    args.active_dev_tpu_cluster = cluster
                    args._managed_dev_tpu_proc = proc
                    args._managed_dev_tpu_pump_thread = pump_thread
                    args._managed_dev_tpu_allocation_log = str(allocation_log)
                    print(
                        f"[gdnctl] managed dev TPU re-acquire is active: dev-tpu-{args.dev_tpu_name} (cluster {cluster})"
                    )
                    return True

                rc = proc.poll()
                if rc is not None:
                    tail = _tail_text(allocation_log)
                    last_error = (
                        "[gdnctl] managed dev TPU re-acquire exited before becoming active "
                        f"(exit={rc}, cluster={cluster}, attempt={attempt}).\n"
                        f"See log: {allocation_log}\n"
                        f"{tail}"
                    )
                    break
                time.sleep(2.0)
            else:
                timed_out = True
                tail = _tail_text(allocation_log)
                last_error = (
                    "[gdnctl] timed out waiting for managed dev TPU re-acquire to become active "
                    f"(cluster={cluster}, attempt={attempt}).\n"
                    f"See log: {allocation_log}\n"
                    f"{tail}"
                )

            if timed_out:
                _terminate_process(proc, graceful_timeout=min(args.dev_tpu_stop_timeout, 15.0))
            else:
                _terminate_process(proc, graceful_timeout=5.0)
            pump_thread.join(timeout=2.0)
            if last_error:
                print(last_error, file=sys.stderr)

        if attempt < attempts and args.dev_tpu_allocate_retry_sleep > 0:
            print(
                f"[gdnctl] retrying managed dev TPU re-acquire in {args.dev_tpu_allocate_retry_sleep:.1f}s",
                file=sys.stderr,
            )
            time.sleep(args.dev_tpu_allocate_retry_sleep)

    print(
        "[gdnctl] managed dev TPU re-acquire failed; continuing with Ray fallback.\n" + (last_error or ""),
        file=sys.stderr,
    )
    return False


@contextmanager
def _hold_dev_tpu_for_loop(
    args: argparse.Namespace,
    *,
    log_dir: Path,
) -> Generator[None, None, None]:
    if not args.hold_dev_tpu:
        yield
        return

    if not args.dev_tpu_name:
        raise SystemExit("[gdnctl] --hold-dev-tpu requires --dev-tpu-name.")

    deduped_clusters = _dev_tpu_cluster_candidates(args)

    allocation_log = Path(args.dev_tpu_allocate_log or (log_dir / "dev_tpu_allocate.log")).resolve()
    args._managed_dev_tpu_allocation_log = str(allocation_log)
    allocation_log.parent.mkdir(parents=True, exist_ok=True)
    if allocation_log.exists():
        allocation_log.unlink()

    print("[gdnctl] starting managed dev TPU allocation for codex-loop")
    print(f"[gdnctl] dev TPU allocation log: {allocation_log}")

    last_error: str | None = None
    for attempt in range(1, args.dev_tpu_allocate_attempts + 1):
        for cluster in deduped_clusters:
            allocate_cmd = _build_dev_tpu_allocate_cmd(args, cluster=cluster)
            print(
                "[gdnctl] managed dev TPU allocation attempt "
                f"{attempt}/{args.dev_tpu_allocate_attempts} on cluster {cluster}"
            )
            _echo_cmd(allocate_cmd)
            proc = subprocess.Popen(
                allocate_cmd,
                cwd=str(REPO_ROOT),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )

            ready_event = threading.Event()
            pump_thread = _stream_subprocess_output_to_file(
                proc,
                output_path=allocation_log,
                ready_markers=DEV_TPU_READY_MARKERS,
                ready_event=ready_event,
                suppress_output_after_ready=True,
            )

            deadline = time.time() + args.dev_tpu_ready_timeout
            timed_out = False
            while time.time() < deadline:
                if ready_event.is_set():
                    args.active_dev_tpu_cluster = cluster
                    args._managed_dev_tpu_proc = proc
                    args._managed_dev_tpu_pump_thread = pump_thread
                    print(f"[gdnctl] dev TPU allocation is active: dev-tpu-{args.dev_tpu_name} (cluster {cluster})")
                    try:
                        yield
                    finally:
                        release_cluster = getattr(args, "active_dev_tpu_cluster", cluster)
                        print(
                            "[gdnctl] releasing managed dev TPU allocation: "
                            f"dev-tpu-{args.dev_tpu_name} (cluster {release_cluster})"
                        )
                        _stop_managed_dev_tpu_process(args, graceful_timeout=args.dev_tpu_stop_timeout)
                        if hasattr(args, "active_dev_tpu_cluster"):
                            delattr(args, "active_dev_tpu_cluster")
                        if hasattr(args, "_managed_dev_tpu_allocation_log"):
                            delattr(args, "_managed_dev_tpu_allocation_log")
                    return

                rc = proc.poll()
                if rc is not None:
                    tail = _tail_text(allocation_log)
                    last_error = (
                        "[gdnctl] managed dev TPU allocation exited before becoming active "
                        f"(exit={rc}, cluster={cluster}, attempt={attempt}).\n"
                        f"See log: {allocation_log}\n"
                        f"{tail}"
                    )
                    break

                time.sleep(2.0)
            else:
                timed_out = True
                tail = _tail_text(allocation_log)
                last_error = (
                    "[gdnctl] timed out waiting for managed dev TPU allocation to become active "
                    f"(cluster={cluster}, attempt={attempt}).\n"
                    f"See log: {allocation_log}\n"
                    f"{tail}"
                )

            if timed_out:
                _terminate_process(proc, graceful_timeout=min(args.dev_tpu_stop_timeout, 15.0))
            else:
                _terminate_process(proc, graceful_timeout=5.0)

            pump_thread.join(timeout=2.0)
            if last_error:
                print(last_error, file=sys.stderr)

        if attempt < args.dev_tpu_allocate_attempts and args.dev_tpu_allocate_retry_sleep > 0:
            print(
                f"[gdnctl] retrying managed dev TPU allocation in {args.dev_tpu_allocate_retry_sleep:.1f}s",
                file=sys.stderr,
            )
            time.sleep(args.dev_tpu_allocate_retry_sleep)

    error_message = last_error or "[gdnctl] managed dev TPU allocation failed without diagnostic output."
    if args.hold_dev_tpu_policy == "best-effort":
        print(
            f"[gdnctl] WARNING: managed dev TPU allocation failed; continuing without held dev TPU.\n{error_message}",
            file=sys.stderr,
        )
        yield
        return
    raise SystemExit(error_message)


def _git_head(cwd: Path) -> str:
    proc = _run_git(["rev-parse", "HEAD"], cwd=cwd, capture_output=True, check=True)
    return proc.stdout.strip()


def _sync_branch_from_remote(
    *,
    workdir: Path,
    remote: str,
    branch: str,
    conflict_policy: str = "fail",
) -> bool:
    target_ref = f"{remote}/{branch}"
    print(f"[gdnctl] syncing branch with {target_ref}")

    fetch_cmd = ["git", "fetch", remote, branch]
    _echo_cmd(fetch_cmd)
    _run_git(["fetch", remote, branch], cwd=workdir, capture_output=True, check=True)

    merge_cmd = ["git", "merge", "--no-edit", target_ref]
    _echo_cmd(merge_cmd)
    merge_proc = _run_git(
        ["merge", "--no-edit", target_ref],
        cwd=workdir,
        capture_output=True,
        check=False,
    )
    merge_output = ((merge_proc.stdout or "") + (merge_proc.stderr or "")).strip()
    if merge_output:
        sink = sys.stdout if merge_proc.returncode == 0 else sys.stderr
        print(f"[gdnctl] {merge_output}", file=sink)

    if merge_proc.returncode == 0:
        return True

    _run_git(["merge", "--abort"], cwd=workdir, capture_output=True, check=False)
    conflict_detected = "CONFLICT" in merge_output or "Automatic merge failed" in merge_output
    if conflict_detected and conflict_policy == "skip":
        print(
            "[gdnctl] WARNING: sync-main merge conflict detected; "
            "continuing without syncing from main this iteration due policy=skip.",
            file=sys.stderr,
        )
        return False
    raise RuntimeError(
        "[gdnctl] failed to merge "
        f"{target_ref} into current branch (exit={merge_proc.returncode}). "
        "Resolve conflicts and rerun."
    )


def _git_dirty(cwd: Path) -> bool:
    proc = _run_git(["status", "--porcelain"], cwd=cwd, capture_output=True, check=True)
    return bool(proc.stdout.strip())


def _stash_dirty_tree(cwd: Path, *, iteration: int) -> tuple[str, str]:
    stash_message = f"gdnctl-codex-loop-iter-{iteration:03d}-{int(time.time())}"
    proc = _run_git(
        ["stash", "push", "-u", "-m", stash_message],
        cwd=cwd,
        capture_output=True,
        check=True,
    )
    output = (proc.stdout + proc.stderr).strip()
    print(f"[gdnctl] stashed dirty tree: {stash_message}")
    if output:
        print(f"[gdnctl] {output}")
    ref_proc = _run_git(["stash", "list", "--format=%gd", "-n", "1"], cwd=cwd, capture_output=True, check=True)
    stash_ref = ref_proc.stdout.strip()
    if not stash_ref:
        raise RuntimeError("[gdnctl] Failed to resolve stash ref after stashing dirty tree.")
    return stash_ref, stash_message


def _stash_apply_would_conflict(cwd: Path, *, stash_ref: str) -> tuple[bool, str]:
    """Preflight stash apply in a temporary worktree.

    This avoids leaving the main working tree in an unmerged state if the stash
    would conflict with iteration-generated commits.
    """

    head = _git_head(cwd)
    with tempfile.TemporaryDirectory(prefix="gdnctl-stash-preflight-") as temp_root:
        probe_worktree = Path(temp_root) / "probe"
        add_proc = _run_git(
            ["worktree", "add", "--detach", str(probe_worktree), head],
            cwd=cwd,
            capture_output=True,
            check=False,
        )
        if add_proc.returncode != 0:
            output = (add_proc.stdout + add_proc.stderr).strip()
            # Fail open if preflight infra itself fails; we'll attempt a normal restore.
            return False, output
        try:
            apply_proc = _run_git(
                ["stash", "apply", "--quiet", stash_ref],
                cwd=probe_worktree,
                capture_output=True,
                check=False,
            )
            output = (apply_proc.stdout + apply_proc.stderr).strip()
            return apply_proc.returncode != 0, output
        finally:
            _run_git(
                ["worktree", "remove", "--force", str(probe_worktree)],
                cwd=cwd,
                capture_output=True,
                check=False,
            )


def _restore_stash_tree(cwd: Path, *, stash_ref: str, stash_message: str) -> bool:
    print(f"[gdnctl] restoring stashed dirty tree: {stash_message} ({stash_ref})")
    would_conflict, preflight_output = _stash_apply_would_conflict(cwd, stash_ref=stash_ref)
    if would_conflict:
        if preflight_output:
            print(f"[gdnctl] stash preflight: {preflight_output}", file=sys.stderr)
        print(
            "[gdnctl] stash restore skipped because preflight detected conflicts; "
            f"stash kept for manual recovery ({stash_ref}, message={stash_message}).",
            file=sys.stderr,
        )
        return False

    if preflight_output:
        # Informative only: preflight setup can emit warnings.
        print(f"[gdnctl] stash preflight: {preflight_output}", file=sys.stderr)

    proc = _run_git(["stash", "apply", stash_ref], cwd=cwd, capture_output=True, check=False)
    output = (proc.stdout + proc.stderr).strip()
    if output:
        sink = sys.stdout if proc.returncode == 0 else sys.stderr
        print(f"[gdnctl] {output}", file=sink)

    if proc.returncode != 0:
        print(
            f"[gdnctl] stash restore failed; stash was kept for manual recovery ({stash_ref}, message={stash_message}).",
            file=sys.stderr,
        )
        return False

    drop_proc = _run_git(["stash", "drop", stash_ref], cwd=cwd, capture_output=True, check=False)
    drop_output = (drop_proc.stdout + drop_proc.stderr).strip()
    if drop_output:
        sink = sys.stdout if drop_proc.returncode == 0 else sys.stderr
        print(f"[gdnctl] {drop_output}", file=sink)
    if drop_proc.returncode != 0:
        print(
            f"[gdnctl] WARNING: stash was applied but could not be dropped automatically ({stash_ref}).",
            file=sys.stderr,
        )
        return False

    return True


def _has_unmerged_paths(cwd: Path) -> bool:
    proc = _run_git(["diff", "--name-only", "--diff-filter=U"], cwd=cwd, capture_output=True, check=True)
    return bool(proc.stdout.strip())


def _recover_from_unmerged_paths(cwd: Path) -> None:
    if not _has_unmerged_paths(cwd):
        return
    print(
        "[gdnctl] WARNING: unmerged paths detected after stash restore attempt; "
        "resetting index/worktree to HEAD to keep loop running (stashes are preserved).",
        file=sys.stderr,
    )
    _run_git(["reset", "--merge"], cwd=cwd, capture_output=True, check=False)


@contextmanager
def _clean_worktree_for_iteration(
    args: argparse.Namespace,
    *,
    workdir: Path,
    iteration: int,
) -> Generator[None, None, None]:
    stash_entry: tuple[str, str] | None = None
    if args.require_clean and _git_dirty(workdir):
        if args.dirty_policy == "stash":
            stash_entry = _stash_dirty_tree(workdir, iteration=iteration)
        else:
            raise RuntimeError(
                "[gdnctl] Working tree is dirty before starting the iteration. "
                "Commit/stash or rerun with --allow-dirty or --dirty-policy stash."
            )
    try:
        yield
    finally:
        if stash_entry is not None:
            stash_ref, stash_message = stash_entry
            restored = _restore_stash_tree(workdir, stash_ref=stash_ref, stash_message=stash_message)
            if not restored:
                _recover_from_unmerged_paths(workdir)
                if _has_unmerged_paths(workdir):
                    raise RuntimeError(
                        "[gdnctl] Unable to recover from unmerged paths after stash restore attempt. "
                        "Resolve conflicts manually before continuing."
                    )
            if not restored and args.stash_restore_policy == "fail":
                raise RuntimeError(
                    "[gdnctl] Failed to restore stashed dirty tree. Resolve stash conflicts manually before continuing."
                )


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


def _read_directive_file(path: Path, *, source: str) -> str:
    if not path.is_absolute():
        path = (REPO_ROOT / path).resolve()
    if not path.exists():
        raise SystemExit(f"[gdnctl] directive file not found for {source}: {path}")
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        raise SystemExit(f"[gdnctl] directive file is empty for {source}: {path}")
    return text


def _load_session_directives(args: argparse.Namespace) -> list[str]:
    directives: list[str] = []

    for preset in args.directive_preset:
        preset_path = SESSION_DIRECTIVE_PRESET_FILES[preset]
        directives.append(_read_directive_file(preset_path, source=f"preset {preset}"))

    for directive_file in args.directive_file:
        directives.append(_read_directive_file(Path(directive_file), source=directive_file))

    for directive in args.directive:
        stripped = directive.strip()
        if stripped:
            directives.append(stripped)

    return directives


def _managed_dev_tpu_session_directive(args: argparse.Namespace) -> str:
    active_cluster = getattr(args, "active_dev_tpu_cluster", args.dev_tpu_cluster)
    validation_requirement = "both validation tests and a profile result"
    if args.validation_mode == "profile-only":
        validation_requirement = "a profile result"
    elif args.validation_mode == "off":
        validation_requirement = "post-check/profile evidence configured for this run"
    return (
        "Managed dev TPU mode is active for this run.\n\n"
        f"- Prefer dev TPU commands for validation/profiling on `--cluster {active_cluster}` "
        f"and `--tpu-name {args.dev_tpu_name}`; if dev TPU fails/unavailable, fall back to Ray TPU commands.\n"
        "- Preferred first:\n"
        f"  - `uv run python scripts/gdn/gdnctl.py dev-tpu-test --cluster {active_cluster} "
        f"--tpu-name {args.dev_tpu_name} --tests both`\n"
        f"  - `uv run python scripts/gdn/gdnctl.py dev-tpu-profile --cluster {active_cluster} "
        f"--tpu-name {args.dev_tpu_name} --tpu {args.dev_tpu_type or 'v5p-8'} ...`\n"
        "- Fallback if dev TPU path fails:\n"
        "  - `uv run python scripts/gdn/gdnctl.py ray-test --cluster <cluster> --tpu auto --tests both`\n"
        "  - `uv run python scripts/gdn/gdnctl.py ray-profile --cluster <cluster> --tpu v5p-8 ...`\n"
        f"- Do not finish an iteration without {validation_requirement}."
    )


def _validation_gate_session_directive(args: argparse.Namespace) -> str:
    if args.validation_mode == "profile-only":
        return (
            "Do not end an iteration without TPU profiling evidence.\n\n"
            "- Validation mode: profile-only (tests are intentionally skipped for this run).\n"
            "- Required profile: one completed profile run per iteration.\n"
            "- If dev TPU path is unavailable/fails, fall back to ray-profile and wait for completion."
        )

    return (
        "Do not end an iteration without TPU validation + profiling evidence.\n\n"
        f"- Required tests target: `{args.validation_tests}`\n"
        "- Required profile: one completed profile run per iteration.\n"
        "- If dev TPU path is unavailable/fails, fall back to ray-test/ray-profile and wait for completion."
    )


def _performance_policy_session_directive(args: argparse.Namespace, *, perf_state_path: Path) -> str:
    window_line = ""
    if args.perf_aggregation == "history-window":
        window_line = (
            "\n"
            f"- Metric aggregation: `history-window` ({args.perf_history_aggregation}, "
            f"steps {args.perf_history_step_start}-{args.perf_history_step_end}, "
            f"min points {args.perf_history_min_points})."
        )
    else:
        window_line = "\n- Metric aggregation: `summary` (final summary value)."

    return (
        "Performance governance is active for this codex-loop run.\n\n"
        f"- Primary metric: `{args.perf_metric}`\n"
        f"- Promote champion only when improvement >= {args.perf_min_improvement_pct:.3f}%.\n"
        f"- Regression threshold: {args.perf_max_regression_pct:.3f}% below champion.\n"
        f"- Regression policy: `{args.perf_regression_policy}`\n"
        f"- State file: `{perf_state_path}`"
        f"{window_line}"
    )


def _warn_if_hold_dev_tpu_with_ray_post_checks(args: argparse.Namespace) -> None:
    if not args.hold_dev_tpu:
        return
    active_cluster = getattr(args, "active_dev_tpu_cluster", args.dev_tpu_cluster)
    for command in args.post_check:
        if "dev-tpu-test" in command or "dev-tpu-profile" in command:
            if f"--cluster {active_cluster}" not in command or f"--tpu-name {args.dev_tpu_name}" not in command:
                print(
                    "[gdnctl] WARNING: --hold-dev-tpu is active but --post-check does not target "
                    f"--cluster {active_cluster} --tpu-name {args.dev_tpu_name}.",
                    file=sys.stderr,
                )


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
    lines.append(
        f"uv pip install --python .venv/bin/python --index-url {shlex.quote(TORCH_CPU_INDEX)} --force-reinstall torch"
    )
    lines.append("(uv pip uninstall --python .venv/bin/python torchvision || true)")
    lines.append(
        f".venv/bin/python -m experiments.speedrun.hackable_transformer_gdn.tiny_profile {shlex.join(profile_args)}"
    )
    return lines


def _parse_profile_env(profile_env: Sequence[str]) -> list[tuple[str, str]]:
    parsed: list[tuple[str, str]] = []
    key_pattern = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
    for raw in profile_env:
        item = raw.strip()
        if not item:
            continue
        if "=" not in item:
            raise SystemExit(f"[gdnctl] invalid --profile-env entry (expected KEY=VALUE): {raw!r}")
        key, value = item.split("=", 1)
        key = key.strip()
        if not key_pattern.match(key):
            raise SystemExit(f"[gdnctl] invalid --profile-env key {key!r}; expected [A-Za-z_][A-Za-z0-9_]*")
        parsed.append((key, value))
    return parsed


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
    extra_env = _parse_profile_env(getattr(args, "profile_env", []))
    for key, value in extra_env:
        cmd += ["-e", f"{key}={value}"]


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


def _run_command_with_output(cmd: Sequence[str], *, cwd: Path | None = None) -> tuple[int, str]:
    proc = _run(cmd, cwd=cwd, capture_output=True, check=False)
    if proc.stdout:
        sys.stdout.write(proc.stdout)
    if proc.stderr:
        sys.stderr.write(proc.stderr)
    return proc.returncode, (proc.stdout or "") + (proc.stderr or "")


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


def _build_codex_exec_cmd(
    *,
    codex_bin: str,
    workdir: Path,
    message_path: Path,
    args: argparse.Namespace,
    search_supported: bool,
) -> list[str]:
    cmd = [
        codex_bin,
        "exec",
        "-C",
        str(workdir),
        "--dangerously-bypass-approvals-and-sandbox",
        "-o",
        str(message_path),
    ]

    if getattr(args, "codex_ephemeral", True):
        cmd.append("--ephemeral")
    if args.model:
        cmd += ["-m", args.model]
    if args.reasoning_effort:
        cmd += ["-c", f"model_reasoning_effort={json.dumps(args.reasoning_effort)}"]
    if args.codex_profile:
        cmd += ["-p", args.codex_profile]
    if args.search and search_supported:
        cmd.append("--search")

    cmd.append("-")
    return cmd


def _run_post_checks(cwd: Path, commands: Sequence[str]) -> None:
    for command in commands:
        _run(["bash", "-lc", command], cwd=cwd)


def _run_post_checks_with_retries(
    cwd: Path,
    commands: Sequence[str],
    *,
    retries: int,
    retry_sleep_seconds: float,
) -> tuple[bool, int]:
    for command in commands:
        for attempt in range(1, retries + 2):
            try:
                _run(["bash", "-lc", command], cwd=cwd)
                break
            except subprocess.CalledProcessError as exc:
                if attempt > retries:
                    return False, exc.returncode or 1
                print(
                    f"[gdnctl] post-check failed (attempt {attempt}/{retries + 1}) for command: {command}",
                    file=sys.stderr,
                )
                if retry_sleep_seconds > 0:
                    print(
                        f"[gdnctl] retrying post-check in {retry_sleep_seconds:.1f}s",
                        file=sys.stderr,
                    )
                    time.sleep(retry_sleep_seconds)
    return True, 0


def _dedupe_clusters(clusters: Sequence[str]) -> list[str]:
    deduped: list[str] = []
    seen: set[str] = set()
    for cluster in clusters:
        if not cluster:
            continue
        if cluster in seen:
            continue
        seen.add(cluster)
        deduped.append(cluster)
    return deduped


def _cluster_supports_tpu_slice(config_path: Path, *, slice_key: str | None) -> bool:
    try:
        text = config_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return False
    if "tpu_slice_" not in text:
        return False
    if slice_key is None:
        return True
    return slice_key in text


def _cluster_supports_any_tpu_family(config_path: Path, *, families: Sequence[str]) -> bool:
    try:
        text = config_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return False
    if "tpu_slice_" not in text:
        return False

    for family in families:
        token = family.strip().lower()
        if not token:
            continue
        if f"tpu_slice_{token}_" in text:
            return True
    return False


def _normalize_tpu_slice_key(tpu_type: str | None) -> str | None:
    if not tpu_type:
        return None
    normalized = tpu_type.strip().lower()
    if not normalized or normalized == "auto":
        return None
    normalized = re.sub(r"[^a-z0-9]+", "_", normalized).strip("_")
    if not normalized:
        return None
    return f"tpu_slice_{normalized}"


def _discover_validation_ray_clusters(
    args: argparse.Namespace,
    *,
    required_tpu_type: str | None,
    modern_only: bool,
) -> list[str]:
    excludes = [token.lower() for token in args.validation_ray_cluster_exclude]
    slice_key = _normalize_tpu_slice_key(required_tpu_type)

    discovered: list[str] = []
    for config_path in sorted((REPO_ROOT / "infra").glob("marin-*.yaml")):
        stem = config_path.stem
        if not stem.startswith("marin-"):
            continue
        cluster = stem[len("marin-") :]
        cluster_lower = cluster.lower()
        if any(token in cluster_lower for token in excludes):
            continue
        if modern_only:
            if not _cluster_supports_any_tpu_family(config_path, families=DEFAULT_MODERN_TPU_FAMILIES):
                continue
        else:
            if not _cluster_supports_tpu_slice(config_path, slice_key=slice_key):
                continue
        discovered.append(cluster)

    return discovered


def _validation_bad_ray_clusters(args: argparse.Namespace) -> set[str]:
    bad = getattr(args, "_validation_bad_ray_clusters", None)
    if bad is None:
        bad = set()
        args._validation_bad_ray_clusters = bad
    return bad


def _validation_ray_clusters(args: argparse.Namespace, *, purpose: str) -> list[str]:
    if purpose not in {"test", "profile"}:
        raise ValueError(f"Invalid validation cluster purpose: {purpose!r}")

    if args.validation_ray_cluster:
        clusters = _dedupe_clusters(args.validation_ray_cluster)
    elif args.validation_ray_cluster_auto:
        active_cluster = getattr(args, "active_dev_tpu_cluster", None)
        if purpose == "profile":
            discovered = _discover_validation_ray_clusters(
                args,
                required_tpu_type=args.validation_profile_tpu,
                modern_only=False,
            )
            report_attr = "_validation_auto_cluster_reported_profile"
            report_label = "profile"
        else:
            allow_cross = bool(args.validation_cross_tpu_test_fallback)
            discovered = _discover_validation_ray_clusters(
                args,
                required_tpu_type=None if allow_cross else args.validation_profile_tpu,
                modern_only=allow_cross,
            )
            report_attr = "_validation_auto_cluster_reported_test"
            if allow_cross:
                report_label = "test-modern"
            else:
                report_label = "test"

        if not getattr(args, report_attr, False):
            print(
                f"[gdnctl] auto-discovered validation Ray clusters ({report_label}): "
                f"{', '.join(discovered) if discovered else '(none)'}"
            )
            setattr(args, report_attr, True)
        clusters = _dedupe_clusters(
            [
                active_cluster,
                args.dev_tpu_cluster,
                *args.dev_tpu_fallback_cluster,
                *discovered,
            ]
        )
    else:
        active_cluster = getattr(args, "active_dev_tpu_cluster", None)
        candidates = [active_cluster, args.dev_tpu_cluster, *args.dev_tpu_fallback_cluster]
        clusters = _dedupe_clusters(candidates)

    bad_clusters = _validation_bad_ray_clusters(args)
    return [cluster for cluster in clusters if cluster not in bad_clusters]


def _run_validation_ray_test_once(args: argparse.Namespace, *, cluster: str) -> tuple[int, bool]:
    remote_cmd = _build_remote_test_command(args.validation_tests, args.validation_pytest_args)
    cmd = [
        *RAY_RUN,
        "--cluster",
        cluster,
        "--tpu",
        args.validation_ray_tpu,
        "-e",
        "EQX_ON_ERROR=nan",
        "-e",
        "WANDB_MODE=offline",
        "--",
        "bash",
        "-lc",
        remote_cmd,
    ]
    proc = _run(cmd, capture_output=True, check=False)
    if proc.stdout:
        sys.stdout.write(proc.stdout)
    if proc.stderr:
        sys.stderr.write(proc.stderr)
    combined = (proc.stdout or "") + (proc.stderr or "")
    if proc.returncode != 0 and "No cluster config found for region" in combined:
        _validation_bad_ray_clusters(args).add(cluster)
        print(
            f"[gdnctl] validation Ray cluster is invalid and will be skipped in future retries: {cluster}",
            file=sys.stderr,
        )
        return proc.returncode, False
    return proc.returncode, True


def _run_validation_tests_once(args: argparse.Namespace) -> tuple[int, bool]:
    active_cluster = getattr(args, "active_dev_tpu_cluster", None)
    if args.dev_tpu_name and active_cluster is not None:
        dev_test_args = argparse.Namespace(
            cluster=active_cluster,
            tpu_name=args.dev_tpu_name,
            tests=args.validation_tests,
            pytest_args=args.validation_pytest_args,
            no_sync=args.validation_dev_no_sync,
        )
        rc = cmd_dev_tpu_test(dev_test_args)
        if rc == 0:
            return 0, True
        if _try_reacquire_managed_dev_tpu(args, reason="validation tests failed on held dev TPU"):
            retry_cluster = getattr(args, "active_dev_tpu_cluster", None)
            if retry_cluster is not None:
                retry_args = argparse.Namespace(
                    cluster=retry_cluster,
                    tpu_name=args.dev_tpu_name,
                    tests=args.validation_tests,
                    pytest_args=args.validation_pytest_args,
                    no_sync=args.validation_dev_no_sync,
                )
                rc = cmd_dev_tpu_test(retry_args)
                if rc == 0:
                    return 0, True
        print(
            "[gdnctl] validation tests failed on held dev TPU; trying Ray fallback.",
            file=sys.stderr,
        )

    clusters = _validation_ray_clusters(args, purpose="test")
    if not clusters:
        print(
            "[gdnctl] no valid Ray clusters remain for validation tests.",
            file=sys.stderr,
        )
        return 1, False

    last_rc = 1
    retryable = False
    for cluster in clusters:
        rc, cluster_retryable = _run_validation_ray_test_once(args, cluster=cluster)
        if rc == 0:
            return 0, True
        last_rc = rc
        retryable = retryable or cluster_retryable
    return last_rc, retryable


def _run_validation_ray_profile_once(
    args: argparse.Namespace,
    *,
    cluster: str,
    run_name_prefix: str,
) -> tuple[int, bool, str]:
    cmd = [
        *RAY_RUN,
        "--cluster",
        cluster,
        "--tpu",
        args.validation_profile_tpu,
    ]
    profile_env_args = argparse.Namespace(
        wandb_mode=args.validation_profile_wandb_mode,
        size=args.validation_profile_size,
        num_steps=args.validation_profile_num_steps,
        profile_start_step=args.validation_profile_start_step,
        profile_num_steps=args.validation_profile_num_steps_window,
        run_name_prefix=run_name_prefix,
        tpu=args.validation_profile_tpu,
        batch_size=args.validation_profile_batch_size,
        chunk_size=args.validation_profile_chunk_size,
        segment_size=args.validation_profile_segment_size,
        profile_env=args.validation_profile_env,
    )
    _append_profile_env(cmd, profile_env_args)

    profile_args = ["--force_run_failed", "true"]
    if args.validation_profile_dry_run:
        profile_args += ["--dry_run", "true"]
    profile_cmd_lines = _profile_command_lines(include_tpu_sync=False, profile_args=profile_args)
    profile_cmd = " && ".join(profile_cmd_lines)
    cmd += ["--", "bash", "-lc", profile_cmd]

    rc, combined = _run_command_with_output(cmd)
    if rc != 0 and "No cluster config found for region" in combined:
        _validation_bad_ray_clusters(args).add(cluster)
        print(
            f"[gdnctl] validation Ray cluster is invalid and will be skipped in future retries: {cluster}",
            file=sys.stderr,
        )
        return rc, False, combined
    return rc, True, combined


def _run_validation_profile_once(args: argparse.Namespace, *, iteration: int) -> tuple[int, bool, dict[str, object]]:
    profile_prefix = f"{args.validation_profile_run_name_prefix}_iter{iteration:03d}"
    active_cluster = getattr(args, "active_dev_tpu_cluster", None)
    if args.dev_tpu_name and active_cluster is not None:

        def _run_on_cluster(cluster: str) -> tuple[int, str]:
            dev_profile_args = argparse.Namespace(
                cluster=cluster,
                tpu_name=args.dev_tpu_name,
                tpu=args.validation_profile_tpu,
                size=args.validation_profile_size,
                num_steps=args.validation_profile_num_steps,
                profile_start_step=args.validation_profile_start_step,
                profile_num_steps=args.validation_profile_num_steps_window,
                batch_size=args.validation_profile_batch_size,
                chunk_size=args.validation_profile_chunk_size,
                segment_size=args.validation_profile_segment_size,
                run_name_prefix=profile_prefix,
                wandb_mode=args.validation_profile_wandb_mode,
                profile_env=args.validation_profile_env,
                marin_prefix=args.validation_profile_marin_prefix,
                dry_run=args.validation_profile_dry_run,
                no_sync=args.validation_dev_no_sync,
            )
            cmd = [
                *DEV_TPU,
                "--cluster",
                dev_profile_args.cluster,
                "--tpu-name",
                dev_profile_args.tpu_name,
                "execute",
            ]
            _append_profile_env(cmd, dev_profile_args)
            if dev_profile_args.marin_prefix:
                marin_prefix = dev_profile_args.marin_prefix
            else:
                marin_prefix = f"gs://marin-{dev_profile_args.cluster}"
            cmd += ["-e", f"MARIN_PREFIX={marin_prefix}"]
            if dev_profile_args.no_sync:
                cmd.append("--no-sync")
            profile_args = ["--force_run_failed", "true"]
            if dev_profile_args.dry_run:
                profile_args += ["--dry_run", "true"]
            profile_cmd_lines = _profile_command_lines(include_tpu_sync=True, profile_args=profile_args)
            cmd += ["--", " && ".join(profile_cmd_lines)]
            return _run_command_with_output(cmd)

        rc, output_text = _run_on_cluster(active_cluster)
        if rc == 0:
            return 0, True, _collect_profile_metrics(args, output_text=output_text, profile_prefix=profile_prefix)
        if _try_reacquire_managed_dev_tpu(args, reason="validation profile failed on held dev TPU"):
            retry_cluster = getattr(args, "active_dev_tpu_cluster", None)
            if retry_cluster is not None:
                rc, output_text = _run_on_cluster(retry_cluster)
                if rc == 0:
                    return (
                        0,
                        True,
                        _collect_profile_metrics(args, output_text=output_text, profile_prefix=profile_prefix),
                    )
        print(
            "[gdnctl] validation profile failed on held dev TPU; trying Ray fallback.",
            file=sys.stderr,
        )

    clusters = _validation_ray_clusters(args, purpose="profile")
    if not clusters:
        print(
            "[gdnctl] no valid Ray clusters remain for validation profile.",
            file=sys.stderr,
        )
        return 1, False, {"profile_prefix": profile_prefix}

    last_rc = 1
    retryable = False
    last_output = ""
    for cluster in clusters:
        rc, cluster_retryable, output_text = _run_validation_ray_profile_once(
            args,
            cluster=cluster,
            run_name_prefix=profile_prefix,
        )
        if rc == 0:
            return 0, True, _collect_profile_metrics(args, output_text=output_text, profile_prefix=profile_prefix)
        last_rc = rc
        last_output = output_text
        retryable = retryable or cluster_retryable
    if last_output:
        return last_rc, retryable, _collect_profile_metrics(args, output_text=last_output, profile_prefix=profile_prefix)
    return last_rc, retryable, {"profile_prefix": profile_prefix}


def _run_validation_gate_for_iteration(
    args: argparse.Namespace,
    *,
    iteration: int,
) -> tuple[bool, int, dict[str, object]]:
    if args.validation_mode == "off":
        return True, 0, {}

    def should_continue(attempt: int) -> bool:
        if args.validation_max_attempts < 0:
            return True
        return attempt < args.validation_max_attempts

    if args.validation_mode == "required":
        attempt = 1
        while True:
            print(f"[gdnctl] validation tests attempt {attempt}")
            rc, retryable = _run_validation_tests_once(args)
            if rc == 0:
                break
            if not retryable:
                return False, rc, {}
            if not should_continue(attempt):
                return False, rc, {}
            if args.validation_retry_sleep > 0:
                print(
                    f"[gdnctl] validation tests failed; waiting {args.validation_retry_sleep:.1f}s before retry.",
                    file=sys.stderr,
                )
                time.sleep(args.validation_retry_sleep)
            attempt += 1

    attempt = 1
    validation_info: dict[str, object] = {}
    while True:
        print(f"[gdnctl] validation profile attempt {attempt}")
        rc, retryable, validation_info = _run_validation_profile_once(args, iteration=iteration)
        if rc == 0:
            return True, 0, validation_info
        if not retryable:
            return False, rc, validation_info
        if not should_continue(attempt):
            return False, rc, validation_info
        if args.validation_retry_sleep > 0:
            print(
                f"[gdnctl] validation profile failed; waiting {args.validation_retry_sleep:.1f}s before retry.",
                file=sys.stderr,
            )
            time.sleep(args.validation_retry_sleep)
        attempt += 1


def _failure_limit_exceeded(*, failures: int, max_failures: int) -> bool:
    if max_failures < 0:
        return False
    return failures > max_failures


def _apply_resilient_defaults(args: argparse.Namespace) -> None:
    if not args.resilient:
        return
    if args.max_failures >= 0:
        args.max_failures = -1
    if args.dirty_policy == "fail":
        args.dirty_policy = "stash"
    if args.no_commit_policy == "fail":
        args.no_commit_policy = "count-failure"
    if args.post_check_policy == "fail":
        args.post_check_policy = "count-failure"
    if args.iteration_error_policy == "fail":
        args.iteration_error_policy = "count-failure"
    if args.hold_dev_tpu_policy == "required":
        args.hold_dev_tpu_policy = "best-effort"
    if args.sync_main_conflict_policy == "fail":
        args.sync_main_conflict_policy = "skip"
    args.stash_restore_policy = "warn-keep"
    args.codex_retries = max(args.codex_retries, 2)
    args.post_check_retries = max(args.post_check_retries, 2)
    if args.validation_mode == "off":
        args.validation_mode = "required"
    args.perf_mode = "required"
    if args.perf_regression_policy == "fail":
        args.perf_regression_policy = "revert-count-failure"
    if args.validation_max_attempts >= 0:
        args.validation_max_attempts = -1
    if not args.validation_ray_cluster:
        args.validation_ray_cluster_auto = True
    args.validation_cross_tpu_test_fallback = True


def _last_iteration_bounds(lines: Sequence[str]) -> tuple[int, int] | None:
    last_start = None
    for idx, line in enumerate(lines):
        if line.startswith("### Iteration "):
            last_start = idx
    if last_start is None:
        return None
    return last_start, len(lines)


def _find_iteration_sequence_issues(lines: Sequence[str]) -> list[str]:
    issues: list[str] = []
    prev: tuple[int, int, str, str] | None = None

    for idx, line in enumerate(lines, start=1):
        match = ITERATION_HEADING_RE.match(line.strip())
        if match is None:
            continue

        num = int(match.group("num"))
        suffix = match.group("suffix") or ""
        heading = line.strip()

        if prev is not None:
            prev_line, prev_num, prev_suffix, prev_heading = prev
            reason: str | None = None

            if num < prev_num:
                reason = "iteration number decreased"
            elif num == prev_num:
                if prev_suffix == "" and suffix == "":
                    reason = "duplicate iteration number"
                elif prev_suffix != "" and suffix == "":
                    reason = "sub-iteration returned to base number"
                elif prev_suffix != "" and suffix != "" and suffix <= prev_suffix:
                    reason = "sub-iteration suffix is not increasing"

            if reason is not None:
                issues.append(f"{reason}: line {prev_line} `{prev_heading}` -> line {idx} `{heading}`")

        prev = (idx, num, suffix, heading)

    return issues


def _stamp_last_log_commit_placeholder(log_path: Path, *, commit_sha: str) -> bool:
    if not log_path.exists():
        return False

    lines = log_path.read_text(encoding="utf-8").splitlines()
    bounds = _last_iteration_bounds(lines)
    if bounds is None:
        return False

    start, end = bounds
    for idx in range(end - 1, start - 1, -1):
        stripped = lines[idx].strip()
        if not stripped.startswith("- Commit:"):
            continue
        value = stripped.split(":", 1)[1].strip()
        if value in {"this commit", "(pending)"}:
            prefix = lines[idx][: lines[idx].index("- Commit:")]
            lines[idx] = f"{prefix}- Commit: {commit_sha}"
            log_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            return True
        return False

    return False


def cmd_lint_log(args: argparse.Namespace) -> int:
    log_path = Path(args.log_file).resolve()
    if not log_path.exists():
        print(f"[gdnctl] log file not found: {log_path}", file=sys.stderr)
        return 2

    lines = log_path.read_text(encoding="utf-8").splitlines()
    pending_lines: list[int]
    this_commit_lines: list[int]
    if args.scope == "last-entry":
        bounds = _last_iteration_bounds(lines)
        if bounds is None:
            pending_lines = []
            this_commit_lines = []
        else:
            start, end = bounds
            pending_lines = [
                idx
                for idx, line in enumerate(lines[start:end], start=start + 1)
                if line.strip() == "- Commit: (pending)"
            ]
            this_commit_lines = [
                idx
                for idx, line in enumerate(lines[start:end], start=start + 1)
                if line.strip() == "- Commit: this commit"
            ]
    else:
        pending_lines = [idx for idx, line in enumerate(lines, start=1) if line.strip() == "- Commit: (pending)"]
        this_commit_lines = [idx for idx, line in enumerate(lines, start=1) if line.strip() == "- Commit: this commit"]

    if pending_lines and not args.allow_pending:
        print(f"[gdnctl] unresolved `Commit: (pending)` entries in {log_path}:", file=sys.stderr)
        for line_no in pending_lines:
            print(f"[gdnctl]   line {line_no}", file=sys.stderr)
        return 1

    if this_commit_lines and not args.allow_this_commit:
        print(f"[gdnctl] unresolved `Commit: this commit` entries in {log_path}:", file=sys.stderr)
        for line_no in this_commit_lines:
            print(f"[gdnctl]   line {line_no}", file=sys.stderr)
        return 1

    if not args.allow_non_monotonic:
        sequence_issues = _find_iteration_sequence_issues(lines)
        if sequence_issues:
            print(f"[gdnctl] non-monotonic iteration headings in {log_path}:", file=sys.stderr)
            for issue in sequence_issues:
                print(f"[gdnctl]   {issue}", file=sys.stderr)
            return 1

    print(f"[gdnctl] log lint passed: {log_path}")
    return 0


def cmd_codex_loop(args: argparse.Namespace) -> int:
    workdir = Path(args.workdir).resolve()
    prompt_template = Path(args.prompt_file).read_text(encoding="utf-8")
    base_session_directives = _load_session_directives(args)
    log_dir = Path(args.log_dir).resolve()
    log_dir.mkdir(parents=True, exist_ok=True)
    perf_state_path = (
        Path(args.perf_state_file).resolve() if args.perf_state_file else (log_dir / "perf_state.json").resolve()
    )

    _apply_resilient_defaults(args)

    if args.hold_dev_tpu and not args.dev_tpu_name:
        if args.resilient:
            print(
                "[gdnctl] WARNING: --hold-dev-tpu requested without --dev-tpu-name; "
                "continuing without managed dev TPU due to --resilient.",
                file=sys.stderr,
            )
            args.hold_dev_tpu = False
        else:
            raise SystemExit("[gdnctl] --hold-dev-tpu requires --dev-tpu-name.")

    if args.dev_tpu_ready_timeout <= 0:
        raise SystemExit("[gdnctl] --dev-tpu-ready-timeout must be > 0.")

    if args.dev_tpu_allocate_attempts <= 0:
        raise SystemExit("[gdnctl] --dev-tpu-allocate-attempts must be > 0.")

    if args.dev_tpu_allocate_retry_sleep < 0:
        raise SystemExit("[gdnctl] --dev-tpu-allocate-retry-sleep must be >= 0.")

    if args.dev_tpu_stop_timeout <= 0:
        raise SystemExit("[gdnctl] --dev-tpu-stop-timeout must be > 0.")

    if args.codex_retries < 0:
        raise SystemExit("[gdnctl] --codex-retries must be >= 0.")

    if args.post_check_retries < 0:
        raise SystemExit("[gdnctl] --post-check-retries must be >= 0.")

    if args.retry_sleep_seconds < 0:
        raise SystemExit("[gdnctl] --retry-sleep-seconds must be >= 0.")

    if args.codex_timeout_seconds is not None and args.codex_timeout_seconds <= 0:
        raise SystemExit("[gdnctl] --codex-timeout-seconds must be > 0 when set.")

    if args.codex_idle_timeout_seconds is not None and args.codex_idle_timeout_seconds <= 0:
        raise SystemExit("[gdnctl] --codex-idle-timeout-seconds must be > 0 when set.")

    if args.validation_max_attempts < -1 or args.validation_max_attempts == 0:
        raise SystemExit("[gdnctl] --validation-max-attempts must be -1 (unlimited) or >= 1.")

    if args.validation_retry_sleep < 0:
        raise SystemExit("[gdnctl] --validation-retry-sleep must be >= 0.")

    args.validation_ray_cluster_exclude = [
        token.strip() for token in args.validation_ray_cluster_exclude if token and token.strip()
    ]

    if args.perf_mode == "required" and args.validation_mode == "off":
        raise SystemExit("[gdnctl] --perf-mode=required requires --validation-mode=required or profile-only.")

    # Fail fast on malformed profile env overrides before entering the loop.
    _parse_profile_env(args.validation_profile_env)

    if args.perf_min_improvement_pct < 0:
        raise SystemExit("[gdnctl] --perf-min-improvement-pct must be >= 0.")

    if args.perf_max_regression_pct < 0:
        raise SystemExit("[gdnctl] --perf-max-regression-pct must be >= 0.")
    if args.perf_history_step_start < 0:
        raise SystemExit("[gdnctl] --perf-history-step-start must be >= 0.")
    if args.perf_history_step_end < args.perf_history_step_start:
        raise SystemExit("[gdnctl] --perf-history-step-end must be >= --perf-history-step-start.")
    if args.perf_history_min_points <= 0:
        raise SystemExit("[gdnctl] --perf-history-min-points must be > 0.")

    if args.sync_main_policy != "off":
        if not args.sync_main_remote.strip():
            raise SystemExit("[gdnctl] --sync-main-remote must be non-empty.")
        if not args.sync_main_branch.strip():
            raise SystemExit("[gdnctl] --sync-main-branch must be non-empty.")

    if args.resilient:
        print(
            "[gdnctl] resilient mode active: unlimited failures, stash dirty handling, "
            "best-effort dev TPU hold, and retry-enabled command paths."
        )

    codex_bin, supported_efforts = _resolve_codex_binary(
        explicit_binary=args.codex_bin,
        reasoning_effort=args.reasoning_effort,
    )
    selected_efforts = ", ".join(supported_efforts) if supported_efforts else "unknown"
    print(f"[gdnctl] using codex binary: {codex_bin}")
    print(f"[gdnctl] supported model_reasoning_effort values: {selected_efforts}")

    search_supported = _codex_exec_supports_search(codex_bin) if args.search else False
    if args.search and not search_supported:
        print(
            "[gdnctl] WARNING: installed `codex exec` does not support `--search`; continuing without it.",
            file=sys.stderr,
        )

    failures = 0
    with _hold_dev_tpu_for_loop(args, log_dir=log_dir):
        session_directives = list(base_session_directives)
        managed_hold_active = hasattr(args, "active_dev_tpu_cluster")
        if managed_hold_active:
            session_directives.append(_managed_dev_tpu_session_directive(args))
            _warn_if_hold_dev_tpu_with_ray_post_checks(args)
        elif args.hold_dev_tpu:
            print(
                "[gdnctl] managed dev TPU hold is not active for this run; continuing without hold-specific directives.",
                file=sys.stderr,
            )
        if args.validation_mode in {"required", "profile-only"}:
            session_directives.append(_validation_gate_session_directive(args))
        if args.perf_mode == "required":
            session_directives.append(_performance_policy_session_directive(args, perf_state_path=perf_state_path))
        if session_directives:
            print(f"[gdnctl] session directives: {len(session_directives)} loaded")

        for iteration in range(1, args.iterations + 1):
            print(f"\n[gdnctl] === codex iteration {iteration}/{args.iterations} ===")
            try:
                with _clean_worktree_for_iteration(args, workdir=workdir, iteration=iteration):
                    should_sync_main = args.sync_main_policy == "each-iteration" or (
                        args.sync_main_policy == "once" and iteration == 1
                    )
                    if should_sync_main:
                        _sync_branch_from_remote(
                            workdir=workdir,
                            remote=args.sync_main_remote,
                            branch=args.sync_main_branch,
                            conflict_policy=args.sync_main_conflict_policy,
                        )

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

                    cmd = _build_codex_exec_cmd(
                        codex_bin=codex_bin,
                        workdir=workdir,
                        message_path=message_path,
                        args=args,
                        search_supported=search_supported,
                    )

                    codex_attempts = args.codex_retries + 1
                    rc = 1
                    for attempt in range(1, codex_attempts + 1):
                        rc = _run_streaming(
                            cmd,
                            cwd=workdir,
                            input_text=prompt,
                            check=False,
                            hide_codex_file_updates=not args.show_file_updates,
                            max_runtime_seconds=args.codex_timeout_seconds,
                            idle_timeout_seconds=args.codex_idle_timeout_seconds,
                        )
                        if rc == 0:
                            break
                        if attempt < codex_attempts:
                            print(
                                "[gdnctl] codex iteration command failed "
                                f"(attempt {attempt}/{codex_attempts}, rc={rc}); retrying.",
                                file=sys.stderr,
                            )
                            if args.retry_sleep_seconds > 0:
                                time.sleep(args.retry_sleep_seconds)

                    if rc != 0:
                        failures += 1
                        print(f"[gdnctl] codex iteration failed with exit code {rc}", file=sys.stderr)
                        if _failure_limit_exceeded(failures=failures, max_failures=args.max_failures):
                            return rc
                        continue

                    head_after = _git_head(workdir)
                    missing_commit = args.require_new_commit and head_after == head_before

                    validation_ok, validation_rc, validation_info = _run_validation_gate_for_iteration(
                        args, iteration=iteration
                    )
                    if not validation_ok:
                        message = (
                            "[gdnctl] Iteration failed required TPU validation/profile gate. "
                            "Use --validation-policy count-failure/continue to keep looping."
                        )
                        if args.validation_policy == "fail":
                            print(message, file=sys.stderr)
                            return validation_rc or 1
                        print(
                            f"{message} Continuing due to policy={args.validation_policy}.",
                            file=sys.stderr,
                        )
                        if args.validation_policy == "count-failure":
                            failures += 1
                            if _failure_limit_exceeded(failures=failures, max_failures=args.max_failures):
                                return validation_rc or 1
                        continue

                    if args.post_check:
                        ok, post_rc = _run_post_checks_with_retries(
                            workdir,
                            args.post_check,
                            retries=args.post_check_retries,
                            retry_sleep_seconds=args.retry_sleep_seconds,
                        )
                        if not ok:
                            message = (
                                "[gdnctl] post-check failed. "
                                "Use --post-check-policy count-failure/continue to avoid exiting."
                            )
                            if args.post_check_policy == "fail":
                                print(message, file=sys.stderr)
                                return post_rc or 1
                            print(
                                f"{message} Continuing due to policy={args.post_check_policy}.",
                                file=sys.stderr,
                            )
                            if args.post_check_policy == "count-failure":
                                failures += 1
                                if _failure_limit_exceeded(failures=failures, max_failures=args.max_failures):
                                    return post_rc or 1
                            continue

                    if missing_commit:
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
                            if _failure_limit_exceeded(failures=failures, max_failures=args.max_failures):
                                return 3
                        continue

                    if _stamp_last_log_commit_placeholder(DEFAULT_HILLCLIMB_LOG, commit_sha=head_after):
                        print(
                            f"[gdnctl] stamped last hill-climb log entry with commit {head_after[:12]}",
                        )

                    perf_ok, perf_count_failure, perf_rc = _apply_performance_policy(
                        args,
                        workdir=workdir,
                        perf_state_path=perf_state_path,
                        iteration=iteration,
                        commit_sha=head_after,
                        validation_info=validation_info,
                    )
                    if not perf_ok:
                        print(
                            "[gdnctl] performance policy failed this iteration.",
                            file=sys.stderr,
                        )
                        return perf_rc or 1
                    if perf_count_failure:
                        failures += 1
                        if _failure_limit_exceeded(failures=failures, max_failures=args.max_failures):
                            return perf_rc or 1
                        continue

                    if iteration < args.iterations and args.sleep_seconds > 0:
                        time.sleep(args.sleep_seconds)
            except Exception as exc:  # pragma: no cover - defensive loop safety
                message = f"[gdnctl] iteration {iteration} raised: {exc!r}"
                if args.iteration_error_policy == "fail":
                    print(message, file=sys.stderr)
                    return 2
                print(f"{message}. Continuing due to policy={args.iteration_error_policy}.", file=sys.stderr)
                if args.iteration_error_policy == "count-failure":
                    failures += 1
                    if _failure_limit_exceeded(failures=failures, max_failures=args.max_failures):
                        return 2
                continue

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
    dev_profile.add_argument(
        "--profile-env",
        action="append",
        default=[],
        help="Extra profile env var override in KEY=VALUE form (repeatable).",
    )
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
    ray_profile.add_argument(
        "--profile-env",
        action="append",
        default=[],
        help="Extra profile env var override in KEY=VALUE form (repeatable).",
    )
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
    codex_loop.add_argument(
        "--sync-main-policy",
        choices=["off", "once", "each-iteration"],
        default="off",
        help="Synchronize branch with --sync-main-remote/--sync-main-branch before loop iterations.",
    )
    codex_loop.add_argument(
        "--sync-main-remote",
        default="origin",
        help="Remote used by --sync-main-policy.",
    )
    codex_loop.add_argument(
        "--sync-main-branch",
        default="main",
        help="Remote branch used by --sync-main-policy.",
    )
    codex_loop.add_argument(
        "--sync-main-conflict-policy",
        choices=["fail", "skip"],
        default="fail",
        help="Behavior when sync-main merge conflicts occur.",
    )
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
    codex_loop.add_argument(
        "--codex-ephemeral",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Run `codex exec` with `--ephemeral` so unattended loops do not persist thousands of "
            "Codex sessions/threads into the app state. Disable only when you explicitly want loop "
            "sessions saved for manual resume/debugging."
        ),
    )
    codex_loop.add_argument("--search", action="store_true", help="Enable Codex web search tool")
    codex_loop.add_argument(
        "--show-file-updates",
        action="store_true",
        help="Show Codex `file update:` diff blocks in loop output.",
    )
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
        choices=sorted(SESSION_DIRECTIVE_PRESET_FILES),
        action="append",
        default=[],
        help="Named per-session directive preset loaded from scripts/gdn/session_directives/*.md (repeatable).",
    )
    codex_loop.add_argument("--sleep-seconds", type=float, default=5.0)
    codex_loop.add_argument(
        "--post-check",
        action="append",
        default=[],
        help="Shell command run after each successful iteration (repeatable)",
    )
    codex_loop.add_argument(
        "--validation-mode",
        choices=["required", "profile-only", "off"],
        default="required",
        help="Validation gate mode: tests+profile, profile-only, or off.",
    )
    codex_loop.add_argument(
        "--validation-policy",
        choices=["fail", "count-failure", "continue"],
        default="fail",
        help="Behavior when required TPU validation/profile gate fails.",
    )
    codex_loop.add_argument(
        "--validation-max-attempts",
        type=int,
        default=-1,
        help="Retry cap for validation tests/profile attempts (-1 = unlimited, default).",
    )
    codex_loop.add_argument(
        "--validation-retry-sleep",
        type=float,
        default=120.0,
        help="Seconds to wait between validation retries.",
    )
    codex_loop.add_argument(
        "--validation-tests",
        choices=["kernels", "layer", "both"],
        default="both",
        help="Test suite used by the per-iteration validation gate.",
    )
    codex_loop.add_argument(
        "--validation-pytest-args",
        default=None,
        help="Extra pytest args for validation tests.",
    )
    codex_loop.add_argument(
        "--validation-ray-cluster",
        action="append",
        default=[],
        help="Ray cluster candidate for validation fallback (repeatable). Defaults to dev cluster candidates.",
    )
    codex_loop.add_argument(
        "--validation-ray-cluster-auto",
        action="store_true",
        help=(
            "Auto-discover validation Ray fallback clusters from infra/marin-*.yaml, "
            "filtered by required TPU type and --validation-ray-cluster-exclude."
        ),
    )
    codex_loop.add_argument(
        "--validation-ray-cluster-exclude",
        action="append",
        default=list(DEFAULT_VALIDATION_CLUSTER_EXCLUDE_SUBSTRINGS),
        help=(
            "Cluster-name substring excluded from auto-discovered validation Ray clusters "
            "(repeatable, case-insensitive)."
        ),
    )
    codex_loop.add_argument(
        "--validation-cross-tpu-test-fallback",
        action="store_true",
        help=(
            "When auto-discovery is enabled, allow Ray test fallback on modern TPU families "
            "(v5p/v5e/v6e) even if they differ from --validation-profile-tpu. "
            "Profile fallback remains pinned to --validation-profile-tpu."
        ),
    )
    codex_loop.add_argument(
        "--validation-ray-tpu",
        default="auto",
        help="TPU type used when running validation tests through ray_run.",
    )
    codex_loop.add_argument(
        "--validation-dev-no-sync",
        action="store_true",
        help="Skip rsync when running validation on held dev TPU.",
    )
    codex_loop.add_argument(
        "--validation-profile-tpu",
        default="v5p-8",
        help="TPU variant for validation profile runs.",
    )
    codex_loop.add_argument(
        "--validation-profile-size",
        choices=["130m", "300m", "520m", "1_2b"],
        default="130m",
        help="Model size used by validation profile runs.",
    )
    codex_loop.add_argument(
        "--validation-profile-num-steps",
        type=int,
        default=20,
        help="Number of steps for validation profile runs.",
    )
    codex_loop.add_argument(
        "--validation-profile-start-step",
        type=int,
        default=2,
        help="Profile start step for validation profile runs.",
    )
    codex_loop.add_argument(
        "--validation-profile-num-steps-window",
        type=int,
        default=6,
        help="Number of profiled steps for validation profile runs.",
    )
    codex_loop.add_argument(
        "--validation-profile-batch-size",
        type=int,
        default=8,
        help="Batch size for validation profile runs.",
    )
    codex_loop.add_argument(
        "--validation-profile-chunk-size",
        type=int,
        default=None,
        help="Optional GDN chunk override for validation profile runs.",
    )
    codex_loop.add_argument(
        "--validation-profile-segment-size",
        type=int,
        default=None,
        help="Optional GDN segment override for validation profile runs.",
    )
    codex_loop.add_argument(
        "--validation-profile-env",
        action="append",
        default=[],
        help="Extra profile env var override for validation profile runs in KEY=VALUE form (repeatable).",
    )
    codex_loop.add_argument(
        "--validation-profile-run-name-prefix",
        default="gdn_loopgate",
        help="Run name prefix for validation profile runs.",
    )
    codex_loop.add_argument(
        "--validation-profile-wandb-mode",
        default="online",
        choices=["online", "offline", "disabled"],
        help="W&B mode for validation profile runs.",
    )
    codex_loop.add_argument(
        "--validation-profile-marin-prefix",
        default=None,
        help="Optional MARIN_PREFIX override for dev TPU validation profile runs.",
    )
    codex_loop.add_argument(
        "--validation-profile-dry-run",
        action="store_true",
        help="Use executor dry-run for validation profile runs.",
    )
    codex_loop.add_argument(
        "--perf-mode",
        choices=["required", "off"],
        default="required",
        help="Whether each validated iteration must include parseable performance metrics.",
    )
    codex_loop.add_argument(
        "--perf-state-file",
        default=None,
        help="Path to champion/baseline performance state JSON (defaults to <log-dir>/perf_state.json).",
    )
    codex_loop.add_argument(
        "--perf-metric",
        default="throughput/mfu",
        help="Primary metric used for champion comparison.",
    )
    codex_loop.add_argument(
        "--perf-aggregation",
        choices=["history-window", "summary"],
        default="history-window",
        help=(
            "How performance metrics are aggregated from W&B: "
            "`history-window` uses a robust step window, `summary` uses final run summary values."
        ),
    )
    codex_loop.add_argument(
        "--perf-history-aggregation",
        choices=["median", "mean"],
        default="median",
        help="Aggregator used for --perf-aggregation=history-window.",
    )
    codex_loop.add_argument(
        "--perf-history-step-start",
        type=int,
        default=10,
        help="Inclusive start step for history-window aggregation.",
    )
    codex_loop.add_argument(
        "--perf-history-step-end",
        type=int,
        default=18,
        help="Inclusive end step for history-window aggregation.",
    )
    codex_loop.add_argument(
        "--perf-history-min-points",
        type=int,
        default=5,
        help="Minimum number of points required per metric in history-window aggregation.",
    )
    codex_loop.add_argument(
        "--perf-min-improvement-pct",
        type=float,
        default=0.25,
        help="Minimum percent gain vs champion required to promote a new champion.",
    )
    codex_loop.add_argument(
        "--perf-max-regression-pct",
        type=float,
        default=1.0,
        help="Maximum allowed regression percent vs champion before policy triggers.",
    )
    codex_loop.add_argument(
        "--perf-regression-policy",
        choices=["revert-count-failure", "revert-continue", "fail", "count-failure", "continue"],
        default="revert-count-failure",
        help="Action when performance regresses beyond threshold.",
    )
    codex_loop.add_argument(
        "--perf-wandb-entity",
        default="marin-community",
        help="W&B entity used when discovering/fetching profile summaries.",
    )
    codex_loop.add_argument(
        "--perf-wandb-project",
        default="marin",
        help="W&B project used when discovering/fetching profile summaries.",
    )
    codex_loop.add_argument(
        "--post-check-policy",
        choices=["fail", "count-failure", "continue"],
        default="fail",
        help="Behavior when post-check commands fail.",
    )
    codex_loop.add_argument("--max-failures", type=int, default=0, help="Max failures before aborting (-1 = unlimited).")
    codex_loop.add_argument(
        "--resilient",
        action="store_true",
        help=(
            "Enable resilient unattended mode: unlimited failures, retries, best-effort managed dev TPU hold, "
            "and non-fatal dirty-tree restore behavior."
        ),
    )
    codex_loop.add_argument(
        "--codex-retries",
        type=int,
        default=0,
        help="Retry attempts for each codex exec iteration command.",
    )
    codex_loop.add_argument(
        "--codex-timeout-seconds",
        type=float,
        default=7200.0,
        help="Maximum wall-clock runtime per codex exec iteration command.",
    )
    codex_loop.add_argument(
        "--codex-idle-timeout-seconds",
        type=float,
        default=900.0,
        help="Maximum seconds without codex output before the iteration command is terminated and retried.",
    )
    codex_loop.add_argument(
        "--post-check-retries",
        type=int,
        default=0,
        help="Retry attempts per post-check command.",
    )
    codex_loop.add_argument(
        "--retry-sleep-seconds",
        type=float,
        default=20.0,
        help="Sleep interval between retry attempts for codex/post-check commands.",
    )
    codex_loop.add_argument("--log-dir", default=str(REPO_ROOT / ".agents/logs/gdn_codex_loop"))
    codex_loop.add_argument("--allow-dirty", action="store_true", help="Allow starting from a dirty tree")
    codex_loop.add_argument(
        "--dirty-policy",
        choices=["fail", "stash"],
        default="fail",
        help="Behavior when the tree is dirty at iteration start (ignored with --allow-dirty).",
    )
    codex_loop.add_argument(
        "--stash-restore-policy",
        choices=["warn-keep", "fail"],
        default="warn-keep",
        help=(
            "Behavior if restoring a stashed dirty tree fails at iteration end: "
            "`warn-keep` keeps the stash and continues, `fail` stops the loop."
        ),
    )
    codex_loop.add_argument("--allow-no-commit", action="store_true", help="Do not require a new commit")
    codex_loop.add_argument(
        "--no-commit-policy",
        choices=["fail", "count-failure", "continue"],
        default="fail",
        help="Behavior when an iteration exits cleanly but does not create a commit.",
    )
    codex_loop.add_argument(
        "--iteration-error-policy",
        choices=["fail", "count-failure", "continue"],
        default="fail",
        help="Behavior when an unexpected iteration exception occurs.",
    )
    codex_loop.add_argument(
        "--hold-dev-tpu",
        action="store_true",
        help="Allocate and hold a dev TPU allocation for the full loop duration.",
    )
    codex_loop.add_argument(
        "--hold-dev-tpu-policy",
        choices=["required", "best-effort"],
        default="required",
        help="Whether managed dev TPU allocation failure should abort or continue without hold.",
    )
    codex_loop.add_argument(
        "--dev-tpu-cluster",
        default="us-central1",
        help="Cluster used for managed dev TPU allocation.",
    )
    codex_loop.add_argument(
        "--dev-tpu-fallback-cluster",
        action="append",
        default=[],
        help="Fallback cluster(s) for managed dev TPU allocation if the primary cluster times out.",
    )
    codex_loop.add_argument(
        "--dev-tpu-name",
        default=None,
        help="TPU name used by managed dev TPU allocation (required with --hold-dev-tpu).",
    )
    codex_loop.add_argument(
        "--dev-tpu-type",
        default=None,
        help="Optional TPU type for managed dev TPU allocation (for example v5p-8).",
    )
    codex_loop.add_argument(
        "--dev-tpu-sync-path",
        default=".",
        help="Sync path passed to managed `dev_tpu allocate`.",
    )
    codex_loop.add_argument(
        "--dev-tpu-ready-timeout",
        type=float,
        default=900.0,
        help="Seconds to wait for managed dev TPU allocation to become active.",
    )
    codex_loop.add_argument(
        "--dev-tpu-allocate-attempts",
        type=int,
        default=1,
        help="Number of full allocation passes across primary+fallback clusters before giving up.",
    )
    codex_loop.add_argument(
        "--dev-tpu-allocate-retry-sleep",
        type=float,
        default=30.0,
        help="Seconds to sleep between managed allocation attempts.",
    )
    codex_loop.add_argument(
        "--dev-tpu-stop-timeout",
        type=float,
        default=120.0,
        help="Seconds to wait for managed dev TPU allocation to release on exit.",
    )
    codex_loop.add_argument(
        "--dev-tpu-allocate-log",
        default=None,
        help="Optional log path for managed dev TPU allocate output (defaults under --log-dir).",
    )
    codex_loop.add_argument(
        "--dev-tpu-verbose",
        action="store_true",
        help="Pass --verbose to managed dev TPU allocation.",
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
    lint_log.add_argument(
        "--allow-this-commit",
        action="store_true",
        help="Do not fail when `Commit: this commit` placeholders are present.",
    )
    lint_log.add_argument(
        "--allow-non-monotonic",
        action="store_true",
        help="Do not fail when iteration headings are not globally monotonic.",
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
