#!/usr/bin/env python3
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Compare a historical Hugging Face trace dataset with local Harbor trials.

This ports the useful parts of OpenThoughts-Agent's ``behavioral_delta`` and
``trace_pair_render`` tools to Harbor's current result/ATIF trajectory layout.
It emits machine-readable aggregate and matched-task statistics plus an HTML
page containing representative regressions.
"""

from __future__ import annotations

import argparse
import html
import json
import math
import os
import re
import statistics
import subprocess
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from datasets import load_dataset


@dataclass(frozen=True)
class Trace:
    task: str
    trial: str
    reward: float | None
    error: str | None
    messages: tuple[dict[str, Any], ...]
    input_tokens: int | None = None
    output_tokens: int | None = None
    episodes: int | None = None
    duration: float | None = None
    api_request_times: tuple[float, ...] = ()
    completion_tokens_by_call: tuple[int, ...] = ()
    timeout_budget: float | None = None


def _number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _iso(value: str | None) -> datetime | None:
    if not value:
        return None
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _historical_trace(row: dict[str, Any]) -> Trace:
    result = row.get("result")
    reward = _number(result)
    error = None if reward is not None or result in (None, "", "null") else str(result)
    messages = row.get("conversations") or row.get("messages") or []
    return Trace(
        task=str(row.get("task") or row.get("task_name") or row.get("trial_name")),
        trial=str(row.get("trial_name") or row.get("episode") or "unknown"),
        reward=reward,
        error=error,
        messages=tuple(message for message in messages if isinstance(message, dict)),
    )


def load_historical(repo: str, trace_source: str = "main") -> list[Trace]:
    """Load the authoritative trace subset from a Hugging Face dataset."""
    rows = load_dataset(repo, split="train")
    return [_historical_trace(row) for row in rows if trace_source == "*" or row.get("trace_source") == trace_source]


def _trajectory_documents(agent_dir: Path) -> list[dict[str, Any]]:
    paths = [agent_dir / "trajectory.json"]
    paths.extend(sorted(agent_dir.glob("trajectory.cont-*.json"), key=lambda path: int(path.stem.rsplit("-", 1)[1])))
    return [json.loads(path.read_text()) for path in paths if path.exists()]


def _trajectory_messages(documents: list[dict[str, Any]]) -> tuple[dict[str, Any], ...]:
    messages = []
    for trajectory in documents:
        for step in trajectory.get("steps", []):
            source = step.get("source")
            if source not in {"user", "agent", "tool", "system"}:
                continue
            messages.append({"role": "assistant" if source == "agent" else source, "content": step.get("message", "")})
            for result in (step.get("observation") or {}).get("results", []):
                if isinstance(result, dict) and result.get("content"):
                    messages.append({"role": "tool", "content": result["content"]})
    return tuple(messages)


def _completion_tokens(documents: list[dict[str, Any]]) -> tuple[int, ...]:
    return tuple(
        int(tokens)
        for document in documents
        for step in document.get("steps", [])
        if step.get("source") == "agent"
        if (tokens := (step.get("metrics") or {}).get("completion_tokens")) is not None
    )


def _local_trace(path: Path) -> Trace:
    row = json.loads(path.read_text())
    rewards = (row.get("verifier_result") or {}).get("rewards") or {}
    reward = _number(rewards.get("reward"))
    exception = row.get("exception_info") or {}
    error = exception.get("exception_type")
    agent = row.get("agent_result") or {}
    metadata = agent.get("metadata") or {}
    documents = _trajectory_documents(path.parent / "agent")
    started, finished = _iso(row.get("started_at")), _iso(row.get("finished_at"))
    duration = (finished - started).total_seconds() if started and finished else None
    request_times = tuple(float(value) / 1000 for value in metadata.get("api_request_times_msec", []))
    return Trace(
        task=str(row.get("task_name") or row.get("trial_name")),
        trial=str(row.get("trial_name") or path.parent.name),
        reward=reward,
        error=str(error) if error else None,
        messages=_trajectory_messages(documents),
        input_tokens=agent.get("n_input_tokens"),
        output_tokens=agent.get("n_output_tokens"),
        episodes=metadata.get("n_episodes"),
        duration=duration,
        api_request_times=request_times,
        completion_tokens_by_call=_completion_tokens(documents),
        timeout_budget=_timeout_budget(exception.get("exception_message")),
    )


def _timeout_budget(message: str | None) -> float | None:
    if not message:
        return None
    match = re.search(r"timed out after ([0-9.]+) seconds", message)
    return float(match.group(1)) if match else None


def load_local(root: Path) -> list[Trace]:
    """Load current Harbor trial directories, including ATIF trajectories."""
    return [_local_trace(path) for path in sorted(root.glob("*/result.json"))]


def _content(message: dict[str, Any]) -> str:
    value = message.get("content", "")
    return value if isinstance(value, str) else json.dumps(value, sort_keys=True)


def _behavior(trace: Trace) -> dict[str, float]:
    assistant = [_content(message) for message in trace.messages if message.get("role") == "assistant"]
    terminal = [
        _content(message)
        for message in trace.messages
        if message.get("role") == "tool" or "New Terminal Output:" in _content(message)
    ]
    actions = []
    xml_messages = 0
    for text in assistant:
        xml_actions = len(re.findall(r"<tool_call>", text, re.I))
        if xml_actions:
            xml_messages += 1
            actions.append(xml_actions)
        else:
            actions.append(text.count('"keystrokes"'))
    return {
        "turns": float(len(trace.messages)),
        "assistant_messages": float(len(assistant)),
        "assistant_characters": float(sum(map(len, assistant))),
        "thinking_characters": float(
            sum(
                len(match)
                for text in assistant
                for match in re.findall(r"<think[^>]*>(.*?)</think[^>]*>", text, re.I | re.S)
            )
        ),
        "tool_responses": float(len(terminal)),
        "tool_error_responses": float(
            sum(
                any(
                    token in text.lower()
                    for token in ("traceback", "command not found", "permission denied", "no such file")
                )
                for text in terminal
            )
        ),
        "task_complete_claims": float(sum('"task_complete": true' in text.lower() for text in assistant)),
        "tool_actions": float(sum(actions)),
        "multi_action_messages": float(sum(value > 1 for value in actions)),
        "no_action_messages": float(sum(value == 0 for value in actions)),
        "xml_messages": float(xml_messages),
    }


def _mean(values: Iterable[float | int | None]) -> float | None:
    clean = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return statistics.fmean(clean) if clean else None


def summarize(traces: list[Trace], max_output_tokens: int | None = None) -> dict[str, Any]:
    """Return score, exception, latency, token, and behavioral aggregates."""
    numeric = [trace.reward for trace in traces if trace.reward is not None]
    behaviors = [_behavior(trace) for trace in traces]
    errors = Counter(trace.error for trace in traces if trace.error)
    api = [value for trace in traces for value in trace.api_request_times]
    completions = [value for trace in traces for value in trace.completion_tokens_by_call]
    timeout_traces = [trace for trace in traces if trace.error == "AgentTimeoutError"]
    timeout_api_fraction = [
        sum(trace.api_request_times) / trace.duration
        for trace in timeout_traces
        if trace.duration and trace.api_request_times
    ]
    reward_error: dict[str, dict[str, float | int | None]] = {}
    total_assistant_messages = sum(row["assistant_messages"] for row in behaviors)
    total_xml_messages = sum(row["xml_messages"] for row in behaviors)
    for error in sorted({trace.error or "none" for trace in traces}):
        group = [trace for trace in traces if (trace.error or "none") == error]
        reward_error[error] = {
            "trials": len(group),
            "reward_sum": sum(trace.reward or 0 for trace in group),
            "mean_reward": _mean(trace.reward or 0 for trace in group),
        }
    return {
        "trials": len(traces),
        "numeric_trials": len(numeric),
        "mean_reward_all_trials": sum(numeric) / len(traces) if traces else None,
        "mean_reward_numeric_trials": _mean(numeric),
        "solved": sum(value > 0 for value in numeric),
        "errors": dict(errors.most_common()),
        "reward_by_error": reward_error,
        "agent_timeout_rate": errors.get("AgentTimeoutError", 0) / len(traces) if traces else None,
        "mean_duration_seconds": _mean(trace.duration for trace in traces),
        "mean_input_tokens": _mean(trace.input_tokens for trace in traces),
        "mean_output_tokens": _mean(trace.output_tokens for trace in traces),
        "mean_episodes": _mean(trace.episodes for trace in traces),
        "timeout_budget_seconds": dict(
            Counter(str(trace.timeout_budget) for trace in timeout_traces if trace.timeout_budget)
        ),
        "timeout_mean_api_fraction": _mean(timeout_api_fraction),
        "api_request_seconds": {
            "count": len(api),
            "mean": _mean(api),
            "p50": statistics.median(api) if api else None,
            "p90": sorted(api)[int(0.9 * (len(api) - 1))] if api else None,
            "over_120_seconds": sum(value >= 120 for value in api),
        },
        "completion_tokens_by_call": {
            "count": len(completions),
            "max": max(completions) if completions else None,
            "p50": statistics.median(completions) if completions else None,
            "p90": sorted(completions)[int(0.9 * (len(completions) - 1))] if completions else None,
            "at_or_above_90_percent_of_limit": (
                sum(value >= 0.9 * max_output_tokens for value in completions) if max_output_tokens else None
            ),
            "configured_limit": max_output_tokens,
        },
        "behavior": {key: _mean(row[key] for row in behaviors) for key in behaviors[0]} if behaviors else {},
        "observed_message_format": {
            "xml_fraction": total_xml_messages / total_assistant_messages if total_assistant_messages else None,
            "dominant_parser": "xml" if total_xml_messages > total_assistant_messages / 2 else "json",
        },
    }


def _task_values(traces: list[Trace]) -> dict[str, list[float]]:
    values: dict[str, list[float]] = defaultdict(list)
    for trace in traces:
        values[trace.task].append(trace.reward or 0.0)
    return values


def matched_tasks(before: list[Trace], after: list[Trace]) -> dict[str, Any]:
    """Compare per-task means over tasks present on both sides."""
    left, right = _task_values(before), _task_values(after)
    common = sorted(left.keys() & right.keys())
    deltas = {task: statistics.fmean(right[task]) - statistics.fmean(left[task]) for task in common}
    return {
        "tasks": len(common),
        "mean_task_delta": _mean(deltas.values()),
        "regressed_tasks": sum(delta < 0 for delta in deltas.values()),
        "improved_tasks": sum(delta > 0 for delta in deltas.values()),
        "unchanged_tasks": sum(delta == 0 for delta in deltas.values()),
        "largest_regressions": sorted(deltas.items(), key=lambda item: item[1])[:20],
        "largest_improvements": sorted(deltas.items(), key=lambda item: item[1], reverse=True)[:20],
    }


def compare(before: list[Trace], after: list[Trace], max_output_tokens: int | None = None) -> dict[str, Any]:
    """Build a structured historical-versus-current comparison."""
    left, right = summarize(before), summarize(after, max_output_tokens)
    return {
        "historical": left,
        "current": right,
        "delta": {
            "mean_reward_all_trials": right["mean_reward_all_trials"] - left["mean_reward_all_trials"],
            "agent_timeout_rate": right["agent_timeout_rate"] - left["agent_timeout_rate"],
        },
        "matched_tasks": matched_tasks(before, after),
    }


def load_registry_job(job_id: str) -> dict[str, Any]:
    """Load authoritative score, token totals, and reward/error overlap."""
    key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY") or os.environ["SUPABASE_ANON_KEY"]
    query = urllib.parse.urlencode({"id": f"eq.{job_id}", "select": "id,metrics,stats,config"})
    request = urllib.request.Request(
        f"{os.environ['SUPABASE_URL']}/rest/v1/sandbox_jobs?{query}",
        headers={"apikey": key, "Authorization": f"Bearer {key}"},
    )
    with urllib.request.urlopen(request) as response:
        rows = json.load(response)
    [row] = rows
    metrics = row.get("metrics") or []
    accuracy = next((entry.get("value") for entry in metrics if entry.get("name") == "accuracy"), None)
    evaluation = next(iter((row.get("stats") or {}).get("evals", {}).values()))
    errors = {trial for trials in evaluation.get("exception_stats", {}).values() for trial in trials}
    rewards = evaluation.get("reward_stats", {}).get("reward", {})
    reward_one = set(rewards.get("1.0", []))
    reward_by_trial = {
        trial: float(reward) for reward, trials in rewards.items() for trial in trials if _number(reward) is not None
    }
    error_reward = sum(reward_by_trial.get(trial, 0) for trial in errors)
    no_error_trials = reward_by_trial.keys() - errors
    no_error_reward = sum(reward_by_trial[trial] for trial in no_error_trials)
    config = row.get("config") or {}
    agent = (config.get("agents") or [{}])[0]
    kwargs = agent.get("kwargs") or {}
    model_info = kwargs.get("model_info") or {}
    return {
        "job_id": job_id,
        "accuracy": accuracy,
        "n_trials": evaluation.get("n_trials"),
        "n_errors": evaluation.get("n_errors"),
        "reward_one": len(reward_one),
        "error_and_reward_one": len(errors & reward_one),
        "no_error_and_reward_one": len(reward_one - errors),
        "mean_reward_with_error": error_reward / len(errors) if errors else None,
        "mean_reward_without_error": no_error_reward / len(no_error_trials) if no_error_trials else None,
        "input_tokens": (row.get("stats") or {}).get("n_input_tokens"),
        "output_tokens": (row.get("stats") or {}).get("n_output_tokens"),
        "config": {
            "parser": kwargs.get("parser"),
            "max_input_tokens": model_info.get("max_input_tokens"),
            "max_output_tokens": model_info.get("max_output_tokens"),
            "timeout_multiplier": config.get("timeout_multiplier"),
            "n_concurrent_trials": config.get("n_concurrent_trials"),
            "enable_thinking": (
                ((kwargs.get("extra_body") or {}).get("chat_template_kwargs") or {}).get("enable_thinking")
            ),
        },
    }


def load_local_config(root: Path) -> dict[str, Any]:
    """Read the effective agent configuration recorded by one local trial."""
    result_path = next(iter(sorted(root.glob("*/result.json"))))
    result = json.loads(result_path.read_text())
    config = result.get("config") or {}
    agent = config.get("agent") or {}
    kwargs = agent.get("kwargs") or {}
    model_info = kwargs.get("model_info") or {}
    trajectory = json.loads((result_path.parent / "agent" / "trajectory.json").read_text())
    return {
        "parser": ((trajectory.get("agent") or {}).get("extra") or {}).get("parser"),
        "max_input_tokens": model_info.get("max_input_tokens"),
        "max_output_tokens": model_info.get("max_output_tokens"),
        "timeout_multiplier": config.get("timeout_multiplier"),
        "enable_thinking": ((kwargs.get("extra_body") or {}).get("chat_template_kwargs") or {}).get("enable_thinking"),
        "agent_version": (result.get("agent_info") or {}).get("version"),
    }


def config_mismatches(historical: dict[str, Any], current: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Return semantic fields whose effective historical/current values differ."""
    return {
        key: {"historical": historical.get(key), "current": current.get(key)}
        for key in sorted(historical.keys() & current.keys())
        if historical.get(key) != current.get(key)
    }


def timeout_counterfactual(historical: dict[str, Any], current: dict[str, Any]) -> dict[str, float] | None:
    """Hold historical conditional rewards fixed and substitute the current timeout rate."""
    current_timeout_rate = _number(current.get("agent_timeout_rate"))
    reward_without_error = _number(historical.get("mean_reward_without_error"))
    reward_with_error = _number(historical.get("mean_reward_with_error"))
    historical_score = _number(historical.get("accuracy"))
    current_score = _number(current.get("mean_reward_all_trials"))
    if current_timeout_rate is None:
        return None
    if reward_without_error is None or reward_with_error is None:
        return None
    if historical_score is None or current_score is None:
        return None
    predicted = (1 - current_timeout_rate) * reward_without_error + current_timeout_rate * reward_with_error
    actual_drop = historical_score - current_score
    return {
        "predicted_score_at_current_timeout_rate": predicted,
        "predicted_drop_from_historical": historical_score - predicted,
        "actual_drop_from_historical": actual_drop,
        "fraction_of_actual_drop_predicted": (historical_score - predicted) / actual_drop if actual_drop else 0,
    }


def load_inference_metrics(job_id: str) -> dict[str, Any]:
    """Query durable finelog counters and saturation gauges for one vLLM job."""
    escaped = job_id.replace("'", "''")
    counter_sql = f"""
WITH peaks AS (
 SELECT name, worker, attempt, json_get(labels, 'engine') AS engine, MAX(value) AS value
 FROM telltale WHERE job_id = '{escaped}' AND name IN (
 'vllm:prompt_tokens_total','vllm:generation_tokens_total',
 'vllm:e2e_request_latency_seconds_count','vllm:e2e_request_latency_seconds_sum',
 'vllm:request_queue_time_seconds_sum','vllm:request_prefill_time_seconds_sum',
 'vllm:request_decode_time_seconds_sum',
 'vllm:num_preemptions_total') GROUP BY 1,2,3,4
)
SELECT name, SUM(value) AS value FROM peaks GROUP BY 1
"""
    gauge_sql = f"""
SELECT name, MAX(value) AS peak, AVG(value) AS average FROM telltale
WHERE job_id = '{escaped}' AND name IN (
'vllm:num_requests_running','vllm:num_requests_waiting','vllm:kv_cache_usage_perc') GROUP BY 1
"""
    counter_rows = _finelog_rows(counter_sql)
    gauge_rows = _finelog_rows(gauge_sql)
    counters = {row["name"]: row["value"] for row in counter_rows}
    gauges = {row["name"]: {"peak": row["peak"], "average": row["average"]} for row in gauge_rows}
    count = counters.get("vllm:e2e_request_latency_seconds_count")
    total = counters.get("vllm:e2e_request_latency_seconds_sum")
    decode = counters.get("vllm:request_decode_time_seconds_sum")
    generation = counters.get("vllm:generation_tokens_total")
    return {
        "job_id": job_id,
        "counters": counters,
        "gauges": gauges,
        "mean_e2e_request_seconds": total / count if total and count else None,
        "decode_fraction_of_e2e": decode / total if decode and total else None,
        "decode_tokens_per_request_second": generation / decode if generation and decode else None,
    }


def _finelog_rows(sql: str) -> list[dict[str, Any]]:
    process = subprocess.run(
        ["uv", "run", "finelog", "query", "marin", sql, "--format", "jsonl", "--max-rows", "100"],
        check=True,
        capture_output=True,
        text=True,
    )
    return [json.loads(line) for line in process.stdout.splitlines() if line.startswith("{")]


def render_regressions(before: list[Trace], after: list[Trace], output: Path, top_n: int = 12) -> None:
    """Render representative before/after conversations for regressed tasks."""
    grouped_before: dict[str, list[Trace]] = defaultdict(list)
    grouped_after: dict[str, list[Trace]] = defaultdict(list)
    for trace in before:
        grouped_before[trace.task].append(trace)
    for trace in after:
        grouped_after[trace.task].append(trace)
    regressions = []
    for task in grouped_before.keys() & grouped_after.keys():
        after_reward = statistics.fmean(t.reward or 0 for t in grouped_after[task])
        before_reward = statistics.fmean(t.reward or 0 for t in grouped_before[task])
        delta = after_reward - before_reward
        if delta < 0:
            regressions.append((delta, task))
    sections = []
    for delta, task in sorted(regressions)[:top_n]:
        left = max(grouped_before[task], key=lambda trace: trace.reward or 0)
        right = min(grouped_after[task], key=lambda trace: trace.reward or 0)
        columns = []
        for label, trace in (("historical", left), ("current", right)):
            messages = "".join(
                f"<details><summary>{html.escape(str(message.get('role')))}</summary><pre>{html.escape(_content(message))}</pre></details>"
                for message in trace.messages
            )
            columns.append(f"<section><h3>{label}: reward={trace.reward}, error={trace.error}</h3>{messages}</section>")
        sections.append(f"<h2>{html.escape(task)} (delta {delta:.3f})</h2><div class='pair'>{''.join(columns)}</div>")
    output.write_text(
        "<html><style>body{font:14px sans-serif}.pair{display:grid;grid-template-columns:1fr 1fr;gap:1rem}"
        "section{min-width:0}pre{white-space:pre-wrap;overflow-wrap:anywhere;max-height:36rem;overflow:auto}"
        "summary{font-weight:bold;cursor:pointer}</style><body>" + "".join(sections) + "</body></html>"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--historical", required=True, help="Hugging Face dataset repository")
    parser.add_argument("--current", required=True, type=Path, help="Directory containing Harbor trial directories")
    parser.add_argument("--output", required=True, type=Path, help="Output JSON path")
    parser.add_argument("--html", type=Path, help="Optional representative-regressions HTML path")
    parser.add_argument("--trace-source", default="main")
    parser.add_argument("--historical-job-id", help="Supabase sandbox_jobs id for authoritative historical metrics")
    parser.add_argument("--inference-job-id", help="Iris job id whose durable vLLM metrics should be queried")
    parser.add_argument("--max-output-tokens", type=int, help="Configured per-request output-token limit")
    args = parser.parse_args()
    before = load_historical(args.historical, args.trace_source)
    after = load_local(args.current)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = compare(before, after, args.max_output_tokens)
    if args.historical_job_id:
        report["historical_registry"] = load_registry_job(args.historical_job_id)
        report["current_config"] = load_local_config(args.current)
        report["config_mismatches"] = config_mismatches(
            report["historical_registry"]["config"], report["current_config"]
        )
        report["observed_parser_mismatch"] = {
            "historical": report["historical"]["observed_message_format"]["dominant_parser"],
            "current": report["current"]["observed_message_format"]["dominant_parser"],
        }
        report["timeout_counterfactual"] = timeout_counterfactual(report["historical_registry"], report["current"])
    if args.inference_job_id:
        report["current_inference"] = load_inference_metrics(args.inference_job_id)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    if args.html:
        render_regressions(before, after, args.html)


if __name__ == "__main__":
    main()
