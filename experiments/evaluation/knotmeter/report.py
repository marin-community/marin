# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Render one campaign report for the terminal and durable summary.md."""

import json
from urllib.parse import quote

import yaml
from rigging.filesystem.buckets import filesystem_for
from rigging.filesystem.storage_path import prefix_join

from experiments.evaluation.knotmeter.artifacts import read_record, write_record
from experiments.evaluation.knotmeter.config import is_confirm_recipe
from experiments.evaluation.knotmeter.recipe import unmeasured_result

QUALIFICATION_SCOPE = "Serving checks do not qualify RL learning or trainer probability ratios."


def evidence_link(uri: str) -> str:
    link = f"[{uri}]({uri})"
    if uri.startswith(("s3://", "gs://")):
        return link + f" ([Filer](https://filer.marina.oa.dev/#{quote(uri, safe='')}))"
    return link


def write_text(uri: str, content: str) -> None:
    # Use the same bucket routing as metadata writes.
    filesystem, path = filesystem_for(uri)
    parent = path.rsplit("/", 1)[0]
    if filesystem.protocol == "file" or filesystem.protocol == ("file", "local"):
        filesystem.makedirs(parent, exist_ok=True)
    with filesystem.open(path, "wt") as stream:
        stream.write(content)


def recorded_checks(result: dict) -> list[dict]:
    """Read the current leader's checks even when no AgentX window started."""
    if result["measurement"] is not None:
        return result["measurement"]["checks"]
    leaders = [attempt for attempt in result["attempts"] if attempt["task_id"].endswith("/0")]
    current = max(leaders, key=lambda attempt: attempt["attempt_number"], default=None)
    return current.get("worker", {}).get("checks", []) if current else []


def comparison_controls(result: dict) -> dict:
    manifest = result["manifest"]
    series = manifest["series"]
    return {
        key: series[key]
        for key in (
            "model",
            "model_revision",
            "tokenizer",
            "tokenizer_revision",
            "harness_version",
            "harness_source",
            "processor_source",
            "dataset_loader",
            "dataset_revision",
            "dataset_sha256",
            "duration",
            "stage",
            "warmup_requests_per_lane",
            "unsafe_override",
            "max_context",
            "random_seed",
            "warmup_grace_period",
            "trace_idle_gap_cap_seconds",
            "max_profiling_error_ratio",
        )
    } | {
        "preparation_recipe": manifest["workload"]["recipe"],
        "gpu_type": series["profile"]["gpus"]["type"],
        "numerical_policies": [
            {"name": check["name"], "reference": check.get("reference")}
            | (
                {"criteria_sha256": check["criteria_sha256"]}
                if check.get("criteria_sha256")
                else (
                    {key: check.get(key) for key in ("policy", "policy_sha256")}
                    if check.get("policy_sha256")
                    else {key: check.get(key) for key in ("command", "evidence")}
                )
            )
            for check in recorded_checks(result)
            if check.get("kind") == "numerical"
        ],
    }


def accepted(result: dict) -> bool:
    """Return whether measurement and scoped checks qualify this trial."""
    measurement = result["measurement"]
    if not measurement or not result["job_succeeded"]:
        return False
    checks = recorded_checks(result)
    numerical = [check for check in checks if check.get("kind") == "numerical"]
    manifest = result["manifest"]
    validity = measurement["validity"]
    required = manifest["series"]["profile"]["checks"]
    return (
        not manifest["diagnostic"]
        and is_confirm_recipe(manifest["series"])
        and validity["protocol_valid"]
        and validity["submission_valid"]
        and validity["gang_valid"]
        and sorted((check["name"], check["kind"]) for check in checks)
        == sorted((check["name"], check["kind"]) for check in required)
        and bool(numerical)
        and all(
            check.get("checkpoint_revision") == manifest["series"]["profile"]["model"]["revision"]
            and check.get("runtime_source") == manifest["vllm_release"]["source_commit"]
            for check in numerical
        )
        and all(check["status"] == "passed" for check in checks)
    )


def format_value(value: float | None) -> str:
    return "unknown" if value is None else f"{value:.3f}"


def render_report(
    results: list[dict], campaign_uri: str, recommendation: str | None = None, baseline: str | None = None
) -> str:
    reference = next((row for row in results if row["trial_id"] == baseline), None)
    if baseline and reference is None:
        raise ValueError(f"No baseline trial {baseline!r} in this campaign")
    lines = [
        "# Inference performance campaign",
        "",
        f"Evidence: {evidence_link(campaign_uri)}",
        "",
        "Throughput counts completed output tokens. Latencies are seconds. Each trial has one measured window; "
        "repeatability has not been measured.",
        "",
        "| Trial | Stage | Runtime | Weights | GPU job | Sessions | GPUs | Output tokens/s/GPU | Change | p90 TTFT | "
        "p90 completion | AgentX | Errors by phase | Checks | GPU-hours |",
        "| --- | --- | --- | --- | --- | ---: | --- | ---: | ---: | ---: | ---: | --- | --- | --- | ---: |",
    ]
    for result in results:
        profile = result["manifest"]["series"]["profile"]
        commit = result["manifest"]["vllm_release"]["source_commit"]
        repository = (profile["vllm"].get("repo") or "https://github.com/marin-community/vllm").removesuffix(".git")
        runtime = (
            f"[{commit[:10]}]({repository}/commit/{commit})"
            if repository.startswith("https://github.com/")
            else commit[:10]
        )
        measurement = result["measurement"]
        metrics = measurement["aggregate"]["request_metrics"] if measurement else {}
        rate = metrics.get("throughput", {}).get("per_gpu", {}).get("output_tput_tps")
        latency = metrics.get("latency", {})
        change = throughput_change(result, reference)
        change_text = f"{change * 100:+.2f}%" if change is not None else "—"
        check_records = recorded_checks(result)
        checks = (
            ", ".join(
                f"{check['name']}: {check['status']}"
                + (" (provisional policy)" if check.get("provisional") else "")
                + (f" (regression {check['regression_status']})" if "regression_status" in check else "")
                for check in check_records
            )
            if check_records
            else ", ".join(f"{check['name']}: pending" for check in profile["checks"]) or "not run"
        )
        agentx = "no window"
        errors = "unknown"
        if measurement:
            validity = measurement["validity"]
            agentx = "native " + ("valid" if validity["submission_valid"] else "invalid")
            agentx += "; protocol " + ("valid" if validity["protocol_valid"] else "invalid")
            counts = validity.get("request_counts_by_phase", {})
            errors = "; ".join(f"{phase}: {counts[phase]['errors']}" for phase in sorted(counts))
            if not errors:
                errors = "profiling: " + str(validity.get("profiling_errors", "unknown"))
        if result["manifest"]["diagnostic"]:
            agentx += "; diagnostic"
        elif result["manifest"]["series"].get("measure_failed_checks") and any(
            check["status"] != "passed" for check in check_records
        ):
            agentx += "; diagnostic performance (failed checks)"
        cost = format_value(result["gpu_seconds"] / 3600) + (" + unknown" if not result["cost_complete"] else "")
        if not result["cost_complete"] and not result["gpu_seconds"]:
            cost = "unknown"
        if any(
            attempt.get("allocation", {}).get("end_basis") == "iris-observed-terminal" for attempt in result["attempts"]
        ):
            cost += " (estimated)"
        record_file = "plan.json" if result["job_state"] == "unrecorded" else "result.json"
        record_uri = prefix_join(result["trial_uri"], record_file)
        weights = profile.get("weight_precision") or profile.get("quantization") or profile["serving"]["dtype"]
        lines.append(
            f"| [{result['trial_id']}]({record_uri}) | "
            f"{result['manifest']['series']['stage']} | {runtime} | {weights} | "
            f"{result['job_state']} | "
            f"{result['manifest']['series']['concurrency']} | {profile['gpus']['count']} {profile['gpus']['type']} | "
            f"{format_value(rate)} | {change_text} | "
            f"{format_value(latency.get('ttft', {}).get('p90'))} | "
            f"{format_value(latency.get('e2el', {}).get('p90'))} | {agentx} | {errors} | {checks} | {cost} |"
        )
    lines += [
        "",
        f"Total recorded GPU-hours: {sum(row['gpu_seconds'] for row in results) / 3600:.3f}.",
        "Cost includes every retained Iris attempt, including setup, warmup, drain and failed attempts. "
        "Iris-observed end times are estimates. Missing allocation timestamps remain unknown; queueing is excluded.",
        "",
    ]
    if any(not row["cost_complete"] for row in results):
        lines.append("Total cost is incomplete: a trial lacks a result or an attempt lacks allocation times.")
    groups = {json.dumps(comparison_controls(row), sort_keys=True) for row in results}
    if len(groups) > 1:
        lines += [
            "",
            "Trials with different controls form separate comparison groups. " "Their throughput changes are omitted.",
        ]
    eligible = [row["trial_id"] for row in results if accepted(row)]
    lines += ["", "Eligible for recommendation: " + (", ".join(eligible) if eligible else "none") + "."]
    if recommendation:
        chosen = next((row for row in results if row["trial_id"] == recommendation), None)
        if chosen is None:
            raise ValueError(f"No recommendation trial {recommendation!r} in this campaign")
        if not accepted(chosen):
            raise ValueError(
                "Recommendation requires a successful GPU job, "
                "a valid Confirm window, unchanged gang and passed required checks"
            )
        lines += [
            "",
            f"Recommended trial: {recommendation}. Settings and runtime: "
            f"{evidence_link(prefix_join(campaign_uri, 'recommended.yaml'))}.",
        ]
    else:
        lines += ["", "No accepted recommendation selected."]
    lines += ["", QUALIFICATION_SCOPE]
    return "\n".join(lines) + "\n"


def throughput_change(result: dict, baseline: dict | None) -> float | None:
    if baseline is None or not baseline["measurement"] or not result["measurement"]:
        return None
    if comparison_controls(result) != comparison_controls(baseline):
        return None
    value = result["measurement"]["aggregate"]["request_metrics"]["throughput"]["per_gpu"]["output_tput_tps"]
    reference = baseline["measurement"]["aggregate"]["request_metrics"]["throughput"]["per_gpu"]["output_tput_tps"]
    return value / reference - 1 if reference else None


def report_campaign(campaign_uri: str, recommendation: str | None = None, baseline: str | None = None) -> str:
    requested_recommendation = recommendation
    filesystem, path = filesystem_for(prefix_join(campaign_uri, "trials/*/result.json"))
    results = [read_record(filesystem.unstrip_protocol(item)) for item in sorted(filesystem.glob(path))]
    recorded = {row["trial_id"] for row in results}
    _, plan_glob = filesystem_for(prefix_join(campaign_uri, "trials/*/plan.json"))
    for item in sorted(filesystem.glob(plan_glob)):
        plan = read_record(filesystem.unstrip_protocol(item))
        if plan["trial_id"] in recorded:
            continue
        results.append(unmeasured_result(plan, "unrecorded"))
    results.sort(key=lambda row: row["trial_id"])
    if recommendation is None:
        try:
            decision = read_record(prefix_join(campaign_uri, "decision.json"))
        except FileNotFoundError:
            pass
        else:
            recommendation = decision["recommended_trial"]
            if baseline is None:
                baseline = decision["baseline_trial"]
    report = render_report(results, campaign_uri, recommendation, baseline)
    if requested_recommendation:
        selected = next(row for row in results if row["trial_id"] == recommendation)
        reference = next((row for row in results if row["trial_id"] == baseline), None)
        if reference and (not accepted(reference) or comparison_controls(selected) != comparison_controls(reference)):
            raise ValueError(
                "A baseline comparison requires a qualified baseline with matched model and workload controls"
            )
        series = selected["manifest"]["series"]
        profile = series["profile"]
        runtime = prefix_join(selected["trial_uri"], "runtime.json")
        write_record(runtime, selected["manifest"]["vllm_release"])
        profile = profile | {
            "session_concurrency": series["concurrency"],
            "vllm": profile["vllm"] | {"release": None, "manifest": runtime},
        }
        policies = [check for check in selected["measurement"]["checks"] if check.get("kind") == "numerical"]
        provisional = any(check.get("provisional") for check in policies)
        prefix = "# Numerical qualification uses a provisional policy. See decision.json.\n" if provisional else ""
        prefix += f"# {QUALIFICATION_SCOPE}\n"
        write_text(prefix_join(campaign_uri, "recommended.yaml"), prefix + yaml.safe_dump(profile, sort_keys=False))
        write_record(
            prefix_join(campaign_uri, "decision.json"),
            {
                "schema_version": 1,
                "recommended_trial": recommendation,
                "baseline_trial": baseline,
                "throughput_change": throughput_change(selected, reference),
                "controls": comparison_controls(selected),
                "provisional": provisional,
                "numerical_checks": policies,
            },
        )
    write_text(prefix_join(campaign_uri, "summary.md"), report)
    return report
