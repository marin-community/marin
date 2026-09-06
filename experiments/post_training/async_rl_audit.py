# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.12,<3.14"
# dependencies = [
#   "wandb==0.26.0", "s3fs==2026.1.0", "hydra-core==1.4.0.dev1",
#   "omegaconf==2.4.0.dev15", "PyYAML==6.0.3",
# ]
# ///
"""Bounded terminal audit. Run on CPU in the artifacts' own region.

Audits W&B and durable results only; verify Finelog independently.
Usage: uv run --locked --script experiments/post_training/async_rl_audit.py --spec spec.json
Run in the artifacts' region with inherited credentials. No training runtime is needed.
The spec contains one to eight runs with label, run_id, attempt_id, receipt_uri,
envelope or envelope_uri, wandb_url, expected_steps, expected_eval_steps, and
expected_eval_rows. Optional expected_request/expected_config maps lock provenance.
require_launcher_success defaults true; false permits verified training evidence
from a failed launcher without calling it clean end-to-end completion.
initial_eval_repeat_count verifies startup_pass_N dumps and W&B namespaces.
Optional locked_validation contains manifest_uri, dataset_revision, offset and
count; its source-row window is verified before reading evaluation dumps.
Alternatively set ASYNC_RL_AUDIT_SPEC to the JSON specification.
No training runtime, credentials, prompt text, or token arrays are emitted.
"""
import argparse
import collections
import datetime
import hashlib
import json
import math
import os
import posixpath
from collections.abc import Iterator, Mapping
from typing import Any, Protocol
from urllib.parse import urlparse

import fsspec
import wandb
import yaml
from hydra.core.override_parser.overrides_parser import OverridesParser

JSONDict = dict[str, Any]
EvalRecords = list[list[Any]]


class HistoryRun(Protocol):
    url: str
    name: str
    state: str
    config: Mapping[str, Any]

    def scan_history(self, *, page_size: int) -> Iterator[JSONDict]: ...


class RunAPI(Protocol):
    def run(self, path: str) -> HistoryRun: ...


MAX_JSON_BYTES = 1024 * 1024
MAX_EVAL_LINE_BYTES = 1024 * 1024
MAX_EVAL_BYTES = 128 * 1024 * 1024
MAX_HISTORY_ROWS = 10000
REQUIRED = (
    "policy/final_loss",
    "policy/policy_loss",
    "reward/avg_raw_reward",
    "policy/behavior_drift/finite_fraction",
    "policy/behavior_drift/token_weight_ess_fraction",
    "policy/behavior_drift/abs_log_ratio_p99",
    "policy/behavior_drift/missing_behavior",
    "consumed/sequences",
)
PREFIXES = ("policy/", "reward/", "consumed/", "async/", "timing/", "tis/")


def check(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def canonical_sha(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def fs_path(uri: str) -> tuple[fsspec.AbstractFileSystem, str]:
    options = {"config_kwargs": {"s3": {"addressing_style": "virtual"}}} if uri.startswith("s3://") else {}
    fs, _, paths = fsspec.get_fs_token_paths(uri, storage_options=options)
    return fs, paths[0]


def read_json(uri: str) -> JSONDict:
    fs, path = fs_path(uri)
    with fs.open(path, "rb") as source:
        data = source.read(MAX_JSON_BYTES + 1)
    check(len(data) <= MAX_JSON_BYTES, f"JSON exceeds audit bound: {uri}")
    value = json.loads(data)
    check(isinstance(value, dict), f"Expected JSON object: {uri}")
    return value


def bounded_lines(source: Any) -> Iterator[bytes]:
    """Bound allocations without relying on fsspec's unsupported readline(size)."""
    pending = b""
    byte_count = 0
    while chunk := source.read(65536):
        byte_count += len(chunk)
        check(byte_count <= MAX_EVAL_BYTES, "Evaluation dump exceeds audit bounds")
        lines = (pending + chunk).split(b"\n")
        pending = lines.pop()
        for line in lines:
            check(len(line) + 1 <= MAX_EVAL_LINE_BYTES, "Evaluation dump exceeds audit bounds")
            yield line + b"\n"
        check(len(pending) <= MAX_EVAL_LINE_BYTES, "Evaluation dump exceeds audit bounds")
    if pending:
        yield pending


def list_entries(uri: str) -> list[JSONDict]:
    fs, path = fs_path(uri)
    if not fs.exists(path):
        return []
    entries = fs.ls(path, detail=True)
    check(len(entries) <= 256, f"Directory listing exceeds audit bound: {uri}")
    return entries


def lookup(config: Mapping[str, Any], path: str) -> Any:
    # W&B may expose nested mappings or flattened paths. Never assume ['trainer'].
    if path in config:
        return config[path]
    if path.replace(".", "/") in config:
        return config[path.replace(".", "/")]
    value = config
    for part in path.split("."):
        if not isinstance(value, dict) or part not in value:
            return None
        value = value[part]
    return value


def wandb_identity(url: str) -> tuple[str, str, str]:
    parsed = urlparse(url)
    check(parsed.hostname in {"wandb.ai", "www.wandb.ai"}, "Supply the actual emitted wandb.ai run URL")
    parts = parsed.path.strip("/").split("/")
    check(len(parts) == 4 and parts[2] == "runs", "W&B URL must identify entity/project/runs/id")
    return parts[0], parts[1], parts[3]


def canonical_entrypoint(value: str) -> str:
    aliases = {
        "fully_async": "fully_async",
        "skyrl_train.entrypoints.fully_async": "fully_async",
        "standard": "standard",
        "skyrl_train.entrypoints.main_base": "standard",
    }
    check(value in aliases, f"Unsupported training entrypoint: {value}")
    return aliases[value]


def parse_hydra_args(arguments: list[str]) -> JSONDict:
    """Use the launcher's override grammar, retaining quoted strings and nulls."""
    return {arg.key_or_group: arg.value() for arg in OverridesParser.create().parse_overrides(arguments)}


def summarize_history(
    run: HistoryRun, expected_steps: int, entrypoint: str, required_metrics: list[str]
) -> tuple[JSONDict, dict[int, JSONDict]]:
    entrypoint = canonical_entrypoint(entrypoint)
    combined = {}
    evaluations = {}
    row_count = 0
    for row in run.scan_history(page_size=100):
        row_count += 1
        check(row_count <= MAX_HISTORY_ROWS, "W&B history exceeds audit bound")
        step = row.get("global_step", row.get("trainer/global_step"))
        if step is None:
            continue
        check(isinstance(step, (int, float)) and math.isfinite(step) and int(step) == step, "Invalid global_step")
        step = int(step)
        values = {
            key: value for key, value in row.items() if key.startswith(PREFIXES) and isinstance(value, (int, float))
        }
        check(all(math.isfinite(value) for value in values.values()), f"Nonfinite metric at optimizer step {step}")
        check(
            not any(
                key.startswith(PREFIXES)
                and isinstance(value, str)
                and value.lower() in {"nan", "inf", "-inf", "infinity", "-infinity"}
                for key, value in row.items()
            ),
            f"String-encoded nonfinite metric at step {step}",
        )
        previous = combined.setdefault(step, {})
        for key, value in values.items():
            check(key not in previous or previous[key] == value, f"Conflicting W&B metric {key} at step {step}")
            previous[key] = value
        metrics = {
            key: value for key, value in row.items() if key.startswith("eval/") and isinstance(value, (int, float))
        }
        if metrics:
            check(all(math.isfinite(value) for value in metrics.values()), f"Nonfinite evaluation at step {step}")
            check(step not in evaluations, f"Duplicate evaluation event at step {step}")
            evaluations[step] = metrics
    updates = {step: row for step, row in combined.items() if "policy/final_loss" in row or "policy/policy_loss" in row}
    check(
        sorted(updates) == list(range(1, expected_steps + 1)),
        "W&B optimizer-step coverage is incomplete or exceeds expectation",
    )
    required = set(REQUIRED) | set(required_metrics)
    if entrypoint == "fully_async":
        required |= {"async/performance/core_seconds", "async/staleness_max"}
    for step, row in updates.items():
        check(required <= row.keys(), f"Missing metrics at step {step}: {sorted(required - row.keys())}")
        check(row["policy/behavior_drift/finite_fraction"] == 1, f"Nonfinite behavior probabilities at step {step}")
        check(row["policy/behavior_drift/missing_behavior"] == 0, f"Missing behavior probabilities at step {step}")
        check(
            all(value == 0 for key, value in row.items() if key.endswith("diagnostics_failed")),
            f"Diagnostic computation failed at step {step}",
        )
        check(0 < row["policy/behavior_drift/token_weight_ess_fraction"] <= 1.0000001, f"Invalid ESS at step {step}")
    ranges = {}
    for key in sorted({key for row in updates.values() for key in row}):
        vals = [row[key] for _, row in sorted(updates.items()) if key in row]
        ranges[key] = {"count": len(vals), "min": min(vals), "max": max(vals), "last": vals[-1]}
    sums = {}
    for key in (
        "consumed/sequences",
        "consumed/length_stop_count",
        "consumed/unknown_stop_count",
        "async/performance/core_seconds",
        "async/performance/cycle_seconds",
        "async/performance/consumed_response_tokens",
        "async/performance/consumed_loss_tokens",
    ):
        if all(key in row for row in updates.values()):
            sums[key] = sum(row[key] for row in updates.values())
    return {"history_rows": row_count, "steps": sorted(updates), "ranges": ranges, "sums": sums}, evaluations


def verify_no_model_files(export_root: str) -> int:
    # Evaluation JSONLs under dumped_evals are allowed. Do not recursively capture them.
    pending = [(export_root, 0)]
    inspected = 0
    while pending:
        uri, depth = pending.pop()
        for entry in list_entries(uri):
            inspected += 1
            check(inspected <= 256, "Export inventory exceeds audit bound")
            name = posixpath.basename(entry["name"].rstrip("/"))
            if depth == 0 and name == "dumped_evals":
                continue
            check(
                not name.endswith((".safetensors", ".bin", ".pt", ".distcp")),
                "Model/checkpoint bytes found in metrics export root",
            )
            check(
                name not in {"config.json", "tokenizer.json", "tokenizer_config.json"},
                "HF export metadata found in metrics export root",
            )
            if entry["type"] == "directory":
                check(depth < 4, "Export inventory depth exceeds audit bound")
                pending.append((posixpath.join(uri, name), depth + 1))
    return inspected


def audit_storage(spec: JSONDict) -> tuple[JSONDict, JSONDict, JSONDict]:
    check(("envelope" in spec) != ("envelope_uri" in spec), "Supply either envelope object or envelope_uri")
    envelope = spec["envelope"] if "envelope" in spec else read_json(spec["envelope_uri"])
    check(
        envelope.get("schema_version") == 2 and isinstance(envelope.get("request"), dict),
        "Expected protocol-2 launch/terminal envelope",
    )
    locator_request = envelope["request"]
    check(locator_request["run_id"] == spec["run_id"], "Envelope run differs from expected run")
    # Read the actual attempt first: failed launchers retain valid training proof
    # there and deliberately do not publish a successful terminal manifest.
    attempt_uri = posixpath.join(locator_request["output"]["attempts_root"], spec["attempt_id"] + ".json")
    attempt = read_json(attempt_uri)
    check(attempt.get("schema_version") == 2, "Invalid attempt protocol schema")
    request = attempt["request"]
    check(
        request["run_id"] == spec["run_id"] and request["attempt_id"] == spec["attempt_id"],
        "Actual attempt identity differs from expected run/attempt",
    )
    check(
        {key: value for key, value in request.items() if key != "attempt_id"}
        == {key: value for key, value in locator_request.items() if key != "attempt_id"},
        "Actual request differs from supplied envelope beyond regenerated attempt_id",
    )
    for key, expected in spec.get("expected_request", {}).items():
        check(lookup(request, key) == expected, f"Actual request differs from declared provenance: {key}")
    check(request["completion_mode"] == "metrics", "Audit requires metrics completion")
    output = request["output"]
    receipt_uri = posixpath.join(
        posixpath.dirname(output["terminal_manifest_uri"]), "receipts", request["attempt_id"] + ".json"
    )
    check(receipt_uri == spec["receipt_uri"], "Receipt URI differs from request-derived URI")
    receipt = read_json(receipt_uri)
    digest = canonical_sha({"schema_version": attempt["schema_version"], "request": request, "receipt_uri": receipt_uri})
    expected_receipt = {
        "schema_version": 1,
        "run_id": spec["run_id"],
        "attempt_id": spec["attempt_id"],
        "request_fingerprint": digest,
        "completion_mode": "metrics",
        "global_step": spec["expected_steps"],
    }
    check(receipt == expected_receipt, "Receipt step/identity/SHA/completion does not match request exactly")
    response = attempt["response"]
    check(response["state"] in {"succeeded", "failed"}, "Attempt is not terminal")
    if spec.get("require_launcher_success", True):
        check(response["state"] == "succeeded" and response["failure"] is None, "Launcher did not succeed")
    check(
        response["run_id"] == spec["run_id"] and response["attempt_id"] == spec["attempt_id"],
        "Launcher identity mismatch",
    )
    check(response["runtime"] == request["runtime"], "Launcher runtime differs from request")
    training = response["training"]
    check(isinstance(training, dict), "Attempt has no independently verifiable training proof")
    check(
        training["global_step"] == spec["expected_steps"] and training["checkpoint"] is None,
        "Launcher result is not expected metrics completion",
    )
    check(
        training["receipt_uri"] == receipt_uri and training["resolved_config_uri"] == output["resolved_config_uri"],
        "Launcher proof locators differ",
    )
    fs, terminal_path = fs_path(output["terminal_manifest_uri"])
    terminal_present = fs.exists(terminal_path)
    clean = response["state"] == "succeeded"
    if clean:
        check(response["failure"] is None, "Successful launcher response contains a failure")
        check(response["iris_job_state"] == "succeeded", "Successful launcher has no recorded Iris success")
        check(terminal_present, "Successful attempt has no terminal manifest")
        terminal = read_json(output["terminal_manifest_uri"])
        check(
            terminal.get("schema_version") == 2 and terminal.get("request") == request,
            "Terminal request/schema differs from actual attempt",
        )
        check(terminal["response"] == response, "Terminal/attempt responses differ")
    else:
        check(
            isinstance(response["failure"], str) and response["failure"], "Failed launcher response omitted its failure"
        )
        check(not terminal_present, "Failed attempt has a terminal manifest; cannot classify as training evidence only")
    resolved = read_json(output["resolved_config_uri"])
    args = parse_hydra_args(resolved["hydra_args"])
    completion = {
        key.removeprefix("trainer.completion."): value
        for key, value in args.items()
        if key.startswith("trainer.completion.")
    }
    check(
        completion
        == {
            "mode": "metrics",
            "run_id": spec["run_id"],
            "attempt_id": spec["attempt_id"],
            "request_fingerprint": digest,
            "receipt_uri": receipt_uri,
        },
        "Resolved completion metadata differs",
    )
    check(
        args.get("trainer.ckpt_interval") == -1 and args.get("trainer.hf_save_interval") == -1,
        "Resolved saving must be disabled",
    )
    check(not list_entries(output["checkpoint_root"]), "Metrics-only run left checkpoint objects")
    inspected = verify_no_model_files(output["export_root"])
    return (
        request,
        resolved,
        {
            "receipt_verified": True,
            "request_fingerprint": digest,
            "actual_attempt_verified": True,
            "terminal_manifest_present": terminal_present,
            "terminal_attempt_verified": clean,
            "clean_end_to_end": clean,
            "launcher": {key: response[key] for key in ("state", "iris_job_state", "failure", "iris_job_id")},
            "iris_job_id": response["iris_job_id"],
            "checkpoint_absent": True,
            "no_model_export": True,
            "non_eval_export_entries_checked": inspected,
            "runtime": request["runtime"],
            "resolved_entrypoint": resolved["entrypoint"],
            "envelope_attempt_id": locator_request["attempt_id"],
            "actual_attempt_id": request["attempt_id"],
            "envelope_matches_actual_except_attempt": True,
        },
    )


def audit_locked_validation(spec: JSONDict, request: JSONDict) -> JSONDict | None:
    """Verify a declared source window before accessing W&B or evaluation rows."""
    window = spec.get("locked_validation")
    if window is None:
        return None
    offset, count = window["offset"], window["count"]
    check(
        type(offset) is int and type(count) is int and offset >= 128 and count > 0 and offset + count <= 1319,
        "Locked validation must exclude test[0:128] and fit within test[128:1319]",
    )
    data = request["validation_data"]
    check(len(data) == 1, "Locked validation requires exactly one declared dataset artifact")
    check(data[0]["relative_path"] == "validation.parquet", "Locked validation request selects a different split file")
    uri = posixpath.join(data[0]["uri"], "selection.json")
    check(uri == window["manifest_uri"], "Locked validation manifest is not in the request's validation artifact")
    manifest = read_json(uri)
    check(
        manifest["dataset"] == "openai/gsm8k" and manifest["revision"] == window["dataset_revision"],
        "Locked validation dataset/revision differs",
    )
    expected_window = {
        "purpose": "locked_holdout",
        "split": "test",
        "offset": offset,
        "count": count,
        "excluded_development_rows": [0, 128],
    }
    check(manifest.get("validation_window") == expected_window, "Locked validation window declaration differs")
    expected_ids = [f"test/{index}" for index in range(offset, offset + count)]
    check(
        manifest["rows"]["test"] == expected_ids,
        "Locked validation source IDs are reordered, missing, duplicated or overlapping",
    )
    check(set(expected_ids).isdisjoint(manifest["rows"]["train"]), "Locked validation overlaps training source IDs")
    check(
        spec["expected_eval_rows"] == count * spec.get("eval_samples_per_prompt", 1),
        "Locked validation row count differs from evaluation contract",
    )
    return {
        "manifest_uri": uri,
        "manifest_sha256": canonical_sha(manifest),
        "source_indices_verified": True,
        "dataset": manifest["dataset"],
        "revision": manifest["revision"],
        **expected_window,
    }


def audit_eval_dump(
    root: str,
    step: int,
    expected_rows: int,
    expected_samples: int,
    wandb_metrics: JSONDict,
    required: bool,
    dump_namespace: str | None = None,
    require_engine_indices: bool = False,
    expected_engine_count: int | None = None,
) -> tuple[JSONDict, EvalRecords]:
    uri = posixpath.join(root, "dumped_evals", f"global_step_{step}_evals")
    if dump_namespace is not None:
        check(
            dump_namespace.startswith("startup_pass_") and dump_namespace.removeprefix("startup_pass_").isdigit(),
            "Invalid startup dump namespace",
        )
        uri = posixpath.join(uri, dump_namespace)
    entries = list_entries(uri)
    if not entries:
        check(not required, f"Missing evaluation dump at step {step}")
        return {"step": step, "present": False, "dump_namespace": dump_namespace}, []
    check(all(entry["type"] == "file" for entry in entries), "Unexpected nested evaluation directory")
    files = [entry for entry in entries if posixpath.basename(entry["name"]) != "aggregated_results.jsonl"]
    check(files and all(entry["name"].endswith(".jsonl") for entry in files), "Unexpected evaluation dump inventory")
    aggregate = read_json(posixpath.join(uri, "aggregated_results.jsonl"))
    ordinal_records = {}
    source_scores = collections.defaultdict(list)
    uid_scores = collections.defaultdict(list)
    source_uids = collections.defaultdict(lambda: collections.defaultdict(list))
    stops = collections.Counter()
    engines = collections.Counter()
    byte_count = 0
    for entry in sorted(files, key=lambda item: item["name"]):
        fs, path = fs_path(posixpath.join(uri, posixpath.basename(entry["name"])))
        with fs.open(path, "rb") as source:
            for line in bounded_lines(source):
                byte_count += len(line)
                check(
                    len(line) <= MAX_EVAL_LINE_BYTES and byte_count <= MAX_EVAL_BYTES,
                    "Evaluation dump exceeds audit bounds",
                )
                row = json.loads(line)
                ordinal = row["row_ordinal"]
                check(
                    type(ordinal) is int and ordinal not in ordinal_records, "Duplicate/invalid evaluation row ordinal"
                )
                check(row["token_provenance"] == "finalized_trajectory", "Unexpected token provenance")
                for tokens_key, digest_key in (
                    ("prompt_token_ids", "prompt_token_ids_sha256"),
                    ("response_ids", "response_ids_sha256"),
                ):
                    tokens = row[tokens_key]
                    check(
                        isinstance(tokens, list) and all(type(token) is int and token >= 0 for token in tokens),
                        "Invalid token IDs",
                    )
                    check(
                        canonical_sha(tokens) == row[digest_key],
                        f"Evaluation token SHA mismatch at step {step}, ordinal {ordinal}",
                    )
                check(row["response_length"] == len(row["response_ids"]), "Response length differs from token count")
                score = sum(row["score"]) if isinstance(row["score"], list) else row["score"]
                check(isinstance(score, (float, int)) and math.isfinite(score), "Invalid evaluation score")
                # The mean uses token-reward sums; NormalizedReward.outcome uses
                # the final reward value. Dumps do not contain unshaped_rewards.
                outcome = (row["score"][-1] if row["score"] else 0.0) if isinstance(row["score"], list) else score
                check(isinstance(row["uid"], str) and row["uid"], "Missing evaluation UID")
                check(len(ordinal_records) < expected_rows, "Evaluation dump has extra rows")
                engine = row.get("generator_engine_index")
                check(
                    engine is None
                    or (
                        type(engine) is int
                        and engine >= 0
                        and (expected_engine_count is None or engine < expected_engine_count)
                    ),
                    "Invalid generator engine index",
                )
                check(not require_engine_indices or engine is not None, "Missing actual generator engine index")
                engines[str(engine)] += 1
                ordinal_records[ordinal] = [
                    row["uid"],
                    row["prompt_token_ids_sha256"],
                    row["response_ids_sha256"],
                    score,
                    row["stop_reason"],
                    engine,
                ]
                dataset = (row["data_source"] or "unknown").replace("/", "_")
                source_scores[dataset].append(score)
                uid_scores[row["uid"]].append(outcome)
                source_uids[dataset][row["uid"]].append(outcome)
                stops[str(row["stop_reason"])] += 1
    check(sorted(ordinal_records) == list(range(expected_rows)), "Evaluation row coverage/order incomplete")
    check(all(len(scores) == expected_samples for scores in uid_scores.values()), "Evaluation UID sample counts differ")
    check(len(uid_scores) * expected_samples == expected_rows, "Evaluation prompt count differs")
    scores = [record[3] for record in ordinal_records.values()]
    reconstructed = {
        "eval/all/avg_score": sum(scores) / len(scores),
        f"eval/all/pass_at_{expected_samples}": (
            sum(any(score > 0 for score in scores) for scores in uid_scores.values()) / len(uid_scores)
        ),
    }
    for dataset, scores in source_scores.items():
        reconstructed[f"eval/{dataset}/avg_score"] = sum(scores) / len(scores)
        reconstructed[f"eval/{dataset}/pass_at_{expected_samples}"] = sum(
            any(score > 0 for score in scores) for scores in source_uids[dataset].values()
        ) / len(source_uids[dataset])
    for key, value in reconstructed.items():
        for label, metrics in (("dump aggregate", aggregate), ("W&B", wandb_metrics)):
            check(
                key in metrics and math.isclose(value, metrics[key], abs_tol=1e-7, rel_tol=1e-7),
                f"Evaluation {label} differs at step {step}: {key}",
            )
    ordered = [ordinal_records[i] for i in range(expected_rows)]
    return {
        "step": step,
        "dump_namespace": dump_namespace,
        "present": True,
        "rows": expected_rows,
        "unique_uids": len(uid_scores),
        "hashes_verified": expected_rows * 2,
        "aggregate_verified": True,
        "ordered_prompt_sha256": canonical_sha([[row[0], row[1]] for row in ordered]),
        "ordered_response_sha256": canonical_sha([[row[0], row[2]] for row in ordered]),
        "ordered_result_sha256": canonical_sha([row[:5] for row in ordered]),
        "stop_counts": dict(stops),
        "ordered_engine_sha256": canonical_sha([[row[0], row[5]] for row in ordered]),
        "generator_engine_index_counts": dict(engines),
        "engine_identity_known_rows": expected_rows - engines.get("None", 0),
        "engine_identity_scope": "dispatcher engine-list index; not a physical GPU identifier",
        "metrics": reconstructed,
        "bytes_streamed": byte_count,
        "reward_reduction": "mean of token-reward sums; pass@n from final reward (checked against aggregate)",
        "unshaped_reward_channel_in_dump": False,
    }, ordered


def compare_eval_records(left: EvalRecords, right: EvalRecords) -> JSONDict:
    check(
        [[row[0], row[1]] for row in left] == [[row[0], row[1]] for row in right],
        "Evaluation comparison prompt/UID order differs",
    )
    pairs = list(zip(left, right, strict=True))
    known = [(a, b) for a, b in pairs if a[5] is not None and b[5] is not None]
    return {
        "rows": len(pairs),
        "response_changed_rows": sum(a[2] != b[2] for a, b in pairs),
        "score_changed_rows": sum(a[3] != b[3] for a, b in pairs),
        "stop_reason_changed_rows": sum(a[4] != b[4] for a, b in pairs),
        "known_engine_pairs": len(known),
        "engine_index_changed_rows": sum(a[5] != b[5] for a, b in known),
        "response_changed_equal_engine_index_rows": sum(a[2] != b[2] and a[5] == b[5] for a, b in known),
        "response_changed_different_engine_index_rows": sum(a[2] != b[2] and a[5] != b[5] for a, b in known),
    }


def compare_startup_runs(left_label: str, right_label: str, snapshots: JSONDict) -> JSONDict:
    left, right = snapshots[left_label], snapshots[right_label]
    for field in ("runtime", "model", "train_data", "validation_data", "topology", "seed"):
        check(left["request"][field] == right["request"][field], f"Frozen pair provenance differs: {field}")
    controls = [json.loads(json.dumps(row["source_config"])) for row in (left, right)]
    probes = [config["trainer"].pop("weight_change_probe", False) for config in controls]
    check(probes == [False, True], "Frozen pair must be ordered probe off, probe on")
    check(controls[0] == controls[1], "Frozen pair source config differs beyond weight_change_probe")
    check(
        left["startup"] and len(left["startup"]) == len(right["startup"]),
        "Frozen pair startup pass counts differ or are absent",
    )
    pairs = []
    for i, a in enumerate(left["startup"]):
        for j, b in enumerate(right["startup"]):
            pairs.append({"left_pass": i, "right_pass": j, **compare_eval_records(a, b)})
    return {
        "left": left_label,
        "right": right_label,
        "provenance_matched": True,
        "only_source_config_difference": "trainer.weight_change_probe",
        "pairwise": pairs,
        "engine_comparison_scope": (
            "indices are local to each run; equal indices across runs do not identify the same actor/GPU"
        ),
    }


def resolved_dispatch_engine_count(config: Mapping[str, Any]) -> int:
    """Local vLLM creates one dispatcher frontend per DP rank of each replica."""
    replicas = lookup(config, "generator.num_inference_engines")
    dp = lookup(config, "generator.inference_engine_data_parallel_size")
    dp = 1 if dp is None else dp
    check(type(replicas) is int and replicas > 0, "Invalid resolved inference replica count")
    check(type(dp) is int and dp > 0, "Invalid resolved inference data parallel size")
    if lookup(config, "generator.backend") == "vllm" and lookup(config, "generator.run_engines_locally") is not False:
        return replicas * dp
    return replicas


def audit_run(spec: JSONDict, api: RunAPI, comparison_snapshots: JSONDict | None = None) -> JSONDict:
    check(type(spec["expected_steps"]) is int and 0 < spec["expected_steps"] <= 10000, "Invalid expected step count")
    request, resolved, storage = audit_storage(spec)
    locked_validation = audit_locked_validation(spec, request)
    entity, project, run_id = wandb_identity(spec["wandb_url"])
    run = api.run(f"{entity}/{project}/{run_id}")
    check(wandb_identity(run.url) == (entity, project, run_id), "W&B resolved identity differs from emitted URL")
    check(run.state == "finished", f"W&B run state is {run.state}, expected finished")
    cfg = dict(run.config)
    engine_count = resolved_dispatch_engine_count(cfg)
    for key, expected in (
        ("trainer.completion.run_id", spec["run_id"]),
        ("trainer.completion.attempt_id", spec["attempt_id"]),
        ("trainer.completion.request_fingerprint", storage["request_fingerprint"]),
        ("trainer.completion.mode", "metrics"),
        ("trainer.seed", request["seed"]),
        ("trainer.max_steps", spec["expected_steps"]),
        ("trainer.ckpt_interval", -1),
        ("trainer.hf_save_interval", -1),
    ):
        check(
            lookup(cfg, key) == expected,
            f"Resolved W&B config missing/different: {key}; top-level keys={sorted(cfg)[:20]}",
        )
    for key, expected in spec.get("expected_config", {}).items():
        check(lookup(cfg, key) == expected, f"Resolved W&B configuration differs from declared control: {key}")
    source_cfg = yaml.safe_load(request["config_yaml"])
    entrypoint = canonical_entrypoint(source_cfg["entrypoint"])
    check(entrypoint == canonical_entrypoint(resolved["entrypoint"]), "Request/resolved entrypoint differs")
    required = list(spec.get("required_metrics", []))
    if lookup(cfg, "trainer.algorithm.use_tis"):
        required += ["tis/batch_skipped_no_logprobs", "tis/skipped_fraction"]
    history, evaluations = summarize_history(run, spec["expected_steps"], entrypoint, required)
    role_plan = request["topology"]["role_plan"]
    expected_sequences = spec["expected_steps"] * role_plan["train_batch_size"] * role_plan["n_samples_per_prompt"]
    check(
        history["sums"]["consumed/sequences"] == expected_sequences,
        "Consumed sequence count differs from fixed batch contract",
    )
    expected_evals = spec["expected_eval_steps"]
    check(sorted(evaluations) == sorted(expected_evals), "W&B initial/periodic/final evaluation coverage differs")
    samples = spec.get("eval_samples_per_prompt", 1)
    # These recipes have unshaped GSM8K rewards. Don't claim pass@n equivalence for shaped objectives.
    check(not lookup(cfg, "trainer.algorithm.use_kl_in_reward"), "This score audit requires unshaped evaluation rewards")
    if lookup(cfg, "trainer.algorithm.use_tis"):
        check(
            lookup(cfg, "trainer.algorithm.require_rollout_logprobs") is True,
            "TIS run did not require behavior coverage",
        )
        for key in ("tis/batch_skipped_no_logprobs", "tis/skipped_fraction"):
            check(history["ranges"][key]["max"] == 0, "TIS correction was skipped")
    if entrypoint == "fully_async":
        age = lookup(cfg, "trainer.fully_async.max_staleness_steps")
        check(history["ranges"]["async/staleness_max"]["max"] <= age, "Consumed update age exceeded configured cap")
    repeats = spec.get("initial_eval_repeat_count", 1)
    check(type(repeats) is int and 1 <= repeats <= 16, "Startup repeat count exceeds audit contract (1-16)")
    configured_repeats = lookup(cfg, "trainer.initial_eval_repeat_count")
    check(
        configured_repeats == repeats or (configured_repeats is None and repeats == 1),
        "Resolved startup repeat count differs from expected contract",
    )
    if repeats > 1:
        check(0 in expected_evals, "Repeated startup evaluation requires step zero in expected_eval_steps")
    dumps, startup_records = [], []
    for step in expected_evals:
        namespaces = [f"startup_pass_{index}" for index in range(repeats)] if step == 0 and repeats > 1 else [None]
        if namespaces != [None]:
            observed = {key.split("/")[1] for key in evaluations[step]}
            check(observed == set(namespaces), "Startup evaluation namespace coverage differs from configured repeats")
            directory = posixpath.join(request["output"]["export_root"], "dumped_evals", f"global_step_{step}_evals")
            entries = list_entries(directory)
            check(
                {posixpath.basename(entry["name"].rstrip("/")) for entry in entries} == set(namespaces)
                and all(entry["type"] == "directory" for entry in entries),
                "Startup dump namespace inventory differs",
            )
        for namespace in namespaces:
            metrics = (
                evaluations[step]
                if namespace is None
                else {
                    "eval/" + key.removeprefix(f"eval/{namespace}/"): value
                    for key, value in evaluations[step].items()
                    if key.startswith(f"eval/{namespace}/")
                }
            )
            summary, records = audit_eval_dump(
                request["output"]["export_root"],
                step,
                spec["expected_eval_rows"],
                samples,
                metrics,
                spec.get("require_eval_dumps", True),
                namespace,
                spec.get("require_engine_indices", False),
                engine_count,
            )
            dumps.append(summary)
            if namespace is not None:
                startup_records.append(records)
    repeatability = None
    if startup_records:
        reference_inputs = [[row[0], row[1]] for row in startup_records[0]]
        check(
            all([[row[0], row[1]] for row in records] == reference_inputs for records in startup_records),
            "Startup evaluation prompt/UID order changed; this is not a frozen-input repeat",
        )
        comparisons = []
        for left in range(repeats):
            for right in range(left + 1, repeats):
                comparisons.append(
                    {
                        "left": f"startup_pass_{left}",
                        "right": f"startup_pass_{right}",
                        **compare_eval_records(startup_records[left], startup_records[right]),
                    }
                )
        repeatability = {
            "passes": repeats,
            "input_hashes_identical": True,
            "response_hashes_identical": all(row["response_changed_rows"] == 0 for row in comparisons),
            "pairwise": comparisons,
            "weight_identity_evidence": (
                "source runs startup passes at step zero before optimizer/producers; "
                "dumps carry no per-request installed-weight identity"
            ),
        }
    if comparison_snapshots is not None:
        comparison_snapshots[spec["label"]] = {
            "startup": startup_records,
            "request": request,
            "source_config": source_cfg,
        }
    config_keys = (
        "trainer.weight_change_probe",
        "trainer.initial_eval_repeat_count",
        "trainer.seed",
        "trainer.max_steps",
        "trainer.eval_before_train",
        "trainer.eval_interval",
        "trainer.algorithm.use_kl_loss",
        "trainer.algorithm.use_kl_in_reward",
        "trainer.algorithm.policy_loss_type",
        "trainer.algorithm.use_tis",
        "trainer.algorithm.tis_imp_ratio_cap",
        "trainer.algorithm.require_rollout_logprobs",
        "trainer.fully_async.weight_sync_interval",
        "trainer.fully_async.max_staleness_steps",
        "generator.num_inference_engines",
        "generator.inference_engine_data_parallel_size",
        "generator.sampling_params",
        "generator.eval_sampling_params",
        "generator.max_input_length",
        "generator.chat_template_kwargs",
    )
    return {
        "training_evidence_pass": True,
        "clean_end_to_end": storage["clean_end_to_end"],
        "require_launcher_success": spec.get("require_launcher_success", True),
        "run_id": spec["run_id"],
        "attempt_id": spec["attempt_id"],
        "wandb_url": run.url,
        "wandb_entity": entity,
        "wandb_name": run.name,
        "state": run.state,
        "storage": storage,
        "resolved_config": {key: lookup(cfg, key) for key in config_keys},
        "provenance": {key: request[key] for key in ("model", "train_data", "validation_data", "topology", "seed")},
        "history": history,
        "eval_dumps": dumps,
        "startup_repeatability": repeatability,
        "locked_validation": locked_validation,
        "resolved_dispatch_engine_count": engine_count,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", help="Local JSON specification; alternatively ASYNC_RL_AUDIT_SPEC")
    args = parser.parse_args()
    specification = read_json(args.spec) if args.spec else json.loads(os.environ["ASYNC_RL_AUDIT_SPEC"])
    runs = specification["runs"]
    check(0 < len(runs) <= 8, "Audit between one and eight runs")
    api = wandb.Api(timeout=45)
    results, errors, snapshots = {}, {}, {}
    for spec in runs:
        label = spec["label"]
        check(label not in results and label not in errors, "Audit labels must be unique")
        try:
            results[label] = audit_run(spec, api, snapshots)
        except Exception as error:
            errors[label] = {"type": type(error).__name__, "message": str(error)[:2000]}
    comparisons = []
    for left, right in specification.get("compare_startup_runs", []):
        try:
            check(left in results and right in results, "Comparison requires two fully audited runs")
            comparisons.append(compare_startup_runs(left, right, snapshots))
        except Exception as error:
            errors[f"comparison:{left}:{right}"] = {"type": type(error).__name__, "message": str(error)[:2000]}
    clean = not errors and all(result["clean_end_to_end"] for result in results.values())
    print(
        "ASYNC_RL_TERMINAL_AUDIT_JSON "
        + json.dumps(
            {
                "observed_at_utc": datetime.datetime.now(datetime.UTC).isoformat(),
                "audit_scope": "W&B and durable results; Finelog requires separate verification",
                "training_evidence_pass": not errors,
                "clean_end_to_end": clean,
                "runs": results,
                "startup_comparisons": comparisons,
                "errors": errors,
            },
            allow_nan=False,
        ),
        flush=True,
    )
    if errors:
        print("ASYNC_RL_TERMINAL_AUDIT_FAIL", flush=True)
    elif clean:
        print("ASYNC_RL_TERMINAL_AUDIT_PASS", flush=True)
    else:
        print("ASYNC_RL_TRAINING_EVIDENCE_PASS", flush=True)
        print("ASYNC_RL_TERMINAL_AUDIT_NOT_CLEAN", flush=True)
    raise SystemExit(bool(errors))


if __name__ == "__main__":
    main()
