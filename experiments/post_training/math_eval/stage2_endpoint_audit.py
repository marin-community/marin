# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Join stage-2 endpoint dumps to frozen inputs and independently scored metrics.

The regional caller must bind the effective config, input manifest and native dump
bytes before calling this audit. This does not establish training or GPU lifecycle.
"""

import hashlib
import json
import math
from collections import Counter, defaultdict

from experiments.post_training.math_eval.thinking_contract_audit import PARSER_VERSION, audit_post_thinking_row

METRIC_PROTOCOL = "post-thinking-native-metrics-v1"


def tokens_sha256(tokens):
    if any(type(token) is not int or token < 0 for token in tokens):
        raise ValueError("Token evidence must contain nonnegative integer IDs")
    return hashlib.sha256(json.dumps(tokens, separators=(",", ":")).encode("ascii")).hexdigest()


def audit_endpoint_rows(rows, decoder, *, endpoint, effective_config, source_rows, native_metrics):
    """Audit one generated endpoint, including source membership and native means."""
    generator = effective_config["generator"]
    modes = generator["non_agentic_eval_endpoints"]
    intervention = generator.get("non_agentic_intervention")
    expected_modes = ["common_off", "package_on"] if intervention is not None else ["package_on"]
    if modes != expected_modes or endpoint not in modes or generator["non_agentic_parser_protocol"] != PARSER_VERSION:
        raise ValueError("Endpoint differs from the frozen generation treatment")
    source = {entry["uid"]: entry for entry in source_rows}
    if (
        not source
        or len(source) != len(source_rows)
        or len({entry["prompt_sha256"] for entry in source_rows}) != len(source)
    ):
        raise ValueError("Frozen endpoint inputs contain repeated or missing identities")
    if len(rows) != len(source) or {row["uid"] for row in rows} != set(source):
        raise ValueError("Endpoint responses differ from frozen source membership")
    shaping = generator["trajectory_reward_shaping"]
    overlong = None
    if shaping["enabled"]:
        if (
            shaping["schema_version"] != 2
            or shaping["passthrough"]["penalty"] != 0
            or shaping["non_termination"]["penalty"] != 0
            or shaping["successful_length"]["penalty_per_token"] != 0
            or shaping["loop"]["advantage_penalty_per_token"] != 0
        ):
            raise ValueError("Endpoint shaping differs from the single overlong treatment")
        overlong = shaping["overlong"]
    if endpoint == "common_off":
        intervention = None
    if sorted(row["row_ordinal"] for row in rows) != list(range(len(source))):
        raise ValueError("Endpoint row ordinals do not cover the original evaluation order")
    if any(row["token_provenance"] != "finalized_trajectory" for row in rows):
        raise ValueError("Endpoint rows lack finalized native token provenance")
    source_ordinals = {entry["uid"]: ordinal for ordinal, entry in enumerate(source_rows)}
    populations = defaultdict(list)
    results = []
    for row in rows:
        expected = source[row["uid"]]
        contract, extras = row["non_agentic_contract"], row["env_extras"]
        if contract.get("metric_protocol") != METRIC_PROTOCOL or contract.get("evaluation_endpoint") != endpoint:
            raise ValueError("Response endpoint or metric protocol differs from its dump scope")
        prompt_hash = tokens_sha256(row["prompt_token_ids"])
        if (
            row["row_ordinal"] != source_ordinals[row["uid"]]
            or extras["extra_info"]["prompt_sha256"] != expected["prompt_sha256"]
            or prompt_hash != row["prompt_token_ids_sha256"]
            or prompt_hash != expected["prompt_token_ids_sha256"]
            or tokens_sha256(row["response_ids"]) != row["response_ids_sha256"]
            or row["env_class"] != expected["env_class"]
            or row["data_source"] != expected["data_source"]
            or extras["reward_model"]["ground_truth"] != expected["ground_truth"]
            or extras["reward_spec"]["ground_truth"] != expected["ground_truth"]
        ):
            raise ValueError("Response tokens, source identity or gold differ from frozen inputs")
        scored = audit_post_thinking_row(
            row, decoder, expected_parser=PARSER_VERSION, intervention=intervention, overlong=overlong
        )
        result = dict(scored, uid=row["uid"], prompt_sha256=expected["prompt_sha256"], endpoint=endpoint)
        results.append(result)
        name = (row["data_source"] or "unknown").replace("/", "_")
        if name == "all":
            raise ValueError("Source name collides with aggregate metric scope")
        populations["all"].append(result)
        populations[name].append(result)
    fields = {
        "contract_correct": "contract_correct",
        "contract_completed": "score_contract_completed",
        "corrected_verifier_reward": "corrected_verifier_reward",
        "score_contract": "score_contract",
        "legacy_full_text_reward_diagnostic": "legacy_full_text_reward_diagnostic",
    }
    metrics = {}
    for name, values in populations.items():
        for metric, field in fields.items():
            key = f"eval/{name}/{metric}"
            mean = sum(row[field] for row in values) / len(values)
            observed = native_metrics[key]
            # Means may be serialized through float arithmetic; individual raw rewards remain verbatim.
            if not math.isfinite(observed) or not math.isclose(mean, observed, rel_tol=0, abs_tol=1e-12):
                raise ValueError("Native endpoint mean differs from independently audited response rows")
            metrics[key] = mean
    return {
        "schema": "e61_stage2_endpoint_audit_v1",
        "endpoint": endpoint,
        "parser_protocol": PARSER_VERSION,
        "metric_protocol": METRIC_PROTOCOL,
        "rows": results,
        "metrics": metrics,
        "stop_counts": dict(Counter(row["stop_reason"] for row in results)),
    }


def canonical_sha256(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def audit_endpoint_capture(
    files, decoder, *, specification, specification_sha256, captured_files_sha256, effective_config, source_rows
):
    """Bind a prospective endpoint declaration to exact native capture bytes.

    The specification is frozen before outcomes; file hashes are observed later.
    The caller authenticates storage and task provenance before supplying files.
    This audit requires native endpoint metadata; a reconstructed filename receipt
    is not a substitute for the producer's installed-version observation.
    """
    if canonical_sha256(specification) != specification_sha256:
        raise ValueError("Endpoint specification changed after freezing")
    if canonical_sha256(effective_config) != specification["config_sha256"]:
        raise ValueError("Endpoint effective configuration differs from its frozen treatment")
    if canonical_sha256(source_rows) != specification["source_rows_sha256"]:
        raise ValueError("Endpoint input membership differs from its frozen declaration")
    if set(files) != set(specification["filenames"]) or set(files) != set(captured_files_sha256):
        raise ValueError("Endpoint capture file membership differs")
    for name, content in files.items():
        if hashlib.sha256(content).hexdigest() != captured_files_sha256[name]:
            raise ValueError("Endpoint capture bytes differ from their bound storage receipt")
    metadata = json.loads(files["endpoint_metadata.json"])
    endpoint, step = specification["endpoint"], specification["step"]
    expected = {
        "schema": "non_agentic_endpoint_metadata_v1",
        "endpoint": endpoint,
        "global_step": step,
        "policy_version": step,
        "parser_protocol": PARSER_VERSION,
        "metric_protocol": METRIC_PROTOCOL,
        "request_fingerprint": specification["request_fingerprint"],
        "config_sha256": specification["config_sha256"],
        "dump_namespace": endpoint,
    }
    if metadata != expected:
        raise ValueError("Native endpoint metadata differs from the frozen endpoint or installed version")
    config = effective_config
    if (
        config["trainer"]["step_wise_training"]
        or config["trainer"]["fully_async"]["eval_mode"] != "blocking"
        or config["trainer"]["fully_async"]["eval_on_installed_weights"]
        or config["trainer"]["completion"]["request_fingerprint"] != specification["request_fingerprint"]
    ):
        raise ValueError("Endpoint configuration is not the bound blocking quality evaluation")
    aggregate_lines = files["aggregated_results.jsonl"].decode().splitlines()
    if len(aggregate_lines) != 1:
        raise ValueError("Endpoint aggregate must contain exactly one native metric record")
    metrics = json.loads(aggregate_lines[0])
    rows = []
    for name, content in sorted(files.items()):
        if name in {"endpoint_metadata.json", "aggregated_results.jsonl"}:
            continue
        if "/" in name or not name.endswith(".jsonl"):
            raise ValueError("Endpoint capture contains an unexpected data file")
        rows.extend(json.loads(line) for line in content.decode().splitlines())
    result = audit_endpoint_rows(
        rows, decoder, endpoint=endpoint, effective_config=config, source_rows=source_rows, native_metrics=metrics
    )
    return result | {
        "arm": specification["arm"],
        "step": step,
        "specification_sha256": specification_sha256,
        "native_endpoint_metadata": metadata,
        "files_sha256": captured_files_sha256,
    }
