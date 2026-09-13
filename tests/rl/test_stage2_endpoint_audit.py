# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import copy
import hashlib
import json

import pytest

from experiments.post_training.math_eval.stage2_endpoint_audit import (
    audit_endpoint_capture,
    audit_endpoint_rows,
    canonical_sha256,
    tokens_sha256,
)
from tests.rl.test_thinking_contract_audit import example


def endpoint_example(*, stop="stop", shaping=False, endpoint="package_on"):
    row, decoder, overlong = example(stop=stop, shaping=shaping)
    row.update(uid="0", data_source="gsm8k/test", row_ordinal=0, token_provenance="finalized_trajectory")
    row["prompt_token_ids_sha256"] = tokens_sha256(row["prompt_token_ids"])
    row["response_ids_sha256"] = tokens_sha256(row["response_ids"])
    row["env_extras"]["extra_info"] = {"prompt_sha256": "a" * 64}
    row["non_agentic_contract"].update(metric_protocol="post-thinking-native-metrics-v1", evaluation_endpoint=endpoint)
    config = {
        "generator": {
            "non_agentic_parser_protocol": row["parser_protocol"],
            "non_agentic_eval_endpoints": ["package_on"],
            "non_agentic_intervention": None,
            "trajectory_reward_shaping": {
                "enabled": shaping,
                "schema_version": 2,
                "overlong": overlong,
                "passthrough": {"penalty": 0},
                "non_termination": {"penalty": 0},
                "successful_length": {"penalty_per_token": 0},
                "loop": {"advantage_penalty_per_token": 0},
            },
        }
    }
    source = [
        {
            "uid": "0",
            "prompt_sha256": "a" * 64,
            "prompt_token_ids_sha256": row["prompt_token_ids_sha256"],
            "env_class": "gsm8k",
            "data_source": "gsm8k/test",
            "ground_truth": "42",
        }
    ]
    metrics = {}
    for name in ["all", "gsm8k_test"]:
        for key, value in {
            "contract_correct": 1,
            "contract_completed": int(stop == "stop"),
            "corrected_verifier_reward": 1,
            "score_contract": 0.5 if shaping else 1,
            "legacy_full_text_reward_diagnostic": 0,
        }.items():
            metrics[f"eval/{name}/{key}"] = value
    return row, decoder, config, source, metrics


@pytest.mark.parametrize("stop", ["stop", "length"])
def test_endpoint_source_and_native_means_keep_shaped_reward_separate(stop):
    row, decoder, config, source, metrics = endpoint_example(stop=stop, shaping=True)
    result = audit_endpoint_rows(
        [row], decoder, endpoint="package_on", effective_config=config, source_rows=source, native_metrics=metrics
    )
    assert result["metrics"]["eval/all/score_contract"] == 0.5
    assert result["metrics"]["eval/all/contract_correct"] == 1
    assert result["metrics"]["eval/all/contract_completed"] == int(stop == "stop")
    assert result["stop_counts"] == {stop: 1}


@pytest.mark.parametrize(
    "poison",
    [
        "endpoint",
        "metric_version",
        "source_gold",
        "prompt_tokens",
        "response_tokens",
        "missing_row",
        "duplicate_row",
        "native_shaped_mapping",
    ],
)
def test_cross_endpoint_source_or_native_metric_contradictions_rejected(poison):
    row, decoder, config, source, metrics = endpoint_example(shaping=True)
    rows = [copy.deepcopy(row)]
    if poison == "endpoint":
        rows[0]["non_agentic_contract"]["evaluation_endpoint"] = "common_off"
    elif poison == "metric_version":
        rows[0]["non_agentic_contract"]["metric_protocol"] = "legacy"
    elif poison == "source_gold":
        source[0]["ground_truth"] = "41"
    elif poison == "prompt_tokens":
        rows[0]["prompt_token_ids"] = [0]
    elif poison == "response_tokens":
        rows[0]["response_ids_sha256"] = "f" * 64
    elif poison == "missing_row":
        rows = []
    elif poison == "duplicate_row":
        rows.append(copy.deepcopy(row))
    else:
        metrics["eval/all/contract_completed"] = 0.5
    with pytest.raises(ValueError):
        audit_endpoint_rows(
            rows, decoder, endpoint="package_on", effective_config=config, source_rows=source, native_metrics=metrics
        )


def test_common_off_disables_declared_token_intervention_without_changing_parser():
    row, decoder, config, source, metrics = endpoint_example(endpoint="common_off")
    config["generator"]["non_agentic_eval_endpoints"] = ["common_off", "package_on"]
    config["generator"]["non_agentic_intervention"] = {"kind": "force_close"}
    result = audit_endpoint_rows(
        [row], decoder, endpoint="common_off", effective_config=config, source_rows=source, native_metrics=metrics
    )
    assert result["rows"][0]["contract_correct"] == 1
    assert result["rows"][0]["forced_positions"] == []
    with pytest.raises(ValueError, match="dump scope"):
        audit_endpoint_rows(
            [row], decoder, endpoint="package_on", effective_config=config, source_rows=source, native_metrics=metrics
        )


def capture_example():
    row, decoder, config, source, metrics = endpoint_example()
    config["trainer"] = {
        "step_wise_training": False,
        "fully_async": {"eval_mode": "blocking", "eval_on_installed_weights": False},
        "completion": {"request_fingerprint": "r" * 64},
    }
    digest = canonical_sha256(config)
    metadata = {
        "schema": "non_agentic_endpoint_metadata_v1",
        "endpoint": "package_on",
        "global_step": 25,
        "policy_version": 25,
        "parser_protocol": row["parser_protocol"],
        "metric_protocol": "post-thinking-native-metrics-v1",
        "request_fingerprint": "r" * 64,
        "config_sha256": digest,
        "dump_namespace": "package_on",
    }
    files = {
        "endpoint_metadata.json": json.dumps(metadata).encode(),
        "aggregated_results.jsonl": json.dumps(metrics).encode(),
        "gsm8k_test.jsonl": json.dumps(row).encode(),
    }
    spec = {
        "arm": "parser_only",
        "step": 25,
        "endpoint": "package_on",
        "request_fingerprint": "r" * 64,
        "config_sha256": digest,
        "source_rows_sha256": canonical_sha256(source),
        "filenames": sorted(files),
    }
    return files, decoder, spec, config, source, {k: hashlib.sha256(v).hexdigest() for k, v in files.items()}


def test_capture_binds_original_metadata_config_inputs_and_bytes():
    files, decoder, spec, config, source, hashes = capture_example()
    result = audit_endpoint_capture(
        files,
        decoder,
        specification=spec,
        specification_sha256=canonical_sha256(spec),
        captured_files_sha256=hashes,
        effective_config=config,
        source_rows=source,
    )
    assert result["step"] == result["native_endpoint_metadata"]["policy_version"] == 25
    assert result["metrics"]["eval/all/contract_completed"] == 1


@pytest.mark.parametrize(
    "poison",
    [
        "version",
        "step",
        "request",
        "config",
        "source",
        "bytes",
        "missing_metadata",
        "extra_file",
        "ordinal",
        "token_origin",
    ],
)
def test_capture_rejects_foreign_endpoint_evidence_even_when_scores_agree(poison):
    files, decoder, spec, config, source, hashes = capture_example()
    if poison in {"version", "step", "request"}:
        metadata = json.loads(files["endpoint_metadata.json"])
        metadata[{"version": "policy_version", "step": "global_step", "request": "request_fingerprint"}[poison]] = 24
        files["endpoint_metadata.json"] = json.dumps(metadata).encode()
        hashes["endpoint_metadata.json"] = hashlib.sha256(files["endpoint_metadata.json"]).hexdigest()
    elif poison == "config":
        config["generator"]["non_agentic_eval_endpoints"] = ["common_off", "package_on"]
    elif poison == "source":
        source[0]["ground_truth"] = "41"
    elif poison == "bytes":
        files["gsm8k_test.jsonl"] += b" "
    elif poison == "missing_metadata":
        del files["endpoint_metadata.json"]
    elif poison == "extra_file":
        files["foreign.jsonl"] = b"{}"
    else:
        row = json.loads(files["gsm8k_test.jsonl"])
        row["row_ordinal" if poison == "ordinal" else "token_provenance"] = 4 if poison == "ordinal" else "raw_engine"
        files["gsm8k_test.jsonl"] = json.dumps(row).encode()
        hashes["gsm8k_test.jsonl"] = hashlib.sha256(files["gsm8k_test.jsonl"]).hexdigest()
    with pytest.raises(ValueError):
        audit_endpoint_capture(
            files,
            decoder,
            specification=spec,
            specification_sha256=canonical_sha256(spec),
            captured_files_sha256=hashes,
            effective_config=config,
            source_rows=source,
        )


def test_endpoint_cannot_permute_question_identity_under_valid_ordinals():
    row, decoder, config, source, metrics = endpoint_example()
    other = copy.deepcopy(row)
    other.update(uid="1", row_ordinal=1)
    other["env_extras"]["extra_info"]["prompt_sha256"] = "b" * 64
    source.append(dict(source[0], uid="1", prompt_sha256="b" * 64))
    row["row_ordinal"], other["row_ordinal"] = 1, 0
    with pytest.raises(ValueError, match="source identity"):
        audit_endpoint_rows(
            [row, other],
            decoder,
            endpoint="package_on",
            effective_config=config,
            source_rows=source,
            native_metrics=metrics,
        )
