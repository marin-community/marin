# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import re

from experiments.evals.synthetic_machine_records_ppl import (
    iter_machine_record_examples,
    synthetic_machine_records_raw_validation_sets,
)

EXPECTED_MACHINE_RECORD_KEYS = {
    "synthetic_machine_records_ppl/ci_logs/github_actions_jobs",
    "synthetic_machine_records_ppl/config_manifests/dockerfile",
    "synthetic_machine_records_ppl/config_manifests/github_actions_yaml",
    "synthetic_machine_records_ppl/config_manifests/kubernetes_yaml",
    "synthetic_machine_records_ppl/config_manifests/package_json",
    "synthetic_machine_records_ppl/config_manifests/pyproject_toml",
    "synthetic_machine_records_ppl/config_manifests/terraform_hcl",
    "synthetic_machine_records_ppl/service_logs/k8s_system_events",
    "synthetic_machine_records_ppl/service_logs/nginx_access_error",
    "synthetic_machine_records_ppl/service_logs/systemd_journal",
    "synthetic_machine_records_ppl/service_logs/zeek_conn_http_dns",
    "synthetic_machine_records_ppl/structured_logs/json_application_logs",
    "synthetic_machine_records_ppl/trace_errors/compiler_errors",
    "synthetic_machine_records_ppl/trace_errors/python_tracebacks",
}


def test_machine_records_registry_uses_supervised_target_only_format():
    datasets = synthetic_machine_records_raw_validation_sets()

    assert set(datasets) == EXPECTED_MACHINE_RECORD_KEYS

    dataset = datasets["synthetic_machine_records_ppl/service_logs/zeek_conn_http_dns"]
    assert dataset.input_key == "input"
    assert dataset.target_key == "target"
    assert "loss:target_only" in dataset.tags
    assert "subset:zeek" in dataset.tags


def test_machine_record_examples_have_required_schema():
    records = list(iter_machine_record_examples(examples_per_config=1))
    instruction_fragments = ("Machine record subset:", "Task:", "User:", "Assistant:")

    assert len(records) == 14
    for record in records:
        assert set(record) == {"input", "target", "id", "subset", "task", "seed", "metadata"}
        assert record["input"]
        assert not any(fragment in record["input"] for fragment in instruction_fragments)
        assert record["target"]
        assert record["id"].endswith("-000000")
        assert record["subset"]
        assert record["task"]
        assert isinstance(record["seed"], int)
        assert isinstance(record["metadata"], dict)


def test_machine_record_examples_keep_native_record_shapes():
    records_by_task = {record["task"]: record for record in iter_machine_record_examples(examples_per_config=1)}

    zeek = records_by_task["zeek_conn_http_dns"]
    assert str(zeek["input"]).startswith("#separator \\x09\n#set_separator\t,\n#fields\tts\tuid")
    assert str(zeek["target"]).startswith("tcp\thttp\t0.")

    systemd = records_by_task["systemd_journal"]
    assert str(systemd["input"]).startswith("May 11 14:10:17 node-")
    assert str(systemd["target"]).startswith("Started ")

    package_json = records_by_task["package_json"]
    assert str(package_json["input"]).startswith('{\n  "name": "@example/service",\n  "version": "0.')
    assert '"dependencies"' in str(package_json["target"])

    dockerfile = records_by_task["dockerfile"]
    assert str(dockerfile["input"]).startswith("FROM python:3.12-slim\nWORKDIR /app")
    assert str(dockerfile["target"]).endswith('CMD ["python", "-m", "service.main"]\n')


def test_machine_record_timestamps_keep_valid_clock_fields():
    records = list(iter_machine_record_examples(examples_per_config=125))
    timestamp_pattern = re.compile(r"14:(?P<minute>\d{2})(?::(?P<second>\d{2}))?")

    for record in records:
        text = f"{record['input']}{record['target']}"
        for match in timestamp_pattern.finditer(text):
            assert int(match.group("minute")) < 60
            if match.group("second") is not None:
                assert int(match.group("second")) < 60
