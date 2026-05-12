# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from experiments.evals.synthetic_machine_records_ppl import (
    MACHINE_RECORD_PPL_SLICES,
    MachineRecordFamily,
    iter_machine_record_examples,
    synthetic_machine_records_raw_validation_sets,
)


def test_machine_records_registry_uses_supervised_target_only_format():
    datasets = synthetic_machine_records_raw_validation_sets()

    assert set(datasets) == {slice_.registry_key for slice_ in MACHINE_RECORD_PPL_SLICES}

    dataset = datasets["synthetic_machine_records_ppl/service_logs/zeek_conn_http_dns"]
    assert dataset.input_key == "input"
    assert dataset.target_key == "target"
    assert "loss:target_only" in dataset.tags
    assert "subset:zeek" in dataset.tags


def test_machine_records_slices_cover_requested_verticals():
    subsets = {slice_.subset for slice_ in MACHINE_RECORD_PPL_SLICES}
    families = {slice_.family for slice_ in MACHINE_RECORD_PPL_SLICES}

    assert families == set(MachineRecordFamily)
    assert {
        "zeek",
        "nginx",
        "k8s",
        "system",
        "python_traceback",
        "compiler",
        "github_actions",
        "json_logs",
        "package_json",
        "pyproject",
        "github_actions_yaml",
        "dockerfile",
        "terraform",
        "kubernetes_yaml",
    }.issubset(subsets)


def test_machine_record_examples_have_required_schema():
    records = list(iter_machine_record_examples(examples_per_config=1))

    assert len(records) == len(MACHINE_RECORD_PPL_SLICES)
    for record in records:
        assert set(record) == {"input", "target", "id", "subset", "task", "seed", "metadata"}
        assert record["input"]
        assert record["target"]
        assert record["id"].endswith("-000000")
        assert record["subset"]
        assert record["task"]
        assert isinstance(record["seed"], int)
        assert isinstance(record["metadata"], dict)
