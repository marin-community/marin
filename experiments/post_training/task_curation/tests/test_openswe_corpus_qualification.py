# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The corpus oracle qualification's deterministic pieces (coordinator ruling
2026-11-28): the population the corpus run qualifies, the shard split, the
fail-closed condensed-arm decision, and the stubbed end-to-end plumbing. The
Daytona execution itself is the shards' own evidence."""

import asyncio
import json

from experiments.post_training.task_curation.datasets.openswe import corpus_qualification
from experiments.post_training.task_curation.datasets.openswe.fidelity_gate import ArmRecord, ControlRecord
from experiments.post_training.task_curation.tests.test_openswe_conversion import CONDENSED
from experiments.post_training.task_curation.tests.test_openswe_grader_repair import (
    GOLDEN,
    TEST_PATCH,
    grader_row,
)


def control(control: str, reward: int) -> ControlRecord:
    return ControlRecord(
        control=control, reward=reward, eval_exit_status="0", protocol="in_place", machine="m", output_tail=""
    )


def test_qualification_population_drops_unrepairable_and_sample_records():
    clean = grader_row(instance_id="acme__clean-1", eval_script='#!/bin/bash\ncd /testbed\npytest -q\necho "OPENSWE_EXIT_CODE=$rc"\n')
    repairable = grader_row(instance_id="acme__golden-2", eval_script=(
        "#!/bin/bash\ncd /testbed\n"
        f"git apply -v --allow-empty - <<'EOF_114329324912'\n{GOLDEN}EOF_114329324912\n"
        f"git apply -v --allow-empty - <<'EOF_114329324912'\n{TEST_PATCH}EOF_114329324912\n"
        'pytest -q\necho "OPENSWE_EXIT_CODE=$rc"\n'
    ))
    stuck = grader_row(
        instance_id="acme__stuck-3",
        eval_script=(
            "#!/bin/bash\ncd /testbed\n"
            f"git apply -v --allow-empty - <<'EOF_114329324912'\n"
            "diff --git a/w.py b/w.py\n--- a/w.py\n+++ b/w.py\n@@ -1 +1 @@\n-forged\n"
            f"EOF_114329324912\n"
            f"git apply -v --allow-empty - <<'EOF_114329324912'\n{TEST_PATCH}EOF_114329324912\n"
            'pytest -q\necho "OPENSWE_EXIT_CODE=$rc"\n'
        ),
    )
    excluded = grader_row(instance_id="acme__gated-4", eval_script='#!/bin/bash\ncd /testbed\npytest -q\necho "OPENSWE_EXIT_CODE=$rc"\n')
    population, counts = corpus_qualification.qualification_population(
        [clean, repairable, stuck, excluded],
        frozenset({"acme__gated-4", "acme__sampled-5"}),
    )
    assert [row.instance_id for row in population] == ["acme__clean-1", "acme__golden-2"]
    assert counts == {"scanned": 4, "sample_records": 1, "converter_unrepairable": 1, "population": 2}


def test_shard_population_splits_deterministically_without_overlap():
    rows = [grader_row(instance_id=f"acme__row-{index}") for index in range(10)]
    shards = [corpus_qualification.shard_population(rows, index, 3) for index in range(3)]
    assert [len(shard) for shard in shards] == [4, 3, 3]
    flattened = [row.instance_id for shard in shards for row in shard]
    assert sorted(flattened) == sorted(row.instance_id for row in rows)
    assert corpus_qualification.shard_population(rows, 0, 3) == corpus_qualification.shard_population(rows, 0, 3)


def test_decide_condensed_fails_closed_on_every_mismatch():
    record = corpus_qualification.decide_condensed(
        "acme__row-1", "corpus", ArmRecord(arm="condensed", noop=control("noop", 0), oracle=control("oracle", 1))
    )
    assert record.status == "qualified"
    record = corpus_qualification.decide_condensed(
        "acme__row-1", "corpus", ArmRecord(arm="condensed", noop=control("noop", 1), oracle=control("oracle", 1))
    )
    assert record.status == "excluded" and record.exclusion_reason == "condensed_noop_mismatch"
    record = corpus_qualification.decide_condensed(
        "acme__row-1", "corpus", ArmRecord(arm="condensed", noop=control("noop", 0), oracle=control("oracle", 0))
    )
    assert record.status == "excluded" and record.exclusion_reason == "condensed_oracle_mismatch"
    record = corpus_qualification.decide_condensed(
        "acme__row-1", "corpus", ArmRecord(arm="condensed", failure_reason="build_failed", failure_detail="boom")
    )
    assert record.status == "excluded" and record.exclusion_reason == "build_failed"
    record = corpus_qualification.decide_condensed("acme__row-1", "corpus", ArmRecord(arm="condensed"))
    assert record.status == "excluded" and record.exclusion_reason == "condensed_control_missing"


def test_condensed_context_renders_the_repaired_package(tmp_path):
    row = grader_row(instance_id="acme__golden-1", eval_script=(
        "#!/bin/bash\ncd /testbed\n"
        f"git apply -v --allow-empty - <<'EOF_114329324912'\n{GOLDEN}EOF_114329324912\n"
        f"git apply -v --allow-empty - <<'EOF_114329324912'\n{TEST_PATCH}EOF_114329324912\n"
        'pytest -q\necho "OPENSWE_EXIT_CODE=$rc"\n'
    ))
    context, workdir, staged = corpus_qualification.condensed_context(row, CONDENSED, tmp_path)
    emitted = (staged / "tests_eval_script.sh").read_text()
    assert GOLDEN not in emitted
    assert TEST_PATCH in emitted
    assert (context / "Dockerfile").exists()
    assert workdir.startswith("/")


def test_stub_run_qualifies_and_excludes_end_to_end(tmp_path, monkeypatch):
    """The driver's plumbing end to end on stubbed sandboxes: the population
    resolves, the shards split, the machines stage the repaired package, the
    no-op control scores 0 and the oracle 1 for every repairable row, and the
    report lands in the gate's shapes. A forged-golden row the classifier
    already excluded never reaches the stub."""
    rows = [
        grader_row(instance_id="acme__golden-1", eval_script=(
            "#!/bin/bash\ncd /testbed\n"
            f"git apply -v --allow-empty - <<'EOF_114329324912'\n{GOLDEN}EOF_114329324912\n"
            f"git apply -v --allow-empty - <<'EOF_114329324912'\n{TEST_PATCH}EOF_114329324912\n"
            'pytest -q\necho "OPENSWE_EXIT_CODE=$rc"\n'
        )),
        grader_row(instance_id="acme__golden-2", eval_script=(
            "#!/bin/bash\ncd /testbed\n"
            f"git apply -v --allow-empty - <<'EOF_114329324912'\n{GOLDEN}EOF_114329324912\n"
            f"git apply -v --allow-empty - <<'EOF_114329324912'\n{TEST_PATCH}EOF_114329324912\n"
            'pytest -q\necho "OPENSWE_EXIT_CODE=$rc"\n'
        )),
    ]
    factory = corpus_qualification.StubMachineFactory()
    population, counts = corpus_qualification.qualification_population(rows, frozenset())
    shard = corpus_qualification.shard_population(population, 0, 1)
    summary = asyncio.run(
        corpus_qualification.run_shard(shard, CONDENSED, factory, report_dir=tmp_path, concurrency=2)
    )
    assert summary["sampled"] == 2
    assert summary["qualified"] == 2
    assert summary["state"] == "ok"
    records = [json.loads(line) for line in (tmp_path / "rows.jsonl").read_text().splitlines()]
    assert all(record["status"] == "qualified" for record in records)
    for record in records:
        arms = record["arms"]["condensed"]
        assert arms["noop"]["reward"] == 0
        assert arms["oracle"]["reward"] == 1
        assert "OPENSWE_EXIT_CODE=0" in arms["oracle"]["output_tail"]
    result = json.loads((tmp_path / "fidelity-gate-result.json").read_text())
    assert result["sampled"] == 2 and result["qualified"] == 2
