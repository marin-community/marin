# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Static retention preserves source contracts without fabricating execution evidence."""

import hashlib
import json

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from finestore.rl.curation import (
    ActorCensusInput,
    ContentEvidenceReference,
    DatasetRelease,
    NoOracleExecutabilityPolicy,
    SourceUse,
)
from taskcompendium.convert.answers import exact_answer_task
from taskcompendium.identity import canonical_sha256
from taskcompendium.models import CommandSemantics, EnvironmentRequirements, NoGrader, ResourceGroups, Source, TaskSpec
from taskcompendium.pipeline.models import RawRow
from taskcompendium.pipeline.oracle_filter import load_oracle_filter, require_oracle_filter
from taskcompendium.pipeline.static_retention import NoOracleSourceContext
from taskcompendium.runtime.resources import inline_resource

from experiments.post_training.task_curation.harbor_inspection import native_task_checksums
from experiments.post_training.task_curation.static_retention import collect_static_retention


def no_oracle_answer(task_id):
    task = exact_answer_task(
        RawRow(
            task_id,
            Source(dataset="open-athena/static-fixture", revision="a" * 40, row=task_id, importer_revision="b" * 40),
            {},
        ),
        prompt="Return the requested label.",
        answers=("label",),
        ignore_case=False,
    )
    assert isinstance(task, TaskSpec)
    return task


def frozen_cohort(tmp_path, tasks, use=SourceUse.TRAIN):
    cohort = tmp_path / "input.parquet"
    rows = [{"task_spec": task.model_dump_json(), "runtime": json.dumps({"source": task.id})} for task in tasks]
    pq.write_table(pa.Table.from_pylist(rows), cohort)
    digest = hashlib.sha256(cohort.read_bytes()).hexdigest()
    release = DatasetRelease(
        repo="open-athena/static-fixture",
        revision="a" * 40,
        url="https://huggingface.co/datasets/open-athena/static-fixture/tree/" + "a" * 40,
    )
    context = NoOracleSourceContext(
        source_id="static-fixture",
        source_release=release,
        input_cohort_sha256=digest,
        current_input=ActorCensusInput(
            dataset_id=release.repo,
            dataset_revision=release.revision,
            converter_version="fixture-v1",
            converter_revision="b" * 40,
            measured_tasks=len(tasks),
            task_specs_sha256=canonical_sha256([task.model_dump(mode="json") for task in tasks]),
            evidence_url="s3://fixtures/static-fixture/current-input.json",
        ),
        intended_use=use,
        policy=NoOracleExecutabilityPolicy(
            version=1,
            executability_proof_required_for_no_oracle=False,
            decision=ContentEvidenceReference(uri="s3://fixtures/no-oracle-ruling.md", sha256="c" * 64),
        ),
    )
    return cohort, rows, context, ContentEvidenceReference(uri=str(cohort), sha256=digest)


def test_static_collection_preserves_task_and_runtime_rows_without_trials_or_resource_projection(tmp_path):
    answer = no_oracle_answer("answer")
    agentic = no_oracle_answer("agentic").model_copy(
        update={
            "environment_requirements": EnvironmentRequirements(
                docker_image="python@sha256:" + "d" * 64,
                command_semantics=CommandSemantics.LINUX_PROCESS,
            )
        }
    )
    cohort, original, context, evidence = frozen_cohort(tmp_path, [answer, agentic])
    output = tmp_path / "retained"
    receipt = collect_static_retention(cohort, context, evidence, output)
    loaded = load_oracle_filter(
        str(output / "oracle-filter.json"),
        source_context=context,
        expected_sha256=hashlib.sha256((output / "oracle-filter.json").read_bytes()).hexdigest(),
    )
    assert loaded == receipt
    with pytest.raises(ValueError, match="independently configured receipt SHA-256"):
        load_oracle_filter(str(output / "oracle-filter.json"), source_context=context)
    assert loaded.uniform_requirements is None
    assert pq.read_table(output / "tasks.parquet").to_pylist() == original
    assert all(task.execution_requirements is None for task in (answer, agentic))
    for task in (answer, agentic):
        assert require_oracle_filter(task, loaded, {"source": task.id}, source_context=context)
        entry = loaded.tasks[task.id].model_dump(mode="json")
        assert entry["outcome"] == "static_retained"
        assert entry["executability"] == "unmeasured"
        assert not {"trials", "calibration", "executability_run", "negative_control"}.intersection(entry)
    summary = json.loads((output / "static-retention.json").read_text())
    assert (summary["static_retained_count"], summary["unique_static_excluded_count"]) == (2, 0)
    assert (summary["source_commands"], summary["provider_creations"], summary["model_requests"]) == (0, 0, 0)


@pytest.mark.parametrize("field", ["source_id", "intended_use", "current_input", "policy"])
def test_public_static_load_rejects_another_source_use_converter_or_ruling(tmp_path, field):
    cohort, _, context, evidence = frozen_cohort(tmp_path, [no_oracle_answer("answer")])
    output = tmp_path / "retained"
    collect_static_retention(cohort, context, evidence, output)
    replacements = {
        "source_id": "foreign-source",
        "intended_use": SourceUse.EVAL,
        "current_input": context.current_input.model_copy(update={"converter_revision": "e" * 40}),
        "policy": context.policy.model_copy(
            update={"decision": ContentEvidenceReference(uri="s3://fixtures/foreign-ruling.md", sha256="f" * 64)}
        ),
    }
    with pytest.raises(ValueError, match="independently bound current source context"):
        load_oracle_filter(
            str(output / "oracle-filter.json"), source_context=context.model_copy(update={field: replacements[field]})
        )
    with pytest.raises(ValueError, match="independently bound current source context"):
        load_oracle_filter(str(output / "oracle-filter.json"))


def test_eval_static_retention_is_measured_but_cannot_supply_training_admission(tmp_path):
    task = no_oracle_answer("eval-answer")
    cohort, original, context, evidence = frozen_cohort(tmp_path, [task], SourceUse.EVAL)
    output = tmp_path / "retained"
    receipt = collect_static_retention(cohort, context, evidence, output)
    assert pq.read_table(output / "tasks.parquet").to_pylist() == original
    assert not require_oracle_filter(task, receipt, {"source": task.id}, source_context=context)


def test_static_native_retention_binds_the_actual_harbor_package_checksum(tmp_path):
    task = no_oracle_answer("native").model_copy(
        update={
            "tags": ("harbor",),
            "grader": NoGrader(reason="Original native verifier"),
            "resources": ResourceGroups(
                verifier=tuple(
                    inline_resource(name, value)
                    for name, value in {
                        "task.toml": (
                            b'[agent]\ntimeout_sec=900\n[verifier]\ntimeout_sec=600\n[environment]\ndocker_image="python@sha256:'
                            + b"d" * 64
                            + b'"\n'
                        ),
                        "instruction.md": b"Apply the original task.\n",
                        "environment/Dockerfile": b"FROM python@sha256:" + b"d" * 64 + b"\n",
                        "tests/test.sh": b"echo 0 > /logs/verifier/reward.txt\n",
                    }.items()
                )
            ),
        }
    )
    cohort, original, context, evidence = frozen_cohort(tmp_path, [task])
    output = tmp_path / "retained"
    receipt = collect_static_retention(cohort, context, evidence, output)
    checksum = native_task_checksums([task])[task.id]
    assert receipt.tasks[task.id].native_task_checksum == checksum
    assert require_oracle_filter(
        task, receipt, {"source": task.id}, source_context=context, native_task_checksum=checksum
    )
    with pytest.raises(ValueError, match="original native package"):
        require_oracle_filter(task, receipt, {"source": task.id}, source_context=context, native_task_checksum="e" * 64)
    assert pq.read_table(output / "tasks.parquet").to_pylist() == original


def test_static_collection_keeps_one_exclusion_per_original_task_and_preserves_an_empty_output(tmp_path):
    task = no_oracle_answer("mutable-native").model_copy(
        update={
            "tags": ("harbor",),
            "grader": NoGrader(reason="Original native verifier"),
            "resources": ResourceGroups(
                verifier=(inline_resource("task.toml", b'[environment]\ndocker_image="python:latest"\n'),)
            ),
        }
    )
    cohort, original, context, evidence = frozen_cohort(tmp_path, [task])
    output = tmp_path / "retained"
    receipt = collect_static_retention(cohort, context, evidence, output)
    assert pq.read_table(output / "tasks.parquet").num_rows == 0
    assert pq.read_table(cohort).to_pylist() == original
    assert receipt.tasks[task.id].outcome == "mechanical"
    summary = json.loads((output / "static-retention.json").read_text())
    assert (summary["static_retained_count"], summary["unique_static_excluded_count"]) == (0, 1)
    assert sum(summary["exclusion_counts"].values()) == 1


@pytest.mark.parametrize("presence", ["original_oracle", "unknown"])
def test_static_collection_rejects_oracle_bearing_or_unknown_population_before_writing(tmp_path, presence):
    task = no_oracle_answer("answer")
    if presence == "original_oracle":
        task = task.model_copy(
            update={
                "resources": ResourceGroups(
                    oracle=(inline_resource("solution/solve.sh", b"printf original-answer > /app/answer.txt\n"),)
                )
            }
        )
    cohort, rows, context, evidence = frozen_cohort(tmp_path, [task])
    if presence == "unknown":
        rows[0]["task_spec"] = None
        pq.write_table(pa.Table.from_pylist(rows), cohort)
        digest = hashlib.sha256(cohort.read_bytes()).hexdigest()
        context = context.model_copy(update={"input_cohort_sha256": digest})
        evidence = evidence.model_copy(update={"sha256": digest})
    with pytest.raises(ValueError, match="complete, unique, known no-oracle population"):
        collect_static_retention(cohort, context, evidence, tmp_path / "retained")
    assert not (tmp_path / "retained").exists()
