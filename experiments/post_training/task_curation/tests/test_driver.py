# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
import os
import shutil
import sys
from dataclasses import replace
from pathlib import Path, PurePosixPath
from typing import cast

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from click.testing import CliRunner
from finestore.rl.curation import CurationCatalog, OversightEvidence, OversightLevel
from fray.current_client import set_current_client
from fray.local_backend import LocalClient
from marin.execution.lazy import run
from rigging.filesystem.storage_path import StoragePath
from shellbox.machine import Command, HostImage, Mount, NetworkPolicy
from taskcompendium.models import CommandSemantics, EnvironmentRequirements
from taskcompendium.pipeline.inputs import SourceFormat
from taskcompendium.pipeline.source_processing import SourceProcessingMode
from taskcompendium.runtime.environment import prepare_machine_spec
from taskcompendium.runtime.local import LocalRuntime, local_runtime
from zephyr.readers import load_parquet

from experiments.post_training.glm import GLM_BULK_TOKEN_ENV
from experiments.post_training.task_curation import driver
from experiments.post_training.task_curation import pipeline as processor
from experiments.post_training.task_curation import sources as source_catalog
from experiments.post_training.task_curation.datasets.skyrl import math as skyrl_math
from experiments.post_training.task_curation.driver import main
from experiments.post_training.task_curation.environment import Environment
from experiments.post_training.task_curation.images.build import environment_artifact
from experiments.post_training.task_curation.pipeline import (
    CampaignMachines,
    CurationRecipe,
    HfSource,
    environment_requirements,
    process_rows,
)
from experiments.post_training.task_curation.source import CatalogConfig, RlDataSource, SourceInfo, SourceOrigin
from experiments.post_training.task_curation.tests.admission_fixtures import with_admission_census
from experiments.post_training.task_curation.tests.image_builds import (
    DATA_PATH,
    REPOSITORY,
    REQUIREMENTS,
    install_fake_archive,
    install_fake_build_tools,
)
from experiments.post_training.task_curation.tests.numbers_pipeline import number_source
from experiments.post_training.task_curation.tests.resource_receipts import resource_recipe, write_fixture_receipt
from lib.taskcompendium.tests.decontamination_fixtures import decontamination_evidence
from lib.taskcompendium.tests.deduplication_fixtures import deduplication_evidence


def math500() -> CurationRecipe:
    recipe = cast(CurationRecipe, next(source.config for source in skyrl_math.sources() if source.name == "math500"))
    return resource_recipe(replace(recipe, source=replace(recipe.source, repo="open-athena/math500-fixture")))


def source_with_input_evidence[ConfigT: CatalogConfig](source: RlDataSource[ConfigT]) -> RlDataSource[ConfigT]:
    dataset = source.dataset
    assert dataset is not None and dataset.revision is not None
    return with_admission_census(
        replace(
            source,
            info=replace(
                source.info,
                lexical_contamination=decontamination_evidence(dataset.name, dataset.revision, 1),
                within_source_dedup=deduplication_evidence(dataset.name, dataset.revision, source.version),
            ),
        )
    )


@pytest.fixture
def catalog(monkeypatch, tmp_path):
    pipelines = {name: replace(math500(), name=name) for name in ("first", "second", "third")}
    monkeypatch.setattr(
        "experiments.post_training.task_curation.driver.runnable_sources",
        lambda: {
            **{
                name: RlDataSource(
                    pipeline=process_rows,
                    info=SourceInfo(
                        within_source_dedup=deduplication_evidence(
                            pipeline.dataset.name, pipeline.dataset.revision, pipeline.version
                        ),
                        id=f"Marin:{name}",
                        upstream_dataset=name,
                        upstream_name=name,
                        build_path="test",
                        origin=SourceOrigin.MARIN,
                        lexical_contamination=decontamination_evidence(
                            pipeline.dataset.name, pipeline.dataset.revision, 1
                        ),
                    ),
                    config=pipeline,
                )
                for name, pipeline in pipelines.items()
            },
            "numbers": number_source(tmp_path / "not-yet-generated"),
        },
    )
    return pipelines


def arguments(tmp_path) -> list[str]:
    options = {
        "--mode": "sample",
        "--review-cache": str(tmp_path / "cache"),
        "--max-workers": "1",
        "--coordinator-memory": "16g",
        "--normalized-shards": "1",
        "--worker-image": "fixture-image",
        "--report-path": str(tmp_path / "report.json"),
    }
    return [item for pair in options.items() for item in pair]


def test_source_option_selects_catalog_order_without_changing_identity(tmp_path, catalog):
    runner = CliRunner()
    full = runner.invoke(main, arguments(tmp_path))
    subset = runner.invoke(main, [*arguments(tmp_path), "--source", "third", "--source", "first"])
    assert full.exit_code == 0, full.output
    assert subset.exit_code == 0, subset.output
    planned = json.loads(full.output)["sources"]
    assert len(planned) == 4 and planned[-1]["name"] == "data/rl/numbers"
    assert json.loads(subset.output)["sources"] == [planned[0], planned[2]]

    unknown = runner.invoke(main, [*arguments(tmp_path), "--source", "unknown", "--run"])
    assert unknown.exit_code == 2
    assert "Unknown source: unknown" in unknown.output


def test_local_canary_cannot_write_an_explicit_production_catalog(tmp_path, catalog):
    production = tmp_path / "production"
    result = CliRunner().invoke(
        main,
        [
            *arguments(tmp_path),
            "--mode",
            "full",
            "--cache-dir",
            str(tmp_path / "canary"),
            "--publish-catalog",
            str(production),
            "--run",
        ],
    )
    assert result.exit_code == 2
    assert not production.exists()


@pytest.mark.skipif(sys.platform != "linux", reason="The local bubblewrap backend requires Linux")
def test_local_environments_grade_in_the_worker_with_the_runtime_built_from_their_lock(tmp_path, monkeypatch, request):
    monkeypatch.setenv("MARIN_PREFIX", str(tmp_path / "prefix"))
    install_fake_build_tools(tmp_path, monkeypatch)
    # The factory probes for a working bubblewrap; CI hosts cannot build sandboxes.
    monkeypatch.setattr(
        "shellbox.backends.local.machine._working_bwrap",
        lambda candidates, system_directories, mounts, environment: Path("/usr/bin/bwrap"),
    )
    environment = Environment(
        requirements=REQUIREMENTS,
        files={DATA_PATH: install_fake_archive(monkeypatch)},
        env={"NLTK_DATA": "/srv/nltk_data"},
    )
    (artifact,) = run(environment_artifact(environment, REPOSITORY))
    monkeypatch.setattr(LocalRuntime, "ensure_built", lambda self: self.root.mkdir(parents=True))
    machines = CampaignMachines()
    requirements = environment_requirements(environment, artifact)
    factory, spec = machines.machine(requirements, 2048)
    spec = prepare_machine_spec(requirements, factory, spec, {})
    runtime = local_runtime(artifact.lock_url)
    request.addfinalizer(lambda: shutil.rmtree(runtime.root, ignore_errors=True))
    assert runtime.root.is_dir()
    data = Mount(runtime.root / "files" / DATA_PATH.removeprefix("/"), PurePosixPath(DATA_PATH))
    assert spec.source == HostImage(read_only=(runtime.root,), bin_dirs=(runtime.bin_dir,), mounts=(data,))
    assert spec.memory_mb is None and spec.network == NetworkPolicy.DENY and spec.workdir == "/app"
    assert spec.env == {"NLTK_DATA": "/srv/nltk_data"}


@pytest.mark.parametrize("root", [None, "s3://marin-us-east-02a/marin/curation-trial"])
def test_remote_planning_uses_cw_instead_of_ambient_gcs(tmp_path, catalog, monkeypatch, root):
    previous = "gs://marin-us-central2/rl-grading"
    monkeypatch.setenv("MARIN_PREFIX", previous)
    options = arguments(tmp_path)
    if root is not None:
        options += ["--artifact-root", root]
    result = CliRunner().invoke(main, options)
    assert result.exit_code == 0, result.output
    plan = json.loads(result.output)
    assert plan["chunk_storage_prefix"] == f"{root or 's3://marin-us-east-02a/marin'}/.zephyr/rl-data-campaign"
    assert os.environ["MARIN_PREFIX"] == previous
    assert not (tmp_path / "report.json").exists()


def test_remote_campaign_plans_with_canonical_execution_defaults(tmp_path, catalog):
    # An Iris campaign uses its pinned reviewer and resolves execution defaults once.
    root = "s3://marin-us-east-02a/marin/curation-trial"
    minimal = [
        "--mode",
        "sample",
        "--report-path",
        str(tmp_path / "report.json"),
        "--artifact-root",
        root,
    ]
    result = CliRunner().invoke(main, minimal)
    assert result.exit_code == 0, result.output
    plan = json.loads(result.output)
    assert plan["max_workers"] == 64
    assert plan["coordinator_resources"]["ram"] == "16g"
    # No image override: Iris starts the workers from the cluster's configured task image.
    assert plan["worker_resources"]["image"] is None
    assert plan["review_cache"] == f"{root}/rl-data/review-cache"

    overrides = ["--max-workers", "8", "--coordinator-memory", "8g", "--worker-image", "fixture-image"]
    overrides += ["--review-cache", str(tmp_path / "cache")]
    result = CliRunner().invoke(main, [*minimal, *overrides])
    assert result.exit_code == 0, result.output
    plan = json.loads(result.output)
    assert plan["max_workers"] == 8
    assert plan["coordinator_resources"]["ram"] == "8g"
    assert plan["worker_resources"]["image"] == "fixture-image"
    assert plan["review_cache"] == str(tmp_path / "cache")


def test_local_campaign_keeps_its_review_cache_in_the_cache_directory(tmp_path, catalog):
    cache = tmp_path / "canary"
    options = ["--mode", "sample"]
    result = CliRunner().invoke(main, [*options, "--cache-dir", str(cache)])
    assert result.exit_code == 0, result.output
    plan = json.loads(result.output)
    assert plan["review_cache"] == str(cache.resolve() / "review-cache")
    assert plan["max_workers"] == 4


def test_gvisor_image_backend_requires_an_in_cluster_driver(tmp_path, catalog, monkeypatch):
    monkeypatch.setattr("experiments.post_training.task_curation.driver.get_job_info", lambda: None)
    options = [*arguments(tmp_path), "--image-backend", "gvisor"]
    missing = CliRunner().invoke(main, options)
    assert missing.exit_code == 2
    assert not (tmp_path / "report.json").exists()


def test_quick_cli_plans_then_converts_once_in_request_order(tmp_path, monkeypatch):
    declared = {source.name: source for source in skyrl_math.sources()}
    sources = {name: source_with_input_evidence(declared[name]) for name in ("math500", "aime24")}
    monkeypatch.setattr("experiments.post_training.task_curation.driver.runnable_sources", lambda: sources)
    monkeypatch.delenv("GLM_BULK_TOKEN", raising=False)
    output, cache = tmp_path / "output", tmp_path / "downloads"
    options = {
        "--mode": "quick",
        "--output-root": str(output),
        "--cache-dir": str(cache),
        "--max-workers": "1",
    }
    args = [item for pair in options.items() for item in pair]
    args += ["--source", "math500", "--source", "aime24", "--source", "math500"]
    runner = CliRunner()
    result = runner.invoke(main, args)
    assert result.exit_code == 0, result.output
    assert [source["name"] for source in json.loads(result.output)["sources"]] == ["math500", "aime24"]
    assert not output.exists()
    assert not cache.exists()

    staged = tmp_path / "input"
    (staged / "data").mkdir(parents=True)
    (staged / "test.jsonl").write_text('{"problem": "Two plus two?", "answer": "4"}\n')
    pq.write_table(
        pa.Table.from_pylist([{"problem": "One plus one?", "answer": "2"}]), staged / "data/train-0000.parquet"
    )
    result = runner.invoke(main, [*args, "--input-root", str(staged), "--run"])
    assert result.exit_code == 0, result.output
    report = json.loads((output / "campaign.json").read_text())
    assert [source["name"] for source in report["sources"]] == ["math500", "aime24"]
    assert report["counts"] == {"completed": 2}
    for name, source_row in (("math500", "test.jsonl:0"), ("aime24", "data/train-0000.parquet:0")):
        rows = [row for shard in (output / name / "normalize").glob("*.parquet") for row in load_parquet(str(shard))]
        assert [row["source_row"] for row in rows] == [source_row]


@pytest.mark.parametrize("option", ["--input-root", "--input-file", "--input", "--output-root"])
def test_reviewed_cli_rejects_local_input_and_output_options(tmp_path, catalog, option):
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    file = inputs / "fixture.jsonl"
    file.write_text('{"prompt": "fixture"}\n')
    values = {
        "--input-root": [str(inputs)],
        "--input-file": ["fixture.jsonl", str(file)],
        "--input": ["answers", str(inputs)],
        "--output-root": [str(tmp_path / "output")],
    }
    result = CliRunner().invoke(main, [*arguments(tmp_path), option, *values[option]])
    assert result.exit_code == 2
    assert not (tmp_path / "report.json").exists()
    assert not (tmp_path / "output").exists()
    assert not (tmp_path / "downloads").exists()


@pytest.mark.parametrize("mode, expected", [("quick", [2, 4, 6]), ("sample", [2, 4]), ("full", [2, 4, 6])])
def test_dataset_cli_runs_own_ingestion_without_review_controller_or_build_settings(
    tmp_path, monkeypatch, mode, expected
):
    monkeypatch.delenv(GLM_BULK_TOKEN_ENV, raising=False)
    monkeypatch.setenv("MARIN_PREFIX", str(tmp_path / "artifacts"))
    monkeypatch.setattr("rigging.filesystem.cluster_config.region_from_metadata", lambda: None)
    input_file = tmp_path / "numbers.txt"
    input_file.write_text("1\n2\n3\n")
    source = number_source(input_file)
    source = source_with_input_evidence(source)
    monkeypatch.setattr("experiments.post_training.task_curation.driver.runnable_sources", lambda: {source.name: source})
    options = [
        "--mode",
        mode,
        "--skip-assessment",
        "--source",
        source.name,
        "--max-workers",
        "1",
        "--artifact-root",
        str(tmp_path / "artifacts"),
    ]
    if mode == "quick":
        output_root = tmp_path / "output"
        report = output_root / "campaign.json"
        options += ["--output-root", str(output_root), "--cache-dir", str(tmp_path / "downloads")]
    else:
        report = tmp_path / "report.json"
        options += ["--coordinator-memory", "1g", "--worker-image", "fixture-image", "--report-path", str(report)]
    runner = CliRunner()
    planned = runner.invoke(main, options)
    assert planned.exit_code == 0, planned.output
    assert len(json.loads(planned.output)["sources"]) == 1
    assert not report.exists()
    assert not (tmp_path / "downloads").exists()
    assert not (tmp_path / "artifacts").exists()
    client = LocalClient()
    try:
        with set_current_client(client):
            executed = runner.invoke(main, [*options, "--run"])
        assert executed.exit_code == 0, executed.output
        outcome = json.loads(report.read_text())["sources"][0]
        result = outcome["result"]
        assert json.loads(StoragePath(result["outputs"]["numbers"]).read_text()) == expected
        evidence = json.loads(StoragePath(result["evidence"]["ingestion"]).read_text())
        assert evidence["mode"] == mode
        assert evidence["catalog_id"] == source.info.id
        if mode == "quick":
            # The generated pipeline really consumes its upstream artifact.
            ingested = list((tmp_path / "downloads").rglob("numbers.txt"))
            assert len(ingested) == 1
            assert ingested[0].read_text() == "1\n2\n3\n"
            input_file.unlink()
            StoragePath(result["outputs"]["numbers"]).write_text("[]")
            repeated = runner.invoke(main, [*options, "--run"])
            assert repeated.exit_code == 0, repeated.output
            assert json.loads(StoragePath(result["outputs"]["numbers"]).read_text()) == expected
            assert Path(os.environ["MARIN_PREFIX"]) == tmp_path / "artifacts"
    finally:
        client.shutdown()


@pytest.mark.parametrize("mode", ["sample", "full"])
@pytest.mark.parametrize("rubric", [None, "Check the reference answer."])
def test_recipe_cli_plans_pinned_review_and_requires_credentials_only_for_execution(tmp_path, monkeypatch, mode, rubric):
    monkeypatch.delenv(GLM_BULK_TOKEN_ENV, raising=False)
    monkeypatch.setenv("MARIN_PREFIX", str(tmp_path / "artifacts"))
    monkeypatch.setattr("rigging.filesystem.cluster_config.region_from_metadata", lambda: None)
    declared = next(source for source in skyrl_math.sources() if source.name == "math500")
    declared = replace(
        declared,
        info=replace(
            declared.info,
            daytona_machine_slots=0,
            daytona_snapshot_count=0,
            daytona_execution_failures=0,
            daytona_count_basis="Fixture numeric rows use a local reply grader and create no actor.",
        ),
    )
    source = replace(
        declared,
        config=replace(
            math500(),
            source=HfSource("open-athena/fixture-questions", "a" * 40, ("rows.jsonl",), SourceFormat.JSONL),
            rubric=rubric,
            controls=None,
            grader=None,
        ),
    )
    remote = tmp_path / "remote"
    remote.mkdir()
    (remote / "rows.jsonl").write_text('{"problem": "Two plus two?", "answer": "4"}\n')
    original = processor.plan_download
    monkeypatch.setattr(
        processor, "plan_download", lambda request: original(replace(request, source_url_override=str(remote)))
    )
    evidence = decontamination_evidence(source.dataset.name, source.dataset.revision, 1)
    decontamination_path = tmp_path / "decontamination.json"
    decontamination_path.write_text(evidence.model_dump_json())
    source = replace(
        source,
        info=replace(
            source.info,
            within_source_dedup=deduplication_evidence(source.dataset.name, source.dataset.revision, source.version),
        ),
    )
    source = with_admission_census(source)
    monkeypatch.setattr("experiments.post_training.task_curation.driver.runnable_sources", lambda: {source.name: source})
    report = tmp_path / "report.json"
    options = [
        "--mode",
        mode,
        "--decontamination-receipt",
        str(decontamination_path),
        "--skip-assessment",
        "--max-workers",
        "1",
        "--coordinator-memory",
        "1g",
        "--worker-image",
        "fixture-image",
        "--normalized-shards",
        "1",
        "--report-path",
        str(report),
        "--artifact-root",
        str(tmp_path / "artifacts"),
    ]
    runner = CliRunner()
    options += write_fixture_receipt(source.config, remote, tmp_path / "resource-receipt.json")
    planned = runner.invoke(main, options)
    assert planned.exit_code == 0, planned.output
    assert not (tmp_path / "artifacts").exists()
    assert not report.exists()
    client = LocalClient()
    try:
        with set_current_client(client):
            executed = runner.invoke(main, [*options, "--run"])
        if rubric is not None:
            assert executed.exit_code == 2, executed.output
            assert not (tmp_path / "artifacts").exists()
            assert not report.exists()
            return
        assert executed.exit_code == 0, executed.output
        result = json.loads(report.read_text())["sources"][0]["result"]
        manifest = json.loads(StoragePath(result["evidence"]["manifest"]).read_text())
        assert manifest["lexical_contamination"] == evidence.model_dump(mode="json")
        rows = [
            row
            for shard in (StoragePath(result["outputs"]["final"]) / "*.parquet").glob()
            for row in load_parquet(str(shard))
        ]
        assert [row["source_row"] for row in rows] == ["rows.jsonl:0"]
    finally:
        client.shutdown()


def test_shell_only_campaign_does_not_load_ports_catalog(tmp_path):
    machines = CampaignMachines(simulator_catalog=(tmp_path / "absent-catalog.json").as_uri())
    factory, spec = machines.machine(EnvironmentRequirements(command_semantics=CommandSemantics.SHELL_SIMULATOR), 2048)

    async def check():
        machine = await factory.create(spec)
        try:
            result = await machine.run(Command(("sh", "-c", "echo hello > answer; cat answer")))
            assert (result.exit_code, result.stdout) == (0, b"hello\n")
        finally:
            await machine.close()

    asyncio.run(check())


def test_local_sample_reviews_selected_tasks_and_reuses_request_cache(tmp_path, monkeypatch, review_server, qa_reviewer):
    declared = next(source for source in skyrl_math.sources() if source.name == "math500")
    declared = replace(
        declared,
        info=replace(
            declared.info,
            daytona_machine_slots=0,
            daytona_snapshot_count=0,
            daytona_execution_failures=0,
            daytona_count_basis="Fixture numeric rows use a local reply grader and create no actor.",
        ),
    )
    source = replace(
        declared,
        config=replace(
            math500(),
            source=HfSource("open-athena/fixture-questions", "a" * 40, ("rows.jsonl",), SourceFormat.JSONL),
            controls=None,
        ),
    )
    source = source_with_input_evidence(source)
    monkeypatch.setattr("experiments.post_training.task_curation.driver.runnable_sources", lambda: {source.name: source})
    monkeypatch.setenv(GLM_BULK_TOKEN_ENV, "fixture-token")
    staged = tmp_path / "input"
    staged.mkdir()
    (staged / "rows.jsonl").write_text(
        "\n".join(json.dumps({"problem": f"What is {value} plus one?", "answer": str(value + 1)}) for value in range(6))
    )
    cache = tmp_path / "cache"
    args = [
        "--mode",
        "sample",
        "--cache-dir",
        str(cache),
        "--source",
        source.name,
        "--input-root",
        str(staged),
        "--quality-sample-size",
        "2",
        "--max-workers",
        "1",
        "--review-concurrency",
        "1",
        "--run",
    ]
    runner = CliRunner()
    args += write_fixture_receipt(source.config, staged, tmp_path / "resource-receipt.json")
    first = runner.invoke(main, [*args, "--normalized-shards", "1"])
    assert first.exit_code == 0, first.output
    outcome = json.loads((cache / "campaigns/campaign.json").read_text())["sources"][0]
    manifest = json.loads((StoragePath(outcome["path"]) / "manifest.json").read_text())
    assert manifest["raw_population_count"] == 6 and manifest["raw_sample_count"] == 2
    assert len(review_server.requests) == 2
    assert (cache / "review-cache").exists()
    repeated = runner.invoke(main, [*args, "--normalized-shards", "2"])
    assert repeated.exit_code == 0, repeated.output
    rebuilt = json.loads((cache / "campaigns/campaign.json").read_text())["sources"][0]
    assert rebuilt["path"] != outcome["path"]
    assert len(review_server.requests) == 2


def test_sample_campaign_assesses_only_with_assess_flag(tmp_path, monkeypatch, review_server, qa_reviewer):
    declared = next(source for source in skyrl_math.sources() if source.name == "math500")
    declared = replace(
        declared,
        info=replace(
            declared.info,
            daytona_machine_slots=0,
            daytona_snapshot_count=0,
            daytona_execution_failures=0,
            daytona_count_basis="Fixture numeric rows use a local reply grader and create no actor.",
        ),
    )
    source = replace(declared, config=replace(math500(), rubric=None, controls=None))
    source = source_with_input_evidence(source)
    monkeypatch.setattr("experiments.post_training.task_curation.driver.runnable_sources", lambda: {source.name: source})
    monkeypatch.setenv(GLM_BULK_TOKEN_ENV, "fixture-token")
    staged = tmp_path / "input"
    staged.mkdir()
    (staged / "test.jsonl").write_text(
        "\n".join(json.dumps({"problem": f"What is {value} plus one?", "answer": str(value + 1)}) for value in range(3))
    )
    cache = tmp_path / "cache"
    args = [
        "--mode",
        "sample",
        "--cache-dir",
        str(cache),
        "--source",
        source.name,
        "--input-root",
        str(staged),
        "--max-workers",
        "1",
        "--quality-sample-size",
        "2",
        "--assessment-sample-size",
        "1",
        "--run",
    ]
    args += write_fixture_receipt(source.config, staged, tmp_path / "resource-receipt.json")
    result = CliRunner().invoke(main, args)
    assert result.exit_code == 0, result.output
    catalog = CurationCatalog(str(cache / "campaigns/catalog"))
    row = next(row for row in catalog.sources(catalog.current().snapshot_id) if row.id == source.info.id)
    assert row.assessment_key is None
    assert review_server.requests == []

    result = CliRunner().invoke(main, [*args, "--assess"])
    assert result.exit_code == 0, result.output
    catalog = CurationCatalog(str(cache / "campaigns/catalog"))
    row = next(row for row in catalog.sources(catalog.current().snapshot_id) if row.id == source.info.id)
    assessment = catalog.assessment(row.assessment_key)
    items = list(catalog.items(row.assessment_key))
    assert [(item.cohort, item.verdict) for item in items] == [("admitted", "good")]
    assert {population.cohort: population.population for population in catalog.populations(row.assessment_key)} == {
        "admitted": 2,
        "excluded": 0,
    }
    assert assessment.scope == "sample"
    assert len(review_server.requests) == 1


def test_full_campaign_requires_qa_transport_availability(tmp_path, monkeypatch, review_server, qa_reviewer):
    declared = next(source for source in skyrl_math.sources() if source.name == "math500")
    declared = replace(
        declared,
        info=replace(
            declared.info,
            daytona_machine_slots=0,
            daytona_snapshot_count=0,
            daytona_execution_failures=0,
            daytona_count_basis="Fixture numeric rows use a local reply grader and create no actor.",
        ),
    )
    source = replace(declared, config=replace(math500(), controls=None))
    source = source_with_input_evidence(source)
    monkeypatch.setattr("experiments.post_training.task_curation.driver.runnable_sources", lambda: {source.name: source})
    monkeypatch.setenv(GLM_BULK_TOKEN_ENV, "fixture-token")
    review_server.review_available = False
    staged = tmp_path / "input"
    staged.mkdir()
    (staged / "test.jsonl").write_text(
        '{"problem":"One plus one?","answer":"2"}\n{"problem":"Two plus one?","answer":"3"}\n'
    )
    cache = tmp_path / "cache"
    args = [
        "--mode",
        "full",
        "--cache-dir",
        str(cache),
        "--input-root",
        str(staged),
        "--max-workers",
        "1",
        "--quality-sample-size",
        "1",
        "--review-concurrency",
        "1",
        "--run",
    ]
    args += write_fixture_receipt(source.config, staged, tmp_path / "resource-receipt.json")
    result = CliRunner().invoke(main, args)
    assert result.exit_code != 0
    campaign = json.loads((cache / "campaigns/campaign.json").read_text())
    outcome = campaign["sources"][0]
    assert campaign["status"] == "failed" and outcome["status"] == "incomplete"
    catalog = CurationCatalog(str(cache / "campaigns/catalog"))
    catalog_source = next(
        row for row in catalog.sources(catalog.current().snapshot_id) if row.declaration.name == "math500"
    )
    assert catalog_source.curation.status == "incomplete"
    assert [(item.cohort, item.status) for item in catalog.items(catalog_source.assessment_key)] == [
        ("excluded", "reviewed"),
    ]
    requests = list(review_server.requests)
    assert sum(request["tool_choice"]["function"]["name"] == "assess_task" for request in requests) == 1
    repeated = CliRunner().invoke(main, args)
    assert repeated.exit_code != 0
    assert review_server.requests == requests
    review_server.review_available = True
    retried = CliRunner().invoke(main, [*args, "--curation-attempt", "1"])
    assert retried.exit_code == 0, retried.output
    recovered = json.loads((cache / "campaigns/campaign.json").read_text())
    assert recovered["sources"][0]["status"] == "completed"
    assert recovered["sources"][0]["path"] != outcome["path"]


def test_mutable_decontamination_receipt_cannot_plan_a_reviewed_campaign(tmp_path, catalog):
    dataset = math500().dataset
    receipt = decontamination_evidence(dataset.name, dataset.revision, 1).model_dump(mode="json")
    receipt["receipt"]["release"]["dataset_revision"] = "main"
    path = tmp_path / "mutable-receipt.json"
    path.write_text(json.dumps(receipt))
    result = CliRunner().invoke(main, [*arguments(tmp_path), "--decontamination-receipt", str(path)])
    assert result.exit_code == 1
    assert isinstance(result.exception, ValueError)
    assert "dataset_revision" in str(result.exception)
    assert not (tmp_path / "report.json").exists()


def test_quick_conversion_can_collect_actor_evidence_before_catalog_publication(tmp_path, monkeypatch):
    input_path = tmp_path / "numbers.txt"
    input_path.write_text("1\n2\n")
    source = number_source(input_path)
    source = replace(source, info=replace(source.info, daytona_machine_slots=None, daytona_count_basis=None))
    monkeypatch.setattr("experiments.post_training.task_curation.driver.all_sources", lambda: {source.info.id: source})
    monkeypatch.setattr("experiments.post_training.task_curation.driver.runnable_sources", lambda: {})
    result = CliRunner().invoke(
        main,
        [
            "--mode",
            "quick",
            "--source",
            source.name,
            "--cache-dir",
            str(tmp_path / "cache"),
            "--output-root",
            str(tmp_path / "output"),
            "--max-workers",
            "1",
            "--run",
        ],
    )
    assert result.exit_code == 0, result.output
    output = tmp_path / "output" / source.name / "numbers.json"
    assert json.loads(output.read_text()) == [2, 4]
    assert source.catalog_declaration is None


def test_quick_selection_does_not_require_admission_for_unrelated_sources(tmp_path, monkeypatch):
    source = number_source(tmp_path / "input")
    unrelated = number_source(tmp_path / "other", "unqualified")
    unrelated = replace(
        unrelated,
        info=replace(
            unrelated.info,
            oversight=OversightEvidence(level=OversightLevel.SAMPLE, verification="Executed fixture controls"),
        ),
    )
    monkeypatch.setattr(driver, "all_sources", lambda: {source.info.id: source, unrelated.info.id: unrelated})

    monkeypatch.setattr(source_catalog, "all_sources", driver.all_sources)
    monkeypatch.setattr(driver, "runnable_sources", source_catalog.runnable_sources)
    assert driver._selected_sources((source.name,), SourceProcessingMode.QUICK) == {source.name: source}
