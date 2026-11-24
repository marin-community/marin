# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Seed the disposable EvalDash catalog and keep its sample files through the session."""

import asyncio
import tempfile
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime

import httpx
import pytest
from evaldash import app as evaldash_app
from evaldash import fixtures, results_db
from evalrecords.records import EvalRunRecord, RunStatus, record_path
from marina.db import DatabaseSpec, engine_for
from marina.journeys import Kernel

APP = "evaldash"
SCALE_COHORT = "fixture-scale"
CATALOG_RELOAD_TIMEOUT = 30.0


def _ingest_fixture_records(database: DatabaseSpec, prefix: str) -> None:
    engine = engine_for(database, APP)
    try:
        with engine.begin() as connection:
            evaldash_app.migrate(connection)
        ingestor = evaldash_app.PostgresIngestor(
            engine,
            (prefix,),
            evaldash_app.DEFAULT_INGEST_INTERVAL,
            evaldash_app.DEFAULT_REVALIDATE_AFTER,
        )
        # Another app's journeys may already have started Playwright's main-thread loop.
        with ThreadPoolExecutor(max_workers=1) as pool:
            pool.submit(asyncio.run, ingestor.run_once()).result()
    finally:
        engine.dispose()


@pytest.fixture(scope="session", autouse=True)
def seeded_records(marina_database: DatabaseSpec) -> Iterator[str]:
    with tempfile.TemporaryDirectory(prefix="evaldash-records-") as prefix, pytest.MonkeyPatch.context() as env:
        env.setenv(evaldash_app.PREFIXES_ENV, prefix)
        env.setenv(evaldash_app.STORE_ENV, evaldash_app.StoreMode.POSTGRES.value)
        env.delenv(evaldash_app.INGEST_JOB_ENV, raising=False)
        fixtures.build_fixtures(prefix)
        fixtures.build_native_fixtures(prefix)
        _ingest_fixture_records(marina_database, prefix)
        yield prefix


def _scale_records(base: EvalRunRecord, prefix: str) -> Iterator[EvalRunRecord]:
    (task,) = base.evaluation.tasks
    assert task.benchmark is not None
    evaluations = []
    for index in range(50):
        name = f"scale-benchmark-{index:02d}"
        benchmark = task.benchmark.model_copy(update={"task": name})
        evaluation_task = task.model_copy(update={"name": name, "benchmark": benchmark})
        evaluations.append(base.evaluation.model_copy(update={"name": name, "tasks": (evaluation_task,)}))

    for index in range(100):
        name = f"scale-model-{index:03d}"
        model = base.model.model_copy(update={"name": name, "location": name})
        for evaluation in evaluations:
            failed = index == 75 and evaluation is evaluations[-1]
            correct = 0 if index == 0 and evaluation is evaluations[0] else 500 + index
            coverage = base.coverage[task.name].model_copy(update={"n_correct": correct})
            score = correct / 1000
            run_id = f"{name}-{evaluation.name}"
            yield base.model_copy(
                update={
                    "run_id": run_id,
                    "group_id": name,
                    "created_at": "2026-06-01T12:00:00+00:00",
                    "timing": None,
                    "version": SCALE_COHORT,
                    "description": None,
                    "model": model,
                    "evaluation": evaluation,
                    "results_path": f"{prefix}/{run_id}/results",
                    "jobs": {},
                    "status": RunStatus.FAILED if failed else RunStatus.SUCCEEDED,
                    "error": "Fixture grader failure" if failed else None,
                    "metrics": {} if failed else {evaluation.name: {"acc,none": score}},
                    "canonical_metrics": {} if failed else {evaluation.name: {"accuracy": score}},
                    "coverage": {} if failed else {evaluation.name: coverage},
                }
            )


def _reload_catalog(kernel: Kernel) -> None:
    response = httpx.post(f"{kernel.origin}/{APP}/api/refresh", timeout=CATALOG_RELOAD_TIMEOUT)
    # The local endpoint reloads its serving snapshot without a scheduled ingestion job.
    assert response.is_success or response.status_code == 503, response.text


@pytest.fixture(scope="module")
def scale_cohort(marina_kernel: Kernel, marina_database: DatabaseSpec, seeded_records: str) -> Iterator[str]:
    prefix = f"{seeded_records}/{SCALE_COHORT}"
    engine = engine_for(marina_database, APP)
    try:
        base = next(
            record for record in results_db.fetch_snapshot(engine).records if record.run_id == "snowball-2026.07.20-mmlu"
        )
        now = datetime.now(UTC)
        observations = [
            results_db.RecordObservation(
                path=record_path(prefix, record.run_id),
                object_version="fixture",
                verified_at=now,
                next_verify_at=now,
                kind=results_db.ObservationKind.CHANGED,
                run_id=record.run_id,
                record=record,
            )
            for record in _scale_records(base, prefix)
        ]
        results_db.configure_prefixes(engine, (seeded_records, prefix))
        results_db.reconcile_prefix(
            engine, prefix, [observation.path for observation in observations], observations, now, 0
        )
        _reload_catalog(marina_kernel)
        yield SCALE_COHORT
    finally:
        results_db.configure_prefixes(engine, (seeded_records,))
        engine.dispose()
        _reload_catalog(marina_kernel)
