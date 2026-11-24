# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# Copyright The Authors
# SPDX-License-Identifier: Apache-2.0

"""The Difficulty coverage classifier: measured, needs-measurement and blocked."""

import json
from datetime import UTC, datetime

import pytest
from finestore.rl.curation import DatasetRelease
from finestore.rl.difficulty import (
    PROTOCOL_ID,
    DifficultyGap,
    DifficultyMeasurement,
    DifficultyObservation,
    MeasurementStatus,
    Solver,
    difficulty_gap_reason,
)
from sqlalchemy import create_engine, text

import infra.marina.apps.rl_data_catalog.load_difficulty as load_difficulty
from infra.marina.apps.rl_data_catalog.load_difficulty import (
    difficulty_records_from_query,
    load_measurements,
    publish_coverage,
)
from infra.marina.apps.rl_data_catalog.server.difficulty_coverage import (
    CoverageStatus,
    DifficultyCoverage,
    campaign_queue,
    difficulty_coverage,
    document_digest,
    source_coverage,
)
from infra.marina.apps.rl_data_catalog.server.schema import SCHEMA_SQL

RELEASE = DatasetRelease(
    repo="open-athena/example-retained",
    revision="a" * 40,
    url="https://huggingface.co/datasets/open-athena/example-retained/tree/" + "a" * 40,
)
CONVERTER_VERSION = "2026.11.16.1"


def declaration(
    status: str = "Available",
    *,
    source_id: str = "Example:example",
    release: DatasetRelease = RELEASE,
    version: str = CONVERTER_VERSION,
) -> dict:
    return {
        "id": source_id,
        "status": status,
        "current_release": release.model_dump(mode="json"),
        "pipeline": {"version": version},
        "actor_census_input": {
            "dataset_id": "example/upstream",
            "dataset_revision": "b" * 40,
            "converter_version": version,
            "converter_revision": "c" * 40,
            "measured_tasks": 100,
            "task_specs_sha256": "d" * 64,
            "evidence_url": "s3://example/census",
        },
    }


PROTOCOL_MODELS = {
    Solver.SMALL: "Qwen/Qwen3-Coder-30B-A3B-Instruct",
    Solver.LARGE: "Qwen/Qwen3.5-122B-A10B",
    Solver.HOSTED: "zai-org/GLM-5.3",
    Solver.SNOWBALL: "open-athena/Marin-Snowball-67B-A2B-v0.1",
}
PROTOCOL_THINKING = {
    Solver.SMALL: "off",
    Solver.LARGE: "on",
    Solver.HOSTED: "low",
    Solver.SNOWBALL: "checkpoint",
}


def _sample(source_id: str, release: DatasetRelease, converter_version: str) -> dict:
    return {
        "source_id": source_id,
        "release": release.model_dump(mode="json"),
        "converter_version": converter_version,
        "converter_revision": "c" * 40,
        "population_count": 100,
        "task_specs_sha256": "d" * 64,
        "drop_manifest": {"uri": "s3://example/drops", "sha256": "0" * 64},
        "seed": 0,
        "pre_oracle_filter": False,
        "resample_after_oracle_filter": False,
    }


def observation(
    solver: Solver,
    *,
    status: MeasurementStatus = MeasurementStatus.MEASURED,
    reason: str | None = None,
    measurement_id: str = "m1",
    release: DatasetRelease = RELEASE,
    converter_version: str = CONVERTER_VERSION,
    source_id: str = "Example:example",
) -> DifficultyObservation:
    measured = status is MeasurementStatus.MEASURED
    return DifficultyObservation(
        measurement_id=measurement_id,
        protocol_id=PROTOCOL_ID,
        created_at=datetime.now(UTC).isoformat(),
        source_id=source_id,
        release=release,
        converter_version=converter_version,
        solver=solver,
        status=status,
        reason=reason,
        sample=_sample(source_id, release, converter_version) if measured else None,
        sample_receipt={"uri": "s3://example/sample", "sha256": "2" * 64} if measured else None,
        model=(
            {
                "model": PROTOCOL_MODELS[solver],
                "revision": "f" * 40,
                "tokenizer": PROTOCOL_MODELS[solver],
                "tokenizer_revision": "main",
                "chat_template_sha256": "1" * 64,
                "serving_receipt": {"uri": "s3://example/serving", "sha256": "3" * 64},
            }
            if measured
            else None
        ),
        settings=(
            {
                "context_tokens": 65536,
                "input_tokens": 49152,
                "output_tokens": 16384,
                "temperature": 0.7,
                "top_p": 0.95,
                "top_k": 20,
                "min_p": 0.0,
                "repetition_penalty": 1.0,
                "presence_penalty": 0.0,
                "frequency_penalty": 0.0,
                "thinking": PROTOCOL_THINKING[solver],
                "deviations": [],
            }
            if measured
            else None
        ),
        n=100 if measured else 0,
        solved=5 if measured else 0,
        wilson_95={"low": 0.02, "high": 0.11} if measured else None,
        infrastructure_attempts=0,
        pending_tasks=0,
        blocked_tasks=0,
        attempts_receipt={"uri": "s3://example/attempts", "sha256": "4" * 64} if measured else None,
        gpu_hours=0.0,
        input_tokens=0,
        output_tokens=0,
    )


def full_measurements(**kwargs) -> list[DifficultyObservation]:
    return [observation(solver, **kwargs) for solver in Solver]


def test_a_source_measured_on_its_current_pin_is_measured_with_measurement_ids():
    coverage = source_coverage(declaration(), full_measurements())
    assert coverage.status is CoverageStatus.MEASURED
    assert len(coverage.measurement_ids) == len(Solver)
    assert all(model.measurement_id for model in coverage.solvers)
    assert coverage.gaps == ()


def test_a_source_re_pinned_after_measurement_needs_measurement():
    """A measurement bound to an older release or converter no longer covers the pin."""
    stale_release = DatasetRelease(
        repo="open-athena/example-retained",
        revision="e" * 40,
        url="https://huggingface.co/datasets/open-athena/example-retained/tree/" + "e" * 40,
    )
    stale_records = [
        observation(solver, release=stale_release, measurement_id=f"stale-{solver.value}") for solver in Solver
    ]
    coverage = source_coverage(declaration(), stale_records)
    assert coverage.status is CoverageStatus.NEEDS_MEASUREMENT
    assert coverage.measurement_ids == ()

    stale_converter = [observation(solver, converter_version="2026.11.1.1") for solver in Solver]
    assert source_coverage(declaration(), stale_converter).status is CoverageStatus.NEEDS_MEASUREMENT


def test_a_rebuilt_environment_artifact_clears_its_gap():
    """Yesterday's missing-artifact block is gone once the artifact exists: the row needs measurement."""
    reason = difficulty_gap_reason(DifficultyGap.ENVIRONMENT_ARTIFACT_MISSING, "s3://example/lock")
    blocked_records = [
        (
            observation(
                solver,
                status=MeasurementStatus.NOT_MEASURED,
                reason=reason,
            )
            if solver is Solver.HOSTED
            else observation(solver)
        )
        for solver in Solver
    ]
    coverage = source_coverage(declaration(), blocked_records)
    assert coverage.status is CoverageStatus.BLOCKED
    assert DifficultyGap.ENVIRONMENT_ARTIFACT_MISSING in coverage.gaps
    assert any("s3://example/lock" in piece for piece in coverage.evidence)

    cleared_records = [
        observation(solver, measurement_id="m2") if solver is Solver.HOSTED else record
        for solver, record in zip(Solver, blocked_records, strict=True)
    ]
    after = source_coverage(declaration(), cleared_records)
    assert after.status is CoverageStatus.MEASURED
    assert after.gaps == ()


def test_an_open_typed_gap_blocks_the_source_with_evidence():
    reason = difficulty_gap_reason(DifficultyGap.ENVIRONMENT_ARTIFACT_MISSING, "s3://example/lock")
    records = [
        (
            observation(solver, status=MeasurementStatus.NOT_MEASURED, reason=reason)
            if solver in (Solver.HOSTED, Solver.SMALL)
            else observation(solver)
        )
        for solver in Solver
    ]
    coverage = source_coverage(declaration(), records)
    assert coverage.status is CoverageStatus.BLOCKED
    assert set(coverage.gaps) == {DifficultyGap.ENVIRONMENT_ARTIFACT_MISSING}
    measured = [model for model in coverage.solvers if model.status is MeasurementStatus.MEASURED]
    assert len(measured) == len(Solver) - 2


def test_the_denominator_is_the_current_catalog_and_other_rows_are_absent():
    declarations = {
        "Example:available": declaration(source_id="Example:available"),
        "Example:pending": declaration("Pending", source_id="Example:pending"),
        "Example:excluded": declaration("Excluded", source_id="Example:excluded"),
        "Example:deprecated": declaration("Deprecated", source_id="Example:deprecated"),
    }
    records = {"Example:available": full_measurements(source_id="Example:available")}
    document = difficulty_coverage(declarations, records)
    assert set(document.sources) == {"Example:available", "Example:pending"}
    assert document.counts["sources"] == 2
    assert document.counts["measured"] == 1
    assert document.counts["needs_measurement"] == 1
    assert document.counts["blocked"] == 0


def test_the_queue_emits_only_needs_measurement_sources_with_their_declarations():
    declarations = {
        "Example:measured": declaration(source_id="Example:measured"),
        "Example:needs": declaration(source_id="Example:needs"),
        "Example:blocked": declaration(source_id="Example:blocked"),
    }
    records = {
        "Example:measured": full_measurements(source_id="Example:measured"),
        # one solver measured at the current pin; the others have no record at all
        "Example:needs": [
            observation(solver, source_id="Example:needs", measurement_id="n1") if solver is Solver.SMALL else None
            for solver in Solver
        ],
        "Example:blocked": [
            observation(
                solver,
                source_id="Example:blocked",
                status=MeasurementStatus.NOT_MEASURED,
                reason=difficulty_gap_reason(DifficultyGap.TOOL_PROVIDER_REQUIRED),
            )
            for solver in Solver
        ],
    }
    records["Example:needs"] = [record for record in records["Example:needs"] if record is not None]
    document = difficulty_coverage(declarations, records)
    assert document.queue_sources() == ["Example:needs"]
    queue = campaign_queue(document, declarations)
    assert len(queue) == 1
    assert queue[0]["declaration"]["id"] == "Example:needs"
    assert queue[0]["coverage"]["status"] == "needs_measurement"
    # the published document carries the queue; the CLI reads it from the served app
    assert len(document.queue) == 1
    assert document.queue[0]["declaration"]["id"] == "Example:needs"
    assert DifficultyCoverage.model_validate(document.model_dump(mode="json")).queue == document.queue


def test_an_unmapped_reason_fails_loudly():
    # The observation constructor validates the reason against the typed gaps,
    # so an unmapped reason is rejected before the server-level check runs.
    with pytest.raises(ValueError, match="Unmapped difficulty gap reason"):
        source_coverage(
            declaration(),
            [
                observation(solver, status=MeasurementStatus.NOT_MEASURED, reason="something made up")
                for solver in Solver
            ],
        )


def test_the_document_digest_binds_the_content():
    first = difficulty_coverage({"Example:example": declaration()}, {"Example:example": full_measurements()})
    second = DifficultyCoverage.model_validate(first.model_dump(mode="json"))
    assert document_digest(first) == document_digest(second)
    assert document_digest(first) != document_digest(difficulty_coverage({"Example:example": declaration()}, {}))


def test_publish_coverage_stores_the_document_for_the_load():
    """The rebuild publishes the computed coverage under its own load id."""
    stored = {}

    class FakeQuery:
        def __init__(self, declarations, records):
            self.declarations = declarations
            self.records = records

        def __call__(self, sql, parameters):
            if "FROM atlas_sources" in sql:
                rows = [
                    {"id": source_id, "declaration": declaration}
                    for source_id, declaration in sorted(self.declarations.items())
                ]
                return {"rows": rows[parameters["offset"] : parameters["offset"] + parameters["limit"]]}
            if "FROM atlas_difficulty_seal_loads" in sql:
                return {"rows": []}
            if "FROM atlas_difficulty" in sql:
                rows = [{"measurement": record.model_dump(mode="json")} for record in self.records]
                return {"rows": rows[parameters["offset"] : parameters["offset"] + parameters["limit"]]}
            if "INSERT INTO atlas_difficulty_coverage" in sql:
                stored[parameters["load_id"]] = json.loads(parameters["coverage"])
                return {"row_count": 1}
            if "FROM atlas_current" in sql:
                return {"rows": [{"load_id": "load-1"}]}
            raise AssertionError(f"Unexpected query: {sql[:60]}")

    declarations = {"Example:example": declaration(), "Example:gone": declaration(status="Excluded")}
    records = [record for record in full_measurements()]
    query = FakeQuery(declarations, records)
    result = publish_coverage(query, "load-1")
    assert result["counts"]["measured"] == 1
    document = stored["load-1"]
    assert document["load_id"] == "load-1"
    assert set(document["sources"]) == {"Example:example"}
    assert document["sources"]["Example:example"]["status"] == CoverageStatus.MEASURED.value


def test_publish_coverage_paginates_bulk_reads_under_the_transport_bound(monkeypatch):
    """Batched bulk reads produce the identical document the unbatched read produces."""
    stored = {}

    class PagedFakeQuery:
        """A transport honoring LIMIT/OFFSET for the two bulk reads."""

        def __init__(self, declarations, records):
            self.declarations = declarations
            self.records = records

        def __call__(self, sql, parameters):
            if "FROM atlas_sources" in sql:
                rows = [
                    {"id": source_id, "declaration": declaration}
                    for source_id, declaration in sorted(self.declarations.items())
                ]
            elif "FROM atlas_difficulty_seal_loads" in sql:
                rows = []
            elif "FROM atlas_difficulty" in sql:
                rows = [{"measurement": record.model_dump(mode="json")} for record in self.records]
            elif "INSERT INTO atlas_difficulty_coverage" in sql:
                stored[parameters["load_id"]] = json.loads(parameters["coverage"])
                return {"row_count": 1}
            elif "FROM atlas_current" in sql:
                return {"rows": [{"load_id": "load-1"}]}
            else:
                raise AssertionError(f"Unexpected query: {sql[:60]}")
            return {"rows": rows[parameters["offset"] : parameters["offset"] + parameters["limit"]]}

    declarations = {
        "Example:example": declaration(),
        "Example:second": declaration(),
        "Example:third": declaration(),
    }
    records = full_measurements()
    query = PagedFakeQuery(declarations, records)
    publish_coverage(query, "load-1")
    unbatched = stored["load-1"]

    monkeypatch.setattr(load_difficulty, "DIFFICULTY_READ_BATCH", 1)
    monkeypatch.setattr(load_difficulty, "DECLARATION_READ_BATCH", 1)
    publish_coverage(query, "load-2")
    # the load id and generation timestamp differ; the document does not
    assert stored["load-2"]["sources"] == unbatched["sources"]
    assert stored["load-2"]["counts"] == unbatched["counts"]


def _measurement_record(index: int) -> DifficultyMeasurement:
    payload = {
        "measurement_id": f"m{index}",
        "protocol_id": "atlas-difficulty-v3",
        "created_at": "2026-11-16T00:00:00+00:00",
        "source_id": "Example:example",
        "release": {"repo": "fixture/repo", "revision": "r1", "url": "https://example.invalid/r1"},
        "converter_version": "1",
        "solver": "small",
        "status": "not_measured",
        "reason": "Awaiting scored coverage",
        "sample": None,
        "sample_receipt": None,
        "model": None,
        "settings": None,
        "n": 0,
        "solved": 0,
        "wilson_95": None,
        "infrastructure_attempts": 0,
        "pending_tasks": 0,
        "blocked_tasks": 0,
        "attempts_receipt": None,
        "gpu_hours": 0,
        "input_tokens": 0,
        "output_tokens": 0,
    }
    return DifficultyMeasurement.model_validate(payload)


def test_load_measurements_batches_inserts_and_readbacks(monkeypatch):
    """Bounded recordset inserts plus a paged typed readback verify every record."""
    monkeypatch.setattr(load_difficulty, "MEASUREMENT_READBACK_BATCH", 2)
    table: dict[str, dict] = {}
    requests: list[str] = []

    class Store:
        def __call__(self, sql: str, parameters: dict):
            requests.append("insert" if "INSERT INTO atlas_difficulty" in sql else "readback")
            if "INSERT INTO atlas_difficulty" in sql:
                rows = json.loads(parameters["rows"])
                inserted = 0
                for row in rows:
                    if row["measurement_id"] not in table:
                        # the transport passes the measurement as a JSON object
                        assert isinstance(row["measurement"], dict)
                        table[row["measurement_id"]] = row["measurement"]
                        inserted += 1
                return {"row_count": inserted}
            if "WHERE measurement_id IN" in sql:
                ids = [row["measurement_id"] for row in json.loads(parameters["ids"])]
                assert len(ids) <= 2, "the readback must page under its bound"
                return {
                    "rows": [
                        {"measurement_id": measurement_id, "measurement": table[measurement_id]}
                        for measurement_id in ids
                        if measurement_id in table
                    ]
                }
            raise AssertionError(f"Unexpected query: {sql[:60]}")

    records = [_measurement_record(index) for index in range(3)]
    assert load_measurements(records, Store()) == {"inserted": len(records), "records": len(records)}
    # idempotent: a rerun inserts nothing and still verifies every record
    assert load_measurements(records, Store()) == {"inserted": 0, "records": len(records)}
    # a pre-existing ID naming different evidence raises
    clash = records[0].model_copy(update={"source_id": "Example:other"})
    with pytest.raises(ValueError, match="already names different evidence"):
        load_measurements([clash], Store())
    # a handful of requests across all three invocations (two per record
    # before), far below the per-record round-trips the transport cost
    assert len(requests) <= 8


def test_load_measurements_rejects_duplicate_ids():
    records = [_measurement_record(index) for index in range(3)]
    with pytest.raises(ValueError, match="Duplicate difficulty measurement IDs"):
        load_measurements([*records, records[0]], None)


def test_loader_reads_stored_measurements_with_sample_tasks(database_url):
    """The loader projects stored measurements exactly like the server's summary shape."""
    record = observation(Solver.SMALL)
    payload = record.model_dump(mode="json")
    # the stored measurement carries the real per-task sample rows
    payload["sample"]["tasks"] = [{"task_id": f"t{index}", "task_sha256": f"{index:064x}"} for index in range(100)]
    measurement = DifficultyMeasurement.model_validate(payload)

    engine = create_engine(database_url)
    with engine.begin() as connection:
        for statement in SCHEMA_SQL.split(";"):
            if statement.strip():
                connection.execute(text(statement))

    def query(sql: str, parameters: dict) -> dict:
        with engine.begin() as connection:
            result = connection.execute(text(sql), parameters)
            if result.returns_rows:
                rows = [dict(row) for row in result.mappings()]
                return {"rows": rows, "row_count": len(rows)}
            return {"rows": [], "row_count": result.rowcount}

    assert load_measurements([measurement], query) == {"inserted": 1, "records": 1}
    with engine.begin() as connection:
        (kind,) = connection.execute(
            text("SELECT jsonb_typeof(measurement->'sample'->'tasks') FROM atlas_difficulty")
        ).one()
    assert kind == "array", "the stored measurement keeps its immutable sample tasks"
    grouped = difficulty_records_from_query(query)
    (read_back,) = grouped["Example:example"]
    assert read_back.measurement_id == measurement.measurement_id
    assert (read_back.sample is not None and not hasattr(read_back.sample, "tasks")) or read_back.sample.tasks is None
    # the coverage consumes the projected observations over the real rows
    # without the sample.tasks rejection (the pre-fix failure mode)
    result = publish_coverage(query, "load-diff-projection")
    assert result["load_id"] == "load-diff-projection"
    engine.dispose()
