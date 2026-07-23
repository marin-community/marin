# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The evaldash matrix/meta aggregation: version cohorts, archive annotation, and suite grouping."""

from marin.evaluation.records import (
    EvalRef,
    EvalRunRecord,
    EvalTaskRef,
    HardwareRef,
    ModelRef,
    Provenance,
    RunStatus,
)

from infra.evaldash.src.metrics import build_matrix, build_meta, eval_suites


def _record(model: str, eval_name: str, version: str | None, created_at: str, value: float | None) -> EvalRunRecord:
    succeeded = value is not None
    return EvalRunRecord(
        run_id=f"{model}-{eval_name}-{created_at}",
        group_id=f"{model}-{created_at}",
        created_at=created_at,
        user="tester",
        version=version,
        model=ModelRef(name=model, location="loc", backend="vllm"),
        evaluation=EvalRef(name=eval_name, mechanism="evalchemy", tasks=(EvalTaskRef(name=eval_name, num_fewshot=0),)),
        hardware=HardwareRef(platform="tpu", accelerator="v6e-8", region_or_cluster="us-central2"),
        status=RunStatus.SUCCEEDED if succeeded else RunStatus.INFRA_FAILED,
        error=None,
        results_path="p",
        metrics={eval_name: {"acc,none": value, "acc_stderr,none": 0.01}} if succeeded else {},
        jobs={},
        log_tails={},
        provenance=Provenance(git_sha="s", evalchemy_image="i", launch_host="h"),
    )


def test_matrix_row_shows_only_the_latest_version_cohort():
    """A model whose newest run is version ``v2`` shows v2's scores, not an older v1 run's, even for
    an eval v2 never re-ran -- the cohort is picked by the newest run's version, then filtered to it."""
    records = [
        _record("m", "mmlu", "v1", "2026-01-01T00:00:00+00:00", 0.50),
        _record("m", "mmlu", "v2", "2026-02-01T00:00:00+00:00", 0.70),
        _record("m", "gsm8k-0shot", "v2", "2026-02-01T00:01:00+00:00", 0.30),
    ]

    matrix = build_matrix(records)

    (row,) = matrix["rows"]
    assert row["version"] == "v2"
    assert row["cells"]["mmlu"]["value"] == 0.70
    assert row["cells"]["gsm8k-0shot"]["value"] == 0.30


def test_matrix_drops_stale_eval_from_a_superseded_version():
    """When the current cohort never ran an eval an older cohort did, that column is absent for the
    model rather than back-filled from the stale version."""
    records = [
        _record("m", "drop", "v1", "2026-01-01T00:00:00+00:00", 0.40),
        _record("m", "mmlu", "v2", "2026-02-01T00:00:00+00:00", 0.70),
    ]

    (row,) = build_matrix(records)["rows"]

    assert row["version"] == "v2"
    assert set(row["cells"]) == {"mmlu"}


def test_matrix_annotates_archived_and_version():
    records = [_record("keep", "mmlu", None, "2026-01-01T00:00:00+00:00", 0.6)]

    matrix = build_matrix(records, frozenset({"keep"}))

    (row,) = matrix["rows"]
    assert row["archived"] is True
    assert row["version"] is None
    assert matrix["leaderboard"][0]["archived"] is True


def test_meta_reports_suites_and_archived_models():
    records = [
        _record("a", "mmlu", None, "2026-01-01T00:00:00+00:00", 0.6),
        _record("b", "math500", None, "2026-01-02T00:00:00+00:00", 0.5),
    ]

    meta = build_meta(records, frozenset({"b"}))

    assert meta["archived_models"] == ["b"]
    assert {group["suite"] for group in meta["suites"]} == {"NLP", "Chat / Math"}


def test_eval_suites_groups_known_evals_and_buckets_the_rest():
    grouped = {group["suite"]: group["evals"] for group in eval_suites({"mmlu", "drop", "math500", "mystery"})}

    assert grouped["NLP"] == ["drop", "mmlu"]
    assert grouped["Chat / Math"] == ["math500"]
    assert grouped["Other"] == ["mystery"]
