# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Behavior tests for the repair-candidate lane: matching, comparative review, decisions."""

import collections
import io
import json
import tarfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import cast

import pytest
from finestore.rl.licenses import LicenseDecision, LicenseDecisionKind, LicenseReceipt
from marin.inference.openai_chat import OpenAIChatClient
from taskcompendium.convert.answers import unsupported
from taskcompendium.models import (
    AnswerType,
    ConversationInput,
    EnvironmentRequirements,
    NoGrader,
    PlainText,
    Source,
    TaskSpec,
    TextMessage,
)
from taskcompendium.pipeline.inputs import SourceFormat
from taskcompendium.pipeline.models import (
    Confidence,
    Defect,
    OracleCommand,
    Quality,
    ReferenceStatus,
    ReviewStatus,
)
from taskcompendium.pipeline.repair import (
    CandidateDropAdvisory,
    ComparativeOutcome,
    ComparativeVerdict,
    MatchCensus,
    MatchReceipt,
    RepairDecision,
    RepairMatchClass,
    repair_decision,
)

from experiments.post_training.task_curation import repair_lane as lane
from experiments.post_training.task_curation.datasets.environments import GRADER_PACKAGES
from experiments.post_training.task_curation.datasets.tasktrove.conversion.archive import (
    archive_files,
    unpack_task_binary,
)
from experiments.post_training.task_curation.full_qc import ReviewerVersion, local_reviewer
from experiments.post_training.task_curation.pipeline import Controls, CurationRecipe, HfSource, IntendedUse
from experiments.post_training.task_curation.repair_lane import matched_pairs
from experiments.post_training.task_curation.repair_match import (
    CONFLICT_THRESHOLD,
    MatchOutcome,
    _classify,
    match_candidates,
    strip_template_sections,
    strip_template_sections_rows,
)
from experiments.post_training.task_curation.repair_review import (
    ComparativePair,
    ComparativePairReviewer,
    pair_alias,
)
from experiments.post_training.task_curation.task_neighbors import PopulationIndex, TaskContent, content_rows

CANDIDATE_SOURCE = "candidate:test"
TRACKED_SOURCE = "tracked:test"


def content(
    task_id: str,
    text: str,
    *,
    instruction: str | None = None,
    state: str = "s",
    tests: str = "t",
    reference: str | None = "r",
) -> TaskContent:
    """One TaskContent row with explicit hashes, so tests control the match classes."""
    return TaskContent(
        task_id=task_id,
        instruction_hash=instruction or f"i:{task_id}",
        text=text,
        reference_hash=reference,
        tests_hash=tests,
        state_hash=state,
    )


def index(rows: list[TaskContent]) -> PopulationIndex:
    return PopulationIndex(rows, {"inputs": "0" * 64})


def spec(task_id: str, instruction: str) -> TaskSpec:
    """A minimal valid TaskSpec for review-workload tests."""
    return TaskSpec(
        id=task_id,
        source=Source(dataset="ds", revision="r1", row=task_id, importer_revision="1"),
        context=ConversationInput(events=(TextMessage(role="user", content=instruction),)),
        environment_requirements=EnvironmentRequirements(),
        answer_type=AnswerType.FILE,
        answer_format=PlainText(),
        grader=NoGrader(reason="test fixture", contract={}),
    )


def verdict(task_id: str, outcome: ComparativeOutcome, *, fixed=(), introduced=()) -> ComparativeVerdict:
    return ComparativeVerdict(
        task_id=task_id,
        outcome=outcome,
        defects_fixed=[Defect(defect) for defect in fixed],
        defects_introduced=[Defect(defect) for defect in introduced],
        confidence=Confidence.HIGH,
        evidence="The candidate repairs the reference without introducing defects.",
    )


def by_class(outcome: MatchOutcome) -> dict[RepairMatchClass, list[MatchReceipt]]:
    return {
        match_class: [receipt for receipt in outcome.receipts if receipt.match_class is match_class]
        for match_class in RepairMatchClass
    }


def outcome_with(receipts: list[MatchReceipt]) -> MatchOutcome:
    counts = {match_class: 0 for match_class in RepairMatchClass}
    for receipt in receipts:
        counts[receipt.match_class] += 1
    census = MatchCensus(
        candidate_source_id=CANDIDATE_SOURCE,
        tracked_source_id=TRACKED_SOURCE,
        candidates=len(receipts),
        identical=counts[RepairMatchClass.IDENTICAL],
        repaired=counts[RepairMatchClass.REPAIRED],
        new=counts[RepairMatchClass.NEW],
        conflicting=counts[RepairMatchClass.CONFLICTING],
        dropped_advisories=0,
    )
    return MatchOutcome(receipts=receipts, advisories=[], census=census)


def receipt(candidate: str, tracked: str | None, match_class: RepairMatchClass) -> MatchReceipt:
    return MatchReceipt(
        candidate_source_id=CANDIDATE_SOURCE,
        candidate_task_id=candidate,
        tracked_source_id=TRACKED_SOURCE,
        tracked_task_id=tracked,
        instruction_match=tracked is not None,
        similarity=1.0 if tracked is not None else 0.1,
        method="exact" if tracked is not None else "neighbor",
        match_class=match_class,
        reason="",
    )


class TestMatchCascade:
    def test_identical_pairs_hash_proven(self):
        tracked = index([content("t1", "text one"), content("t2", "text two")])
        candidates = index([content("c1", "text three", instruction="i:t1")])
        outcome = match_candidates(
            tracked, candidates, tracked_source_id=TRACKED_SOURCE, candidate_source_id=CANDIDATE_SOURCE
        )
        matched = by_class(outcome)
        assert [receipt.candidate_task_id for receipt in matched[RepairMatchClass.IDENTICAL]] == ["c1"]
        matched_receipt = matched[RepairMatchClass.IDENTICAL][0]
        assert matched_receipt.tracked_task_id == "t1"
        assert matched_receipt.method.value == "exact"
        assert matched_receipt.similarity == 1.0
        assert outcome.census.identical == 1 and outcome.census.candidates == 1

    def test_same_instruction_with_changed_contract_is_repaired(self):
        tracked = index([content("t1", "text one")])
        candidates = index([content("c1", "text two", instruction="i:t1", tests="t2")])
        outcome = match_candidates(
            tracked, candidates, tracked_source_id=TRACKED_SOURCE, candidate_source_id=CANDIDATE_SOURCE
        )
        matched = by_class(outcome)
        assert len(matched[RepairMatchClass.REPAIRED]) == 1
        assert matched[RepairMatchClass.REPAIRED][0].tests_match is False
        assert matched[RepairMatchClass.REPAIRED][0].instruction_match is True

    def test_near_duplicate_via_neighbor_is_repaired(self):
        base = " ".join(f"token{position}" for position in range(60))
        edited = base.replace("token30", "token30a")
        tracked = index([content("t1", base)])
        candidates = index([content("c1", edited)])
        outcome = match_candidates(
            tracked, candidates, tracked_source_id=TRACKED_SOURCE, candidate_source_id=CANDIDATE_SOURCE
        )
        matched = by_class(outcome)
        assert len(matched[RepairMatchClass.REPAIRED]) == 1, outcome.census
        assert matched[RepairMatchClass.REPAIRED][0].method.value == "neighbor"
        assert outcome.advisories == []

    def test_mutual_best_neighbor_is_repaired_and_non_mutual_conflicts(self):
        base = " ".join(f"token{position}" for position in range(60))
        near = base.replace("token30", "token30a")
        farther = base.replace("token30", "token30a").replace("token7", "token7b")
        tracked = index([content("t1", base)])
        candidates = index([content("c1", near), content("c2", farther)])
        outcome = match_candidates(
            tracked, candidates, tracked_source_id=TRACKED_SOURCE, candidate_source_id=CANDIDATE_SOURCE
        )
        matched = by_class(outcome)
        # Both candidates prefer t1, but t1's own best candidate is c1: only c1 is mutual.
        repaired = {receipt.candidate_task_id for receipt in matched[RepairMatchClass.REPAIRED]}
        conflicting = {receipt.candidate_task_id for receipt in matched[RepairMatchClass.CONFLICTING]}
        assert repaired == {"c1"}
        assert conflicting == {"c2"}
        assert all("mutual" in receipt.reason.casefold() for receipt in matched[RepairMatchClass.REPAIRED])

    def test_two_candidates_on_one_tracked_task_conflict(self):
        tracked = index([content("t1", "text one")])
        candidates = index(
            [
                content("c1", "text two", instruction="i:t1"),
                content("c2", "text three", instruction="i:t1", tests="t2"),
            ]
        )
        outcome = match_candidates(
            tracked, candidates, tracked_source_id=TRACKED_SOURCE, candidate_source_id=CANDIDATE_SOURCE
        )
        matched = by_class(outcome)
        assert {receipt.candidate_task_id for receipt in matched[RepairMatchClass.CONFLICTING]} == {"c1", "c2"}
        assert outcome.census.conflicting == 2

    def test_content_identical_candidates_collapse_before_matching(self):
        tracked = index([content("t1", "text one")])
        candidates = index(
            [
                content("c1", "text two", instruction="i:t1"),
                content("c2", "text four", instruction="i:t1"),
            ]
        )
        outcome = match_candidates(
            tracked, candidates, tracked_source_id=TRACKED_SOURCE, candidate_source_id=CANDIDATE_SOURCE
        )
        matched = by_class(outcome)
        assert [receipt.candidate_task_id for receipt in matched[RepairMatchClass.IDENTICAL]] == ["c1"]
        assert outcome.census.candidates == 2

    def test_unrelated_candidate_is_new(self):
        tracked = index([content("t1", "completely unrelated tracked text about calendars and events " * 8)])
        candidates = index([content("c1", "a bash pipeline task asking for the disk usage summary " * 8)])
        outcome = match_candidates(
            tracked, candidates, tracked_source_id=TRACKED_SOURCE, candidate_source_id=CANDIDATE_SOURCE
        )
        matched = by_class(outcome)
        assert [receipt.candidate_task_id for receipt in matched[RepairMatchClass.NEW]] == ["c1"]
        assert matched[RepairMatchClass.NEW][0].tracked_task_id is None

    def test_tracked_task_absent_from_candidate_becomes_advisory(self):
        tracked = index([content("t1", "kept tracked task"), content("t2", "dropped by the candidate filter")])
        candidates = index([content("c1", "text three", instruction="i:t1")])
        outcome = match_candidates(
            tracked, candidates, tracked_source_id=TRACKED_SOURCE, candidate_source_id=CANDIDATE_SOURCE
        )
        assert [advisory.tracked_task_id for advisory in outcome.advisories] == ["t2"]
        advisory = outcome.advisories[0]
        assert isinstance(advisory, CandidateDropAdvisory)
        assert advisory.signal == "absent_from_candidate"
        assert advisory.candidate_source_id == CANDIDATE_SOURCE
        assert "advisory" in advisory.reason

    def test_candidate_with_no_overlap_emits_no_advisories(self):
        tracked = index([content("t1", "tracked task one about calendars " * 6)])
        candidates = index([content("c1", "an unrelated candidate about bash pipelines " * 6)])
        outcome = match_candidates(
            tracked, candidates, tracked_source_id=TRACKED_SOURCE, candidate_source_id=CANDIDATE_SOURCE
        )
        assert outcome.census.overlapped == 0
        assert outcome.advisories == []
        assert outcome.census.dropped_advisories == 0
        assert outcome.census.new == 1

    def test_census_counts_follow_receipts(self):
        tracked = index([content("t1", "one"), content("t2", "dropped by the candidate filter")])
        candidates = index(
            [content("c1", "two", instruction="i:t1"), content("c2", "an entirely new candidate task " * 5)]
        )
        outcome = match_candidates(
            tracked, candidates, tracked_source_id=TRACKED_SOURCE, candidate_source_id=CANDIDATE_SOURCE
        )
        census = outcome.census
        assert (
            census.candidates,
            census.identical,
            census.repaired,
            census.new,
            census.conflicting,
            census.dropped_advisories,
        ) == (2, 1, 0, 1, 0, 1)


class TestMatchThresholdPolicy:
    """Pin the standing mutual-best rule at the classification boundary."""

    def test_mutual_best_band_pinned(self):
        row = content("c1", "some candidate text")
        partner = content("t1", "some tracked text")

        def classify(similarity: float, *, mutual: bool):
            return _classify(
                row,
                partner,
                similarity,
                "lsh",
                mutual_best=mutual,
                tracked_source_id=TRACKED_SOURCE,
                candidate_source_id=CANDIDATE_SOURCE,
            )

        assert classify(CONFLICT_THRESHOLD, mutual=True).match_class is RepairMatchClass.REPAIRED
        assert classify(0.9, mutual=True).match_class is RepairMatchClass.REPAIRED
        # The old one-way 0.9 tier is gone: non-mutual neighbors stay conflicting.
        assert classify(0.9, mutual=False).match_class is RepairMatchClass.CONFLICTING
        unmatched = _classify(
            row,
            None,
            0.0,
            "lsh",
            mutual_best=False,
            tracked_source_id=TRACKED_SOURCE,
            candidate_source_id=CANDIDATE_SOURCE,
        )
        assert unmatched.match_class is RepairMatchClass.NEW and unmatched.tracked_task_id is None
        below = classify(CONFLICT_THRESHOLD - 0.01, mutual=True)
        assert below.match_class is RepairMatchClass.NEW


class TestTemplateSectionNormalization:
    """The approved cross-conversion normalization: strip shared leading setup sections."""

    SETUP_BLOCK = (
        "## Environment Setup (run before starting)\n\n```bash\nbash /setup_files/setup_seeds.sh\n```\n\n---\n\n"
    )
    BODY = (
        "# Terminal Automation Request\n\n## Goal\n"
        "Normalize all input lines to lowercase and count them.\n\n## Capture\nSave stdout and stderr.\n"
    )

    def test_leading_setup_section_is_stripped(self):
        stripped = strip_template_sections("User:\n" + self.SETUP_BLOCK + self.BODY)
        assert stripped == "User:\n" + self.BODY.strip()

    def test_body_without_setup_is_untouched(self):
        assert strip_template_sections(self.BODY) == self.BODY.strip()

    def test_midtext_setup_section_is_kept(self):
        text = self.BODY + "\n## Setup\n\nRun the verifier script.\n"
        assert "## Setup" in strip_template_sections(text)

    def test_consecutive_leading_sections_strip(self):
        text = "## Setup\n\nInstall packages.\n\n---\n\n" + self.SETUP_BLOCK + self.BODY
        assert strip_template_sections(text) == self.BODY.strip()

    def test_normalization_aligns_a_preamble_pair(self):
        tracked = index(strip_template_sections_rows([content("t1", "User:\n" + self.SETUP_BLOCK + self.BODY)]))
        candidates = index(strip_template_sections_rows([content("c1", "User:\n" + self.BODY)]))
        outcome = match_candidates(
            tracked, candidates, tracked_source_id=TRACKED_SOURCE, candidate_source_id=CANDIDATE_SOURCE
        )
        matched = by_class(outcome)
        # Same stripped body and grading contract: the pair is content-identical.
        assert len(matched[RepairMatchClass.IDENTICAL]) == 1, outcome.census

    def test_normalization_preserves_original_hashes_for_exact_joins(self):
        rows = strip_template_sections_rows([content("t1", self.SETUP_BLOCK + self.BODY)])
        assert rows[0].instruction_hash == "i:t1"
        assert rows[0].text == self.BODY.strip()


class TestLicenseGate:
    """Stage 0 gates every candidate folder on its bound lineage receipt (repackager ruling)."""

    @staticmethod
    def _receipt(approved: bool) -> LicenseReceipt:
        return LicenseReceipt(
            source_id="FWeindel:validated-tasks/nl2bash_v2",
            dataset_id="FWeindel/validated-tasks",
            revision="eafe9b38",
            resolved_at="2026-11-17T00:00:00+00:00",
            resolver_version="atlas-license-check-v10",
            evidence=[],
            bundled_third_party=True,
            model_generated_outputs=False,
            decisions=[
                LicenseDecision(
                    kind=LicenseDecisionKind.EXCEPTION,
                    approved=approved,
                    who="user via coordinator",
                    when="2026-11-17T00:00:00+00:00",
                    source_id="FWeindel:validated-tasks/nl2bash_v2",
                    dataset_id="FWeindel/validated-tasks",
                    revision="eafe9b38",
                    scope=["train"],
                    rationale="Repackager ruling: the folder's original upstream carries the license.",
                    evidence_urls=["https://huggingface.co/datasets/open-thoughts/TaskTrove"],
                    issue_url="https://github.com/marin-community/marin/issues/11497",
                )
            ],
        )

    def test_folder_without_a_bound_receipt_is_refused_before_any_loading(self, monkeypatch):
        monkeypatch.setattr(lane, "read_license_receipts", lambda index: {})

        def tripwire(**kwargs):
            raise AssertionError("candidate loading ran before the license gate")

        monkeypatch.setattr(lane, "load_candidate_population", tripwire)
        with pytest.raises(ValueError, match=r"no bound lineage receipt"):
            lane.folder_license_gate("FWeindel/validated-tasks", ("nl2bash_v2",))
        with pytest.raises(ValueError, match=r"no bound lineage receipt"):
            lane.dry_run(
                tracked_source_id=TRACKED_SOURCE,
                tracked_uris=(),
                candidate_source_id=CANDIDATE_SOURCE,
                candidate_revision="eafe9b38",
                folders=("nl2bash_v2",),
                repo="FWeindel/validated-tasks",
                reviewer_version=ReviewerVersion.V3,
                run=False,
            )

    def test_banned_folder_is_refused(self, monkeypatch):
        receipt = self._receipt(approved=True)
        monkeypatch.setattr(lane, "read_license_receipts", lambda index: {receipt.source_id: receipt})
        monkeypatch.setattr(
            lane,
            "require_unbanned",
            lambda source_id, dataset_id, receipt=None: (_ for _ in ()).throw(
                ValueError(f"License banned: {source_id}: per-folder ban")
            ),
        )
        with pytest.raises(ValueError, match="License banned"):
            lane.folder_license_gate("FWeindel/validated-tasks", ("nl2bash_v2",))

    def test_unapproved_folder_is_refused(self, monkeypatch):
        receipt = self._receipt(approved=False)
        monkeypatch.setattr(lane, "read_license_receipts", lambda index: {receipt.source_id: receipt})
        with pytest.raises(ValueError, match=r"no approved per-folder decision"):
            lane.folder_license_gate("FWeindel/validated-tasks", ("nl2bash_v2",))

    def test_approved_receipt_binds_and_records_upstream(self, monkeypatch):
        receipt = self._receipt(approved=True)
        monkeypatch.setattr(lane, "read_license_receipts", lambda index: {receipt.source_id: receipt})
        bound = lane.folder_license_gate("FWeindel/validated-tasks", ("nl2bash_v2",))
        assert bound["FWeindel:validated-tasks/nl2bash_v2"] is receipt


class TestCohortConversion:
    """The merge stage's first step: winners converted through the tracked source's own converter."""

    @staticmethod
    def _report() -> dict:
        return {
            "candidate_source_id": CANDIDATE_SOURCE,
            "candidate_revision": "eafe9b38",
            "decisions": {
                "pairs": [
                    {
                        "candidate_task_id": f"{CANDIDATE_SOURCE}:task_{index}",
                        "tracked_task_id": f"tracked-{index}",
                        "decision": "ACCEPTED",
                    }
                    for index in range(3)
                ]
            },
        }

    @staticmethod
    def _single_winner_report() -> dict:
        return {
            "candidate_source_id": CANDIDATE_SOURCE,
            "candidate_revision": "eafe9b38",
            "decisions": {
                "pairs": [
                    {
                        "candidate_task_id": f"{CANDIDATE_SOURCE}:task_0",
                        "tracked_task_id": "tracked-0",
                        "decision": "ACCEPTED",
                    }
                ]
            },
        }

    @staticmethod
    def _recipe(monkeypatch, tmp_path):
        def convert(row, context):
            files = archive_files(row.data)
            if files.text("instruction.md") == "reject me":
                return unsupported("reviewed_defect", "fixture rejection")
            return spec(f"converted-{row.data['path']}", files.text("instruction.md"))

        return CurationRecipe(
            name="tasktrove-fixture",
            source=HfSource(
                repo="fixture/validated-tasks",
                revision="r1",
                files=("x/tasks.parquet",),
                format=SourceFormat.PARQUET,
                decode=unpack_task_binary,
            ),
            convert=convert,
            version="1",
            intended_use=IntendedUse.TRAIN,
            grader=GRADER_PACKAGES,
            controls=Controls(
                golden=lambda task: OracleCommand("bash /solution/solve.sh", answer_file="/output/command_capture.txt")
            ),
        )

    @staticmethod
    def _runtime_join():
        """A runtime join covering the two fixture tasks that convert successfully."""
        join = {}
        for path, instruction in (("task_0", "keep me"), ("task_1", "keep me too")):
            task = spec(f"converted-{path}", instruction)
            row = next(content_rows([task.model_dump(mode="json")]))
            join[row.instruction_hash] = {"shape": {"agent_timeout_sec": 300}}
        return join

    def test_converts_accepted_winners_and_skips_the_rest(self, monkeypatch, tmp_path):
        monkeypatch.setattr(
            lane,
            "configured_sources",
            lambda: [
                type(
                    "S",
                    (),
                    {"info": type("I", (), {"id": TRACKED_SOURCE})(), "config": self._recipe(monkeypatch, tmp_path)},
                )()
            ],
        )
        monkeypatch.setattr(
            lane,
            "load_folder_rows",
            lambda repo, folders, revision=None: [
                {"path": "task_0", "task_binary": _fixture_binary("keep me")},
                {"path": "task_1", "task_binary": _fixture_binary("keep me too")},
                {"path": "task_2", "task_binary": _fixture_binary("reject me")},
            ],
        )
        monkeypatch.setattr(lane, "tracked_runtimes", lambda uris: self._runtime_join())
        output_root = str(tmp_path / "cohort")
        manifest = lane.convert_cohort(
            tracked_source_id=TRACKED_SOURCE,
            report=self._report(),
            folders=("fixture_v1",),
            repo="fixture/validated-tasks",
            candidate_revision="eafe9b38",
            tracked_spec_uris=(),
            output_root=output_root,
        )
        assert manifest["counts"] == {"converted": 2, "rejected": 1}
        entries = [json.loads(line) for line in (Path(output_root) / "task-map.jsonl").read_text().splitlines()]
        assert [(entry["status"], entry["tracked_task_id"]) for entry in entries] == [
            ("converted", "tracked-0"),
            ("converted", "tracked-1"),
            ("rejected", "tracked-2"),
        ]
        assert entries[2]["rejection"]["reason"] == "reviewed_defect"
        shards = sorted((Path(output_root) / "converted-pending").glob("*.parquet"))
        assert len(shards) == 1

    def test_rerun_resumes_from_the_task_map(self, monkeypatch, tmp_path):
        calls = []

        def counting_rows(repo, folders, revision=None):
            calls.append(1)
            return [{"path": "task_0", "task_binary": _fixture_binary("keep me")}]

        monkeypatch.setattr(
            lane,
            "configured_sources",
            lambda: [
                type(
                    "S",
                    (),
                    {"info": type("I", (), {"id": TRACKED_SOURCE})(), "config": self._recipe(monkeypatch, tmp_path)},
                )()
            ],
        )
        monkeypatch.setattr(lane, "load_folder_rows", counting_rows)
        monkeypatch.setattr(lane, "tracked_runtimes", lambda uris: self._runtime_join())
        output_root = str(tmp_path / "cohort")
        report = self._single_winner_report()
        lane.convert_cohort(
            tracked_source_id=TRACKED_SOURCE,
            report=report,
            folders=("fixture_v1",),
            repo="fixture/validated-tasks",
            candidate_revision="eafe9b38",
            tracked_spec_uris=(),
            output_root=output_root,
        )
        manifest = lane.convert_cohort(
            tracked_source_id=TRACKED_SOURCE,
            report=report,
            folders=("fixture_v1",),
            repo="fixture/validated-tasks",
            candidate_revision="eafe9b38",
            tracked_spec_uris=(),
            output_root=output_root,
        )
        assert manifest["counts"] == {"already_done": 1}

    def test_missing_runtime_binding_fails_loudly(self, monkeypatch, tmp_path):
        monkeypatch.setattr(
            lane,
            "configured_sources",
            lambda: [
                type(
                    "S",
                    (),
                    {"info": type("I", (), {"id": TRACKED_SOURCE})(), "config": self._recipe(monkeypatch, tmp_path)},
                )()
            ],
        )
        monkeypatch.setattr(
            lane,
            "load_folder_rows",
            lambda repo, folders, revision=None: [{"path": "task_0", "task_binary": _fixture_binary("keep me")}],
        )
        monkeypatch.setattr(lane, "tracked_runtimes", lambda uris: collections.defaultdict(dict))
        with pytest.raises(ValueError, match="No tracked runtime binding"):
            lane.convert_cohort(
                tracked_source_id=TRACKED_SOURCE,
                report=self._single_winner_report(),
                folders=("fixture_v1",),
                repo="fixture/validated-tasks",
                candidate_revision="eafe9b38",
                tracked_spec_uris=(),
                output_root=str(tmp_path / "cohort"),
            )

    def test_missing_winner_row_fails_loudly(self, monkeypatch, tmp_path):
        monkeypatch.setattr(
            lane,
            "configured_sources",
            lambda: [
                type(
                    "S",
                    (),
                    {"info": type("I", (), {"id": TRACKED_SOURCE})(), "config": self._recipe(monkeypatch, tmp_path)},
                )()
            ],
        )
        monkeypatch.setattr(lane, "load_folder_rows", lambda repo, folders, revision=None: [])
        with pytest.raises(ValueError, match="missing from the candidate folders"):
            lane.convert_cohort(
                tracked_source_id=TRACKED_SOURCE,
                report=self._report(),
                folders=("fixture_v1",),
                repo="fixture/validated-tasks",
                candidate_revision="eafe9b38",
                tracked_spec_uris=(),
                output_root=str(tmp_path / "cohort"),
            )


def _fixture_binary(instruction: str, *, solve_sh: str | None = None) -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        for name, data in (("instruction.md", instruction.encode()), ("task.toml", b"")):
            member = tarfile.TarInfo(name)
            member.size = len(data)
            archive.addfile(member, io.BytesIO(data))
        if solve_sh is not None:
            member = tarfile.TarInfo("solution/solve.sh")
            member.size = len(solve_sh.encode())
            archive.addfile(member, io.BytesIO(solve_sh.encode()))
    return buffer.getvalue()


class TestDropWrapperSolveSh:
    """Ruling case (a): the FWeindel wrapper solve.sh is dropped after verification."""

    WRAPPER = "#!/bin/bash\nset -euo pipefail\ncp /solution/solution.py /app/solution.py\n"

    def test_dropped_when_pure_wrapper(self):
        rewritten = lane.strip_wrapper_solve_sh(_fixture_binary("keep me", solve_sh=self.WRAPPER))
        assert rewritten is not None
        with tarfile.open(fileobj=io.BytesIO(rewritten), mode="r:*") as archive:
            names = {member.name.removeprefix("./") for member in archive if member.isfile()}
        assert "solution/solve.sh" not in names
        assert "instruction.md" in names

    def test_none_when_absent(self):
        assert lane.strip_wrapper_solve_sh(_fixture_binary("keep me")) is None

    def test_fails_loudly_when_not_the_pure_wrapper(self):
        rogue = "#!/bin/bash\nset -euo pipefail\ncp /solution/solution.py /app/solution.py\nrm -rf /workspace\n"
        with pytest.raises(ValueError, match="not the verified pure wrapper"):
            lane.strip_wrapper_solve_sh(_fixture_binary("keep me", solve_sh=rogue))

    def test_cohort_records_the_normalization_and_converts(self, monkeypatch, tmp_path):
        monkeypatch.setattr(
            lane,
            "configured_sources",
            lambda: [
                type(
                    "S",
                    (),
                    {
                        "info": type("I", (), {"id": TRACKED_SOURCE})(),
                        "config": TestCohortConversion._recipe(monkeypatch, tmp_path),
                    },
                )()
            ],
        )
        monkeypatch.setattr(
            lane,
            "load_folder_rows",
            lambda repo, folders, revision=None: [
                {"path": "task_0", "task_binary": _fixture_binary("keep me", solve_sh=self.WRAPPER)}
            ],
        )
        monkeypatch.setattr(lane, "tracked_runtimes", lambda uris: TestCohortConversion._runtime_join())
        output_root = str(tmp_path / "cohort")
        manifest = lane.convert_cohort(
            tracked_source_id=TRACKED_SOURCE,
            report=TestCohortConversion._single_winner_report(),
            folders=("fixture_v1",),
            repo="fixture/validated-tasks",
            candidate_revision="eafe9b38",
            tracked_spec_uris=(),
            output_root=output_root,
            drop_wrapper_solve_sh=True,
        )
        assert manifest["normalization"] == {"dropped_wrapper_solve_sh": 1}
        assert manifest["counts"] == {"converted": 1}


class TestPrepareModeArgs:
    """The prepare mode consumes no candidate args; argparse must not demand them."""

    def test_prepare_argv_parses_without_candidate_args(self):
        args = lane.build_parser().parse_args(
            [
                "--tracked-source-id",
                TRACKED_SOURCE,
                "--source-revision",
                "eafe9b38",
                "--cohort-root",
                "s3://bucket/cohort",
                "--prepare-output-root",
                "s3://bucket/harbor",
                "--report",
                "s3://bucket/report.json",
            ]
        )
        assert args.prepare_output_root == "s3://bucket/harbor"

    def test_report_mode_still_requires_candidate_args(self):
        parser = lane.build_parser()
        args = parser.parse_args(["--tracked-source-id", TRACKED_SOURCE, "--report", "s3://bucket/report.json"])
        with pytest.raises(SystemExit):
            lane.require_candidate_args(parser, args, prepare_mode=False)


class TestRepairDecision:
    def decision(
        self,
        outcome: ComparativeOutcome,
        *,
        introduced=(),
        oracle=True,
        nop=True,
        quality=Quality.GOOD,
        status=ReferenceStatus.CONSISTENT,
    ):
        return repair_decision(
            verdict=verdict("c1", outcome, introduced=introduced),
            oracle_qualified=oracle,
            nop_qualified=nop,
            candidate_quality=quality,
            candidate_reference_status=status,
        )

    def test_worse_is_kept(self):
        decision, reasons = self.decision(ComparativeOutcome.WORSE)
        assert decision is RepairDecision.KEPT_ORIGINAL
        assert "comparative_outcome_worse" in reasons

    def test_incomparable_is_kept(self):
        decision, reasons = self.decision(ComparativeOutcome.INCOMPARABLE)
        assert decision is RepairDecision.KEPT_ORIGINAL
        assert "comparative_outcome_incomparable" in reasons

    def test_introduced_defect_blocks_replacement(self):
        decision, reasons = self.decision(ComparativeOutcome.BETTER, introduced=["ambiguity"])
        assert decision is RepairDecision.KEPT_ORIGINAL
        assert "defects_introduced" in reasons

    def test_unqualified_execution_blocks_replacement(self):
        for oracle, nop in ((False, True), (True, False), (None, None)):
            decision, reasons = self.decision(ComparativeOutcome.BETTER, oracle=oracle, nop=nop)
            assert decision is RepairDecision.KEPT_ORIGINAL
            assert "oracle_not_qualified" in reasons or "nop_not_qualified" in reasons

    def test_bad_quality_or_reference_conflict_blocks_replacement(self):
        for quality, status in ((Quality.BAD, ReferenceStatus.CONSISTENT), (Quality.GOOD, ReferenceStatus.CONFLICT)):
            decision, reasons = self.decision(ComparativeOutcome.BETTER, quality=quality, status=status)
            assert decision is RepairDecision.KEPT_ORIGINAL
            assert len(reasons) == 1

    def test_missing_verdict_blocks_replacement(self):
        decision, reasons = repair_decision(
            verdict=None,
            oracle_qualified=True,
            nop_qualified=True,
            candidate_quality=Quality.GOOD,
            candidate_reference_status=ReferenceStatus.CONSISTENT,
        )
        assert decision is RepairDecision.KEPT_ORIGINAL
        assert reasons == ["comparative_review_missing"]

    def test_clean_better_candidate_replaces(self):
        decision, reasons = repair_decision(
            verdict=verdict("c1", ComparativeOutcome.BETTER, fixed=["wrong_reference"]),
            oracle_qualified=True,
            nop_qualified=True,
            candidate_quality=Quality.GOOD,
            candidate_reference_status=ReferenceStatus.CONSISTENT,
        )
        assert decision is RepairDecision.REPLACED
        assert reasons == []

    def test_clean_tie_replaces(self):
        decision, reasons = self.decision(ComparativeOutcome.TIE, quality=Quality.ISSUES, status=ReferenceStatus.UNKNOWN)
        assert decision is RepairDecision.REPLACED
        assert reasons == []


class ComparativeReviewServer(ThreadingHTTPServer):
    """A local judge stub returning scripted comparative verdicts."""

    def __init__(self):
        self.requests: list[dict] = []
        self.verdicts: dict[str, dict] = {}
        self.failures: dict[str, int] = {}
        super().__init__(("127.0.0.1", 0), ComparativeHandler)


class ComparativeHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        server = cast(ComparativeReviewServer, self.server)
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        server.requests.append(body)
        payload = json.loads(body["messages"][-1]["content"])
        candidate_id = payload["candidate"]["id"]
        if server.failures.get(candidate_id, 0) > 0:
            server.failures[candidate_id] -= 1
            self.send_error(503)
            return
        arguments = server.verdicts[candidate_id]
        result = {
            "id": "chatcmpl-fixture",
            "object": "chat.completion",
            "created": 0,
            "model": body["model"],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "tool_calls",
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "verdict",
                                "type": "function",
                                "function": {"name": "comparative_verdict", "arguments": json.dumps(arguments)},
                            }
                        ],
                    },
                }
            ],
        }
        encoded = json.dumps(result).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def log_message(self, message, *args):
        pass


@pytest.fixture
def judge():
    server = ComparativeReviewServer()
    thread = threading.Thread(target=server.serve_forever)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        thread.join()
        server.server_close()


def reviewer(server: ComparativeReviewServer) -> ComparativePairReviewer:
    client = OpenAIChatClient(f"http://127.0.0.1:{server.server_port}/v1", "unused", timeout=5)
    return ComparativePairReviewer(client, local_reviewer(ReviewerVersion.V2), max_concurrent=4)


def pair(candidate_id: str, tracked_id: str) -> ComparativePair:
    return ComparativePair(
        candidate=spec(candidate_id, "candidate instruction"), original=spec(tracked_id, "original instruction")
    )


def aliased_verdict(candidate_id: str, outcome: ComparativeOutcome, *, fixed=(), introduced=()) -> dict:
    """A judge reply echoing the candidate's pair alias, as the provider would."""
    return verdict(pair_alias(candidate_id), outcome, fixed=fixed, introduced=introduced).model_dump(mode="json")


class TestComparativeReview:
    def test_reviewed_pairs_return_typed_verdicts_and_usage(self, judge):
        judge.verdicts[pair_alias("c1")] = aliased_verdict("c1", ComparativeOutcome.BETTER, fixed=["wrong_reference"])
        judge.verdicts[pair_alias("c2")] = aliased_verdict("c2", ComparativeOutcome.WORSE, introduced=["ambiguity"])
        result = reviewer(judge).review_pairs([pair("c1", "t1"), pair("c2", "t2")])
        assert [record.status for record in result.records] == [ReviewStatus.REVIEWED, ReviewStatus.REVIEWED]
        assert [record.task_id for record in result.records] == ["c1", "c2"]
        assert result.records[0].verdict.outcome is ComparativeOutcome.BETTER
        assert [defect.value for defect in result.records[0].verdict.defects_fixed] == ["wrong_reference"]
        assert result.records[1].verdict.outcome is ComparativeOutcome.WORSE
        assert result.usage.prompt_tokens == 20 and result.usage.completion_tokens == 10

    def test_verdict_task_id_must_match_the_pair_alias(self, judge):
        judge.verdicts[pair_alias("c1")] = verdict("other-task", ComparativeOutcome.TIE).model_dump(mode="json")
        result = reviewer(judge).review_pairs([pair("c1", "t1")])
        assert result.records[0].status is ReviewStatus.INVALID

    def test_payload_shows_the_alias_as_the_candidate_id(self, judge):
        judge.verdicts[pair_alias("c1")] = aliased_verdict("c1", ComparativeOutcome.TIE)
        reviewer(judge).review_pairs([pair("c1", "t1")])
        payload = json.loads(judge.requests[0]["messages"][-1]["content"])
        assert payload["candidate"]["id"] == pair_alias("c1")
        assert payload["original"]["id"] == "t1"

    def test_invalid_verdicts_retry(self, judge):
        judge.verdicts[pair_alias("c1")] = aliased_verdict("c1", ComparativeOutcome.TIE)
        judge.failures[pair_alias("c1")] = 1
        result = reviewer(judge).review_pairs([pair("c1", "t1")])
        assert result.records[0].status is ReviewStatus.REVIEWED
        assert len(judge.requests) == 2

    def test_persistent_failure_exhausts_attempts(self, judge):
        judge.verdicts[pair_alias("c1")] = aliased_verdict("c1", ComparativeOutcome.TIE)
        judge.failures[pair_alias("c1")] = 10
        result = reviewer(judge).review_pairs([pair("c1", "t1")])
        assert result.records[0].status is ReviewStatus.UNAVAILABLE

    def test_pair_requests_carry_both_sides(self, judge):
        judge.verdicts[pair_alias("c1")] = aliased_verdict("c1", ComparativeOutcome.TIE)
        reviewer(judge).review_pairs([pair("c1", "t1")])
        payload = json.loads(judge.requests[0]["messages"][-1]["content"])
        assert set(payload) == {"original", "candidate"}
        assert payload["candidate"]["id"] == pair_alias("c1") and payload["original"]["id"] == "t1"
        system = judge.requests[0]["messages"][0]["content"]
        assert "atlas-repair-comparative" in system

    def test_empty_workload_short_circuits(self, judge):
        result = reviewer(judge).review_pairs([])
        assert result.records == [] and result.usage.prompt_tokens == 0

    def test_duplicate_candidate_ids_rejected(self, judge):
        with pytest.raises(ValueError):
            reviewer(judge).review_pairs([pair("c1", "t1"), pair("c1", "t2")])


class TestMatchedPairWorkload:
    def test_only_repaired_pairs_become_review_workload(self):
        receipts = [
            receipt("c-ident", "t-ident", RepairMatchClass.IDENTICAL),
            receipt("c-repair", "t-repair", RepairMatchClass.REPAIRED),
            receipt("c-new", None, RepairMatchClass.NEW),
        ]
        workload = matched_pairs(
            outcome_with(receipts), {"t-repair": spec("t-repair", "instruction")}, [spec("c-repair", "instruction")]
        )
        assert [workload_pair.candidate.id for workload_pair in workload.pairs] == ["c-repair"]
        assert workload.tracked_by_candidate == {"c-repair": "t-repair"}

    def test_missing_tracked_specs_count_as_unresolved(self):
        workload = matched_pairs(
            outcome_with([receipt("c-repair", "t-repair", RepairMatchClass.REPAIRED)]),
            {},
            [spec("c-repair", "instruction")],
        )
        assert workload.pairs == [] and workload.unresolved == 1

    def test_content_joined_specs_resolve_prepared_pins(self):
        tracked_spec = spec("t-spec-id", "original instruction")
        instruction_hash = next(iter(content_rows([tracked_spec.model_dump(mode="json")]))).instruction_hash
        tracked_row = content("t-prepared-id", "unused text", instruction=instruction_hash)
        workload = matched_pairs(
            outcome_with([receipt("c-repair", "t-prepared-id", RepairMatchClass.REPAIRED)]),
            {},
            [spec("c-repair", "candidate instruction")],
            tracked_rows=[tracked_row],
            tracked_specs_by_instruction={instruction_hash: tracked_spec},
        )
        assert [workload_pair.candidate.id for workload_pair in workload.pairs] == ["c-repair"]
        assert workload.pairs[0].original.id == "t-spec-id"
        assert workload.unresolved == 0
