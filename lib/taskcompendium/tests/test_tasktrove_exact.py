# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""TaskTrove exact imports with direct-chat Harbor coverage."""

import json
from pathlib import Path

from tasktrove_verify.grade import grade as source_grade
from tasktrove_verify.spec import ExactSpec, parse_spec

from taskcompendium.grading import Outcome
from taskcompendium.importers.tasktrove.exact import import_task
from taskcompendium.lowering import HarborEnvironmentConfig, lower_to_harbor
from taskcompendium.models import AnswerType, ConversationTrace, TextMessage, VerifierKind
from taskcompendium.submission import GradingAttempt, PlainText, render_instruction
from taskcompendium.verifier_registry import grade_answer

from .harbor_replay import run_replay_trial
from .tasktrove_fixtures import FIXTURE_DATASET_URI, FIXTURE_REVISION, exact_archive

FIXTURES = Path(__file__).parent / "fixtures/tasktrove/clean-alpha"


def _archive():
    return exact_archive()


def _attempt(specification, response: str):
    return GradingAttempt(
        ConversationTrace(events=(*specification.context.events, TextMessage(role="assistant", content=response))),
        {},
        object(),
    )


async def test_exact_import_matches_source_list_verifier_and_tags(tmp_path):
    archive = _archive()
    specification = import_task(archive)
    source_contract = parse_spec(archive.files["tests/verifier.toml"].decode())
    assert isinstance(source_contract, ExactSpec)
    assert specification.answer_type is AnswerType.TEXT
    assert specification.verifier.kind is VerifierKind.EXACT_LIST_ANSWER
    assert specification.tags == ("test", "puzzle", "ordered-list")
    assert specification.source.dataset == FIXTURE_DATASET_URI
    assert specification.source.revision == FIXTURE_REVISION
    assert specification.source.row == "laion__all-puzzles-v2:synthetic-exact-task"
    instruction = render_instruction(specification, PlainText(id="plain"))
    assert "/app/answer.txt" not in instruction
    assert "verifier reads" not in instruction.lower()
    assert ", ".join(source_contract.expected) not in instruction

    correct = ", ".join(source_contract.expected)
    incorrect_order = ", ".join(reversed(source_contract.expected))
    answer_path = tmp_path / "answer.txt"
    contract = ExactSpec(
        expected=source_contract.expected,
        ignore_case=source_contract.ignore_case,
        ignore_whitespace=source_contract.ignore_whitespace,
        ordered=source_contract.ordered,
        output=str(answer_path),
    )
    for answer, reward in ((correct, 1.0), (incorrect_order, 0.0)):
        answer_path.write_text(answer)
        source_result = source_grade(contract, tmp_path, tmp_path)
        imported_result = await grade_answer(specification, PlainText(id="plain"), _attempt(specification, answer))
        assert source_result.reward == reward
        assert (imported_result.status, imported_result.reward) == (Outcome.GRADED, reward)

    task_dir = lower_to_harbor(
        specification,
        PlainText(id="plain"),
        HarborEnvironmentConfig(),
        tmp_path / "harbor-task",
    )
    for response, name, reward in ((correct, "valid-exact-list", 1.0), (incorrect_order, "wrong-exact-list", 0.0)):
        harbor_result = await run_replay_trial(
            task_dir,
            {"role": "assistant", "content": response},
            tmp_path / "trials",
            name,
        )
        outcome = json.loads((tmp_path / f"trials/{name}/verifier/taskcompendium-result.json").read_text())
        assert harbor_result.exception_info is None, harbor_result.exception_info
        assert outcome == {"status": "graded", "reward": reward, "error": None}


def test_exact_sample_provenance_records_revision_manifest_and_hash():
    provenance = json.loads((FIXTURES / "provenance.json").read_text())
    sample = next(row for row in provenance["samples"] if row["source"] == "laion__all-puzzles-v2")
    assert provenance["dataset_revision"] == "9065fa568394f286dab0081e43dc76fc87c48984"
    assert provenance["manifest_sha256"] == "5b8b0337d1473fbf2dd6bc2e682b4c597ca988edcb398e5abcd6f009c3aef75a"
    assert provenance["parquet_lfs_sha256"] == "83320be884448b96ec715738520c0989fbc99a914cef6b97811f6d66c43a1a94"
    assert provenance["upstream_revision"] == "0292300"
    assert sample["source_path"] == "all_puzzles-5039"
    assert sample["archive_sha256"] == "2fa5479915b7aebc0d45149ae8b00db6ea32d1f858cda9820dcf832c071e7b71"
    assert sample["redistribution_status"] == "source-level rights review pending"
    assert _archive().source.revision == FIXTURE_REVISION
