# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Contracts for bounded shared-verifier NeMo static imports."""

import hashlib
import json
from pathlib import Path

import msgspec
import pytest
from tasktrove_verify.spec import Mode

from taskcompendium.grading import grade_attempt
from taskcompendium.importers.nemo_static import StaticCorpus, import_hub_row
from taskcompendium.judging import JudgeReply
from taskcompendium.models import (
    AssistantFinal,
    JudgeConfig,
    JudgeModelPolicy,
    JudgeView,
    Outcome,
    Rejected,
    Rendering,
    ResourceRole,
    TaskTroveVerifier,
)
from taskcompendium.rendering import render_task

FIXTURES = Path(__file__).parent / "fixtures/nemo/static"
JUDGE = JudgeConfig(JudgeModelPolicy("fixture", "small", "fixture", "https://fixture.invalid/v1"), JudgeView())


class FakeJudge:
    def __init__(self, score: str):
        self.score = score

    def complete(self, prompt: str, policy: JudgeModelPolicy, timeout: float) -> JudgeReply:
        return JudgeReply(f"Fixture verdict.\nSCORE: {self.score}", policy.model, "fixture")


def _fixture(name: str) -> bytes:
    return (FIXTURES / name).read_bytes()


def _import(name: str, corpus: StaticCorpus, offset: int, *, judge: JudgeConfig | None = JUDGE):
    result = import_hub_row(
        _fixture(name),
        corpus=corpus,
        split="train" if corpus is not StaticCorpus.SCIENCE else "so_openq",
        offset=offset,
        judge=judge,
    )
    assert not isinstance(result, Rejected), result
    return result


def test_static_fixtures_are_hash_pinned():
    provenance = json.loads(_fixture("provenance.json"))
    for name, record in provenance.items():
        assert hashlib.sha256(_fixture(name)).hexdigest() == record["raw_sha256"]


@pytest.mark.parametrize(
    ("name", "corpus", "offset", "mode"),
    [
        ("mcqa-132738.json", StaticCorpus.MCQA, 132738, Mode.MCQ),
        ("mcqa-441248.json", StaticCorpus.MCQA, 441248, Mode.MCQ),
        ("open-math-0.json", StaticCorpus.OPEN_MATH, 0, Mode.MATH),
        ("open-math-1.json", StaticCorpus.OPEN_MATH, 1, Mode.MATH),
        ("stack-math-0.json", StaticCorpus.STACK_MATH, 0, Mode.MATH),
        ("stack-math-1.json", StaticCorpus.STACK_MATH, 1, Mode.MATH),
        ("open-qa-25704.json", StaticCorpus.OPEN_QA, 25704, Mode.JUDGE),
        ("open-qa-86898.json", StaticCorpus.OPEN_QA, 86898, Mode.JUDGE),
        ("science-144924.json", StaticCorpus.SCIENCE, 144924, Mode.JUDGE),
        ("reasoning-gym-0.json", StaticCorpus.REASONING_GYM, 0, Mode.JUDGE),
        ("reasoning-gym-1.json", StaticCorpus.REASONING_GYM, 1, Mode.MATH),
    ],
)
def test_static_rows_use_shared_verifiers_and_keep_source_private(name, corpus, offset, mode):
    specification = _import(name, corpus, offset)
    verifier = specification.steps[0].verifier
    assert isinstance(verifier, TaskTroveVerifier)
    assert verifier.mode is mode
    private = {
        resource.path: resource for resource in specification.resources if ResourceRole.VERIFIER in resource.roles
    }
    assert set(private) == {"source-row.json", "source-provenance.json"}
    task = render_task(specification, (Rendering("plain", AssistantFinal()),))
    public = msgspec.json.encode(task).decode()
    for hidden in ("expected_answer", "reward_profiles", "source-row.json", "references"):
        assert hidden not in public
    assert r"\boxed" not in specification.steps[0].instructions
    assert "judge" not in specification.steps[0].instructions.lower()


@pytest.mark.parametrize(
    ("name", "corpus", "offset", "wrong"),
    [
        ("mcqa-132738.json", StaticCorpus.MCQA, 132738, "A"),
        ("mcqa-441248.json", StaticCorpus.MCQA, 441248, "A"),
        ("open-math-0.json", StaticCorpus.OPEN_MATH, 0, "31"),
        ("open-math-1.json", StaticCorpus.OPEN_MATH, 1, "0"),
        ("stack-math-0.json", StaticCorpus.STACK_MATH, 0, "0"),
        ("stack-math-1.json", StaticCorpus.STACK_MATH, 1, "1"),
        ("reasoning-gym-1.json", StaticCorpus.REASONING_GYM, 1, "5"),
    ],
)
def test_mcqa_and_math_grade_private_gold(name, corpus, offset, wrong, tmp_path):
    specification = _import(name, corpus, offset)
    source = json.loads(_fixture(name))
    expected = source.get("expected_answer", source.get("answer"))
    assert grade_attempt(specification, Rendering("plain", AssistantFinal()), expected, tmp_path).reward == 1.0
    assert grade_attempt(specification, Rendering("plain", AssistantFinal()), wrong, tmp_path).reward == 0.0


@pytest.mark.parametrize(
    ("name", "corpus", "offset"),
    [
        ("open-qa-25704.json", StaticCorpus.OPEN_QA, 25704),
        ("open-qa-86898.json", StaticCorpus.OPEN_QA, 86898),
        ("science-144924.json", StaticCorpus.SCIENCE, 144924),
        ("reasoning-gym-0.json", StaticCorpus.REASONING_GYM, 0),
    ],
)
def test_reference_rows_exact_gate_and_infrastructure_failure(name, corpus, offset, tmp_path):
    specification = _import(name, corpus, offset)
    reference = specification.steps[0].verifier.parameters["references"][0]
    exact = grade_attempt(specification, Rendering("plain", AssistantFinal()), reference, tmp_path)
    assert exact.status is Outcome.GRADED and exact.reward == 1.0
    unavailable = grade_attempt(specification, Rendering("plain", AssistantFinal()), "fluorine", tmp_path)
    assert unavailable.status is Outcome.INFRA_ERROR and unavailable.reward is None
    judged = grade_attempt(
        specification, Rendering("plain", AssistantFinal()), "fluorine", tmp_path, judge_client=FakeJudge("1")
    )
    assert judged.status is Outcome.GRADED and judged.reward == 1.0


def test_reasoning_gym_is_limited_to_reviewed_answer_shapes():
    result = import_hub_row(
        _fixture("reasoning-gym-1.json"), corpus=StaticCorpus.REASONING_GYM, split="train", offset=2, judge=JUDGE
    )
    assert isinstance(result, Rejected)
    assert result.reason.value == "broken_grader"


def test_reference_rows_require_a_declared_judge_policy():
    result = import_hub_row(_fixture("open-qa-25704.json"), corpus=StaticCorpus.OPEN_QA, split="train", offset=25704)
    assert isinstance(result, Rejected)
    assert result.reason.value == "broken_grader"
