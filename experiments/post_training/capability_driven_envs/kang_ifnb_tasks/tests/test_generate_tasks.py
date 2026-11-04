# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json

from taskcompendium.grading import grade_answer
from taskcompendium.grading_result import Outcome
from taskcompendium.models import ConversationTrace, GradingAttempt, TextMessage, VerifyitGrader, verifyit_spec
from verifyit.spec import StructuredExactSpec

from experiments.post_training.capability_driven_envs.kang_ifnb_tasks.generate_tasks import (
    FOCAL_CELL_TYPES,
    HOUSEKEEPING_GENES,
    IFN_GENES,
    REQUIRED_COLUMNS,
    generate,
)
from experiments.post_training.capability_driven_envs.kang_ifnb_tasks.taskforge_flow import run_campaign, write_release


def fixture_counts() -> bytes:
    lines = [",".join(REQUIRED_COLUMNS)]
    genes = (*HOUSEKEEPING_GENES, *IFN_GENES)
    for donor_index in range(6):
        for condition in ("ctrl", "stim"):
            for cell_index, cell_type in enumerate(FOCAL_CELL_TYPES):
                total = 100_000 + 1000 * donor_index + 100 * cell_index
                for gene_index, gene in enumerate(genes):
                    baseline = 10 + donor_index + gene_index
                    if condition == "stim" and gene in IFN_GENES:
                        count = baseline * (3 + cell_index)
                    else:
                        count = baseline
                    lines.append(
                        f"donor_{donor_index},{condition},{cell_type},{100 + donor_index},{total},{gene},{count}"
                    )
    return ("\n".join(lines) + "\n").encode()


def attempt(task, answer: str) -> GradingAttempt:
    return GradingAttempt(
        conversation=ConversationTrace(events=(*task.context.events, TextMessage(role="assistant", content=answer)))
    )


def test_generates_reproducible_balanced_objective_tasks():
    data = fixture_counts()

    first = generate(data, source_revision="sha256:fixture", count=25)
    second = generate(data, source_revision="sha256:fixture", count=25)

    assert [task.model_dump(mode="json") for task in first] == [task.model_dump(mode="json") for task in second]
    assert len({task.id for task in first}) == 25
    assert {task.tags[-1] for task in first} == {
        "analysis_unit",
        "cell_type_response_comparison",
        "ifn_panel_gene_rank",
        "ifn_signature_cell_type_rank",
        "paired_gene_response",
    }
    for task in first:
        assert isinstance(task.grader, VerifyitGrader)
        assert isinstance(verifyit_spec(task.grader), StructuredExactSpec)
        assert len(task.resources.worker) == 3

    task = first[0]
    spec = verifyit_spec(task.grader)
    assert isinstance(spec, StructuredExactSpec)
    positive = grade_answer(task, attempt(task, json.dumps(spec.expected)))
    negative = grade_answer(task, attempt(task, "{}"))
    assert (positive.status, positive.reward) == (Outcome.GRADED, 1.0)
    assert (negative.status, negative.reward) == (Outcome.GRADED, 0.0)


def test_taskforge_rebuilds_and_releases_candidates(tmp_path):
    data = fixture_counts()
    direct = generate(data, source_revision="sha256:fixture", count=2)
    root = tmp_path / "taskforge"

    accepted = asyncio.run(run_campaign(data, "sha256:fixture", 2, root))

    assert [task.model_dump(mode="json") for task in accepted] == [task.model_dump(mode="json") for task in direct]
    release = tmp_path / "release"
    write_release(release, accepted, data, "sha256:fixture")
    manifest = json.loads((release / "manifest.json").read_text())
    assert manifest["count"] == 2
    assert manifest["status"] == "structural_only_unreviewed_do_not_publish"
