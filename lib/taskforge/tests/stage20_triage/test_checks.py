# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The triage's structural checks on a proposal document.

Invariants: none (pins the deterministic preflight checks; the preflight's verdicts are in ``test_preflight.py``).
"""

from dataclasses import replace

import pytest

from taskforge.stage10_proposal.model import ResearchItem, ResearchKind, parse
from taskforge.stage20_triage.checks import (
    CHECKS,
    CheckStatus,
    grader_matches_verification,
    required_sections,
    research_has_purpose,
    resources_in_build_plan,
    run_checks,
)

PROSE = "The task asks for a concrete deliverable with stated inputs, outputs, and a checked format. " * 6


def null_version(p, reason="the capability has no gradable artifact in this environment"):
    return replace(p, header=replace(p.header, research=(), build=(), resources=(), null_reason=reason))


@pytest.mark.parametrize(
    ("verification", "resource", "grader_sentence", "expected"),
    [
        ("code", "prompt.md", "A judge scores the essay against anchored criteria. ", CheckStatus.FAIL),
        ("composite", "prompt.md", "The checker parses the output and recomputes every total. ", CheckStatus.FAIL),
        ("composite", "prompt.md", "The checker parses the output; a judge rubric scores the memo. ", CheckStatus.PASS),
        ("judge", "prompt.md", "The checker parses the output and recomputes every total. ", CheckStatus.FAIL),
        ("simple", "prompt.md", "The answer key is compared after whitespace normalization. ", CheckStatus.PASS),
        # A code file among the resources counts as the executable check.
        ("code", "grader/score.py", "Every item is worth one point toward the total. ", CheckStatus.PASS),
    ],
)
def test_grader_section_must_match_verification(
    make_document, section_replaced, verification, resource, grader_sentence, expected
):
    p = parse(make_document(verification=verification, resources=f'["{resource}"]'))
    p = section_replaced(p, "Grader design and controls", grader_sentence * 10)
    assert grader_matches_verification(p)[0] is expected


def test_required_sections_flags_thin_and_placeholder_sections(proposal, section_replaced):
    thin = section_replaced(proposal, "Risks and null conditions", "None.")
    result, reason = required_sections(thin)
    assert result is CheckStatus.FAIL and "Risks and null conditions" in reason
    placeholder = section_replaced(proposal, "Task", "<...>")
    result, reason = required_sections(placeholder)
    assert result is CheckStatus.FAIL and "placeholder" in reason


@pytest.mark.parametrize(
    "task",
    [
        "Answer with `injective: <yes|no>` and `rate: <...>` per row. " * 10,
        "The deliverable looks like:\n```markdown\n## Build plan\n```\n" + PROSE,
    ],
    ids=["answer-format-in-angle-brackets", "fenced-heading-example"],
)
def test_a_task_section_with_bracketed_formats_or_fenced_headings_passes(proposal, section_replaced, task):
    assert required_sections(section_replaced(proposal, "Task", task))[0] is CheckStatus.PASS


def test_fenced_heading_example_cannot_fill_an_empty_section(proposal, section_replaced):
    p = section_replaced(proposal, "Risks and null conditions", "")
    example = "## Task\n```markdown\n## Risks and null conditions\n" + PROSE + "\n```\n"
    p = replace(p, body=p.body.replace("## Task\n", example, 1))
    result, reason = required_sections(p)
    assert result is CheckStatus.FAIL and "'## Risks and null conditions' has 0 words" in reason


def test_research_purpose_must_be_substantive(proposal):
    items = (ResearchItem(ResearchKind.WEB, "<what to find and why>"), ResearchItem(ResearchKind.GITHUB, "schemas"))
    p = replace(proposal, header=replace(proposal.header, research=items))
    status, reason = research_has_purpose(p)
    assert status is CheckStatus.FAIL
    assert all(f"research[{index}]" in reason for index in range(2))


@pytest.mark.parametrize(
    ("resource", "build_plan", "expected"),
    [
        ("grader/check.py", "Write `grader/check.py`.", CheckStatus.PASS),
        ("grader/check.py", "Write `check.py` beside the fixtures.", CheckStatus.PASS),
        ("build/gold/key.csv", "Produce `build/gold/` per the tables.", CheckStatus.PASS),
        ("fixtures/*.json", "Generate 30 matrices into `fixtures/`.", CheckStatus.PASS),
        ("build/controls/*.md", "Write the controls into `controls/`.", CheckStatus.PASS),
        ("build/controls/*.md", "Write the negative-control answers.", CheckStatus.FAIL),
        ("eval/answer_key.json", "Derive the answer key by hand.", CheckStatus.FAIL),
    ],
)
def test_resource_reference_in_build_plan(proposal, section_replaced, resource, build_plan, expected):
    p = section_replaced(proposal, "Build plan", build_plan + " " + "Each session has acceptance checks. " * 10)
    p = replace(p, header=replace(p.header, resources=(resource,)))
    assert resources_in_build_plan(p)[0] is expected


def test_null_proposal_skips_body_checks(proposal):
    results = {r.name: r.status for r in run_checks(null_version(proposal), CHECKS)}
    assert results["required_sections"] is CheckStatus.SKIP
    assert results["grader_matches_verification"] is CheckStatus.SKIP
