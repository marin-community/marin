# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from dataclasses import replace

import pytest

from taskforge.stage10_proposal.model import (
    CompositePart,
    ResearchItem,
    ResearchKind,
    Verification,
    parse,
)
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


def test_valid_proposal_passes_every_check(proposal):
    results = {r.name: r.status for r in run_checks(proposal, CHECKS)}
    assert results == {c.name: CheckStatus.PASS for c in CHECKS}


@pytest.mark.parametrize(
    ("verification", "grader_text", "expected"),
    [
        ("code", "A judge scores the essay against anchored criteria. " * 10, CheckStatus.FAIL),
        ("composite", "The checker parses the output and recomputes every total. " * 10, CheckStatus.FAIL),
        ("composite", "The checker parses the output; a judge rubric scores the memo. " * 10, CheckStatus.PASS),
        ("llm_as_judge", "The checker parses the output and recomputes every total. " * 10, CheckStatus.FAIL),
        ("decision_model", "The checker parses the output and recomputes every total. " * 10, CheckStatus.FAIL),
        ("decision_model", "Each rubric item compares the answer with the reference. " * 10, CheckStatus.PASS),
        ("answer_in_box", "The answer key is compared after whitespace normalization. " * 10, CheckStatus.PASS),
    ],
)
def test_grader_section_must_match_verification(make_document, section_replaced, verification, grader_text, expected):
    p = parse(make_document(verification=verification, resources='["prompt.md"]'))
    p = section_replaced(p, "Grader design and controls", grader_text)
    assert grader_matches_verification(p)[0] is expected


def test_a_composite_needs_the_evidence_of_each_of_its_parts(make_document, section_replaced):
    p = parse(make_document(verification="composite", resources='["prompt.md"]'))
    rubric_only = "Each rubric item compares the answer with the reference. " * 10
    p = section_replaced(p, "Grader design and controls", rubric_only)
    status_, reason = grader_matches_verification(p)
    assert status_ is CheckStatus.FAIL and "an executable check" in reason

    decisions = (CompositePart(Verification.DECISION_MODEL, 0.5), CompositePart(Verification.DECISION_MODEL, 0.5))
    two_rubrics = replace(p, header=replace(p.header, composite=decisions))
    assert grader_matches_verification(two_rubrics)[0] is CheckStatus.PASS


def test_code_file_resource_counts_as_an_executable_check(make_document, section_replaced):
    p = parse(make_document(verification="code", resources='["grader/score.py"]'))
    p = section_replaced(p, "Grader design and controls", "Every item is worth one point toward the total. " * 10)
    assert grader_matches_verification(p)[0] is CheckStatus.PASS


def test_required_sections_flags_thin_and_placeholder_sections(proposal, section_replaced):
    thin = section_replaced(proposal, "Risks and null conditions", "None.")
    result, reason = required_sections(thin)
    assert result is CheckStatus.FAIL and "Risks and null conditions" in reason
    placeholder = section_replaced(proposal, "Task", "<...>")
    result, reason = required_sections(placeholder)
    assert result is CheckStatus.FAIL and "placeholder" in reason


def test_answer_format_in_angle_brackets_is_not_a_placeholder(proposal, section_replaced):
    p = section_replaced(proposal, "Task", "Answer with `injective: <yes|no>` and `rate: <...>` per row. " * 10)
    assert required_sections(p)[0] is CheckStatus.PASS


def test_fenced_heading_example_does_not_end_the_section(proposal, section_replaced):
    task = "The deliverable looks like:\n```markdown\n## Build plan\n```\n" + PROSE
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
    assert research_has_purpose(p) == (
        CheckStatus.FAIL,
        "['research[0]', 'research[1]'] lack a substantive purpose (placeholder or fewer than 5 words)",
    )


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
