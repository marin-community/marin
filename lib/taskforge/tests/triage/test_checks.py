# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from dataclasses import replace

import pytest

from taskforge.proposal.model import BuildItem, Environment, ResearchItem, ResearchKind, Verification, parse
from taskforge.triage.checks import (
    ALL_COMBINATIONS,
    CHECKS,
    CHECKS_BY_NAME,
    CheckContext,
    CheckStatus,
    run_checks,
)

CTX = CheckContext(allowed_combinations=ALL_COMBINATIONS)


def status(name, p, ctx=CTX):
    result = CHECKS_BY_NAME[name].run(p, ctx)
    return result.status, result.reason


def null_version(p, reason="the capability has no gradable artifact in this environment"):
    return replace(p, header=replace(p.header, research=(), build=(), resources=(), null_reason=reason))


def test_valid_proposal_passes_every_applicable_check(proposal):
    results = {r.name: r.status for r in run_checks(proposal, CHECKS, CTX)}
    assert results == {c.name: CheckStatus.PASS for c in CHECKS} | {"null_reason_only": CheckStatus.SKIP}


def test_header_rejects_paths_escaping_the_build_root(proposal):
    p = replace(proposal, header=replace(proposal.header, resources=("../secrets.txt",)))
    assert status("header_schema_valid", p)[0] is CheckStatus.FAIL


def test_header_rejects_a_value_the_parser_would_refuse(proposal):
    p = replace(proposal, header=replace(proposal.header, id="  "))
    result, reason = status("header_schema_valid", p)
    assert result is CheckStatus.FAIL and "id" in reason


def test_header_rejects_repeated_build_names(proposal):
    items = (BuildItem("grader", "one"), BuildItem("grader", "two"))
    p = replace(proposal, header=replace(proposal.header, build=items))
    assert status("header_schema_valid", p) == (CheckStatus.FAIL, "build names repeat: ['grader']")


def test_disallowed_combination_fails_and_null_skips(proposal):
    ctx = CheckContext(allowed_combinations=ALL_COMBINATIONS - {(Environment.REASONING, Verification.CODE)})
    assert status("combination_allowed", proposal, ctx)[0] is CheckStatus.FAIL
    assert status("combination_allowed", null_version(proposal), ctx)[0] is CheckStatus.SKIP


@pytest.mark.parametrize(
    ("verification", "grader_text", "expected"),
    [
        ("code", "A judge scores the essay against anchored criteria. " * 10, CheckStatus.FAIL),
        ("composite", "The checker parses the output and recomputes every total. " * 10, CheckStatus.FAIL),
        ("composite", "The checker parses the output; a judge rubric scores the memo. " * 10, CheckStatus.PASS),
        ("judge", "The checker parses the output and recomputes every total. " * 10, CheckStatus.FAIL),
        ("simple", "The answer key is compared after whitespace normalization. " * 10, CheckStatus.PASS),
    ],
)
def test_grader_section_must_match_verification(make_document, section_replaced, verification, grader_text, expected):
    p = parse(make_document(verification=verification, resources='["prompt.md"]'))
    p = section_replaced(p, "Grader design and controls", grader_text)
    assert status("grader_matches_verification", p)[0] is expected


def test_code_file_resource_counts_as_an_executable_check(make_document, section_replaced):
    p = parse(make_document(verification="code", resources='["grader/score.py"]'))
    p = section_replaced(p, "Grader design and controls", "Every item is worth one point toward the total. " * 10)
    assert status("grader_matches_verification", p)[0] is CheckStatus.PASS


def test_required_sections_flags_thin_and_placeholder_sections(proposal, section_replaced):
    thin = section_replaced(proposal, "Risks and null conditions", "None.")
    result, reason = status("required_sections", thin)
    assert result is CheckStatus.FAIL and "Risks and null conditions" in reason
    placeholder = section_replaced(proposal, "Task", "<...>")
    result, reason = status("required_sections", placeholder)
    assert result is CheckStatus.FAIL and "placeholder" in reason


def test_answer_format_in_angle_brackets_is_not_a_placeholder(proposal, section_replaced):
    p = section_replaced(proposal, "Task", "Answer with `injective: <yes|no>` and `rate: <...>` per row. " * 10)
    assert status("required_sections", p)[0] is CheckStatus.PASS


def test_research_purpose_must_be_substantive(proposal):
    items = (ResearchItem(ResearchKind.WEB, "<what to find and why>"), ResearchItem(ResearchKind.GITHUB, "schemas"))
    p = replace(proposal, header=replace(proposal.header, research=items))
    assert status("research_has_purpose", p) == (
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
    assert status("resources_in_build_plan", p)[0] is expected


def test_null_proposal_carries_only_a_reason(proposal):
    assert status("null_reason_only", null_version(proposal))[0] is CheckStatus.PASS
    keeps_resources = replace(null_version(proposal), header=replace(null_version(proposal).header, resources=("a",)))
    assert status("null_reason_only", keeps_resources) == (
        CheckStatus.FAIL,
        "null proposal still declares ['resources']",
    )
    assert status("null_reason_only", null_version(proposal, "too hard"))[0] is CheckStatus.FAIL


def test_null_proposal_skips_body_checks(proposal):
    results = {r.name: r.status for r in run_checks(null_version(proposal), CHECKS, CTX)}
    assert results["required_sections"] is CheckStatus.SKIP
    assert results["grader_matches_verification"] is CheckStatus.SKIP
