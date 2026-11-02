# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The deterministic pieces of the arms: the probe battery, embed_file, the L4 shadow checks and the L3 guard."""

import ast
import asyncio
import json

import pytest

pytest.importorskip("taskforge")

from taskcompendium.grading_result import Outcome
from taskforge.stage40_build_with_program.agentic_author import ToolRefusal
from taskforge.stage50_validate.adversary import Defect
from taskforge.stage60_review.rules import default_review

from experiments.post_training.capability_driven_envs.acceptance_measure.author import embedded
from experiments.post_training.capability_driven_envs.acceptance_measure.dryrun import synthetic_round
from experiments.post_training.capability_driven_envs.acceptance_measure.probes import (
    Contract,
    Expect,
    Witness,
    battery,
    met,
)
from experiments.post_training.capability_driven_envs.acceptance_measure.review import keyword_class, saturation
from experiments.post_training.capability_driven_envs.acceptance_measure.runtime import TAXONOMY, arm_runtime
from experiments.post_training.capability_driven_envs.acceptance_measure.shadows import (
    Framing,
    contract_cite,
    delivery,
    frames,
    framing_of,
    read_taxonomy,
)
from experiments.post_training.capability_driven_envs.acceptance_measure.treatment import load_arms

REFERENCE = json.dumps({"label": "converges", "limit": 1.5, "terms": 3})


def by_name(probes) -> dict:
    return {p.name: p for p in probes}


def test_a_json_reference_gets_structural_probes_whose_expectations_follow_the_contract():
    probes = by_name(battery(REFERENCE, {}, Contract()))

    assert probes["reference"].expect is Expect.FULL and probes["reference"].reply == REFERENCE
    assert probes["keys_reordered"].expect is Expect.FULL
    assert json.loads(probes["keys_reordered"].reply) == json.loads(REFERENCE)
    for name in (
        "duplicate_key_wrong_first",
        "one_field_wrong",
        "extra_key",
        "number_as_string",
        "label_case",
        "fenced",
    ):
        assert probes[name].expect is Expect.AT_MOST_CEILING, name
    assert probes["integral_float"].reply.count('"terms": 3.0') == 1
    assert probes["bom"].expect is Expect.FULL and probes["bom"].reply.startswith("﻿")


def test_relaxed_contract_fields_turn_rejections_into_equivalences_or_drop_them():
    probes = by_name(
        battery(
            REFERENCE, {}, Contract(framing="fenced", closed_schema=False, labels="aliases", transport_noise="reject")
        )
    )

    assert probes["fenced"].expect is Expect.FULL
    assert "extra_key" not in probes and "label_case" not in probes
    assert probes["bom"].expect is Expect.AT_MOST_CEILING


def test_a_file_channel_mutates_the_answer_file_and_witnesses_replay_with_their_polarity():
    contract = Contract(channel="file", answer_path="/workspace/answer.json")
    witnesses = [
        Witness("r0-fa", "false_accept", '{"label": "CONVERGES"}', {}),
        Witness("r0-fr", "false_reject", "", {"/workspace/answer.json": REFERENCE}),
    ]
    probes = by_name(battery("", {"/workspace/answer.json": REFERENCE}, contract, witnesses))

    assert probes["one_field_wrong"].reply == "" and "/workspace/answer.json" in probes["one_field_wrong"].files
    assert probes["reply_channel_only"].files == {} and probes["reply_channel_only"].reply == REFERENCE
    assert probes["witness_r0-fa"].expect is Expect.BELOW_FULL and probes["witness_r0-fr"].expect is Expect.FULL


def test_contract_and_expectation_rules():
    with pytest.raises(ValueError, match="answer_path is given exactly when"):
        Contract(channel="file")
    assert met(Expect.FULL, Outcome.GRADED, 1.0, 0.0) and not met(Expect.FULL, Outcome.GRADED, 0.99, 0.0)
    assert met(Expect.AT_MOST_CEILING, Outcome.SUBMISSION_FAILURE, None, 0.0)
    assert not met(Expect.AT_MOST_CEILING, Outcome.GRADED, 0.5, 0.25)
    assert met(Expect.BELOW_FULL, Outcome.GRADED, 0.5, 0.25)


def test_embed_file_rewrites_the_marked_assignment_to_the_exact_text():
    source = "X = 1\n# embed: reference/helpers/grade_kit.py\nKIT = ''\n# end embed\nY = 2\n"
    text = "def f():\n    return '''quoted'''\\n"

    updated = embedded(source, "reference/helpers/grade_kit.py", text)

    namespace: dict = {}
    exec(compile(ast.parse(updated), "program.py", "exec"), namespace)
    assert namespace["KIT"] == text and namespace["Y"] == 2
    with pytest.raises(ToolRefusal, match=r"has no '# embed: other\.py' marker"):
        embedded(source, "other.py", text)


def test_the_fa_contract_cite_needs_a_verbatim_quote_of_the_instruction():
    instruction = "Reply with one JSON object.\nThe label must be one of  converges or diverges, exactly as written."
    cited = 'The grader folds case although the task says "the label must be one of converges or diverges, exactly".'
    invented = 'A correct answer would use "SI units with three significant figures" which the grader ignores.'

    assert contract_cite(cited.replace("the label", "The label"), instruction)["precondition"] == "pass"
    assert contract_cite(invented, instruction) == {
        "quotes": 1, "quote": "SI units with three significant figures", "verbatim": False, "precondition": "fail",
    }  # fmt: skip
    assert contract_cite("no quote at all", instruction)["precondition"] == "fail"


def test_the_fr_delivery_check_reads_the_reference_framing_after_transport_noise():
    assert framing_of(REFERENCE) is Framing.EXACT
    assert framing_of(f"```json\n{REFERENCE}\n```") is Framing.FENCED
    assert framing_of(f"The answer:\n{REFERENCE}") is Framing.LAST
    assert framing_of("42 hours") is Framing.TEXT
    assert frames("﻿" + REFERENCE.replace(",", ",\r\n") + "\n", Framing.EXACT)
    assert not frames(f"Here it is: {REFERENCE}", Framing.EXACT)


def test_the_l3_guard_needs_all_trials_solved_and_a_passing_fa_search():
    review = arm_runtime(load_arms()["C"]).review

    def read(result):
        return saturation(result, asyncio.run(default_review(result, band=review.band)), 8)

    assert read(synthetic_round("w", 8))["accept"]
    assert not read(synthetic_round("w", 7))["accept"]
    flagged = read(synthetic_round("w", 8, (Defect((1,), "case folding", True),)))
    assert not flagged["accept"] and not flagged["fa_passed"] and not flagged["band_only_modify"]


def test_keyword_classes_fall_back_to_other():
    taxonomy = read_taxonomy(TAXONOMY)

    assert keyword_class("The grader accepts an extra key in the object", taxonomy) == "open_schema"
    assert keyword_class("something unforeseen", taxonomy) == "other"


def test_delivery_requires_the_reference_answer_files_and_framing():
    exact = delivery("\ufeff" + REFERENCE, ["/workspace/x"], REFERENCE, [])
    assert exact == {"framing": "exact", "framed": True, "missing_files": [], "precondition": "pass"}
    assert delivery(f"My answer is {REFERENCE}", [], REFERENCE, [])["precondition"] == "fail"
    missing = delivery("", ["/workspace/notes.md"], "", ["/workspace/answer.json"])
    assert missing["missing_files"] == ["/workspace/answer.json"] and missing["precondition"] == "fail"
