# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import pytest
from tasktrove_verify.modes import xml_elements
from tasktrove_verify.reward import InvalidTask, Status
from tasktrove_verify.spec import XmlElementsSpec

ORDER = """
<order id="A-1">
  <customer><name>Ada</name></customer>
  <quantity>3</quantity>
</order>
"""


@pytest.fixture
def workspace(tmp_path):
    directory = tmp_path / "app"
    directory.mkdir()
    return directory


def answer(workspace, text):
    (workspace / "answer.txt").write_text(text)


def grade(workspace, spec):
    return xml_elements.grade(spec, workspace.parent / "tests", workspace)


def test_document_with_every_required_name_scores_one(workspace):
    answer(workspace, ORDER)
    reward = grade(workspace, XmlElementsSpec(required=("order", "name", "quantity")))
    assert (reward.reward, reward.status) == (1.0, Status.SCORED)


def test_required_name_matches_an_attribute_as_well_as_a_tag(workspace):
    answer(workspace, ORDER)
    assert grade(workspace, XmlElementsSpec(required=("id",))).reward == 1.0


def test_missing_required_name_scores_zero_and_names_it(workspace):
    answer(workspace, ORDER)
    reward = grade(workspace, XmlElementsSpec(required=("order", "email")))
    assert reward.reward == 0.0
    assert reward.detail["missing"] == ["email"]


def test_namespaced_tags_match_their_local_name(workspace):
    answer(workspace, '<ns:order xmlns:ns="http://example.com/o"><ns:quantity>3</ns:quantity></ns:order>')
    assert grade(workspace, XmlElementsSpec(required=("order", "quantity"))).reward == 1.0


def test_document_inside_a_code_fence_is_unwrapped(workspace):
    answer(workspace, f"Here is the document:\n\n```xml\n{ORDER}\n```\n")
    assert grade(workspace, XmlElementsSpec(required=("order",))).reward == 1.0


def test_any_of_needs_one_name_present(workspace):
    answer(workspace, ORDER)
    assert grade(workspace, XmlElementsSpec(any_of=("quantity", "email"))).reward == 1.0
    reward = grade(workspace, XmlElementsSpec(any_of=("email", "phone")))
    assert reward.reward == 0.0
    assert reward.detail["expected"] == ["email", "phone"]


def test_malformed_xml_scores_zero_without_raising(workspace):
    answer(workspace, "<order><quantity>3</order>")
    reward = grade(workspace, XmlElementsSpec(required=("order",)))
    assert (reward.reward, reward.status) == (0.0, Status.SCORED)
    assert reward.detail["reason"] == "parse_error"


def test_prose_that_is_not_xml_scores_zero(workspace):
    answer(workspace, "I could not produce the document, sorry.")
    assert grade(workspace, XmlElementsSpec(required=("order",))).reward == 0.0


@pytest.mark.parametrize("text", [None, "", "   \n\n"])
def test_absent_or_blank_output_scores_zero_with_no_output(workspace, text):
    if text is not None:
        answer(workspace, text)
    reward = grade(workspace, XmlElementsSpec(required=("order",)))
    assert reward.reward == 0.0
    assert reward.detail == {"reason": "no_output"}


def test_spec_naming_nothing_is_an_invalid_task(workspace):
    answer(workspace, ORDER)
    with pytest.raises(InvalidTask, match="required or any_of"):
        grade(workspace, XmlElementsSpec())
