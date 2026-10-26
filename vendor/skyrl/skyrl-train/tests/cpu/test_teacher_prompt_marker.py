import pytest

from marinskyrl.distillation import TeacherPromptInsertion, _teacher_prompt_insertions, conditioned_teacher_prompt


def test_marker_rule_inserts_after_a_variable_length_header():
    rule = TeacherPromptInsertion(after_prefix=(1, 2), tokens=(90, 91), after_marker=(7, 8))
    plain = [1, 2, 3, 7, 8, 40, 41]
    with_tools = [1, 2, 3, 55, 56, 57, 7, 8, 40]
    assert conditioned_teacher_prompt(plain, (rule,)) == [1, 2, 3, 7, 8, 90, 91, 40, 41]
    assert conditioned_teacher_prompt(with_tools, (rule,)) == [1, 2, 3, 55, 56, 57, 7, 8, 90, 91, 40]


def test_prefix_rule_is_unchanged():
    rule = TeacherPromptInsertion(after_prefix=(1, 2), tokens=(90,))
    assert conditioned_teacher_prompt([1, 2, 3], (rule,)) == [1, 2, 90, 3]


def test_marker_rule_without_its_marker_fails():
    rule = TeacherPromptInsertion(after_prefix=(1, 2), tokens=(90,), after_marker=(7, 8))
    with pytest.raises(ValueError, match="no insertion marker"):
        conditioned_teacher_prompt([1, 2, 7, 3, 8], (rule,))


def test_marker_is_parsed_from_the_teacher_config():
    rules = _teacher_prompt_insertions(
        [{"after_prefix": [1, 2], "tokens": [9], "after_marker": [7, 8]}, {"after_prefix": [1, 3], "tokens": [9]}],
        "teachers.identity.prompt_insertions",
    )
    assert rules == (TeacherPromptInsertion((1, 2), (9,), (7, 8)), TeacherPromptInsertion((1, 3), (9,)))
