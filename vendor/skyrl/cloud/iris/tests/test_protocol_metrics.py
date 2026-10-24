"""Report each protocol result per source, over only the attempts that assessed it."""

from skyrl_train.evaluate import protocol_metrics


def test_protocol_rates_separate_results_and_exclude_unassessed_attempts():
    diagnostics = [
        {"protocol": {"check": "tool_execution", "label": "executed", "protocol_compliance": True, "answer_correctness": False}},
        {"protocol": {"check": "tool_execution", "label": "no_invocation", "protocol_compliance": False, "answer_correctness": True}},
        {"protocol": {"check": "tool_execution", "not_evaluable": "grade infra_error: machine lost"}},
        {"protocol": {"check": "required_file", "label": "submitted", "submission_completeness": True}},
        {"other": 1},
    ]
    sources = ["probes", "probes", "probes", "probes", "math"]

    metrics = protocol_metrics(diagnostics, sources)

    assert metrics == {
        "eval/probes/protocol/tool_execution/protocol_compliance": 0.5,
        "eval/probes/protocol/tool_execution/protocol_compliance_assessed": 2.0,
        "eval/probes/protocol/tool_execution/answer_correctness": 0.5,
        "eval/probes/protocol/tool_execution/answer_correctness_assessed": 2.0,
        "eval/probes/protocol/required_file/submission_completeness": 1.0,
        "eval/probes/protocol/required_file/submission_completeness_assessed": 1.0,
    }
