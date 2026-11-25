# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the cohort loop audit: Doom's tail detector wiring plus the agent-loop rules."""

import json

import pytest
from vendor.skyrl.scripts.loop_audit import (
    FREQUENT,
    INTERRUPT,
    RUN8,
    STALL,
    TERMINUS8,
    audit_candidate,
    audit_cohort,
    audit_domain,
    audit_trial,
    prompt_hash,
    trial_turns,
)


def cohort_and_row(tokens, *, interrupt=False):
    cohort = {
        "budget": 32768, "max_period_tokens": 4096, "interrupt": interrupt,
        "rows": [{"task_id": "a", "prompt_hash": prompt_hash([42])}],
    }
    row = {"task_id": "a", "prompt_token_ids": [42], "response_token_ids": tokens,
           "verification": {"status": "verified", "passed": False}}
    return cohort, row


def assistant(content):
    return {"role": "assistant", "content": content}


def user(text):
    return {"role": "user", "content": text}


def terminus(command, analysis="a"):
    return json.dumps({"analysis": analysis, "plan": "p", "commands": [{"keystrokes": command, "duration": 1.0}]})


def wait_message():
    return json.dumps({"commands": [{"keystrokes": "", "duration": 10.0}]})


def message_row(messages, tokens=None, task_id="a"):
    return {
        "task_id": task_id, "messages": messages,
        "response_token_ids": tokens if tokens is not None else [1, 2, 3],
        "prompt_token_ids": [42], "verification": {"status": "verified", "passed": False},
    }


def stall_conversation(command="cat a.txt", copies=3):
    """An assistant action repeated ``copies`` times with an identical observation each time."""
    messages = []
    for _ in range(copies):
        messages += [assistant(terminus(command)), user("same output")]
    return messages


@pytest.mark.parametrize("tokens,expected", [([7] * 8, 1), ([7] * 7, 0), (list(range(4096)) * 8, 1), ([7] * 80 + [9], 0)])
def test_audit_detects_only_exact_suffix_with_eight_copies(tokens, expected):
    cohort, row = cohort_and_row(tokens)
    incidence, _ = audit_cohort(cohort, [row])
    assert (incidence.confirmed, incidence.unassessed, incidence.UB) == (expected, 0, expected)


@pytest.mark.parametrize("change", [{"response_token_ids": None}, {"server_error": "failed"},
                                    {"error_treatment": "zero"}, {"exception_type": "TimeoutError"},
                                    {"prompt_token_ids": [99]}])
def test_audit_damaged_generation_enters_upper_bound_and_fails_closed(change):
    cohort, row = cohort_and_row([7] * 8)
    row.update(change)
    row["verification"] = {"status": "verified", "passed": True}
    incidence, _ = audit_cohort(cohort, [row])
    assert (incidence.confirmed, incidence.unassessed, incidence.UB, incidence.passes) == (0, 1, 1, 0)


def test_audit_missing_generation_is_unassessed():
    cohort, _ = cohort_and_row([7] * 8)
    incidence, _ = audit_cohort(cohort, [])
    assert (incidence.N, incidence.unassessed, incidence.unavailable, incidence.UB) == (1, 1, 1, 1)


def test_missing_native_pass_flag_is_unavailable_despite_positive_score():
    cohort, row = cohort_and_row([1, 2, 3])
    row["verification"] = {"status": "verified", "score": 1.0, "passed": None}
    incidence, _ = audit_cohort(cohort, [row])
    assert (incidence.confirmed, incidence.unassessed, incidence.passes, incidence.unavailable, incidence.errors) == (
        0, 0, 0, 1, 0,
    )


@pytest.mark.parametrize("response,expected", [("still reasoning", 1), ("<|end_think|>answer", 0)])
def test_open_thinking_at_cap_includes_the_rendered_prompt(response, expected):
    cohort, row = cohort_and_row(list(range(32)))
    row.update(input_prompt="<|start_think|>", output_response=response, stop_reason="length")
    incidence, _ = audit_cohort(cohort, [row])
    assert (incidence.length_stop, incidence.open_think) == (1, expected)


def test_low_incidence_cohort_is_not_a_target():
    cohort, row = cohort_and_row([7] * 8)
    incidence, _ = audit_cohort(cohort, [row])
    assert incidence.confirmed == 1
    audited = audit_domain("tb2", cohort, [row])
    assert audited.target_status == "non-target"
    candidate = audit_candidate(cohort, [{**row, "response_token_ids": [1, 2, 3]}], audited)
    assert candidate.target_status == "non-target"
    assert candidate.incidence.confirmed == 0


def test_candidate_task_set_must_match_the_sealed_baseline():
    cohort, row = cohort_and_row([7] * 8)
    baseline = audit_domain("tb2", cohort, [row])
    other = {**cohort, "rows": [{"task_id": "b", "prompt_hash": prompt_hash([43])}]}
    with pytest.raises(ValueError, match="sealed baseline"):
        audit_candidate(other, [row], baseline)


def test_agent_loop_trial_share_counts_confirmed_and_upper_bound():
    cohort, _ = cohort_and_row([1, 2, 3])
    cohort["rows"] = [{"task_id": str(index), "prompt_hash": prompt_hash([index])} for index in range(2)]
    rows = [
        dict(message_row(stall_conversation(), tokens=[7] * 8, task_id="0"), prompt_token_ids=[0]),
        dict(message_row([assistant("done")], task_id="1"), prompt_token_ids=[1]),
    ]
    incidence, agent = audit_cohort(cohort, rows)
    assert (incidence.confirmed, incidence.unassessed, incidence.UB) == (1, 0, 1)
    assert agent.trial_share == 0.5
    assert agent.rule_fires[STALL] == 1


def test_damaged_looping_trial_stays_out_of_confirmed_but_in_ub():
    cohort, _ = cohort_and_row([1, 2, 3])
    looping = message_row(stall_conversation())
    looping["exception_type"] = "TimeoutError"
    incidence, agent = audit_cohort(cohort, [looping])
    assert (incidence.confirmed, incidence.unassessed, incidence.UB) == (0, 1, 1)
    assert agent.trial_share == 0.0
    assert agent.trial_share_ub == 1.0


def test_repeat_turn_share_counts_repeats_beyond_first_copies():
    messages = [assistant(terminus(f"cmd {index}")) for index in range(3)] + stall_conversation(copies=4)
    trial = audit_trial(message_row(messages))
    # Four copies of one action: three repeats inside the run, over seven assistant turns.
    assert (trial.repeat_turns, trial.assistant_turns) == (3, 7)


def test_run8_fires_on_eight_identical_actions_despite_changed_replies():
    messages = []
    for index in range(8):
        messages += [assistant(terminus("cat a.txt")), user(f"output {index}")]
    trial = audit_trial(message_row(messages))
    assert RUN8 in trial.rules
    assert STALL not in trial.rules
    assert trial.repeat_turns == 7


def test_seven_identical_actions_do_not_fire_run8():
    row = message_row([assistant(terminus("cat a.txt"))] * 7)
    trial = audit_trial(row)
    assert RUN8 not in trial.rules
    # The observations never change, so the stall rule still fires.
    assert STALL in trial.rules


def test_frequent_rule_needs_count_share_and_a_run():
    frequent = json.dumps({"commands": [{"keystrokes": "cat a.txt", "duration": 1.0}]})
    # Ten copies of one action over 17 turns (59% share) with a run of three identical copies.
    messages = [assistant(frequent), user("same")] * 3
    for index in range(7):
        messages += [assistant(frequent), user(f"out {index}"), assistant(terminus(f"x {index}"))]
    trial = audit_trial(message_row(messages))
    assert STALL in trial.rules
    assert FREQUENT in trial.rules
    # Ten copies but no three identical consecutive actions: the frequent rule does not fire.
    scattered = []
    for index in range(10):
        scattered += [assistant(frequent), user(f"out {index}"), assistant(terminus(f"y {index}"))]
    trial = audit_trial(message_row(scattered))
    assert trial.rules == frozenset()


def test_terminus8_needs_identical_messages_and_a_repeated_observation():
    repeated = []
    for _ in range(8):
        repeated += [assistant(terminus("cat a.txt")), user("same output")]
    trial = audit_trial(message_row(repeated))
    assert TERMINUS8 in trial.rules
    # Churning analysis text breaks the message run even with identical keystrokes.
    churning = []
    for index in range(8):
        churning += [assistant(terminus("cat a.txt", analysis=f"thought {index}")), user("same output")]
    trial = audit_trial(message_row(churning))
    assert TERMINUS8 not in trial.rules


def test_interrupt_rule_fires_only_when_the_cohort_measures_it():
    messages = []
    for _ in range(3):
        messages += [assistant(json.dumps({"commands": [{"keystrokes": "C-c", "duration": 1.0}]})), user("unchanged")]
    cohort, _ = cohort_and_row([1, 2, 3], interrupt=True)
    incidence, agent = audit_cohort(cohort, [message_row(messages)])
    assert agent.rule_fires[INTERRUPT] == 1
    assert agent.interrupt_trial_share == 1.0
    plain_cohort, _ = cohort_and_row([1, 2, 3], interrupt=False)
    _, agent = audit_cohort(plain_cohort, [message_row(messages)])
    assert agent.interrupt_trial_share is None
    assert agent.interrupt_repeat_share is None


def test_interrupt_run_needs_an_unchanged_observation_and_repeated_keystrokes():
    messages = []
    for index in range(3):
        messages += [
            assistant(json.dumps({"commands": [{"keystrokes": "C-c", "duration": 1.0}]})),
            user(f"screen {index}"),
        ]
    trial = audit_trial(message_row(messages))
    assert INTERRUPT not in trial.rules
    resent = [{"keystrokes": "C-c"}, {"keystrokes": "python run.py"}]
    repeated = []
    for _ in range(3):
        repeated += [assistant(json.dumps({"commands": resent})), user("unchanged")]
    trial = audit_trial(message_row(repeated))
    assert INTERRUPT in trial.rules


def test_wait_only_runs_are_exempt_from_the_stall_rule():
    row = message_row([assistant(wait_message()), user("same")] * 4)
    trial = audit_trial(row)
    assert trial.rules == frozenset()
    # A wait between two identical actions leaves only two identical pairs: no stall.
    row = message_row(
        [assistant(terminus("cat a.txt")), user("same"), assistant(wait_message()), user("same"),
         assistant(terminus("cat a.txt")), user("same")]
    )
    trial = audit_trial(row)
    assert trial.rules == frozenset()


def test_trial_turns_split_on_assistant_messages_and_keep_trailing_observations():
    turns = trial_turns([assistant(terminus("ls")), user("out"), assistant(terminus("ls"))])
    assert len(turns) == 2
    assert turns[0].observation != ""
    assert turns[1].observation == ""


def test_duplicate_fraction_reports_median_and_p90():
    cohort, _ = cohort_and_row([1, 2, 3])
    cohort["rows"] = [{"task_id": str(index), "prompt_hash": prompt_hash([index])} for index in range(2)]
    duplicate = [assistant(terminus("cat a.txt")), user("same")] * 3
    duplicate_row = message_row(duplicate + [assistant(terminus("unique")), user("out")], task_id="0")
    fresh_row = message_row([assistant(terminus("one")), user("out"), assistant(terminus("two"))], task_id="1")
    rows = [
        dict(duplicate_row, prompt_token_ids=[0], response_token_ids=[7] * 8),
        dict(fresh_row, prompt_token_ids=[1]),
    ]
    _, agent = audit_cohort(cohort, rows)
    # The duplicate trial repeats two of its four calls; the fresh trial has none (0.0).
    # The cohort median and p90 run over both trials' fractions.
    assert agent.dup_tool_call_frac_median == 0.25
    assert agent.dup_tool_call_frac_p90 == 0.45


def test_cost_and_stop_reason_histogram_are_reported():
    cohort, _ = cohort_and_row([1, 2, 3])
    row = message_row([assistant("12")])
    row.update(stop_reason="repeat_stall", score=0.0, env_metrics={"timing/attempt": 12.5})
    _, agent = audit_cohort(cohort, [row])
    assert agent.stop_reasons == {"repeat_stall": 1}
    assert agent.cost.turns_p50 == 1
    assert agent.cost.wall_seconds_p50 == 12.5
    assert agent.mean_reward == 0.0

