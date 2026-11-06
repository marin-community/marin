# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json

from infra.loom.gdoc.doc_changes import DocVersion, Watched, goal_doc_id, next_check, next_round

BASE = DocVersion("rev-1", "comments-1")
EDITED = DocVersion("rev-2", "comments-1")
EDITED_AGAIN = DocVersion("rev-3", "comments-1")
COMMENTED = DocVersion("rev-1", "comments-2")


def looks(timeline: list[tuple]) -> tuple[Watched, list[str | None]]:
    """The state after each look at (seconds, doc) or (seconds, doc, completed turns) and each look's message, with
    the state carried through JSON as the watch engine carries it. A doc of None is a look while the session works;
    the turns default to 1."""
    state = Watched("doc").to_json()
    messages = []
    for now, version, *turns in timeline:
        watched, message = next_round(Watched.from_json(json.loads(json.dumps(state))), version, now, *(turns or [1]))
        state = watched.to_json()
        messages.append(message)
    return Watched.from_json(state), messages


def test_a_change_while_the_session_waits_is_sent_once_after_it_stays_the_same_for_ten_seconds():
    _, messages = looks([(0, BASE), (5, EDITED), (10, EDITED), (15, EDITED), (20, EDITED)])

    assert messages[:3] == [None, None, None]
    assert messages[3] is not None and messages[3].startswith("The doc changed: a person changed the text.")
    assert messages[4] is None


def test_edits_made_while_the_session_works_are_not_sent():
    # The agent edits the doc during its turn (None: the session works); it then waits on its own last edit.
    _, messages = looks([(0, BASE), (5, None), (10, None), (15, EDITED), (20, EDITED), (60, EDITED)])

    assert messages == [None] * 6


def test_a_turn_that_no_look_saw_starts_a_new_wait():
    # Regression: a turn that started and ended between two looks left the baseline from before it, so the agent's
    # own edits in that turn were reported as a person's change.
    _, messages = looks([(0, BASE, 1), (30, EDITED, 2), (40, EDITED, 2), (60, EDITED, 2)])
    assert messages == [None] * 4

    _, messages = looks([(0, BASE, 1), (30, EDITED, 2), (35, EDITED_AGAIN, 2), (45, EDITED_AGAIN, 2)])
    assert messages[3] is not None


def test_a_change_in_progress_is_sent_when_it_stops():
    _, messages = looks([(0, BASE), (5, EDITED), (10, EDITED_AGAIN), (15, EDITED_AGAIN), (20, EDITED_AGAIN)])

    assert messages[:4] == [None] * 4
    assert messages[4] is not None


def test_the_message_names_a_comment_change():
    _, messages = looks([(0, BASE), (5, COMMENTED), (15, COMMENTED)])

    assert messages[2] is not None and "changed the comments." in messages[2]


def test_checks_start_every_five_seconds_and_slow_to_every_thirty_while_nothing_changes():
    waiting, _ = looks([(0, BASE)])

    assert [next_check(waiting, now) for now in (0, 59, 60, 120, 180, 600)] == [5, 5, 10, 20, 30, 30]
    assert next_check(Watched("doc"), 0) is None


def test_a_change_brings_the_checks_back_to_every_five_seconds():
    pending, _ = looks([(0, BASE), (600, BASE), (605, EDITED)])
    assert next_check(pending, 605) == 5

    sent, messages = looks([(0, BASE), (605, EDITED), (615, EDITED)])
    assert messages[2] is not None
    assert next_check(sent, 620) == 5


def test_the_doc_id_is_the_first_google_doc_that_the_goal_links():
    goal = (
        "The goal is the doc https://docs.google.com/document/d/15VKTbBk0FCezs7O5IpZrvJ-aoF2zYKN8sqJdNon00qY/edit. "
        "See also https://docs.google.com/document/d/other/edit."
    )

    assert goal_doc_id(goal) == "15VKTbBk0FCezs7O5IpZrvJ-aoF2zYKN8sqJdNon00qY"
    assert goal_doc_id("See https://docs.google.com/document/u/0/d/account-link/edit") == "account-link"
    assert goal_doc_id("Fix the flaky test.") is None
