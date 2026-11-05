# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json

from infra.loom.gdoc.doc_changes import DocVersion, Watched, goal_doc_id, next_round

BASE = DocVersion("rev-1", "comments-1")
EDITED = DocVersion("rev-2", "comments-1")
EDITED_AGAIN = DocVersion("rev-3", "comments-1")
COMMENTED = DocVersion("rev-1", "comments-2")


def rounds(versions: list[DocVersion | None]) -> list[str | None]:
    """The message of each round, with the state carried through JSON as the watch engine carries it."""
    state = Watched("doc").to_json()
    messages = []
    for version in versions:
        watched, message = next_round(Watched.from_json(json.loads(json.dumps(state))), version)
        state = watched.to_json()
        messages.append(message)
    return messages


def test_a_change_while_the_session_waits_is_sent_once_after_it_holds_for_a_round():
    messages = rounds([BASE, EDITED, EDITED, EDITED])

    assert messages[:2] == [None, None]
    assert messages[2] is not None and messages[2].startswith("The doc changed: a person changed the text.")
    assert messages[3] is None


def test_edits_made_while_the_session_works_are_not_sent():
    # The agent edits the doc during its turn (None: the session works); it then waits on its own last edit.
    assert rounds([BASE, None, None, EDITED, EDITED]) == [None] * 5


def test_a_change_in_progress_is_sent_when_it_stops():
    messages = rounds([BASE, EDITED, EDITED_AGAIN, EDITED_AGAIN])

    assert messages[:3] == [None, None, None]
    assert messages[3] is not None


def test_the_message_names_a_comment_change():
    message = rounds([BASE, COMMENTED, COMMENTED])[2]

    assert message is not None and "changed the comments." in message


def test_the_doc_id_is_the_first_google_doc_that_the_goal_links():
    goal = (
        "The goal is the doc https://docs.google.com/document/d/15VKTbBk0FCezs7O5IpZrvJ-aoF2zYKN8sqJdNon00qY/edit. "
        "See also https://docs.google.com/document/d/other/edit."
    )

    assert goal_doc_id(goal) == "15VKTbBk0FCezs7O5IpZrvJ-aoF2zYKN8sqJdNon00qY"
    assert goal_doc_id("Fix the flaky test.") is None
