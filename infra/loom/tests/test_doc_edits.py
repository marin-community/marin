# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from infra.loom.doc_edits import edit_message


def test_a_persons_edit_after_the_last_look_is_sent_and_the_agents_own_writes_are_not():
    versions = [
        {"rev": 1, "author": "agent"},
        {"rev": 2, "author": "user"},
        {"rev": 3, "author": "agent"},
    ]

    latest, message = edit_message(versions, seen_rev=1)
    assert latest == 3
    assert message is not None and message.startswith("A person edited the doc: revision 2")

    assert edit_message(versions, seen_rev=2) == (3, None)


def test_the_message_names_the_newest_of_several_edits():
    versions = [{"rev": 1, "author": "agent"}, {"rev": 2, "author": "user"}, {"rev": 3, "author": "user"}]

    _, message = edit_message(versions, seen_rev=1)

    assert message is not None and "revision 3" in message
