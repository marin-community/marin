# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from marin.datakit.chat_render import render_chat_record
from marin.datakit.download.identity_data import row_to_chat_doc


def test_identity_data_uses_structured_turns_as_sft_targets():
    row = {
        "seed_id": "identity-example",
        "content": "User: Who trained you?\n\nAssistant: An incorrect rendered answer.",
        "messages": [
            {"role": "user", "content": "Who trained you?"},
            {"role": "assistant", "content": "I was trained by the Marin Community."},
        ],
    }

    [document] = row_to_chat_doc(row)
    rendered = render_chat_record(document)["text"]

    assert document["source_id"] == "identity-example"
    assert document["messages"][-1]["channel"] == "final"
    assert "Who trained you?" in rendered
    assert "I was trained by the Marin Community." in rendered
    assert "An incorrect rendered answer." not in rendered
