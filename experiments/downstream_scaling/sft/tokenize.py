# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Chat-template and resource tier map for GSM8K Q+A SFT.

The chat template renders to the canonical lm-eval GSM8K shape:

    Question: <q>\n
    Answer: <a>

The `{% generation %}` marker pairs with `mask_user_turns=True` to mask loss on
the user turn so loss is only computed on the assistant response (the leading
space + the full gold answer + #### N marker).
"""

from __future__ import annotations

from levanter.data.text import ChatLmDatasetFormat

GSM8K_QA_CHAT_TEMPLATE = "Question: {{ messages[0]['content'] }}\nAnswer:{% generation %} {{ messages[1]['content'] }}{% endgeneration %}"  # noqa: E501

GSM8K_QA_CHAT_FORMAT = ChatLmDatasetFormat(
    messages_field="messages",
    chat_template=GSM8K_QA_CHAT_TEMPLATE,
    # pack=1: each sequence carries exactly one Q+A pair (padded to seq_len).
    # Equivalent intent to pack=False, but Levanter's dataset builder has a
    # bool-subclass-int bug at datasets.py:485-487 that turns `pack=False` into
    # `max_segments_per_example=int(False)=0`, which fails validation. pack=1
    # routes through `int(1)=1` → max_segments=1, the correct "no packing" path.
    # With packing on, one 1024-token sequence holds ~6 Q+A pairs (avg ~161 tok)
    # -> 360 steps * 64 batch * 6.3 pack ~= 145k pair-views ~= 19 epochs of
    # GSM8K's 7,473-problem train split. With pack=1, num_train_steps maps 1:1
    # to pair-views so 117 steps = 1 true epoch.
    pack=1,
    mask_user_turns=True,
)


SFT_RESOURCES: dict[str, str] = {
    # base ladder (Rohith's models/delphi.py minus 1e23)
    "3e18": "v5p-8",
    "9e18": "v5p-8",
    "2e19": "v5p-8",
    "3e19": "v5p-8",
    "9e19": "v5p-8",
    "2e20": "v5p-8",
    "3e20": "v5p-8",
    "1e21": "v5p-8",
    "1e22": "v5p-32",
    # 1e20 iso stand-in base (1.9B params, matches midtrain base architecture)
    "1e20_iso": "v5p-8",
    # midtrained variants from issue #4547 (best-LR per mix per scale)
    "1e20_p33m67_lr0.67": "v5p-8",
    "1e20_p67m33_lr0.33": "v5p-8",
    "1e21_p33m67_lr0.67": "v5p-8",
    "1e21_p67m33_lr0.33": "v5p-8",
    "1e22_p33m67_lr0.67": "v5p-32",
    "1e22_p67m33_lr0.33": "v5p-32",
}
