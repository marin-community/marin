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
    # pack=True: Levanter packs ~6 Q+A pairs into each 1024-token sequence
    # (GSM8K Q+A averages ~161 tokens). One step at batch=64 covers ~403
    # pair-views, so 19 steps ≈ 7,473 ≈ one true epoch of GSM8K's train split.
    # (pack=False trips a Levanter bool-subclass-int bug — use pack=True or
    # pack=1, not pack=False.)
    pack=True,
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
