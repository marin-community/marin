# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from typing import Any

from marin.core.conversation import OpenAIChatMessage


@dataclass
class PreferenceTransformAdapter:
    source: str
    chosen_column: str = "chosen"
    rejected_column: str = "rejected"
    role_key: str = "role"
    content_key: str = "content"

    def extract_preference_example(self, row: dict[str, Any]) -> dict[str, list[OpenAIChatMessage]] | None:
        """
        Convert a row with 'chosen' and 'rejected' columns (each a list of messages)
        into a standardized dict with OpenAIChatMessage lists for each.
        """
        chosen = row.get(self.chosen_column)
        rejected = row.get(self.rejected_column)
        if not chosen or not rejected:
            return None

        def convert(messages):
            return [OpenAIChatMessage(role=msg[self.role_key], content=msg[self.content_key]) for msg in messages]

        return {"chosen": convert(chosen), "rejected": convert(rejected)}


preference_transform_templates: dict[str, PreferenceTransformAdapter] = {}


def register_preference_adapter(adapter: PreferenceTransformAdapter):
    preference_transform_templates[adapter.source] = adapter


def get_preference_adapter(source: str) -> PreferenceTransformAdapter | None:
    return preference_transform_templates.get(source)


register_preference_adapter(
    PreferenceTransformAdapter(
        source="HuggingFaceH4/ultrafeedback_binarized",
        chosen_column="chosen",
        rejected_column="rejected",
        role_key="role",
        content_key="content",
    )
)

register_preference_adapter(
    PreferenceTransformAdapter(
        source="allenai/olmo-2-1124-7b-preference-mix",
        chosen_column="chosen",
        rejected_column="rejected",
        role_key="role",
        content_key="content",
    ),
)
