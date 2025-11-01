# Copyright 2025 The Marin Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass
from typing import Any

from marin.core.conversation import OpenAIChatMessage


@dataclass
class PreferenceDatasetFormat:
    """Format of the Preference Dataset (DPO, RM, etc)."""

    CHOSEN_REJECTED: str = "chosen_rejected"


@dataclass
class PreferenceTransformAdapter:
    source: str
    dataset_format: str = PreferenceDatasetFormat.CHOSEN_REJECTED
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
        dataset_format=PreferenceDatasetFormat.CHOSEN_REJECTED,
        chosen_column="chosen",
        rejected_column="rejected",
        role_key="role",
        content_key="content",
    )
)

register_preference_adapter(
    PreferenceTransformAdapter(
        source="allenai/olmo-2-1124-7b-preference-mix",
        dataset_format=PreferenceDatasetFormat.CHOSEN_REJECTED,
        chosen_column="chosen",
        rejected_column="rejected",
        role_key="role",
        content_key="content",
    ),
)
