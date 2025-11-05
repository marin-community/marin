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

"""
Registry of custom attribute functions.

Each function should take a document, a list of attributes, and an optional additional kwargs.
It should return a dictionary with the attribute name as the key and the attribute value as the value.
We use this registry so that we can version the attribute functions within ExecutorSteps.
"""

from collections.abc import Callable
from typing import Any

from marin.classifiers.types import Attribute, Document

REGISTRY = {}


def register(func: Callable[[Document, list[Attribute], Any | None], dict]):
    REGISTRY[func.__name__] = func
    return func


@register
def max_quality_score(
    doc: list[Document],
    input_attrs: list[Attribute],
    input_attr_names: list[str],
    score_name: str,
    output_attr_name: str,
):
    """
    Take the maximum of the input attributes.
    """
    return {
        output_attr_name: {
            "score": max(
                attr["attributes"][input_attr_names[classifier_id]][score_name]
                for classifier_id, attr in enumerate(input_attrs)
            )
        }
    }
