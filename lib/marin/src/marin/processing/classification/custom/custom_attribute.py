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
custom_attribute.py

Function for creating a custom attribute.

Computes the value of a custom attribute for each document in a collection of documents. The attribute can
be an arbitrary (user-specified) function of document itself as well as some number of other
pre-existing document attributes.

Example 1: Count the number of words in each document (depends only on the text of the document).

Example 2: Compute the maximum quality score over two quality classifiers (depends on the attribute files
    generated from running inference with both quality classifiers).

"""

from dataclasses import dataclass
from typing import Any

from marin.classifiers.custom.registry import REGISTRY
from marin.classifiers.utils import label_documents


@dataclass
class CustomAttributeConfig:
    """
    Configuration class for creating a custom attribute.

    Attributes:
        input_doc_path (str): Path to documents (i.e., gs://$BUCKET/documents/...).
        output_attr_path (str): Path to write attributes (i.e., gs://$BUCKET/attributes/...).
        attribute_func_name (str): Name of attribute function to use in registry.
        attribute_func_kwargs (dict[str, Any]): Keyword arguments to pass to the attribute function.
            Should be serializeable to JSON.
        input_attr_paths (list[str]): Path to attributes needed to determine new attribute.
    """

    input_doc_path: str
    output_attr_path: str
    attribute_func_name: str
    attribute_func_kwargs: dict[str, Any] | None = None
    input_attr_paths: list[str] | None = None


def create_custom_attribute(cfg: CustomAttributeConfig):
    """
    Create a custom attribute for each document in a collection of documents.

    Args:
        cfg (CustomAttributeConfig): Configuration for creating a custom attribute.
    """

    def label_func(doc, attrs):
        return REGISTRY[cfg.attribute_func_name](doc, attrs, **cfg.attribute_func_kwargs)

    label_documents(
        input_doc_path=cfg.input_doc_path,
        output_attr_path=cfg.output_attr_path,
        label_func=label_func,
        input_attr_paths=cfg.input_attr_paths,
    )
