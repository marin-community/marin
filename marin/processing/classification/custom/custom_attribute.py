"""
custom_attribute.py

Function for creating custom attributes.
"""

from collections.abc import Callable
from dataclasses import dataclass

from marin.classifiers.types import Attribute, Document
from marin.classifiers.utils import label_documents


@dataclass
class CustomAttributeConfig:
    """
    Configuration class for main process.

    Attributes:
        input_doc_path (str): Path to documents (i.e., gs://$BUCKET/documents/...).
        output_attr_path (str): Path to write attributes (i.e., gs://$BUCKET/attributes/...).
        label_func (Callable[[Document, list[Attribute]], dict]): Generates attribute dict from
            document and other input attributes.
        input_attr_paths (list[str]): Path to attributes needed to determine new attribute.
    """

    input_doc_path: str
    output_attr_path: str
    label_func: Callable[[Document, list[Attribute]], dict]
    input_attr_paths: list[str] | None = None


def create_custom_attribute(cfg: CustomAttributeConfig):
    label_documents(
        input_doc_path=cfg.input_doc_path,
        output_attr_path=cfg.output_attr_path,
        label_func=cfg.label_func,
        input_attr_paths=cfg.input_attr_paths,
    )
