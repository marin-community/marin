"""
custom_attribute.py

Demo for creating custom attributes.
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
        output_attr_path (str): Path to write attributes (i.e., gs://$BUCKET/attributes/...).
        input_doc_path (str): Path to documents (i.e., gs://$BUCKET/documents/...).
        label_func (Callable[[Document, list[Attribute]], dict]): Generates attribute dict from
            document and other input attributes.
        input_attr_paths (list[str]): Path to attributes needed to determine new attribute.
    """

    output_attr_path: str
    input_doc_path: str
    label_func: Callable[[Document, list[Attribute]], dict]
    input_attr_paths: list[str] | None = None


def create_custom_attribute(cfg: CustomAttributeConfig):
    label_documents(
        output_attr_path=cfg.output_attr_path,
        input_doc_path=cfg.input_doc_path,
        label_func=cfg.label_func,
        input_attr_paths=cfg.input_attr_paths,
    )


def main():
    """
    Demo for creating custom attributes.

    First creates an attribute based on the length of the text in a document.
    Then increments that attribute by 1.
    """
    create_custom_attribute(
        CustomAttributeConfig(
            output_attr_path="gs://marin-us-central2/custom_attribute_demo/part1",
            input_doc_path="gs://marin-us-central2/documents/instruct/tulu_v2_mix/text",
            label_func=lambda doc, attrs: {"label": len(doc["text"])},
        )
    )
    create_custom_attribute(
        CustomAttributeConfig(
            output_attr_path="gs://marin-us-central2/custom_attribute_demo/part2",
            input_doc_path="gs://marin-us-central2/documents/instruct/tulu_v2_mix/text/",
            label_func=lambda doc, attrs: {"label": attrs[0]["label"] + 1},
            input_attr_paths=["gs://marin-us-central2/scratch/custom_attribute_demo/part1"],
        )
    )


if __name__ == "__main__":
    main()
