"""
custom_attribute.py

Demo for creating custom attributes.
"""

from collections.abc import Callable
from dataclasses import dataclass

from marin.classifiers.utils import label_documents


@dataclass
class CustomAttributeConfig:
    """
    Configuration class for main process.

    Attributes:
        input_doc_path (str): Path to documents (i.e., gs://$BUCKET/documents/...).
        output_attr_path (str): Path to write attributes (i.e., gs://$BUCKET/attributes/...).
        attribute_func (Callable[[dict],dict]): Function to construct attributes from documents.
        input_attr_paths (list[str]): Path to attributes needed to determine new attribute.
    """

    input_doc_path: str
    output_attr_path: str
    attribute_func: Callable[[dict, list[dict] | None], dict]
    input_attr_paths: list[str] | None = None


def create_custom_attribute(cfg: CustomAttributeConfig):
    label_documents(
        input_doc_path=cfg.input_doc_path,
        output_attr_path=cfg.output_attr_path,
        attribute_func=cfg.attribute_func,
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
            input_doc_path="gs://marin-us-central2/documents/instruct/tulu_v2_mix/text",
            output_attr_path="gs://marin-us-central2/custom_attribute_demo/part1",
            attribute_func=lambda doc, attrs: {"label": len(doc["text"])},
        )
    )
    create_custom_attribute(
        CustomAttributeConfig(
            input_doc_path="gs://marin-us-central2/documents/instruct/tulu_v2_mix/text/",
            output_attr_path="gs://marin-us-central2/custom_attribute_demo/part2",
            attribute_func=lambda doc, attrs: {"label": attrs[0]["label"] + 1},
            input_attr_paths=["gs://marin-us-central2/scratch/custom_attribute_demo/part1"],
        )
    )


if __name__ == "__main__":
    main()
