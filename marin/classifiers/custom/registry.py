from marin.classifiers.types import Attribute, Document

REGISTRY = {}


def register(func):
    REGISTRY[func.__name__] = func
    return func


@register
def max_quality_score(
    doc: list[Document], input_attrs: list[Attribute], input_attr_names: list[str], output_attr_name: str
):
    return {
        output_attr_name: {
            "score": max(
                attr["attributes"][f"{input_attr_names[classifier_id]}"]["__label__hq"]
                for classifier_id, attr in enumerate(input_attrs)
            )
        }
    }
