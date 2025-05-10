from typing import TypedDict


class LabeledExample(TypedDict):
    text: str
    label: str


class Document(TypedDict):
    id: str
    source: str
    text: str


class Attribute(TypedDict):
    id: str
    source: str
    attributes: dict
