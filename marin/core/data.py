from dataclasses import dataclass


@dataclass
class QAExampleMetadata:
    """
    Dataclass representing metadata for a document.

    Attributes:
        subset (str): subset of dataset
        split (str): split of dataset
        provenance (str): URL of source of data, usually HF
        answer (str): text of answer
        answer_idx (str): index into list of answer options corresponding to correct answer
        answer_label (str): label of correct answer (e.g. A)
        options (list[str]): list of potential options for multiple choice question
        answer_labels (list[str]): list of labels for options (e.g. A,B,C,D)
    """

    subset: str | None = None
    split: str | None = None
    provenance: str | None = None
    answer: str | None = None
    answer_idx: int | None = None
    answer_label: str | None = None
    options: list[str] | None = None
    answer_labels: list[str] | None = None


@dataclass
class QAExample:
    """
    Dataclass representing a document

    Attributes:
        id (str): Unique identifier for the record.
        source (str): The name of the dataset.
        metadata (QAExampleMetadata): Metadata related to the dataset.
        text (str): The text of the document
        prompt (str): If document is prompt/response, the prompt component
        response (str): If document is prompt/response, the expected response component
    """

    id: str
    source: str
    metadata: QAExampleMetadata
    text: str | None = None
    prompt: str | None = None
    response: str | None = None
