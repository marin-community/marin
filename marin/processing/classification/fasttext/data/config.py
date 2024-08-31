from dataclasses import dataclass
from typing import Optional


@dataclass
class DatasetConfig:
    """Configuration to generate the final fasttext-formatted data.

    filepath (str): Path to the dataset file.

    label (str): Label for the dataset.

    max_num_samples (Optional[int]): Maximum number of samples to use. If the filepath is a directory, then
    we will perform uniform sampling across the files in the directory.

    preprocessed (bool): Whether the dataset is preprocessed. If the dataset is already in fasttext format or not.
    If it is already in fasttext format, then there is no need to preprocess it.
    """

    filepath: str
    label: str
    max_num_samples: Optional[int] = None
    preprocessed: bool = False
