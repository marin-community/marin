from typing import Optional


class DatasetConfig:
    def __init__(self, filepath: str, label: str, max_num_samples: Optional[int] = None, preprocessed: bool = False):
        self.filepath = filepath
        self.label = label
        self.max_num_samples = max_num_samples
        self.preprocessed = preprocessed
