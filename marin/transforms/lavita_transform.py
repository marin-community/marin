import datasets

def transform_lavita_dataset(dataset: datasets.Dataset) -> datasets.Dataset:
    """
    Transforms a Hugging Face Dataset by concatenating 'instruction', 'input', and 'output' columns.

    Args:
        dataset: The input Hugging Face Dataset.

    Returns:
        The transformed Hugging Face Dataset with a new 'text' column.
    """
    def transform_example(example):
        example['text'] = f"{example['instruction']}\n\n{example['input']}\n\n{example['output']}"
        return example

    return dataset.map(transform_example)
