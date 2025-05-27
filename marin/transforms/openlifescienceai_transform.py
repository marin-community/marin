import datasets

def transform_openlifescienceai_dataset(dataset: datasets.Dataset) -> datasets.Dataset:
    """
    Transforms a Hugging Face Dataset by concatenating 'question' and 'exp' columns.

    Args:
        dataset: The input Hugging Face Dataset.

    Returns:
        The transformed Hugging Face Dataset with a new 'text' column.
    """
    def transform_example(example):
        example['text'] = f"{example['question']}\n\n{example['exp']}"
        return example

    return dataset.map(transform_example)
