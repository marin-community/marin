from datasets import Dataset


def _category_documents(documents: Dataset, category: str) -> Dataset:
    return documents.filter(lambda document: document["category"] == category)


def wikipedia_english(documents: Dataset) -> Dataset:
    return _category_documents(documents, "wikipedia_english")


def wikipedia_nonenglish(documents: Dataset) -> Dataset:
    return _category_documents(documents, "wikipedia_nonenglish")


def github_python(documents: Dataset) -> Dataset:
    return _category_documents(documents, "github_python")


def github_cpp(documents: Dataset) -> Dataset:
    return _category_documents(documents, "github_cpp")


def github_javascript(documents: Dataset) -> Dataset:
    return _category_documents(documents, "github_javascript")


def github_markdown(documents: Dataset) -> Dataset:
    return _category_documents(documents, "github_markdown")


def github_other(documents: Dataset) -> Dataset:
    return _category_documents(documents, "github_other")


def bbc_news(documents: Dataset) -> Dataset:
    return _category_documents(documents, "bbc_news")


def arxiv_physics(documents: Dataset) -> Dataset:
    return _category_documents(documents, "arxiv_physics")


def arxiv_computer_science(documents: Dataset) -> Dataset:
    return _category_documents(documents, "arxiv_cs")


def arxiv_math(documents: Dataset) -> Dataset:
    return _category_documents(documents, "arxiv_math")


def arxiv_other(documents: Dataset) -> Dataset:
    return _category_documents(documents, "arxiv_other")


def biorxiv_all(documents: Dataset) -> Dataset:
    return _category_documents(documents, "biorxiv_all")


def ao3_english(documents: Dataset) -> Dataset:
    return _category_documents(documents, "ao3_english")


def ao3_nonenglish(documents: Dataset) -> Dataset:
    return _category_documents(documents, "ao3_nonenglish")
