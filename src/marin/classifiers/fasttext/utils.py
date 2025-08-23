"""
utils.py

Utility functions for training fastText models.
"""

import unicodedata

import regex as re


def preprocess(text: str) -> str:
    """
    Preprocesses text for fastText training by stripping newline characters.
    """
    return re.sub(r"[\n\r]", " ", text)


def preprocess_normalize(text: str) -> str:
    """
    Comprehensive text normalization for fastText training compatibility.

    Applies the following normalization techniques:
    - Unicode normalization (NFKC)
    - Case folding (lowercase conversion)
    - Digit normalization (replace digits with placeholder)
    - Whitespace normalization and cleanup
    - Special character handling
    - Punctuation normalization

    Args:
        text (str): Input text to normalize

    Returns:
        str: Normalized text suitable for fastText training
    """
    if not text or not isinstance(text, str):
        return ""

    # Unicode normalization - NFKC provides compatibility decomposition + canonical composition
    # This handles various Unicode representations consistently
    text = unicodedata.normalize("NFKC", text)

    # Remove control characters but preserve basic whitespace
    text = "".join(char for char in text if not unicodedata.category(char).startswith("C") or char in "\t\n\r ")

    # Case folding - more aggressive than just lowercase, handles special Unicode cases
    text = text.casefold()

    # Digit normalization - replace all digits with a placeholder token
    # This helps the model generalize better across different numbers
    text = re.sub(r"\d", "0", text)

    # Normalize common punctuation and special characters
    # Replace various quote types with standard quotes
    text = re.sub(r"[`\u00B4]", "'", text)
    text = re.sub(r'[""„"]', '"', text)

    # Normalize dashes and hyphens
    text = re.sub(r"[\u2013\u2014\u2015]", "-", text)

    # Normalize ellipsis
    text = re.sub(r"…", "...", text)

    # Handle common ligatures and special characters
    text = re.sub(r"[ﬀﬁﬂﬃﬄ]", lambda m: {"ﬀ": "ff", "ﬁ": "fi", "ﬂ": "fl", "ﬃ": "ffi", "ﬄ": "ffl"}[m.group()], text)

    # Whitespace normalization
    # Replace various whitespace characters with standard space
    text = re.sub(r"[\t\n\r\f\v\u00a0\u1680\u2000-\u200b\u2028\u2029\u202f\u205f\u3000\ufeff]+", " ", text)

    # Collapse multiple spaces into single space
    text = re.sub(r" +", " ", text)

    # Remove leading/trailing whitespace
    text = text.strip()

    # Additional tokenization-friendly preprocessing
    # Add spaces around punctuation to help with tokenization
    text = re.sub(r"([.!?;:,])", r" \1 ", text)
    text = re.sub(r"([()[\]{}])", r" \1 ", text)

    # Clean up extra spaces created by punctuation spacing
    text = re.sub(r" +", " ", text)
    text = text.strip()

    return text


def normalization(text):
    from nltk.tokenize import wordpunct_tokenize

    tokens = wordpunct_tokenize(text)

    processed_tokens = []
    for token in tokens:
        token = token.lower()

        if token.isdigit():
            processed_tokens.append("<NUM>")
        elif len(token) <= 100:
            processed_tokens.append(token)

    preprocessed_text = " ".join(processed_tokens)

    preprocessed_text = re.sub(r"[\n\r]+", " ", preprocessed_text)
    preprocessed_text = re.sub(r"[-_]+", " ", preprocessed_text)
    preprocessed_text = re.sub(r"[^a-zA-Z0-9\s<NUM>]", "", preprocessed_text)
    preprocessed_text = re.sub(r"\s+", " ", preprocessed_text).strip()

    return preprocessed_text


def megamath_preprocess(text):
    if isinstance(text, bytes):
        text = text.decode("utf-8")

    text = unicodedata.normalize("NFKC", text)

    text = re.sub(r"\s", " ", text)

    text = text.replace("\n", " <EOS> ")

    text = re.sub(r"\s+", " ", text)

    text = normalization(text)

    MAX_LINE_SIZE = 1024
    lines = text.split("<EOS>")
    processed_lines = []
    for line in lines:
        tokens = line.split()
        if len(tokens) > MAX_LINE_SIZE:
            processed_lines.extend(
                [" ".join(tokens[i : i + MAX_LINE_SIZE]) for i in range(0, len(tokens), MAX_LINE_SIZE)]
            )
        else:
            processed_lines.append(line)

    text = " <EOS> ".join(processed_lines)

    return text.strip()


def get_preprocess_fn(preprocess_fn_type: str):
    if preprocess_fn_type == "megamath":
        return megamath_preprocess
    elif preprocess_fn_type == "normalize":
        return preprocess_normalize
    else:
        return preprocess


def make_format_example_fn(preprocess_fn_type: str):
    """
    Creates a function that formats examples for fastText training.
    """
    preprocess_fn = get_preprocess_fn(preprocess_fn_type)

    def format_example(data: dict) -> str:
        return f'__label__{data["label"]}' + " " + preprocess_fn(data["text"])

    return format_example
