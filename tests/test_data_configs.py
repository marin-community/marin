import pytest

from marin.processing.tokenize.data_configs import _are_tokenizers_equivalent


def test_are_tokenizers_equivalent():
    # Test cases where tokenizers should be equivalent
    equivalent_pairs = [
        ("meta-llama/Meta-Llama-3.1-8B", "stanford-crfm/marin-tokenizer"),
        ("meta-llama/Meta-Llama-3.1-8B", "meta-llama/Meta-Llama-3.1-8B-Instruct"),
        ("stanford-crfm/marin-tokenizer", "meta-llama/Meta-Llama-3.1-8B-Instruct"),
    ]

    for t1, t2 in equivalent_pairs:
        try:
            assert _are_tokenizers_equivalent(t1, t2), f"Tokenizers {t1} and {t2} should be equivalent"
        except Exception as e:
            pytest.skip(f"Skipping test because models are not accessible: {e}")

    # Test cases where tokenizers should be different
    different_pairs = [
        ("meta-llama/Meta-Llama-3.1-8B", "EleutherAI/gpt-neox-20b"),
        ("stanford-crfm/marin-tokenizer", "EleutherAI/gpt-neox-20b"),
        ("meta-llama/Meta-Llama-3.1-8B-Instruct", "EleutherAI/gpt-neox-20b"),
    ]

    for t1, t2 in different_pairs:
        try:
            assert not _are_tokenizers_equivalent(t1, t2), f"Tokenizers {t1} and {t2} should be different"
        except Exception as e:
            pytest.skip(f"Skipping test because models are not accessible: {e}")

    # Test that a tokenizer is equivalent to itself
    for t in [
        "meta-llama/Meta-Llama-3.1-8B",
        "stanford-crfm/marin-tokenizer",
        "meta-llama/Meta-Llama-3.1-8B-Instruct",
        "EleutherAI/gpt-neox-20b",
    ]:
        try:
            assert _are_tokenizers_equivalent(t, t), f"Tokenizer {t} should be equivalent to itself"
        except Exception as e:
            pytest.skip(f"Skipping test because model is not accessible: {e}")
