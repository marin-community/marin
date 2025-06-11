import pytest
import tempfile

from experiments.create_marin_tokenizer import load_llama3_tokenizer, create_marin_tokenizer, special_tokens_injection_check
from transformers import AutoTokenizer, PreTrainedTokenizer
from experiments.marin_models import MARIN_CUSTOM_SPECIAL_TOKENS


@pytest.fixture
def marin_tokenizer():
    """Fixture that provides a configured marin tokenizer for testing."""
    llama3_tokenizer = load_llama3_tokenizer()
    tokenizer = create_marin_tokenizer(llama3_tokenizer)

    # Roundtrip write-read to ensure consistency
    with tempfile.TemporaryDirectory() as temp_path:
        tokenizer.save_pretrained(temp_path)
        tokenizer = AutoTokenizer.from_pretrained(temp_path, local_files_only=True)

    return tokenizer


def test_special_tokens_injection(marin_tokenizer: PreTrainedTokenizer):
    """Test that special tokens are correctly replaced."""
    special_tokens_injection_check(marin_tokenizer)



