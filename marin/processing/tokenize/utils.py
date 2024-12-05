import logging

logger = logging.getLogger(__name__)


def get_vocab_size_for_tokenizer(tokenizer: str) -> int:

    logger.info("Tokenizer: ", tokenizer)
    if tokenizer == "EleutherAI/gpt-neox-20b":
        vocab_size = 50_257
    elif tokenizer == "meta-llama/Meta-Llama-3.1-8B":
        vocab_size = 128_256
    elif tokenizer == "meta-llama/Llama-2-7b":
        vocab_size = 32_000
    elif tokenizer == "gpt2":
        vocab_size = 50_257
    else:
        raise ValueError(f"Unknown tokenizer: {tokenizer}")

    logger.info("Vocab size: ", vocab_size)
    return vocab_size
