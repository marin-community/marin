"""Debug logging utilities for evaluation."""
import logging
import json
from typing import Any

logger = logging.getLogger(__name__)


def log_tokenizer_details(tokenizer: Any, model_name: str) -> None:
    """Log detailed tokenizer information."""
    logger.info("=" * 80)
    logger.info(f"TOKENIZER DETAILS FOR: {model_name}")
    logger.info("=" * 80)
    
    # Basic info
    logger.info(f"Tokenizer class: {type(tokenizer).__name__}")
    logger.info(f"Vocab size: {len(tokenizer)}")
    
    # Special tokens
    logger.info("\nSpecial tokens:")
    logger.info(f"  BOS: {tokenizer.bos_token!r} (id={tokenizer.bos_token_id})")
    logger.info(f"  EOS: {tokenizer.eos_token!r} (id={tokenizer.eos_token_id})")
    logger.info(f"  PAD: {tokenizer.pad_token!r} (id={tokenizer.pad_token_id})")
    
    # Chat template
    if hasattr(tokenizer, 'chat_template') and tokenizer.chat_template:
        logger.info("\nChat template (first 1000 chars):")
        logger.info(tokenizer.chat_template[:1000])
        
        # Check for problematic patterns
        if "{% generation %}" in tokenizer.chat_template:
            logger.warning("⚠️  Chat template contains {% generation %} tags!")
            logger.warning("   These are Levanter-specific and may not work with vLLM")
        
        # Test formatting
        test_messages = [
            {"role": "user", "content": "What is 2+2?"}
        ]
        
        try:
            formatted = tokenizer.apply_chat_template(
                test_messages,
                tokenize=False,
                add_generation_prompt=True
            )
            logger.info("\nTest prompt formatting:")
            logger.info(f"Input: {test_messages}")
            logger.info(f"Formatted output:\n{formatted}")
            logger.info(f"Output length: {len(formatted)} chars")
        except Exception as e:
            logger.error(f"Error testing chat template: {e}")
    else:
        logger.warning("⚠️  No chat template found!")
    
    logger.info("=" * 80)


def log_vllm_initialization(model_path: str, model_args: str, engine_kwargs: dict) -> None:
    """Log vLLM initialization parameters."""
    logger.info("=" * 80)
    logger.info("VLLM INITIALIZATION")
    logger.info("=" * 80)
    logger.info(f"Model path: {model_path}")
    logger.info(f"Model args string: {model_args}")
    logger.info(f"Engine kwargs: {json.dumps(engine_kwargs, indent=2)}")
    logger.info("=" * 80)


def log_sample_generation(
    prompt: str,
    response: str,
    task_name: str,
    instance_idx: int
) -> None:
    """Log a sample prompt and response."""
    logger.info("=" * 80)
    logger.info(f"SAMPLE GENERATION - Task: {task_name}, Instance: {instance_idx}")
    logger.info("=" * 80)
    logger.info("PROMPT:")
    logger.info(prompt)
    logger.info("-" * 80)
    logger.info("RESPONSE:")
    logger.info(response[:2000])  # First 2000 chars
    if len(response) > 2000:
        logger.info(f"... (truncated, full length: {len(response)} chars)")
    logger.info("=" * 80)