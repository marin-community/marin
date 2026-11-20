import traceback
from typing import Any, Dict, List, Optional, Union

from loguru import logger

POSITIVE_RESPONSE_COLOR = "green"
NEGATIVE_RESPONSE_COLOR = "yellow"
BASE_PROMPT_COLOR = "cyan"


def format_exception_text(error: BaseException) -> str:
    """Format an exception without retaining it in a queued logging record."""
    return "".join(traceback.format_exception(type(error), error, error.__traceback__)).rstrip()


def log_exception_as_text(context: str, error: BaseException) -> None:
    """Emit a traceback as text so Loguru never needs to pickle the exception."""
    logger.error("{}:\n{}", context, format_exception_text(error))


def _color_block_format_and_kwargs(
    text: str,
    color: str,
    field_prefix: str,
) -> tuple[str, dict]:
    """Build a format string and kwargs for a multi-line colored block.

    The format string will look like:
        "<color>{p0}</color>\n<color>{p1}</color>\n..."

    where "p0", "p1", ... are placeholder names starting with `field_prefix`.
    """
    # Ensure at least one line
    lines = text.splitlines() or [""]

    fmt_lines = []
    kwargs: dict[str, str] = {}

    for i, line in enumerate(lines):
        key = f"{field_prefix}{i}"
        # NOTE: double braces {{ }} so that {key} survives into str.format
        fmt_lines.append(f"<{color}>{{{key}}}</{color}>")
        kwargs[key] = line

    fmt = "\n".join(fmt_lines)
    return fmt, kwargs


def log_example(
    logger: Any,
    prompt: List[Dict[str, Any]],
    response: str,
    reward: Optional[Union[float, List[float]]] = None,
) -> None:
    """
    Log a single example prompt and response with formatting and colors.

    Args:
        logger: The logger instance to use (expected to be loguru logger or compatible).
        prompt: The input prompt in OpenAI message format.
        response: The output response string.
        reward: The reward value(s) associated with the response.
    """
    reward_val = 0.0
    reward_str = "N/A"
    try:
        prompt_str = str(prompt)
        response_str = str(response)
        # --- Reward handling ---
        if reward is not None:
            if isinstance(reward, list):
                reward_val = float(sum(reward))
            else:
                reward_val = float(reward)
            reward_str = f"{reward_val:.4f}"

        # --- Color selection ---
        if reward is not None and reward_val > 0:
            response_color = POSITIVE_RESPONSE_COLOR
        else:
            response_color = NEGATIVE_RESPONSE_COLOR

        # --- Build per-line colored blocks in the *format string* ---
        prompt_fmt, prompt_kwargs = _color_block_format_and_kwargs(prompt_str, BASE_PROMPT_COLOR, "p")
        response_fmt, response_kwargs = _color_block_format_and_kwargs(response_str, response_color, "r")

        # Single format string with only our own markup and placeholders
        log_format = f"Example:\n  Input: {prompt_fmt}\n  Output (Total Reward: {{reward}}):\n{response_fmt}"

        # Merge all args for str.format
        format_kwargs = {**prompt_kwargs, **response_kwargs, "reward": reward_str}

        # Let Loguru parse tags in log_format and then substitute arguments.
        logger.opt(colors=True).info(log_format, **format_kwargs)
    except Exception as e:
        logger.warning("Error pretty printing example; logging plain text instead: {}", e)
        logger.info(
            "Example:\n  Input: {}\n  Output (Total Reward: {}):\n{}",
            prompt,
            reward_str,
            response,
        )
