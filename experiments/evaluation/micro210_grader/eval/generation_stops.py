"""Shared stop policies for generation and scoring."""

from collections.abc import Sequence

TOKEN_SENTINEL_SEQUENCES: tuple[str, ...] = (
    "<|im_end|>",
    "<|eot_id|>",
    "<|end_of_text|>",
    "<|endoftext|>",
    "</s>",
)
END_OF_TURN_SEQUENCES: tuple[str, ...] = (
    *TOKEN_SENTINEL_SEQUENCES,
    "\nYou are an AI assistant",
    "\nQuestion:",
    "\nQ:",
    "\n[Question]",
    "\nUser:",
    "\nuser\n",
    "\nAssistant:",
)
# lm-eval imports these through its ``!function`` YAML tag and requires ``until``
# to be a list. Its task factory deep-copies the config before models may append EOS.
GSM8K_STOP_SEQUENCES: list[str] = [  # noqa: ml-module-globals
    "Question:" if stop == "\nQuestion:" else stop for stop in END_OF_TURN_SEQUENCES
]
DROP_STOP_SEQUENCES: list[str] = list(END_OF_TURN_SEQUENCES)  # noqa: ml-module-globals
SHORT_ANSWER_STOP_SEQUENCES: list[str] = list(END_OF_TURN_SEQUENCES)  # noqa: ml-module-globals
OPENAI_COMPLETIONS_MAX_STOP_SEQUENCES = 4
HUMANEVAL_STOP_SEQUENCES: list[str] = [  # noqa: ml-module-globals
    "\nclass",
    "\ndef",
    "\n#",
    "\nif",
    "\nprint",
    "\n```",
    *END_OF_TURN_SEQUENCES,
]
def _is_token_sentinel(stop: str) -> bool:
    return stop in TOKEN_SENTINEL_SEQUENCES


def bounded_request_stops(
    stops: Sequence[str], max_stops: int = OPENAI_COMPLETIONS_MAX_STOP_SEQUENCES
) -> list[str]:
    """Bound request stops while retaining task-semantic boundaries.

    OpenAI-compatible APIs accept at most four stop sequences. Tokenizer
    sentinels are useful fallbacks, but models need not emit them; textual task
    boundaries therefore take precedence when the complete scorer policy does
    not fit in the request. Not for the chat route: servers apply stop strings
    to reasoning content, so it uses :func:`chat_request_stops` instead.
    """
    unique_stops = list(dict.fromkeys(stops))
    semantic = [stop for stop in unique_stops if not _is_token_sentinel(stop)]
    token_sentinels = [stop for stop in unique_stops if _is_token_sentinel(stop)]
    return (semantic + token_sentinels)[:max_stops]


def chat_request_stops(
    stops: Sequence[str], max_stops: int = OPENAI_COMPLETIONS_MAX_STOP_SEQUENCES
) -> list[str]:
    """Choose chat-route request stops: tokenizer sentinels only.

    Chat-completions endpoints end the turn at EOS, and servers apply stop
    strings to the entire generation stream, reasoning content included.
    Textual few-shot boundaries (``Question:``, ``\\nQ:``, ``\\nUser:`` ...)
    therefore truncate reasoning models mid-chain when the model restates the
    question it is answering. Only tokenizer sentinels are safe to send; a
    request without any is fine -- the endpoint stops at EOS, and the scorer
    still applies the complete policy after generation.
    """
    return [stop for stop in dict.fromkeys(stops) if _is_token_sentinel(stop)][:max_stops]


# The scorer still applies the complete set after generation, including the
# omitted ``print``, code-fence, and chat turn boundaries.
HUMANEVAL_REQUEST_STOP_SEQUENCES: list[str] = bounded_request_stops(  # noqa: ml-module-globals
    HUMANEVAL_STOP_SEQUENCES
)


def truncate_at_stop(text: str, stops: Sequence[str] = END_OF_TURN_SEQUENCES) -> str:
    stop_indexes = [index for stop in stops if (index := text.find(stop)) >= 0]
    return text[: min(stop_indexes)] if stop_indexes else text
