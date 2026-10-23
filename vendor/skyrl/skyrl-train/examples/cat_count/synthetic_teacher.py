"""A programmed CatCount teacher served through the OpenAI-compatible teacher protocol.

The teacher knows the right answer, so it is not a language model. For each prompt it reads N, then gives every
response token one log-probability:

- log ``correct_probability`` while the response is still a prefix of ``cat`` repeated N times, and for a stop
  token declared by the tokenizer or serving engine once the response is exactly that;
- for the first token that leaves that prefix, the log of the remaining probability spread evenly over the rest of
  the vocabulary, as if the teacher were a full distribution;
- after that first error, the correct score for a stop token and the wrong score for everything else, so a student
  that went wrong is taught to stop.

The wrong score must scale with the vocabulary. A fixed 0.001 suits a tiny word-level vocabulary but is above the
student's own probability for most of a 150k-token vocabulary, so it would push rarely sampled junk tokens up.

Noise makes the teacher imperfect:

- ``jitter`` adds Gaussian noise with that standard deviation to every response log-probability, capped at 0;
- ``error_rate`` is the probability of attempting an N+1 or N-1 target for a whole response; target counts stay at least 1.

Noise is causal and deterministic. Each response is seeded from ``seed`` and its prompt, the wrong-count draw comes
first, and each response token then consumes one jitter draw, so a token's score depends only on the prompt and
the tokens before it, as a language model's would, and a retried request gets the same scores.

Only end-of-sequence tokens stop a reply: the tokenizer's by default, or the ones the inference engine stops on
(``stop_token_ids``; the server adds the model's generation-config EOS ids, such as Qwen's <|endoftext|> beside
<|im_end|>). Any other special token the student writes, such as a chat-template header, is a wrong token. ``flipped`` swaps the correct and wrong log-probabilities; a student trained on a flipped teacher should
get worse, which tests that learning depends on the teacher's signal.

Serve it for a training run on the policy's tokenizer::

    uv run --project .. python examples/cat_count/synthetic_teacher.py --tokenizer <policy-dir> --port 18080
"""

from __future__ import annotations

import argparse
import hashlib
import math
import random
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

from aiohttp import web
from skyrl_gym.envs.cat_count.reward import TARGET_WORD
from transformers import AutoTokenizer, GenerationConfig, PreTrainedTokenizerBase

# Tokenizers that split digits decode "12" as "1 2".
_N_PATTERN = re.compile(r"exactly (\d[\d ]*) times")


@dataclass(frozen=True)
class TeacherNoise:
    correct_probability: float = 0.95
    jitter: float = 0.0
    error_rate: float = 0.0
    flipped: bool = False
    seed: int = 0

    def __post_init__(self):
        if not 0 < self.correct_probability < 1:
            raise ValueError("correct_probability must be in (0, 1)")
        if self.jitter < 0 or not 0 <= self.error_rate <= 1:
            raise ValueError("jitter must be non-negative and error_rate must be in [0, 1]")


class CatCountTeacher:
    """Score CatCount responses token by token against the known answer."""

    def __init__(self, tokenizer: PreTrainedTokenizerBase, noise: TeacherNoise, stop_token_ids: Iterable[int] = ()):
        self.tokenizer = tokenizer
        self.noise = noise
        self.correct_logprob = math.log(noise.correct_probability)
        self.wrong_logprob = math.log((1 - noise.correct_probability) / (len(tokenizer) - 1))
        self.stop_token_ids = frozenset(stop_token_ids) if stop_token_ids else frozenset({tokenizer.eos_token_id})
        self.special_token_ids = frozenset(tokenizer.all_special_ids)
        message = [{"role": "user", "content": "x"}]
        with_header = tokenizer.apply_chat_template(
            message, tokenize=True, return_dict=False, add_generation_prompt=True
        )
        without_header = tokenizer.apply_chat_template(
            message, tokenize=True, return_dict=False, add_generation_prompt=False
        )
        if with_header[: len(without_header)] != without_header or len(with_header) == len(without_header):
            raise ValueError("the chat template must append an assistant header for the generation prompt")
        self.assistant_header = list(with_header[len(without_header) :])

    def response_start(self, sequence: Sequence[int]) -> int:
        """Return where the response begins: after the first assistant header, since prompts are single-turn.

        A header the student generates later is part of its response and must not move this boundary.
        """
        header = self.assistant_header
        for start in range(len(sequence) - len(header) + 1):
            if list(sequence[start : start + len(header)]) == header:
                return start + len(header)
        raise ValueError("the scored sequence has no assistant header")

    def target_count(self, prompt_ids: Sequence[int], rng: random.Random) -> int:
        match = _N_PATTERN.search(self.tokenizer.decode(prompt_ids, skip_special_tokens=True))
        if match is None:
            raise ValueError("the scored prompt does not name a CatCount N")
        n = int(match.group(1).replace(" ", ""))
        if rng.random() < self.noise.error_rate:
            n = max(1, n + rng.choice((-1, 1)))
        return n

    def correct_tokens(self, response_ids: Sequence[int], n: int) -> list[bool]:
        """Return, for each response token, whether the teacher prefers it."""
        target = " ".join([TARGET_WORD] * n)
        verdicts = []
        on_track = True
        for index, token_id in enumerate(response_ids):
            if token_id in self.stop_token_ids:
                text = self.tokenizer.decode(response_ids[:index], skip_special_tokens=True).strip()
                verdicts.append(not on_track or text == target)
                on_track = False
                continue
            if on_track and token_id not in self.special_token_ids:
                text = self.tokenizer.decode(response_ids[: index + 1], skip_special_tokens=True).lstrip()
                on_track = target.startswith(text) or text.rstrip() == target
                verdicts.append(on_track)
            elif on_track:
                on_track = False
                verdicts.append(False)
            else:
                verdicts.append(False)
        return verdicts

    def score(self, sequence: Sequence[int]) -> list[float | None]:
        """Return one log-probability per sequence position; the first position has none, as in vLLM."""
        start = self.response_start(sequence)
        identity = repr((self.noise.seed, list(sequence[:start]))).encode()
        rng = random.Random(int.from_bytes(hashlib.sha256(identity).digest()[:8], "big"))
        n = self.target_count(sequence[:start], rng)
        high, low = self.correct_logprob, self.wrong_logprob
        if self.noise.flipped:
            high, low = low, high
        scores: list[float | None] = [None, *([0.0] * (start - 1))]
        for correct in self.correct_tokens(sequence[start:], n):
            value = high if correct else low
            if self.noise.jitter:
                value = min(0.0, value + rng.gauss(0.0, self.noise.jitter))
            scores.append(value)
        return scores


def engine_stop_token_ids(model: str, tokenizer: PreTrainedTokenizerBase) -> frozenset[int]:
    """Return the EOS ids vLLM stops on: the tokenizer's and any in the model's generation config."""
    ids = {tokenizer.eos_token_id}
    try:
        eos = GenerationConfig.from_pretrained(model).eos_token_id
    except OSError:
        return frozenset(ids)
    ids.update(eos if isinstance(eos, list) else [eos] if eos is not None else [])
    return frozenset(ids)


def completion_choice(index: int, sequence: Sequence[int], scores: Sequence[float | None]) -> dict[str, Any]:
    return {
        "index": index,
        "text": "",
        "finish_reason": "length",
        "logprobs": {
            "tokens": [f"token_id:{token_id}" for token_id in sequence],
            "token_logprobs": list(scores),
            "top_logprobs": [None, *({} for _ in sequence[1:])],
            "text_offset": [0] * len(sequence),
        },
    }


def application(teacher: CatCountTeacher) -> web.Application:
    """Serve ``/v1/completions`` with ``echo`` scoring of token-ID prompts, as vLLM does."""

    async def completions(request: web.Request) -> web.Response:
        body = await request.json()
        prompts = body.get("prompt")
        if not isinstance(prompts, list) or not all(
            isinstance(sequence, list) and sequence and all(isinstance(token, int) for token in sequence)
            for sequence in prompts
        ):
            return web.json_response({"error": "prompt must be a non-empty batch of token-ID sequences"}, status=400)
        try:
            choices = [completion_choice(index, seq, teacher.score(seq)) for index, seq in enumerate(prompts)]
        except ValueError as error:
            return web.json_response({"error": str(error)}, status=400)
        return web.json_response({"choices": choices})

    app = web.Application()
    app.router.add_post("/v1/completions", completions)
    return app


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tokenizer", required=True, help="the student policy's tokenizer directory or HF name")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--correct-probability", type=float, default=TeacherNoise.correct_probability)
    parser.add_argument("--jitter", type=float, default=0.0)
    parser.add_argument("--error-rate", type=float, default=0.0)
    parser.add_argument("--flipped", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    noise = TeacherNoise(
        correct_probability=args.correct_probability,
        jitter=args.jitter,
        error_rate=args.error_rate,
        flipped=args.flipped,
        seed=args.seed,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    teacher = CatCountTeacher(tokenizer, noise, engine_stop_token_ids(args.tokenizer, tokenizer))
    web.run_app(application(teacher), host=args.host, port=args.port, print=None)


if __name__ == "__main__":
    main()
