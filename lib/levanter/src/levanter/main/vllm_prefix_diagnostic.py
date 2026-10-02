# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Untimed teacher-forced next-token logprobs in the pinned vLLM runtime."""

import argparse
import asyncio
import json
from pathlib import Path

from vllm import AsyncEngineArgs, SamplingParams
from vllm.v1.engine.async_llm import AsyncLLM


async def diagnose(engine: AsyncLLM, sequences: list[list[int]]) -> list[dict]:
    async def request(index: int, tokens: list[int]) -> dict:
        params = SamplingParams(temperature=0, max_tokens=1, ignore_eos=True, logprobs=20, detokenize=False)
        async for output in engine.generate({"prompt_token_ids": tokens}, params, f"prefix-{index}"):
            if output.finished:
                choice = output.outputs[0]
                assert choice.logprobs is not None
                return {
                    "token_ids": list(choice.token_ids),
                    "top_logprobs": {token: value.logprob for token, value in choice.logprobs[0].items()},
                }
        raise RuntimeError("vLLM did not finish the teacher-forced diagnostic")

    return await asyncio.gather(*(request(index, tokens) for index, tokens in enumerate(sequences)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine-args", type=Path, required=True)
    parser.add_argument("--prefixes", type=Path, required=True)
    parser.add_argument("--provenance", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    inputs = json.loads(args.prefixes.read_text())
    sequences = inputs["sequences"]
    if not sequences or len(sequences) > 8 or any(not row or len(row) > 128 for row in sequences):
        raise ValueError("Diagnostic requires at most eight nonempty prefixes of at most128 tokens")
    engine_args = json.loads(args.engine_args.read_text())
    if engine_args["enable_prefix_caching"]:
        raise ValueError("Disable prefix caching for the shared-prefix diagnostic")
    engine = AsyncLLM.from_engine_args(AsyncEngineArgs(**engine_args))
    try:
        with asyncio.Runner() as runner:
            results = runner.run(diagnose(engine, sequences))
        result = {
            "boundary": "untimed_teacher_forced_single_prefill",
            "checkpoint": json.loads(args.provenance.read_text())["checkpoint"],
            "prefixes": inputs,
            "engine_args": engine_args,
            "effective_dtype": str(engine.vllm_config.model_config.dtype),
            "results": results,
        }
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result), flush=True)
    finally:
        engine.shutdown()


if __name__ == "__main__":
    main()
