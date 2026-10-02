# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Benchmark the same token workload in a separately provisioned vLLM environment."""

import argparse
import asyncio
import dataclasses
import importlib.metadata
import json
import logging
import time
from pathlib import Path

import torch
from vllm import AsyncEngineArgs, SamplingParams
from vllm.sampling_params import RequestOutputKind
from vllm.v1.engine.async_llm import AsyncLLM

from levanter.inference.benchmark import BatchMeasurement, TokenWorkload, measure_batches

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine-args", type=Path, required=True, help="JSON keyword arguments for vLLM EngineArgs")
    parser.add_argument("--workload", type=Path, required=True)
    parser.add_argument("--provenance", type=Path, required=True, help="JSON checkpoint/config/hardware manifest")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmup-batches", type=int, default=2)
    parser.add_argument("--measured-batches", type=int, default=5)
    args = parser.parse_args()
    workload = TokenWorkload(**json.loads(args.workload.read_text()))
    provenance = json.loads(args.provenance.read_text())
    for field in ["checkpoint", "model_config", "dtype", "hardware_label"]:
        if not provenance.get(field):
            raise ValueError(f"Provenance requires {field}; use an immutable checkpoint revision or weight digest")
    engine_args = json.loads(args.engine_args.read_text())
    if engine_args.get("enable_prefix_caching", False):
        raise ValueError("Disable prefix caching so repeated warmup prompts do not bypass prefill")
    engine_args["enable_prefix_caching"] = False
    setup_start = time.perf_counter()
    engine = AsyncLLM.from_engine_args(AsyncEngineArgs(**engine_args))
    setup_elapsed = time.perf_counter() - setup_start
    batch_index = 0

    async def generate(tokens: TokenWorkload) -> BatchMeasurement:
        nonlocal batch_index
        params = SamplingParams(
            temperature=0,
            max_tokens=tokens.output_tokens,
            ignore_eos=True,
            output_kind=RequestOutputKind.CUMULATIVE,
            detokenize=False,
        )
        request_ids = [f"{batch_index}:{i}" for i in range(len(tokens.prompts))]
        first = {}
        outputs = {}
        start = time.perf_counter()

        async def request(request_id: str, prompt: list[int]) -> None:
            async for output in engine.generate({"prompt_token_ids": prompt}, params, request_id):
                generated = list(output.outputs[0].token_ids)
                if generated:
                    first.setdefault(request_id, time.perf_counter() - start)
                if output.finished:
                    outputs[request_id] = generated

        async with asyncio.TaskGroup() as tasks:
            for request_id, prompt in zip(request_ids, tokens.prompts, strict=True):
                tasks.create_task(request(request_id, prompt))
        elapsed = time.perf_counter() - start
        batch_index += 1
        return BatchMeasurement(elapsed, [first[rid] for rid in request_ids], [outputs[rid] for rid in request_ids])

    with asyncio.Runner() as runner:
        try:
            result = measure_batches(
                workload,
                lambda tokens: runner.run(generate(tokens)),
                warmup_batches=args.warmup_batches,
                measured_batches=args.measured_batches,
            )
        finally:
            engine.shutdown()
    result = dataclasses.asdict(result)
    vllm_distribution = importlib.metadata.distribution("vllm")
    direct_url = vllm_distribution.read_text("direct_url.json")
    result["provenance"] = {
        **provenance,
        "backend": "vllm",
        "evidence_kind": "checkpoint",
        "vllm_distribution_source": None if direct_url is None else json.loads(direct_url),
        "engine_args": engine_args,
        "versions": {name: importlib.metadata.version(name) for name in ["vllm", "torch"]},
        "devices": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
    }
    result["model_and_cache_setup"] = setup_elapsed
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    logger.info("Wrote benchmark result to %s", args.output)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
