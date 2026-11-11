# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Compare a checkpoint's serving logprobs with its Megatron score."""

import argparse
import json
import math

import requests
from iris.rpc import job_pb2
from marin.evaluation.eval_env import EVAL_RUNTIME_ENV_KEYS, env_vars_from_keys
from marin.evaluation.hardware import AcceleratorChoice, Platform
from marin.evaluation.serving_config import inference_config_for_model
from marin.inference.iris import remote_inference
from rigging.filesystem.buckets import filesystem_for
from rigging.filesystem.s3_compat import configure_coreweave_s3

from experiments.post_training.stage0.protocol_gate import checkpoint_model


def read_json(url: str) -> dict:
    filesystem, path = filesystem_for(url)
    with filesystem.open(path) as handle:
        return json.load(handle)


def run(checkpoint: str, input_url: str, score_url: str, output_url: str, model_name: str, max_model_len: int) -> None:
    configure_coreweave_s3()
    filesystem, path = filesystem_for(input_url)
    with filesystem.open(path) as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    expected = read_json(score_url)
    if len(rows) != expected["rows"] or sum(row["completion_tokens"] for row in rows) != expected["positions"]:
        raise ValueError("serving parity rows do not match the frozen Megatron score shard")
    accelerator = AcceleratorChoice(platform=Platform.GPU, gpu_type="H100", gpu_count=8, target_cluster="cw-rno2a")
    inference = inference_config_for_model(
        checkpoint_model(model_name, checkpoint, max_model_len=max_model_len),
        accelerator,
        env_vars=env_vars_from_keys(EVAL_RUNTIME_ENV_KEYS),
        priority=job_pb2.PRIORITY_BAND_INHERIT,
    )
    scored = []
    with remote_inference(inference) as session:
        endpoint = session.model.endpoint
        headers = {"Authorization": f"Bearer {endpoint.api_key}"} if endpoint.api_key else {}
        for row in rows:
            token_ids = row["token_ids"]
            body = {
                "model": endpoint.model,
                "prompt": token_ids,
                "add_special_tokens": False,
                "temperature": 0.0,
                "max_tokens": 1,
                "prompt_logprobs": 1,
                "return_tokens_as_token_ids": True,
                "return_token_ids": True,
            }
            response = requests.post(endpoint.url("completions"), json=body, headers=headers, timeout=(30, 900))
            response.raise_for_status()
            (choice,) = response.json()["choices"]
            echoed = [int(str(value).removeprefix("token_id:")) for value in choice["prompt_token_ids"]]
            if echoed != token_ids:
                raise ValueError(f"{row['sample_id']}: vLLM echoed different token IDs")
            prompt_logprobs = choice["prompt_logprobs"]
            if len(prompt_logprobs) != len(token_ids):
                raise ValueError(f"{row['sample_id']}: vLLM returned incomplete prompt logprobs")
            nll_sum = 0.0
            for position in range(row["prompt_tokens"], len(token_ids)):
                entry = prompt_logprobs[position]
                scores = {
                    int(str(key).removeprefix("token_id:")): float(
                        value["logprob"] if isinstance(value, dict) else value
                    )
                    for key, value in entry.items()
                }
                if token_ids[position] not in scores:
                    raise ValueError(f"{row['sample_id']}: realized token absent at position {position}")
                nll_sum -= scores[token_ids[position]]
            if not math.isfinite(nll_sum):
                raise ValueError(f"{row['sample_id']}: nonfinite NLL")
            scored.append({"sample_id": row["sample_id"], "positions": row["completion_tokens"], "nll_sum": nll_sum})
    positions = sum(item["positions"] for item in scored)
    serving_nll = sum(item["nll_sum"] for item in scored) / positions
    megatron_nll = expected["chosen_nll_sum"] / expected["positions"]
    result = {
        "checkpoint": checkpoint,
        "input": input_url,
        "megatron_score": score_url,
        "rows": len(scored),
        "positions": positions,
        "vllm_nll": serving_nll,
        "megatron_nll": megatron_nll,
        "difference": serving_nll - megatron_nll,
        "per_row": scored,
    }
    filesystem, path = filesystem_for(output_url)
    if filesystem.exists(path):
        raise ValueError(f"parity output already exists: {output_url}")
    with filesystem.open(path, "w") as handle:
        json.dump(result, handle, sort_keys=True)
    print(json.dumps({key: result[key] for key in ("rows", "positions", "vllm_nll", "megatron_nll", "difference")}))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--megatron-score", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--model-name", default="opd-1b-m1-parity")
    parser.add_argument("--max-model-len", type=int, default=65_536)
    args = parser.parse_args()
    run(args.checkpoint, args.input, args.megatron_score, args.out, args.model_name, args.max_model_len)


if __name__ == "__main__":
    main()
