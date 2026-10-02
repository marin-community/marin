# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Untimed teacher-forced next-token logprobs in the pinned vLLM runtime."""

import argparse
import asyncio
import hashlib
import json
import tempfile
from functools import partial
from pathlib import Path
from typing import cast

import torch

from vllm import AsyncEngineArgs, SamplingParams
from vllm.v1.engine.async_llm import AsyncLLM
from vllm.v1.worker.worker_base import WorkerBase


def install_final_state_capture(model) -> None:
    """Install untimed eager hooks after engine startup; leave outputs and weights unchanged."""
    if model.model.config.is_hero or len(model.model.layers) > 2 or model.model.config.hidden_dim > 512:
        raise ValueError("Stage capture supports at most two Snowball layers of width512")
    gate = model.model.embed_gated_norm
    weights = {"norm": model.model.embed_norm.weight, "down": gate.down_proj.weight, "up": gate.up_proj.weight}
    records: list[dict] = [
        {
            "site": "embedding_gate_weights",
            "sha256": {
                name: hashlib.sha256(weight.detach().float().cpu().contiguous().numpy().tobytes()).hexdigest()
                for name, weight in weights.items()
            },
        }
    ]
    handles = []
    restores = []
    model.prefix_diagnostic_capture = (records, handles, restores)

    def capture_inputs(_module, _args, kwargs):
        records.append(
            {
                "site": "model_inputs",
                "token_ids": kwargs["input_ids"].detach().cpu().tolist(),
                "positions": kwargs["positions"].detach().cpu().tolist(),
            }
        )

    def capture_norm(_module, args, output):
        records.append(
            {
                "site": "final_norm",
                "pre_final_norm": args[0].detach().float().cpu().tolist(),
                "post_final_norm": output.detach().float().cpu().tolist(),
            }
        )

    def capture_gate(_module, _args, output):
        records.append({"site": "final_gate", "post_final_gate": output.detach().float().cpu().tolist()})

    def capture_projection(module, args, output):
        head, hidden = args[:2]
        weight = head.weight[: model.config.vocab_size].detach().float().cpu()
        hidden_cpu = hidden.detach().float().cpu()
        records.append(
            {
                "site": "projection",
                "post_final_gate": hidden_cpu.tolist(),
                "logits": output.detach().float().cpu().tolist(),
                "projection_fp64": (hidden_cpu.double() @ weight.double().T).tolist(),
                "head_weight_sha256": hashlib.sha256(weight.contiguous().numpy().tobytes()).hexdigest(),
                "logits_dtype": str(output.dtype),
                "hidden_dtype": str(hidden.dtype),
                "weight_dtype": str(head.weight.dtype),
                "head_dtype_override": None if module.head_dtype is None else str(module.head_dtype),
            }
        )

    def output_hook(site, layer_index=None):
        def capture(_module, _args, output):
            records.append(
                {"site": site, "layer_index": layer_index, "values": output.detach().float().cpu().tolist()}
            )

        return capture

    def input_hook(site, layer_index):
        def capture(_module, args):
            records.append(
                {"site": site, "layer_index": layer_index, "values": args[0].detach().float().cpu().tolist()}
            )

        return capture

    def capture_gate_projection(site, _module, _args, output):
        projected, _ = output
        records.append({"site": site, "layer_index": None, "values": projected.detach().float().cpu().tolist()})
        if site == "embedding_gate_up":
            records.append(
                {
                    "site": "embedding_gate_sigmoid_recomputed",
                    "layer_index": None,
                    "values": torch.sigmoid(projected).detach().float().cpu().tolist(),
                }
            )

    handles.append(model.model.embed_norm.register_forward_hook(output_hook("embedding_norm")))
    handles.append(gate.down_proj.register_forward_hook(partial(capture_gate_projection, "embedding_gate_down")))
    handles.append(gate.up_proj.register_forward_pre_hook(input_hook("embedding_gate_silu", None)))
    handles.append(gate.up_proj.register_forward_hook(partial(capture_gate_projection, "embedding_gate_up")))
    handles.append(model.model.embed_tokens.register_forward_hook(output_hook("embedding")))
    handles.append(model.model.embed_gated_norm.register_forward_hook(output_hook("post_embed_norm_gate")))
    for layer_index, layer in enumerate(model.model.layers):
        handles.append(layer.input_layernorm.register_forward_pre_hook(input_hook("layer_input", layer_index)))
        handles.append(layer.attn_gated_norm.register_forward_hook(output_hook("attention_input", layer_index)))
        handles.append(layer.self_attn.register_forward_hook(output_hook("attention_output", layer_index)))
        handles.append(
            layer.post_attention_layernorm.register_forward_pre_hook(
                input_hook("post_attention_residual", layer_index)
            )
        )
        handles.append(layer.mlp_gated_norm.register_forward_hook(output_hook("mlp_input", layer_index)))
        handles.append(layer.mlp.register_forward_hook(output_hook("routed_output", layer_index)))
        handles.append(layer.shared_expert.register_forward_hook(output_hook("shared_output", layer_index)))
        handles.append(layer.register_forward_hook(output_hook("layer_output", layer_index)))
        router = layer.mlp.experts.router
        original = router.select_experts

        def capture_routing(*args, _original=original, _layer_index=layer_index, **kwargs):
            weights, indices = _original(*args, **kwargs)
            records.append(
                {
                    "site": "router",
                    "layer_index": _layer_index,
                    "router_logits": kwargs["router_logits"].detach().float().cpu().tolist(),
                    "expert_ids": indices.detach().cpu().tolist(),
                    "expert_weights": weights.detach().float().cpu().tolist(),
                    "weight_dtype": str(weights.dtype),
                }
            )
            return weights, indices

        router.select_experts = capture_routing
        restores.append(partial(setattr, router, "select_experts", original))

    handles.append(model.register_forward_pre_hook(capture_inputs, with_kwargs=True))
    handles.append(model.model.norm.register_forward_hook(capture_norm))
    handles.append(model.model.final_gated_norm.register_forward_hook(capture_gate))
    handles.append(model.logits_processor.register_forward_hook(capture_projection))


def remove_final_state_capture(model) -> list[dict]:
    records, handles, restores = model.prefix_diagnostic_capture
    for handle in handles:
        handle.remove()
    for restore in restores:
        restore()
    del model.prefix_diagnostic_capture
    return records


class PrefixDiagnosticWorkerExtension:
    """Named RPC methods mixed into vLLM's worker for this untimed probe only."""

    def install_prefix_diagnostic_capture(self) -> None:
        install_final_state_capture(cast(WorkerBase, self).get_model())

    def save_prefix_diagnostic_capture(self, directory: str) -> None:
        worker = cast(WorkerBase, self)
        rank = worker.vllm_config.parallel_config.data_parallel_rank
        records = remove_final_state_capture(worker.get_model())
        # DPLBAsyncMPClient broadcasts utility RPCs but returns only rank zero's result.
        # All diagnostic ranks share this host directory; retain every rank before returning.
        output = {"data_parallel_rank": rank, "records": records}
        (Path(directory) / f"rank-{rank}.json").write_text(json.dumps(output) + "\n")


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
    if not engine_args.get("enforce_eager") or engine_args["tensor_parallel_size"] != 1:
        raise ValueError("Final-state capture requires eager execution and TP1")
    if engine_args["data_parallel_size"] != engine_args["data_parallel_size_local"]:
        raise ValueError("Final-state capture requires all DP ranks on the same host")
    extension = "levanter.main.vllm_prefix_diagnostic.PrefixDiagnosticWorkerExtension"
    if engine_args.get("worker_extension_cls") not in (None, "", extension):
        raise ValueError("The prefix diagnostic cannot replace another worker extension")
    engine_args["worker_extension_cls"] = extension
    engine = AsyncLLM.from_engine_args(AsyncEngineArgs(**engine_args))
    try:
        with (
            asyncio.Runner() as runner,
            tempfile.TemporaryDirectory(prefix="vllm-prefix-capture-") as capture_directory,
        ):
            runner.run(engine.collective_rpc("install_prefix_diagnostic_capture"))
            results = runner.run(diagnose(engine, sequences))
            runner.run(engine.collective_rpc("save_prefix_diagnostic_capture", args=(capture_directory,)))
            final_states = [
                json.loads((Path(capture_directory) / f"rank-{rank}.json").read_text())
                for rank in range(engine_args["data_parallel_size"])
            ]
        result = {
            "boundary": "untimed_teacher_forced_single_prefill",
            "final_states": final_states,
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
