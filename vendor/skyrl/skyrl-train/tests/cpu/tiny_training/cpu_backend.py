"""CPU policy worker, strategy, and inference engine for running the production trainer locally.

These classes stand in for the Megatron workers and vLLM engines so the real entrypoint, trainer,
trajectory runners, and weight-sync protocol can run end to end on a tiny Hugging Face model.
"""

import asyncio
import hashlib
import itertools
import os
import time
from typing import Any

import torch
import torch.distributed as dist
from skyrl_train.distributed.dispatch import MeshRank
from skyrl_train.distributed.strategy import DistributedStrategy
from skyrl_train.inference_engines.base import (
    CHOICE_POLICY_STEPS_KEY,
    SESSION_ID_HEADER,
    InferenceEngineInput,
    InferenceEngineInterface,
    InferenceEngineOutput,
    NamedWeightsUpdateRequest,
)
from skyrl_train.inference_engines.chat_continuation import EXACT_PROMPT_TOKEN_IDS_KEY
from skyrl_train.inference_engines.policy_steps import PolicyStepStamps
from skyrl_train.io import io
from skyrl_train.utils.torch_utils import chunked_entropy_from_logits, logprobs_from_logits
from skyrl_train.workers.worker import PolicyWorkerBase
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer

CHECKPOINT_FILE_TEMPLATE = "rank_{rank}.pt"
# Sampling controls the CPU engine implements; any other non-neutral control fails fast.
NEUTRAL_SAMPLING_PARAMS = {"top_p": 1.0, "top_k": -1, "min_p": 0.0, "repetition_penalty": 1.0}


class CausalLMPolicy(nn.Module):
    """Adapt a Hugging Face causal LM to the policy worker's log-probability forward contract."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    @torch.autocast(device_type="cpu", enabled=False)
    def forward(
        self,
        sequences: torch.Tensor,
        num_actions: int,
        attention_mask: torch.Tensor,
        temperature: float = 1.0,
        return_output: bool = False,
        compute_entropy: bool = False,
        entropy_requires_grad: bool = True,
        rollout_routed_experts: torch.Tensor | None = None,
    ):
        if rollout_routed_experts is not None:
            raise ValueError("router replay requires a mixture-of-experts policy")
        # FP32 keeps the CPU reference's backward arithmetic stable across batch splits.
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)
        # Callers read only the trailing response-aligned positions, so skip the vocabulary
        # projection for the prompt; it dominates the cost of a tiny model with a full vocabulary.
        kept = num_actions + 1
        logits = self.model(
            sequences, attention_mask=attention_mask, position_ids=position_ids, logits_to_keep=kept
        ).logits
        logits = logits.float() / temperature
        labels = torch.roll(sequences, shifts=-1, dims=1)[:, -kept:]
        action_log_probs = logprobs_from_logits(logits, labels)[:, :-1]
        if not return_output:
            return action_log_probs
        output = {"logits": logits}
        if compute_entropy:
            output["entropy"] = chunked_entropy_from_logits(
                logits, requires_grad=entropy_requires_grad, attention_mask=attention_mask[:, -kept:]
            )
        return action_log_probs, output


class CPUStrategy(DistributedStrategy):
    """Plain PyTorch optimization over a gloo process group, averaging gradients across ranks."""

    def __init__(self, max_grad_norm: float):
        self.max_grad_norm = max_grad_norm
        self.world_size = dist.get_world_size()

    def setup_distributed(self):
        """The worker creates the gloo group before the strategy exists."""

    def collective_device(self) -> torch.device:
        return torch.device("cpu")

    def backward(self, loss: torch.Tensor, model, optimizer, **kwargs):
        loss.backward()

    def optimizer_step(self, optimizer, model, scheduler, name="model", **kwargs):
        if self.world_size > 1:
            for parameter in model.parameters():
                if parameter.grad is not None:
                    dist.all_reduce(parameter.grad)
                    parameter.grad /= self.world_size
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), self.max_grad_norm)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        return grad_norm

    def save_checkpoint(
        self, model, ckpt_dir, node_local_rank, optimizer, scheduler, client_state=None, tokenizer=None
    ):
        io.makedirs(ckpt_dir, exist_ok=True)
        states = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "client_state": client_state or {},
            "rng": self.get_rng_state(),
        }
        torch.save(states, os.path.join(ckpt_dir, CHECKPOINT_FILE_TEMPLATE.format(rank=dist.get_rank())))
        dist.barrier()

    def load_checkpoint(
        self, model, ckpt_dir, optimizer=None, scheduler=None, load_module_strict=True, load_training_state=True
    ):
        path = os.path.join(ckpt_dir, CHECKPOINT_FILE_TEMPLATE.format(rank=dist.get_rank()))
        states = torch.load(path, weights_only=False)
        model.load_state_dict(states["model"], strict=load_module_strict)
        if load_training_state:
            optimizer.load_state_dict(states["optimizer"])
            scheduler.load_state_dict(states["scheduler"])
            self.load_rng_state(states["rng"])
        return ckpt_dir, states

    def save_hf_model(self, model, output_dir: str, tokenizer=None, **kwargs):
        if self.is_rank_0():
            model.model.save_pretrained(output_dir)
            if tokenizer is not None:
                tokenizer.save_pretrained(output_dir)
        dist.barrier()


class CPUPolicyWorker(PolicyWorkerBase):
    """Policy worker that trains a Hugging Face model on CPU and ships weights through Ray."""

    @property
    def device(self) -> torch.device:
        return torch.device("cpu")

    def init_worker_process_group(self, master_addr: str, master_port: int):
        os.environ["MASTER_ADDR"] = master_addr
        os.environ["MASTER_PORT"] = str(master_port)
        dist.init_process_group("gloo", rank=self._rank, world_size=self._world_size)
        self.mesh_rank = MeshRank(
            dp=self._rank,
            sp=0,
            tp=0,
            pp=0,
            world_size=self._world_size,
            dp_size=self._world_size,
            pp_size=1,
        )

    def init_model(self, model_path, num_training_steps: int | None = None):
        optimizer_config = self.cfg.trainer.policy.optimizer_config
        self.model = CausalLMPolicy(AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.float32))
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=optimizer_config.lr,
            betas=tuple(optimizer_config.adam_betas),
            weight_decay=optimizer_config.weight_decay,
        )
        self.scheduler = torch.optim.lr_scheduler.ConstantLR(self.optimizer, factor=1.0, total_iters=0)
        self.strategy = CPUStrategy(max_grad_norm=optimizer_config.max_grad_norm)
        self._normalize_mini_batch_size()

    def _set_pad_token_id(self, pad_token_id):
        """The Hugging Face model reads its pad token from the saved config."""

    def offload_to_cpu(self, pin_memory=True, non_blocking=True, **kwargs):
        """State already lives on the CPU."""

    def backload_to_gpu(self, non_blocking=True, **kwargs):
        """State already lives on the CPU."""

    async def init_weight_sync_state(self, inference_engine_client):
        """Weights travel through the Ray object store, which needs no communicator."""

    async def barrier_all(self) -> None:
        """Drain the actor event loop and synchronize the CPU worker group."""
        await asyncio.to_thread(dist.barrier)

    async def broadcast_to_inference_engines(self, inference_engine_client):
        if dist.get_rank() == 0:
            state = {name: tensor.detach().clone() for name, tensor in self.model.model.state_dict().items()}
            await inference_engine_client.update_named_weights(
                {
                    "names": list(state),
                    "dtypes": [str(tensor.dtype) for tensor in state.values()],
                    "shapes": [list(tensor.shape) for tensor in state.values()],
                    "extras": [{"tensor": tensor} for tensor in state.values()],
                }
            )
        dist.barrier()


class CPURefWorker(CPUPolicyWorker):
    """Reference worker: the CPU policy forward without optimizer state."""

    def init_model(self, model_path, num_training_steps: int | None = None):
        self.model = CausalLMPolicy(AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.float32).eval())
        self.optimizer = None
        self.scheduler = None

    def training_step(self, *args, **kwargs):
        raise RuntimeError("the reference model never trains")

    def save_checkpoint(self, ckpt_dir, tokenizer=None):
        raise RuntimeError("the reference model checkpoints with the trainer state")


class CPUInferenceEngine(InferenceEngineInterface):
    """Sample from a Hugging Face causal LM with vLLM's pause, abort, and weight-update semantics.

    Pausing preserves in-flight requests at their next token, matching the production keep policy;
    requests that arrive while paused wait for the resume. Like the vLLM engine, each token carries the
    policy step installed when it was sampled.
    """

    def __init__(self, model_path: str, seed: int):
        self.seed = seed
        self.request_count = 0
        self.model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.float32).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.eos_token_id = self.model.config.eos_token_id
        self.max_model_len = self.model.config.max_position_embeddings
        torch.set_num_threads(1)
        self._pending = []
        self._wake = asyncio.Event()
        self._decoder = None
        self._resumed = asyncio.Event()
        self._resumed.set()
        self._policy_steps = PolicyStepStamps()
        self._request_ids = itertools.count()
        self._live_requests: set[str] = set()

    async def generate(self, input_batch: InferenceEngineInput) -> InferenceEngineOutput:
        await self._resumed.wait()
        sampling_params = input_batch["sampling_params"]
        _check_sampling_params(sampling_params)
        prompts = input_batch["prompt_token_ids"]
        session_ids = input_batch.get("session_ids") or [None] * len(prompts)
        results = await asyncio.gather(
            *(
                self._generate_one(prompt_ids, sampling_params, session_id)
                for prompt_ids, session_id in zip(prompts, session_ids, strict=True)
            )
        )
        response_ids, response_logprobs, stop_reasons, policy_steps = map(list, zip(*results, strict=True))
        return InferenceEngineOutput(
            responses=[self.tokenizer.decode(ids, skip_special_tokens=True) for ids in response_ids],
            response_ids=response_ids,
            stop_reasons=stop_reasons,
            response_logprobs=response_logprobs if sampling_params.get("logprobs") is not None else None,
            prompt_logprobs=None,
            response_policy_steps=policy_steps,
        )

    async def _generate_one(self, prompt_ids, sampling_params, session_id):
        if self._decoder is None:
            self._decoder = asyncio.get_running_loop().create_task(self._decode_loop())
        future = asyncio.get_running_loop().create_future()
        if session_id is None:
            session_id = self.request_count
            self.request_count += 1
        seed = sampling_params.get("seed")
        if seed is None:
            seed = self.seed
        identity = repr((seed, session_id, list(prompt_ids))).encode()
        generator = torch.Generator().manual_seed(int.from_bytes(hashlib.sha256(identity).digest()[:8], "big"))
        request_id = str(next(self._request_ids))
        self._live_requests.add(request_id)
        self._pending.append((list(prompt_ids), dict(sampling_params), future, generator, request_id))
        self._wake.set()
        try:
            response_ids, logprobs, stop_reason = await future
        finally:
            self._live_requests.discard(request_id)
        return response_ids, logprobs, stop_reason, self._policy_steps.finish(request_id, len(response_ids))

    async def _decode_loop(self):
        while True:
            await self._wake.wait()
            # Let the other concurrent requests of this wave enqueue before decoding.
            for _ in range(3):
                await asyncio.sleep(0)
            await self._resumed.wait()
            self._wake.clear()
            batch, self._pending = self._pending, []
            if not batch:
                continue
            groups = {}
            for item in batch:
                sampling_params = item[1]
                key = (float(sampling_params["temperature"]), int(sampling_params.get("min_tokens", 1)))
                groups.setdefault(key, []).append(item)
            for (temperature, min_tokens), items in groups.items():
                try:
                    results = await self._decode_batch(items, temperature, min_tokens)
                    for (_, _, future, _, _), result in zip(items, results):
                        if not future.done():
                            future.set_result(result)
                except Exception as error:  # noqa: BLE001
                    for _, _, future, _, _ in items:
                        if not future.done():
                            future.set_exception(error)

    async def _decode_batch(self, items, temperature, min_tokens):
        batch_size = len(items)
        max_response_tokens = [int(sampling_params["max_tokens"]) for _, sampling_params, _, _, _ in items]
        prompt_lengths = [len(prompt_ids) for prompt_ids, _, _, _, _ in items]
        request_ids = [request_id for _, _, _, _, request_id in items]
        prompt_width = max(prompt_lengths)
        pad_token_id = self.model.config.pad_token_id or 0
        input_ids = torch.full((batch_size, prompt_width), pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros((batch_size, prompt_width), dtype=torch.long)
        for index, (prompt_ids, _, _, _, _) in enumerate(items):
            input_ids[index, prompt_width - len(prompt_ids) :] = torch.tensor(prompt_ids)
            attention_mask[index, prompt_width - len(prompt_ids) :] = 1
        response_ids = [[] for _ in range(batch_size)]
        response_logprobs = [[] for _ in range(batch_size)]
        stop_reasons = ["length"] * batch_size
        live_sequences = torch.ones(batch_size, dtype=torch.bool)
        past_key_values = None
        input_tokens = input_ids
        position_ids = (attention_mask.cumsum(-1) - 1).clamp(min=0)
        for token_index in range(max(max_response_tokens)):
            await self._resumed.wait()
            sampled_at = time.monotonic()
            with torch.no_grad(), torch.autocast(device_type="cpu", dtype=torch.bfloat16, enabled=True):
                output = self.model(
                    input_tokens,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    use_cache=True,
                )
            past_key_values = output.past_key_values
            logits = output.logits[:, -1].float()
            if token_index < min_tokens:
                logits[:, self.eos_token_id] = float("-inf")
            if temperature == 0.0:
                log_probs = torch.log_softmax(logits, -1)
                token_ids = logits.argmax(-1)
            else:
                log_probs = torch.log_softmax(logits / temperature, -1)
                token_ids = torch.full((batch_size,), pad_token_id, dtype=torch.long)
                for index, (_, _, _, generator, _) in enumerate(items):
                    if live_sequences[index]:
                        token_ids[index] = torch.multinomial(log_probs[index].exp(), 1, generator=generator)[0]
            token_logprobs = log_probs.gather(-1, token_ids[:, None]).squeeze(-1)
            for index in range(batch_size):
                if not live_sequences[index]:
                    continue
                response_ids[index].append(int(token_ids[index]))
                response_logprobs[index].append(float(token_logprobs[index]))
                self._policy_steps.observe(request_ids[index], 1, sampled_at)
                if int(token_ids[index]) == self.eos_token_id:
                    stop_reasons[index] = "stop"
                    live_sequences[index] = False
                elif len(response_ids[index]) >= max_response_tokens[index]:
                    live_sequences[index] = False
            if not live_sequences.any():
                break
            input_tokens = torch.where(live_sequences, token_ids, torch.full_like(token_ids, pad_token_id))[:, None]
            attention_mask = torch.cat([attention_mask, live_sequences.long()[:, None]], 1)
            position_ids = position_ids[:, -1:] + 1
            await asyncio.sleep(0)
        return [(response_ids[index], response_logprobs[index], stop_reasons[index]) for index in range(batch_size)]

    async def pause_generation(self) -> None:
        self._resumed.clear()

    async def resume_generation(self, policy_step: int) -> None:
        self._policy_steps.install(policy_step, time.monotonic())
        self._policy_steps.forget_unclaimed(self._live_requests)
        self._resumed.set()

    async def update_named_weights(self, request: NamedWeightsUpdateRequest):
        state = self.model.state_dict()
        with torch.no_grad():
            for name, extra in zip(request["names"], request["extras"], strict=True):
                state[name].copy_(extra["tensor"])

    async def chat_completion(self, request_payload: dict[str, Any]) -> dict[str, Any]:
        body = request_payload["json"]
        prompt = body.get(EXACT_PROMPT_TOKEN_IDS_KEY)
        if prompt is None:
            prompt = (await self.tokenize({"json": {**body, "add_generation_prompt": True}}))["tokens"]
        sampling = {
            "temperature": body.get("temperature", 1.0),
            "max_tokens": body["max_completion_tokens"],
            "logprobs": body.get("logprobs"),
            **{key: body[key] for key in (*NEUTRAL_SAMPLING_PARAMS, "min_tokens", "stop") if key in body},
        }
        result = await self.generate(
            InferenceEngineInput(
                prompt_token_ids=[prompt],
                sampling_params=sampling,
                session_ids=[request_payload.get("headers", {}).get(SESSION_ID_HEADER)],
            )
        )
        tokens = result["response_ids"][0]
        logprobs = result["response_logprobs"]
        return {
            "id": request_payload.get("headers", {}).get("x-request-id", "cpu-completion"),
            "object": "chat.completion",
            "model": body["model"],
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": result["responses"][0]},
                    "token_ids": tokens,
                    CHOICE_POLICY_STEPS_KEY: result["response_policy_steps"][0],
                    "finish_reason": result["stop_reasons"][0],
                    "logprobs": None
                    if logprobs is None
                    else {
                        "content": [
                            {"token": self.tokenizer.decode([token]), "logprob": score, "top_logprobs": []}
                            for token, score in zip(tokens, logprobs[0], strict=True)
                        ]
                    },
                }
            ],
            "usage": {
                "prompt_tokens": len(prompt),
                "completion_tokens": len(tokens),
                "total_tokens": len(prompt) + len(tokens),
            },
        }

    async def tokenize(self, request_payload: dict[str, Any]) -> dict[str, Any]:
        body = request_payload["json"]
        encoded = self.tokenizer.apply_chat_template(
            body["messages"],
            tokenize=True,
            return_dict=False,
            add_generation_prompt=body.get("add_generation_prompt", False),
        )
        return {"tokens": encoded}

    async def completion(self, request_payload: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError("the CPU engine serves token-in-token-out generation only")

    async def wake_up(self, *args: Any, **kwargs: Any):
        """The CPU engine never releases its memory."""

    async def sleep(self, *args: Any, **kwargs: Any):
        """The CPU engine never releases its memory."""

    async def init_weight_update_communicator(
        self, master_addr, master_port, rank_offset, world_size, group_name, backend, override_existing: bool = False
    ):
        """Weights arrive through the Ray object store."""

    async def begin_weight_reload(self):
        """Weights load directly into the Hugging Face parameters."""

    async def finish_weight_reload(self):
        """Weights load directly into the Hugging Face parameters."""

    async def reset_prefix_cache(self):
        """The CPU engine keeps no prefix cache across requests."""

    async def teardown(self):
        if self._decoder is not None:
            self._decoder.cancel()
            await asyncio.gather(self._decoder, return_exceptions=True)

    def tp_size(self) -> int:
        return 1

    def pp_size(self) -> int:
        return 1

    def dp_size(self) -> int:
        return 1


def _check_sampling_params(sampling_params: dict[str, Any]) -> None:
    for key, neutral in NEUTRAL_SAMPLING_PARAMS.items():
        if sampling_params.get(key, neutral) != neutral:
            raise ValueError(f"the CPU engine does not implement {key}={sampling_params[key]}")
    if sampling_params.get("stop"):
        raise ValueError("the CPU engine does not implement stop strings")
