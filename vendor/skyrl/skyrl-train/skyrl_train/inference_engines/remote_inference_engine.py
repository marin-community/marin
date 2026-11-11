import aiohttp
from skyrl_train.inference_engines.base import (
    InferenceEngineInterface,
    InferenceEngineInput,
    InferenceEngineOutput,
    NamedWeightsUpdateRequest,
)
from skyrl_train.weight_sync.weight_loader import WeightLoader
from typing import List, Optional, Any, Dict
import json
from transformers import PreTrainedTokenizerBase


class RemoteWeightLoader(WeightLoader):
    """Loads weights into remote inference engine via HTTP.

    This loader coordinates weight updates with remote inference servers
    (vLLM or SGLang) via their HTTP APIs.
    """

    def __init__(self, url: str, engine_backend: str) -> None:
        """Initialize the loader.

        Args:
            url: Base URL of the remote inference server.
            engine_backend: Backend type ("vllm" or "sglang").
        """
        self._url = url
        self._engine_backend = engine_backend

    async def init_communicator(
        self,
        master_address: str,
        master_port: int,
        rank_offset: int,
        world_size: int,
        group_name: str,
        backend: str,
        override_existing: bool = False,
    ) -> Dict[str, Any]:
        """Initialize the distributed process group for syncing weights.

        Args:
            master_address: Master address for the process group.
            master_port: Master port for the process group.
            rank_offset: Rank offset for this process.
            world_size: Total world size.
            group_name: Name of the process group.
            backend: Backend to use (e.g., "nccl", "gloo").
            override_existing: Whether to override an existing group.

        Returns:
            Response from the remote server.
        """
        path = "/init_weights_update_group" if self._engine_backend == "sglang" else "/init_weight_update_communicator"
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self._url}{path}",
                json={
                    "master_address": master_address,
                    "master_port": master_port,
                    "rank_offset": rank_offset,
                    "world_size": world_size,
                    "group_name": group_name,
                    "backend": backend,
                    "override_existing": override_existing,
                },
            ) as response:
                return await response.json()

    async def load_weights(self, request: NamedWeightsUpdateRequest) -> Dict[str, Any]:
        """Load weights via HTTP to the remote inference server.

        Remote engines only support broadcast weight updates (no IPC).
        Each request should contain a single weight to update.

        Args:
            request: Weight update request containing names, dtypes, shapes.

        Returns:
            Response from the remote server.
        """
        if self._engine_backend == "vllm":
            weight_update_method = "update_weights"
        elif self._engine_backend == "sglang":
            weight_update_method = "update_weights_from_distributed"
        else:
            raise ValueError(f"Invalid engine backend: {self._engine_backend}")

        async with aiohttp.ClientSession() as session:
            name = request["names"][0]
            dtype = request["dtypes"][0]
            shape = request["shapes"][0]

            resp = await session.post(
                f"{self._url}/{weight_update_method}",
                json={
                    "name": name,
                    "dtype": dtype,
                    "shape": shape,
                },
            )
            return await resp.json()

    async def destroy_group(self) -> Dict[str, Any]:
        """Destroy the weights update group.

        Returns:
            Response from the remote server.
        """
        async with aiohttp.ClientSession() as session:
            resp = await session.post(f"{self._url}/destroy_weights_update_group")
            return await resp.json()


class RemoteInferenceEngine(InferenceEngineInterface):
    """
    Lightweight client to call into an OpenAI-compatible server over HTTP with a customizable backend.
    """

    def __init__(
        self,
        url: str,
        model_name: str,
        engine_backend: str,
        tokenizer: PreTrainedTokenizerBase,
        tp_size: Optional[int] = None,
        pp_size: Optional[int] = None,
        dp_size: Optional[int] = None,
        ep_size: Optional[int] = None,
        dcp_size: Optional[int] = None,
    ):
        """Initialize the InferenceEngine.

        Note on `dcp_size` (decode context parallel): for remote engines SkyRL does NOT
        launch the inference server, so DCP must be enabled on the EXTERNAL `vllm serve`
        command via `-dcp <n>`. We carry `dcp_size` here as client metadata only (mirroring
        `tp_size`) for geometry validation / GPU-accounting consistency; it is never sent to
        or used to spawn the server. DCP reuses the TP group's GPUs (adds no GPUs — G4).
        """
        self.url = f"http://{url}"
        self.model_name = model_name
        self.engine_backend = engine_backend
        self._tp_size = tp_size
        self._pp_size = pp_size
        self._dp_size = dp_size
        self._ep_size = ep_size
        self._dcp_size = dcp_size
        self.tokenizer = tokenizer

        # Create weight loader for coordinating weight updates
        self._weight_loader = RemoteWeightLoader(self.url, engine_backend)

    def tp_size(self) -> int:
        return self._tp_size

    def pp_size(self) -> int:
        return self._pp_size

    def dp_size(self) -> int:
        return self._dp_size

    def ep_size(self) -> int:
        return self._ep_size

    def dcp_size(self) -> int:
        return self._dcp_size

    async def generate(self, input_batch: InferenceEngineInput) -> InferenceEngineOutput:
        # 1. Prepare inputs
        prompts = input_batch.get("prompts")
        prompt_token_ids: Optional[List[List[int]]] = input_batch.get("prompt_token_ids")
        request_sampling_params = input_batch.get("sampling_params")

        assert prompts is None and prompt_token_ids is not None, (
            "RemoteInferenceEngine only accepts `prompt_token_ids`, not `prompts`."
        )

        sampling_params = request_sampling_params if request_sampling_params is not None else {}
        if "n" in sampling_params and sampling_params["n"] > 1:
            raise ValueError(
                "n is not supported yet for remote inference engines. "
                "You can set `config.generator.n_samples_per_prompt` instead."
            )

        # 2. Send a batched request to the server
        response = None
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=None)) as session:
            headers = {"Content-Type": "application/json"}
            payload = {}
            request_url = ""
            if self.engine_backend == "vllm":
                # vLLM does not support /generate, use /completions instead. It supports batch generation.
                payload = sampling_params.copy()
                payload["model"] = self.model_name
                payload["prompt"] = prompt_token_ids
                payload["return_token_ids"] = True
                payload.setdefault("logprobs", 0)
                request_url = f"{self.url}/v1/completions"
            elif self.engine_backend == "sglang":
                # SGLang supports /generate, works exactly like its Python `async_generate()` method
                # and can do batch generation.
                payload = {
                    "input_ids": prompt_token_ids,
                    "sampling_params": sampling_params,
                }
                request_url = f"{self.url}/generate"
            else:
                raise ValueError(f"Invalid engine backend: {self.engine_backend}")
            async with session.post(request_url, json=payload, headers=headers) as resp:
                resp.raise_for_status()
                response = await resp.json()

        # 3. Parse outputs
        outputs = []
        output_ids = []
        finish_reasons = []
        response_logprobs = None

        if self.engine_backend == "vllm":
            if len(response["choices"]) != len(prompt_token_ids):
                raise ValueError("Remote completion count does not match the prompt batch")
            if payload["logprobs"] is not None:
                response_logprobs = []
            for i, choice in enumerate(response.get("choices", [])):
                # Since n=1, index i represents the output for `prompt[i]`
                assert choice["index"] == i, "Expect the choices to be ordered by index."
                text = choice["text"]
                outputs.append(text)
                finish_reasons.append(choice["finish_reason"])
                token_ids = choice["token_ids"]
                output_ids.append(token_ids)
                if response_logprobs is not None:
                    logprobs = choice["logprobs"]["token_logprobs"]
                    if len(logprobs) != len(token_ids) or any(value is None for value in logprobs):
                        raise ValueError("Remote sampled-token logprobs must align with the returned token IDs")
                    response_logprobs.append(logprobs)
        elif self.engine_backend == "sglang":
            # since prompt_token_ids is a list of lists, response is a list of dicts
            for output in response:
                cur_output_ids = output["output_ids"]
                output_ids.append(cur_output_ids)
                # SGLang only returns tokens not text when skip_tokenizer_init is True, so
                # we manually decode it.
                outputs.append(self.tokenizer.decode(cur_output_ids, skip_special_tokens=True))
                finish_reasons.append(output["meta_info"]["finish_reason"]["type"])
        else:
            raise ValueError(f"Invalid engine backend: {self.engine_backend}")

        return InferenceEngineOutput(
            responses=outputs, stop_reasons=finish_reasons, response_ids=output_ids, response_logprobs=response_logprobs
        )

    async def chat_completion(self, request_payload: Dict[str, Any]) -> Dict[str, Any]:
        body = request_payload.get("json", {})
        # NOTE(Charlie): cannot reuse payload["headers"] since we are posting a new request.
        # Otherwise will lead to json decode error.
        headers = {"Content-Type": "application/json"}
        response = None
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=None)) as session:
            request_url = f"{self.url}/v1/chat/completions"
            async with session.post(request_url, json=body, headers=headers) as resp:
                response = await resp.json()

        return response

    async def completion(self, request_payload: Dict[str, Any]) -> Dict[str, Any]:
        body = request_payload.get("json", {})
        headers = {"Content-Type": "application/json"}
        response = None
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=None)) as session:
            request_url = f"{self.url}/v1/completions"
            async with session.post(request_url, json=body, headers=headers) as resp:
                response = await resp.json()

        return response

    async def wake_up(self, *args: Any, **kwargs: Any):
        async with aiohttp.ClientSession() as session:
            resp = await session.post(f"{self.url}/wake_up", json={"tags": kwargs.get("tags", 1)})
            return await resp.json()

    async def sleep(self, *args: Any, **kwargs: Any):
        async with aiohttp.ClientSession() as session:
            # TODO(Charlie): this is vLLM's API, not SGLang (which uses tags). Fix when need to
            # support sleeping with remote engines.
            resp = await session.post(f"{self.url}/sleep", json={"level": kwargs.get("level", 1)})
            return await resp.json()

    async def init_weight_update_communicator(
        self, master_addr, master_port, rank_offset, world_size, group_name, backend, override_existing: bool = False
    ):
        """Initialize the distributed process group for syncing weights."""
        return await self._weight_loader.init_communicator(
            master_addr, master_port, rank_offset, world_size, group_name, backend, override_existing
        )

    async def update_named_weights(self, request: NamedWeightsUpdateRequest):
        if "names" not in request:
            raise ValueError(f"Expected update weight request with 'names' entry, got keys: {request.keys()}")

        assert len(request["names"]) == 1, (
            f"Remote inference engines support only requests with a single named weight at a time , got request with {len(request['names'])} entries"
        )

        if request.get("extras") and "ipc_handles" in request["extras"][0]:
            raise ValueError(
                "Remote inference engines do not support CUDA IPC weight updates. Only local engines support IPC."
            )

        return await self._weight_loader.load_weights(request)

    # TODO(tgriggs): Come up with a (more) elegant way to handle text or json responses, and test it and handle errors.
    async def reset_prefix_cache(self):
        if self.engine_backend == "vllm":
            reset_prefix_cache_method = "reset_prefix_cache"
        elif self.engine_backend == "sglang":
            reset_prefix_cache_method = "flush_cache"
        else:
            raise ValueError(f"Invalid engine backend: {self.engine_backend}")

        async with aiohttp.ClientSession() as session:
            resp = await session.post(f"{self.url}/{reset_prefix_cache_method}")
            text = await resp.text()

        # First try to parse it as JSON
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            # If invalid JSON, return raw text plus status
            return {
                "status": resp.status,
                "body": text,
            }

    async def teardown(self):
        await self._weight_loader.destroy_group()

    async def pause_generation(self) -> None:
        raise NotImplementedError("Pausing generation is not supported for remote inference engines.")

    async def resume_generation(self, policy_step: int) -> None:
        raise NotImplementedError("Resuming generation is not supported for remote inference engines.")


def create_remote_inference_engines(
    urls: List[str],
    model_name: str,
    engine_backend: str,
    tokenizer: PreTrainedTokenizerBase,
    tensor_parallel_size: Optional[int] = None,
    pipeline_parallel_size: Optional[int] = None,
    data_parallel_size: Optional[int] = None,
    expert_parallel_size: Optional[int] = None,
    decode_context_parallel_size: Optional[int] = None,
):
    # `decode_context_parallel_size` is metadata only for remote engines: SkyRL does not
    # launch the server, so the operator must pass `-dcp <n>` on the external `vllm serve`
    # command. DCP reuses the TP GPUs (adds no GPUs — G4); it never changes the client's
    # GPU accounting or the request payload.
    return [
        RemoteInferenceEngine(
            url=url,
            model_name=model_name,
            tokenizer=tokenizer,
            engine_backend=engine_backend,
            tp_size=tensor_parallel_size,
            pp_size=pipeline_parallel_size,
            dp_size=data_parallel_size,
            ep_size=expert_parallel_size,
            dcp_size=decode_context_parallel_size,
        )
        for url in urls
    ]
