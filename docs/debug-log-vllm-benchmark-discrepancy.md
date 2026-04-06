# Debugging log for vLLM benchmark discrepancy

Investigate why the HTTP-based GRPO benchmark (`BVF-060/061`) showed the forked TPU stack as slower, while the in-process RL-path benchmark (`BVF-072/073`) showed it as much faster.

## Initial status

Observed discrepancy:

- HTTP/OpenAI server benchmark on candidate (`BVF-061R`) was slower than baseline.
- In-process RL-path benchmark on candidate (`BVF-073`) was much faster than baseline.

The user asked whether this means `VllmEnvironment` is broken.

## Hypothesis 1

The two benchmarks were not actually measuring the same workload shape.

## Changes to make

Read and compare:

- `.agents/tmp/bvf_grpo_realism.py`
- `.agents/tmp/bvf_grpo_inprocess.py`
- `lib/marin/src/marin/rl/environments/inference_ctx/vllm.py`
- `lib/marin/src/marin/inference/vllm_server.py`

## Results

Confirmed multiple major workload-shape mismatches:

1. HTTP benchmark serialized prompt requests.
   - `bvf_grpo_realism.py` uses `asyncio.Semaphore(args.prompt_concurrency)`.
   - Its default is `--prompt-concurrency 1`.
   - So each mini-batch issued `64` prompt requests one at a time over HTTP, each with `n=16`.
   - The in-process benchmark calls `inference_ctx.batch_completions(...)` once for the whole 64-prompt batch, which lets vLLM batch work across prompts directly.

2. The in-process benchmark changed the sampled dataset slice.
   - HTTP benchmark warmup consumes `1` example.
   - In-process benchmark warmup consumes a full `64`-prompt mini-batch.
   - So the measured examples are offset by `63` prompts between the two harnesses.
   - This means the two runs did not process the same prompt distribution.

3. The in-process benchmark’s warmup batch contained much longer prompts than its measured batches.
   - Extracted from `BVF-073` final report:
     - warmup `prompt_tokens_max=1001`
     - warmup `prompt_tokens_p95=318.35`
   - Measured batch 10 had:
     - `prompt_tokens_max=366`
     - `prompt_tokens_p95=271.8`
   - So the in-process benchmark not only shifted the dataset slice, it moved some long prompts into the unscored warmup window.

4. The HTTP benchmark used the OpenAI server path, which does stricter prompt+completion validation.
   - The earlier HTTP GRPO benchmark produced `400 BadRequestError` for some prompts because `prompt_len + max_tokens > max_model_len`.
   - The in-process path uses `LLM.generate(...)` on token IDs and does not fail through the same OpenAI request-validation path.

## Hypothesis 2

Even beyond transport, the two paths may have been using different internal vLLM execution backends.

## Changes to make

Inspect environment defaults and direct-engine construction:

- `lib/marin/src/marin/inference/vllm_server.py`
- `lib/marin/src/marin/rl/environments/inference_ctx/vllm.py`

## Results

Confirmed a likely backend mismatch:

1. `VllmEnvironment` sets canonical subprocess defaults in `_VLLM_ENV_DEFAULTS`.
   - This includes `("MODEL_IMPL_TYPE", "vllm")`.

2. The in-process RL path does not use `VllmEnvironment`.
   - `vLLMInferenceContext._get_llm_engine(...)` directly constructs `LLM(...)`.
   - The successful `BVF-073` logs showed:
     - `Loading model with MODEL_IMPL_TYPE=auto`
     - `Resolved MODEL_IMPL_TYPE 'auto' to 'flax_nnx'`

Implication:

- The server benchmark and in-process benchmark may have used different internal model implementations (`vllm` vs `flax_nnx`) on TPU.
- That makes the comparison even less apples-to-apples.

## Hypothesis 3

`VllmEnvironment` does not translate enough engine kwargs to CLI flags, so the HTTP benchmark may have been launching the server with weaker settings than the RL path.

## Changes to make

Inspect `_engine_kwargs_to_cli_args(...)` and compare against in-process engine config.

## Results

Confirmed a likely parity gap:

1. `_engine_kwargs_to_cli_args(...)` only forwards:
   - `load_format`
   - `max_model_len`
   - `gpu_memory_utilization`

2. It does **not** forward key RL-path settings such as:
   - `tensor_parallel_size`
   - `enforce_eager`

3. The HTTP GRPO benchmark created:
   - `ModelConfig(..., engine_kwargs={"max_model_len": args.max_model_len})`
   - then passed that into `VllmEnvironment(...)`

4. The in-process RL path explicitly created:
   - `LLM(..., max_model_len=2048, tensor_parallel_size=4, ... , enforce_eager=True)`

Implication:

- The HTTP benchmark almost certainly did not launch the server with the same TP/eagerness config as the RL path.
- This is the strongest evidence that `VllmEnvironment` is not suitable as-is for parity benchmarking against the RL in-process path.
- This is also the most credible “real bug” uncovered by the discrepancy investigation.

## Hypothesis 4

Prompt rendering itself caused the giant gap.

## Changes to make

Compare prompt token counts for the same message list through:

- Marin’s `Llama3Renderer`
- Hugging Face tokenizer `apply_chat_template(..., add_generation_prompt=True)`

## Results

Measured on a sample of real Hendrycks MATH prompts with the same few-shot structure:

- renderer:
  - `min=127`
  - `p50=156`
  - `p95=210.05`
  - `max=845`
- HF chat template:
  - `min=152`
  - `p50=181`
  - `p95=235.05`
  - `max=870`
- HF minus renderer:
  - constant `25` tokens

Implication:

- Prompt rendering overhead exists, but it is small and constant.
- It does **not** explain the large benchmark discrepancy by itself.

## Future Work

- [ ] Patch `VllmEnvironment` CLI arg translation to include `tensor_parallel_size` and `enforce_eager`.
- [ ] Add a parity benchmark harness that uses the same exact sampled examples on both server and in-process paths.
- [ ] Fix the in-process benchmark warmup so it does not consume a full measured mini-batch.
- [ ] Rerun the server benchmark with high client concurrency instead of `prompt_concurrency=1`.
- [ ] Rerun the server benchmark with the same backend selection as the RL path, or make backend selection explicit in both.
- [ ] Decide whether the HTTP/OpenAI path is worth benchmarking for RL at all, since real RL does not use it.
