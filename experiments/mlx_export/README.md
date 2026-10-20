# MLX builds of Grug MoE models

This directory turns a Grug MoE Hugging Face export (`model_type: grug_moe`, artifact schema 1) into
8-bit and 4-bit MLX builds that run on Apple Silicon under `mlx_lm.server`, with thinking and tool
calls. It also checks each build against the Marin vLLM fork before it is published.

**Most of this only runs on a Mac.** MLX has no Linux wheels, so this directory is a separate uv
project (`pyproject.toml` here, excluded from the root workspace). Only `golden_vllm.py` runs from the
root Marin environment, because it submits a GPU job to Iris. On Linux, the root test run skips
`test_grug_moe_mlx.py`.

Dense Marin models do not need any of this: stock `mlx_lm.convert` handles Llama-style exports.

## Files

| File | Runs on | Purpose |
|---|---|---|
| `grug_moe_mlx.py` | Mac | The architecture `mlx-lm` loads through `model_file`. It is shipped inside every MLX repo. |
| `convert.py` | Mac | Download a pinned revision, quantize with a recipe, enable tool-call parsing, check the build loads. |
| `prompts.py` | Mac | Build the parity prompts from the source tokenizer and chat template. |
| `golden_vllm.py` | root env | Serve the source in bf16 on H100s through Iris and record top-25 logprobs. |
| `dump_logprobs.py` | Mac | Score an MLX build on the same prompts, teacher-forced on vLLM's continuation. |
| `compare_parity.py` | Mac | Gate the build and write a Markdown report. |
| `smoke_server.sh` | Mac | Check `mlx_lm.server` end to end: thinking, tools, streaming, a 20k-token prompt. |
| `model_card.md` | | Card template for the published MLX repo. |
| `test_grug_moe_mlx.py` | Mac | Cache, mask, routing and XSA tests on a tiny random config. |

## Recipes

| Recipe | Routed experts | Everything else | Snowball 67B-A2B size |
|---|---|---|---|
| `8bit` | 8-bit | 8-bit | 66 GB |
| `mixed4` | 4-bit | 8-bit | 36 GB |

Publish `mixed4` as the `-MLX-4bit` repo. Routed experts hold almost all the parameters, so keeping
attention, the shared expert, embeddings and `lm_head` at 8 bits costs about 0.5 GB. A uniform 4-bit
build of Snowball failed the short-case gate: p95 probability error 0.193 against a 0.15 limit.

## Build and qualify a model

Run the graph from the repository root on a Mac. `MARIN_PREFIX` is the local artifact directory;
allow about 2.5 times the bf16 model size in free space. The command prints the graph without
running it. Add `--run` to download the source, submit the GPU reference job, convert the model,
and gate the result.

```bash
export MARIN_PREFIX="$HOME/mlx-work/artifacts"
uv run python -m experiments.mlx_export.pipeline \
  --repo open-athena/Marin-Snowball-67B-A2B-v0.1 \
  --revision e4aa78c7c9729dea7238c2e7d1ac7fa57492e0a6 \
  --recipe mixed4 --gpus 8 --version 2026.10.20 \
  --remote-prefix s3://marin-us-east-02a/marin/evals/mlx-export
```

The graph binds prompts, the bf16 golden reference, the converted model, the MLX logprob dump,
and the parity report. The golden stage waits for the Iris job to succeed before downloading its
small JSON output. The remote prefix must be in the GPU cluster's bucket; private models need
`HF_TOKEN`, and the download needs CoreWeave S3 credentials. Size `--gpus` to fit the bf16 weights
on one node. Conversion and scoring use this directory's frozen Mac environment.

Use a new `--version` after changing the code, prompt fixture, dependency lock, or vLLM release.
The artifact fingerprint records these inputs. Conversion downloads into a fresh temporary
source directory and refuses to overwrite an existing build. After an interrupted conversion,
remove that failed model artifact directory before retrying.

Each model carries `mlx-build.json`, which records the source repository, full commit SHA,
recipe, and file hashes, excluding the root `README.md` and `LICENSE` added at publication.
Scoring checks this manifest; comparison rejects a dump from another
build, recipe, source, or golden reference. The report includes the build and golden hashes.
These checks establish which inputs were compared; they do not replace the numerical parity gates.

Run the local tests from the repository root:

```bash
uv run --project experiments/mlx_export --frozen python -m pytest \
  -c experiments/mlx_export/pyproject.toml --noconftest \
  experiments/mlx_export/test_grug_moe_mlx.py experiments/mlx_export/test_provenance.py
```

After parity passes, run `experiments/mlx_export/smoke_server.sh <model-directory> <work-directory>`.
Fill in `model_card.md` and use the `upload-hf-model` skill to publish the build with its report,
license, and research-artifact warning. Name the repositories `<source>-MLX-8bit` and
`<source>-MLX-4bit`. Add publication documents after qualification; changing model files requires
rebuilding and rerunning parity.

## Parity gates

`compare_parity.py` gates two groups separately. Pooled, the long case's thousands of positions would
hide errors on the short chat and tool cases.

| Recipe | top-1 match on decisive positions | p95 probability error |
|---|---|---|
| `8bit` | >= 0.99 | <= 0.075 |
| `mixed4` | >= 0.97 | <= 0.15 |

- **Short:** every position of the prompts no longer than the sliding window.
- **Long:** positions at or past the sliding window in the longer prompts.

A position is decisive when vLLM's top token leads the runner-up by at least 0.2 in probability.

## When a new model fails

- `ValueError: ... this mlx port does not implement these GrugMoE features` at load time: the export
  uses a Grug feature `grug_moe_mlx.py` doesn't have (short conv, PKO, latent experts, split KV heads,
  more than one shared expert, tied embeddings). Port it, add a test, and rerun parity.
- `convert.py` asserts on quantization overrides: module names in `grug_moe_mlx.py` changed, and the
  `.mlp.experts.` match in `quant_params` needs updating.
- Thinking missing from `reasoning`: `mlx-lm` is not 0.32.2. The thinking-marker patch in
  `grug_moe_mlx.py` replaces `mlx_lm.tokenizer_utils._infer_thinking`. Re-check it before bumping the
  pin.

## Running a build

```bash
uvx --from mlx-lm==0.32.2 mlx_lm.server --model <mlx repo or dir> --trust-remote-code --max-tokens 32768
```

Requests must use `"model": "default_model"`. Without `--max-tokens` the server stops at 512 tokens,
mid-thought. pi needs `compat.chatTemplateKwargs: {"enable_thinking": {"$var": "thinking.enabled"}}`
in `models.json` for its thinking toggle to reach the template.
