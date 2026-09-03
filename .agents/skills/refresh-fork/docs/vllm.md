# The vllm forks: group, pins, release pipeline, e2e

Fork-specific guidance for the `vllm` (TPU), `tpu-inference`, and `vllm-gpu` pins.
The generic workflow lives in `../SKILL.md`; this file holds what is particular to
these forks.

## The vllm/tpu-inference group

Refresh the `vllm`/`tpu-inference` group as one unit and one PR; `vllm-gpu` tracks
the fork's independent `main` branch. The TPU launcher installs both grouped pins,
and vLLM's TPU base derives from the tpu-inference release; splitting them can
produce an unvalidated stack.

On the vllm fork the GPU and TPU pins promote onto their own stable branches
(`main` and `tpu`) independently.

## Base selection detail

For `base_select = derived` (`vllm`): read the SHA at `derived_from`
(`tpu-inference:.buildkite/vllm_lkg.version`) from the selected `tpu-inference`
release. That exact SHA is the base; verify it resolves in the fork's `upstream`.
Inspect its TPU build metadata (`requirements/tpu.txt`, `pyproject.toml`,
`setup.py`) for dependency implications.

## Pinning

- `pin = descriptor:<path>#<section>` (`vllm`, `tpu-inference`): push `<branch>-next` to
  the fork. Set the section's `commit` to its tip and `upstream_base` to the selected
  base in `vllm/tpu.toml`. This stack resolves entirely inside the `uvx` env from
  the two forks, so there are no `uv.lock` changes and `jax`/`jaxlib`/`libtpu`/`torch`
  come from the forks' own dependencies — do not touch `marin-core`, `marin-levanter`,
  or `marin-fray`.
- `pin = release:<path>` (`vllm-gpu`): the pin is a prebuilt wheel, so the refresh builds
  and promotes one through the fork's own release pipeline, then re-pins from the promoted
  manifest. Dispatching fork workflows needs `actions:write` on `marin-community/vllm`,
  which the fork-ferry profile grants.
  1. Push `main-next` to the fork.
  2. Build the candidate on that ref:
     `gh workflow run marin-gpu-candidate.yaml --repo marin-community/vllm --ref main-next`.
     The workflow otherwise builds on `push: main`; dispatching on `main-next` compiles the
     staged tip into both arches under a `marin-vllm-gpu-candidate-<sha>` prerelease.
  3. Once the candidate prerelease exists, promote it:
     `gh workflow run marin-gpu-release.yaml --repo marin-community/vllm --ref main-next -f candidate_tag=<tag>`.
     The release job validates the exact wheel bytes on real GPUs and publishes an immutable
     release carrying `marin-vllm-gpu-manifest.json`.
  4. Download that manifest and re-pin without hand-editing `gpu.toml`:
     `gh release download <release_tag> --repo marin-community/vllm --pattern marin-vllm-gpu-manifest.json`,
     then `uv run config/update-external.py --promote-gpu-release marin-vllm-gpu-manifest.json`.
     The helper writes `gpu.toml` (release tag, source commit, version, torch backend,
     per-arch url+sha256) and regenerates `external_dependencies.py`; it re-encodes the wheel
     URLs the way the pin loader validates, which a hand copy gets wrong.

A base that crosses a CUDA/torch or vLLM stable-ABI boundary is a migration. Re-audit the
wheel verifier and the fork's release gate for the extension name (`vllm._C_stable_libtorch`
on CUDA 13) when the base moves across such a boundary.

## Fork suite caveat

Without its compiled CUDA/TPU stack, `py_compile` and a conflict-marker sweep are
structural checks only for vLLM; do not call them behavioral validation.

## End-to-end validation

- **`experiments/evals/served_qwen3.py::QWEN3_TPU_INFERENCE`** (`vllm`,
  `tpu-inference`) — a bounded brokered TPU serve+eval smoke. Run TPU workloads
  through Iris on the `marin` cluster at interactive priority, `v6e-4` in GCP
  `europe-west4`. Confirm the proxy served completions, lm-eval wrote metrics and
  sample outputs, and no TPU/vLLM build, import, or runtime tracebacks occurred:

```sh
uv run iris --config lib/iris/config/marin.yaml job run \
  --job-name served-qwen3-<run-id> --cpu 1 --memory 2G --extra cpu \
  --priority interactive --no-wait -- python -c \
  "from dataclasses import replace; from fray.types import ResourceConfig; from marin.execution.lazy import lower; from marin.execution.step_runner import StepRunner; from experiments.evals.lm_eval_suite import lm_eval_suite; from experiments.evals.served_qwen3 import QWEN3_TPU_INFERENCE; inference = replace(QWEN3_TPU_INFERENCE, iris=replace(QWEN3_TPU_INFERENCE.iris, worker_resources=ResourceConfig.with_tpu('v6e-4', ram='96g', regions=['europe-west4']))); StepRunner().run([lower(lm_eval_suite(inference, model_name='qwen3-0.6b-refresh-smoke', version='<run-id>-dev', limit=8))])"
```

- **`tests/cluster/vllm/test_snowball_backend_parity.py`** (`vllm-gpu`) — the
  Snowball-67B next-token logprob parity gate on H100s. It is `-m cluster` marked, so
  run it with `-o addopts= --import-mode=importlib`; the H100s live on CoreWeave and are
  reached through the marin federation hub (`target_cluster`), not a direct controller.
  Confirm the Levanter reference and `vllm-gpu-pp1` (single-node 8×H100) match the
  goldens within `max_probability_error` (`pp2` is a 16×H100 multi-node variant). Pair
  it with a bounded qwen3-0.6B GPU serve+eval to exercise the brokered serving path:

```sh
uv run pytest tests/cluster/vllm/test_snowball_backend_parity.py \
  -m cluster -o addopts= --import-mode=importlib -vv -s
```
