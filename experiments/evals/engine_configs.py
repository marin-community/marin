# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Engine + run defaults for vLLM-backed evals.

- `DEFAULT_VLLM_DEPLOYMENT_KWARGS`: vLLM server flags. Feeds `ModelDeployment.engine_kwargs`.
- `DEFAULT_LM_EVAL_EXTRA_MODEL_ARGS`: per-request / lm-eval client knobs.
  Feeds `LmEvalRun.extra_model_args` as `k=v` strings.
"""

DEFAULT_VLLM_DEPLOYMENT_KWARGS: dict = {"max_model_len": 4096}

DEFAULT_LM_EVAL_EXTRA_MODEL_ARGS: tuple[str, ...] = ("max_gen_toks=4096",)
