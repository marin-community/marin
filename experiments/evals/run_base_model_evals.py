# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Base model evaluations across multiple LLMs.

Evaluates OLMo Base 8B, LLAMA 3.1 8B, Deeper Starling 8B, MAP-NEO 7B, and Amber Base 7B on the
``base_model_evals`` suite (CORE + leaderboard, MMLU 0/5-shot, MMLU-Pro, and generation), then
compiles one ``EvalReport`` per model.
"""

from marin.execution.lazy import ArtifactStep, run
from marin.training.training import LevanterCheckpoint

from experiments.evals.evals import base_model_evals, eval_report, eval_steps
from experiments.models import amber_base_7b, llama_3_1_8b, map_neo_7b, olmo_2_base_8b

if __name__ == "__main__":
    # Adopt the externally-produced Deeper Starling checkpoint as a typed handle: resolves to the
    # source path, records provenance, no copy/recompute.
    deeper_starling = ArtifactStep.adopt(
        "checkpoints/deeper-starling-8b",
        "2026.06.30",
        "gs://marin-us-central2/checkpoints/tootsie-8b-deeper-starling/hf/step-1419999",
        kind=LevanterCheckpoint,
    )
    models = [
        (deeper_starling, base_model_evals()),
        (llama_3_1_8b, base_model_evals()),
        (olmo_2_base_8b, base_model_evals()),
        (amber_base_7b, base_model_evals(engine_kwargs={"max_model_len": 2048, "max_gen_toks": 2048})),
        (
            map_neo_7b,
            base_model_evals(engine_kwargs={"trust_remote_code": True, "max_model_len": 4096, "max_gen_toks": 4096}),
        ),
    ]
    reports = [eval_report(eval_steps(model, groups), name=f"{model.name}/base") for model, groups in models]
    run(*reports)
