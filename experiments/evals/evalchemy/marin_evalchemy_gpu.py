# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""CoreWeave-GPU driver for the Marin canonical baseline evals (A3B MoEs).

LOCAL / UNCOMMITTED (marin is upstream marin-community/marin — do NOT push/merge).
GPU sibling of ``marin_evalchemy_tpu.py``, but built on the DECOUPLED ``serve_and_eval``
path so the serve child runs the marin VllmBackend/vLLM-fork stack (MoE / exotic-arch
support) and the eval child is a CPU-only ``eval.eval --model local-chat-completions``
client against the served OpenAI URL (:evalchemy-gpu image, has fsspec/s3fs).

Submits ONE parent CPU orchestrator to ``cw-us-east-02a``; serve_and_eval spawns:
  * serve child  (8xH100)  — marin VllmBackend, TP=$TP, max_model_len=32768, bf16.
  * eval  child  (CPU)     — one ``eval.eval`` per task (own num_fewshot), chat route,
                             --apply_chat_template, --log_samples, --verbosity INFO.

Run from the marin repo root with the marin .venv (kubernetes 35.0.0 present) + secrets
sourced + the CoreWeave GPU kubeconfig:
    export DC_AGENT_SECRET_ENV=/Users/benjaminfeuer/Documents/secrets.env
    set -a; source "$DC_AGENT_SECRET_ENV"; set +a
    export KUBECONFIG=~/.kube/coreweave-iris-gpu
    ./.venv/bin/python experiments/evals/evalchemy/marin_evalchemy_gpu.py

Env knobs (all optional; defaults = the Qwen3-30B-A3B-Thinking-2507 pathfinder, tier 1):
    EVAL_MODEL        HF id to serve+eval          (default Qwen/Qwen3-30B-A3B-Thinking-2507)
    EVAL_TOKENIZER    HF tokenizer id              (default = EVAL_MODEL)
    EVAL_TP           tensor_parallel_size         (default 8; must divide num_attention_heads)
    EVAL_TIER         1 | 2                         (default 1 = 14 lm-harness tasks)
    EVAL_OUT_PATH     fsspec out_path              (default pod-local /tmp/<slug>-tier<N>out)
    EVAL_MAX_MODEL_LEN                              (default 32768)
    EVAL_MAX_GEN_TOKS                              (default 32768)
    EVAL_NUM_CONCURRENT                            (default 16)
    EVAL_TRUST_REMOTE_CODE  "1" to note it (serve child trust_remote_code — see NOTE below)
    EVAL_JOB_SUFFIX   extra tag on the job name
"""
from __future__ import annotations

import os
import re
import time
import uuid
from pathlib import Path

from iris.cli.connect import open_iris_client
from iris.cluster.types import Entrypoint, EnvironmentSpec, ResourceSpec
from iris.rpc import job_pb2

from experiments.evals.evalchemy.serve_and_eval import (
    EvalchemyEvalConfig,
    ServeBackend,
    ServeSpec,
    serve_and_eval,
)
from marin.evaluation.evaluation_config import EvalTaskConfig

CLUSTER = "cw-us-east-02a"
# :evalchemy-gpu (already built, PR #18) — CPU-only eval client; python at /opt/eval/evalchemy/.venv.
EVAL_IMAGE = os.environ.get(
    "EVAL_IMAGE", "ghcr.io/open-thoughts/openthoughts-agent:evalchemy-gpu-73c19cf"
)
EVAL_PYTHON = os.environ.get("EVAL_PYTHON", "/opt/eval/evalchemy/.venv/bin/python")

# --- Tier 1: 14 lm-eval-harness tasks (POLICY §3; per-task shots). 1 run, seed 42. ---
TIER1_TASKS = (
    EvalTaskConfig("gsm8k", 0),
    EvalTaskConfig("mmlu", 5),
    EvalTaskConfig("hellaswag", 10),
    EvalTaskConfig("arc_challenge", 25),
    EvalTaskConfig("arc_easy", 0),
    EvalTaskConfig("piqa", 0),
    EvalTaskConfig("winogrande", 5),
    EvalTaskConfig("openbookqa", 0),
    EvalTaskConfig("boolq", 0),
    EvalTaskConfig("truthfulqa_mc2", 0),
    EvalTaskConfig("lambada_openai", 0),
    EvalTaskConfig("triviaqa", 5),
    EvalTaskConfig("nq_open", 5),
    EvalTaskConfig("drop", 3),
)

# --- Tier 2: evalchemy chat benchmarks (POLICY §3). Names VERIFIED against the evalchemy fork's
#     eval/chat_benchmarks/ dirs (the authoritative registry). ⚠ OlympiadBench + FinanceBench are NOT
#     in this fork → omitted (flagged to operator). IFBench→IFEval (the fork's constraint benchmark).
#     AIME24 = 10-seed (42..51) via EVAL_TIER=2 seed handling in the driver/client. ---
TIER2_TASKS = (
    EvalTaskConfig("MATH500", 0),
    # AIME24 is NOT here — it runs as a dedicated 10-seed μ±σ set (EVAL_TASK_SET=aime24_seeds).
    EvalTaskConfig("HumanEvalPlus", 0),
    EvalTaskConfig("MBPPPlus", 0),
    # MMLUPro DEFERRED — fork construction load_dataset fails (not a clean pip dep); N/A in RESULTS.
    EvalTaskConfig("GPQADiamond", 0),
    # CruxEval DEFERRED — the fork's CruxEval does `from execution import ...` (local-module import,
    # not a pip dep) → registration fails; not a clean install. Marked N/A in RESULTS.
    EvalTaskConfig("IFEval", 0),
)


def _slug(model: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", model.lower()).strip("-")


def _is_thinking(model: str) -> bool:
    """Thinking/reasoning models emit long CoT under the chat template → they truncate tier-2
    generative benchmarks at 20480 (MATH500/HumanEval → 0) and need the near-full window. The 3
    thinking baselines: Qwen3-*-Thinking + Qwen3.5 (thinking-on default). Instruct/chat models don't."""
    m = model.lower()
    return "thinking" in m or "qwen3.5" in m or "qwen3-next" in m


def _auto_serve_overrides(model: str, requested_max_model_len: int):
    """Per-model serve mitigations derived from the model's config.json (durable + automatic, so both
    tiers + every launch carry them identically). Returns (extra_args_tuple, effective_max_model_len).

    - GDN hybrid (gated-delta-net linear attention: Qwen3-Next, Qwen3.5) → `--gdn-prefill-backend triton`.
      Maintainer-blessed mitigation (marin #7373 — Romain: pass via ServeSpec.vllm_extra_args; the
      FlashInfer GDN prefill kernel is JIT-compiled and needs nvcc, absent in the uvx serve env).
    - Multimodal wrapper (vision tower / *ForConditionalGeneration: Qwen3.5) → `--limit-mm-per-prompt
      {"image":0,"video":0}` for text-only serving.
    - Cap max_model_len at the model's native context (Moonlight = 8192; else vLLM ValidationError).
    """
    extra: list[str] = []
    mml = requested_max_model_len
    cfg: dict = {}
    try:
        import json as _json
        from huggingface_hub import hf_hub_download
        cfg = _json.load(open(hf_hub_download(model, "config.json")))
    except Exception as e:  # noqa: BLE001
        print(f"[warn] config.json fetch failed for {model}: {e!r}; using request defaults")
    txt = __import__("json").dumps(cfg).lower()
    arch = " ".join(cfg.get("architectures") or []).lower()
    tcfg = cfg.get("text_config", cfg) if isinstance(cfg.get("text_config", cfg), dict) else cfg
    if ("gated_delta_net" in txt or "linear_attn" in txt or "qwen3next" in arch
            or "qwen3_5" in arch or "qwen3.5" in model.lower() or "qwen3-next" in model.lower()):
        extra += ["--gdn-prefill-backend", "triton"]
    if cfg.get("vision_config") or "forconditionalgeneration" in arch:
        extra += ["--limit-mm-per-prompt", '{"image":0,"video":0}']
    # Reasoning parser for thinking models (POLICY §5): the model emits <think>…</think> then the
    # answer; without a parser the chat endpoint returns empty `content` for evalchemy graders → 0.
    # `qwen3` parser splits reasoning_content from content so the grader sees the post-</think> answer.
    ml = model.lower()
    if ("thinking" in ml or "qwen3.5" in ml or "qwen3-next" in ml) and "qwen" in ml:
        extra += ["--reasoning-parser", "qwen3"]
    mpe = tcfg.get("max_position_embeddings") or cfg.get("max_position_embeddings")
    if isinstance(mpe, (int, float)) and int(mpe) < mml:
        mml = int(mpe)
    return tuple(extra), mml


def build_config(model: str, tokenizer: str, tier: int) -> EvalchemyEvalConfig:
    tp = int(os.environ.get("EVAL_TP", "8"))
    max_model_len = int(os.environ.get("EVAL_MAX_MODEL_LEN", "32768"))
    # ⚠ max_gen_toks MUST be < max_model_len (prompt + max_tokens ≤ context, else vLLM 400s every
    # request). Default 20480 leaves ~12k for even 25-shot prompts. lm-harness loglikelihood/MC tasks
    # ignore it (max_tokens=0); it only bounds the 4 generative tasks (gsm8k/triviaqa/nq_open/drop).
    # Tier-2 budget is MODEL-CLASS-dependent: thinking/GDN models need ~30720 (near-full 32k) for
    # their CoT + boxed answer under the chat template; instruct models finish at 20480. Tier-1 uses
    # 20480 (completion-style, shorter). AIME already forces 30720 via EVAL_MAX_GEN_TOKS in the launcher.
    _default_gen = 30720 if (tier == 2 and _is_thinking(model)) else 20480
    max_gen_toks = int(os.environ.get("EVAL_MAX_GEN_TOKS", str(_default_gen)))
    num_concurrent = int(os.environ.get("EVAL_NUM_CONCURRENT", "16"))
    # Per-model serve mitigations (GDN triton / limit-mm / native-context cap) derived automatically
    # from config.json — durable + identical across tiers, no hand-passed flags.
    env_extra = tuple(os.environ["EVAL_VLLM_EXTRA_ARGS"].split()) if os.environ.get("EVAL_VLLM_EXTRA_ARGS") else ()
    auto_extra, max_model_len = _auto_serve_overrides(model, max_model_len)
    # Merge: env-passed flags win; append an auto flag only if its flag-name isn't already present.
    merged = list(env_extra)
    i = 0
    while i < len(auto_extra):
        flag = auto_extra[i]
        pair = auto_extra[i:i + 2] if (i + 1 < len(auto_extra) and not auto_extra[i + 1].startswith("--")) else auto_extra[i:i + 1]
        if flag not in merged:
            merged += list(pair)
        i += len(pair)
    vllm_extra_args = tuple(merged)
    # max_gen_toks MUST stay < max_model_len (else vLLM 400s / decoder-prompt-empty). Cap with ~4k
    # prompt headroom — critical for small-context models (Moonlight 8192 → 4096).
    if max_gen_toks >= max_model_len:
        max_gen_toks = max(1024, max_model_len - 4096)
    # Tier 1 (lm-harness) → local-completions (/completions logprob endpoint, NO chat template): the
    # 10 MC/loglikelihood tasks REQUIRE /completions (chat/completions can't score forced continuations),
    # matching marin's canonical `marin/evaluation/lm_eval.py` default (LOCAL_COMPLETIONS, apply_chat_template=False).
    # Tier 2 (evalchemy chat benchmarks) → chat template ON (generative graders). Override via EVAL_APPLY_CHAT_TEMPLATE.
    _act_env = os.environ.get("EVAL_APPLY_CHAT_TEMPLATE")
    apply_chat_template = (_act_env == "1") if _act_env is not None else (tier == 2)
    tasks = TIER1_TASKS if tier == 1 else TIER2_TASKS
    # Fast diagnostic smoke: a generative task (gsm8k) + an MC/loglikelihood task (hellaswag), small
    # --limit → proves BOTH request types score non-empty over local-completions before the full suite.
    if os.environ.get("EVAL_SMOKE") == "1":
        tasks = (EvalTaskConfig("gsm8k", 0), EvalTaskConfig("hellaswag", 10))
    # AIME24 10-seed μ±σ (POLICY §3): one process per seed 42..51, distinct dir per seed (task_alias)
    # so results don't overwrite. Harvest = mean±std over the 10 pass@1 values.
    if os.environ.get("EVAL_TASK_SET") == "math500":  # one-benchmark chat smoke (reasoning-parser test)
        tasks = (EvalTaskConfig("MATH500", 0),)
        apply_chat_template = True
    if os.environ.get("EVAL_TASK_SET") == "aime24_seeds":
        tasks = tuple(
            EvalTaskConfig("AIME24", 0, task_alias=f"AIME24_seed{s}", task_kwargs={"seed": s})
            for s in range(42, 52)
        )
        apply_chat_template = True
    max_eval_instances = int(os.environ["EVAL_LIMIT"]) if os.environ.get("EVAL_LIMIT") else None
    # Durable CoreWeave LOTA object store (marin-us-east-02a): the eval child writes each task's
    # results_*.json here (run_evalchemy_client passes the LOTA virtual-addressing storage_options).
    # Unique per-run sub-path so re-runs don't collide. Mac can't read LOTA directly (in-cluster only) —
    # harvest via the EVALCHEMY_RESULT log-print; the s3 JSONs are the durable deliverable/provenance.
    run_id = os.environ.get("EVAL_RUN_ID", uuid.uuid4().hex[:6])
    default_out = f"s3://marin-us-east-02a/iris/marinbase-eval/{_slug(model)}/tier{tier}-{run_id}"
    out_path = os.environ.get("EVAL_OUT_PATH", default_out)
    return EvalchemyEvalConfig(
        model=model,
        tokenizer=tokenizer,
        tasks=tasks,
        out_path=out_path,
        serve=ServeSpec(
            backend=ServeBackend.VLLM,
            tpu_type=None,
            gpu_type="H100",
            gpu_count=8,
            tensor_parallel_size=tp,
            dtype="bfloat16",
            max_model_len=max_model_len,
            serve_cpu=48.0,
            serve_memory="1400g",
            serve_disk="200g",
            # Auto-derived per-model serve args (GDN triton / limit-mm) merged with any EVAL_VLLM_EXTRA_ARGS;
            # rides ServeSpec.vllm_extra_args — the maintainer-blessed surface (marin #7373).
            vllm_extra_args=vllm_extra_args,
        ),
        apply_chat_template=apply_chat_template,
        max_gen_toks=max_gen_toks,
        max_eval_instances=max_eval_instances,
        num_concurrent=num_concurrent,
        eval_image=EVAL_IMAGE,
        eval_cpu=8.0,
        eval_memory="32g",
        eval_disk="50g",
    )


def main() -> None:
    model = os.environ.get("EVAL_MODEL", "Qwen/Qwen3-30B-A3B-Thinking-2507")
    tokenizer = os.environ.get("EVAL_TOKENIZER", model)
    tier = int(os.environ.get("EVAL_TIER", "1"))
    suffix = os.environ.get("EVAL_JOB_SUFFIX", "")
    config = build_config(model, tokenizer, tier)

    # Priority: EVAL_PRIORITY (batch|interactive|production; default batch). Forwarded to the parent
    # env as MARINBASE_EVAL_PRIORITY so serve_and_eval sets the serve+eval CHILDREN to match — bump to
    # interactive for slow models that keep getting preempted mid-run (GDN 80B/35B).
    priority = os.environ.get("EVAL_PRIORITY", "batch").strip().lower()
    priority_band = {
        "production": job_pb2.PRIORITY_BAND_PRODUCTION,
        "interactive": job_pb2.PRIORITY_BAND_INTERACTIVE,
        "batch": job_pb2.PRIORITY_BAND_BATCH,
    }.get(priority, job_pb2.PRIORITY_BAND_BATCH)

    env_vars = {"EVALCHEMY_PYTHON": EVAL_PYTHON, "MARINBASE_EVAL_PRIORITY": priority}
    for k in ("HF_TOKEN", "WANDB_API_KEY", "WANDB_ENTITY", "WANDB_PROJECT", "EVAL_RAW_PROBE"):
        if os.environ.get(k):
            env_vars[k] = os.environ[k]

    name = f"marinbase-eval-{_slug(model)}-t{tier}{('-' + suffix) if suffix else ''}-{uuid.uuid4().hex[:6]}"
    print(f"MODEL={model}  TIER={tier}  TP={config.serve.tensor_parallel_size}  "
          f"max_model_len={config.serve.max_model_len}  out_path={config.out_path}")
    print(f"TASKS={[t.name + '@' + str(t.num_fewshot) for t in config.tasks]}")
    print(f"EVAL_IMAGE={EVAL_IMAGE}")

    with open_iris_client(cluster_name=CLUSTER, workspace=Path(__file__).resolve().parents[3]) as client:
        submit_ms = int(time.time() * 1000)
        job = client.submit(
            entrypoint=Entrypoint.from_callable(serve_and_eval, config),
            name=name,
            resources=ResourceSpec(cpu=1.0, memory="8g", disk="20g"),
            environment=EnvironmentSpec(env_vars=env_vars),
            max_retries_failure=0,
            priority_band=priority_band,
        )
        job_id = getattr(job, "job_id", None) or getattr(job, "name", None)
        print("SUBMITTED_JOB_ID:", job_id)
        print("JOB_NAME:", name)
        print("SUBMIT_MS:", submit_ms)
        print(f"LOG_CMD: KUBECONFIG=~/.kube/coreweave-iris-gpu "
              f"/Users/benjaminfeuer/miniconda3/envs/otagent/bin/iris --cluster={CLUSTER} "
              f"job logs {job_id} --since-ms {submit_ms} --max-lines 400 --no-tail")


if __name__ == "__main__":
    main()
