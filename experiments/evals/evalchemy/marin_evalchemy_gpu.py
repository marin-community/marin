# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""CoreWeave-GPU driver for the Marin canonical baseline evals (A3B MoEs).

LOCAL / UNCOMMITTED (marin is upstream marin-community/marin — do NOT push/merge).
GPU sibling of ``marin_evalchemy_tpu.py``, but built on the DECOUPLED ``serve_and_eval``
path so the serve child runs the marin VllmBackend/vLLM-fork stack (MoE / exotic-arch
support) and the eval child is a CPU-only ``eval.eval --model local-chat-completions``
client against the served OpenAI URL (:evalchemy-gpu image, has fsspec/s3fs).

Submits ONE parent CPU orchestrator to the selected CoreWeave H100 cluster; ``EVAL_CLUSTER``
defaults to ``cw-us-east-02a`` and may be set to ``cw-rno2a``. ``serve_and_eval`` spawns:
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
    EVAL_CLUSTER      Iris GPU cluster              (default cw-us-east-02a; e.g. cw-rno2a)
    EVAL_TIER         1 | 2                         (default 1 = 14 lm-harness tasks)
    EVAL_OUT_PATH     fsspec out_path              (default pod-local /tmp/<slug>-tier<N>out)
    EVAL_MAX_MODEL_LEN                              (default 32768)
    EVAL_MAX_GEN_TOKS                              (default 32768)
    EVAL_NUM_CONCURRENT                            (default 16)
    EVAL_TRUST_REMOTE_CODE  "1" to note it (serve child trust_remote_code — see NOTE below)
    EVAL_JOB_SUFFIX   extra tag on the job name
"""
from __future__ import annotations

import json
import os
import re
import time
import uuid
from pathlib import Path

from huggingface_hub import hf_hub_download
from iris.cli.connect import open_iris_client
from iris.cluster.types import Entrypoint, EnvironmentSpec, ResourceSpec
from iris.rpc import job_pb2
from marin.evaluation.evaluation_config import EvalTaskConfig

from experiments.evals.evalchemy.serve_and_eval import (
    EvalchemyEvalConfig,
    ServeBackend,
    ServeSpec,
    serve_and_eval,
)

CLUSTER = os.environ.get("EVAL_CLUSTER", "cw-us-east-02a")
# :evalchemy-gpu (built via evalchemy infra/docker/build_evalchemy_gpu_kaniko.sh, PR #18) — CPU-only eval
# client; python at /opt/eval/evalchemy/.venv. Pinned to evalchemy main HEAD 676fb85f which carries #28
# (per-sample records now persist for lm-eval-native tasks under --log_samples → offline rescore, e.g. drop).
EVAL_IMAGE = os.environ.get(
    "EVAL_IMAGE",
    "ghcr.io/open-thoughts/openthoughts-agent@sha256:5da405afbc9341f9c813e8a8df9e2cc5a371ab47d055dba2d9735573eb87783b",
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
    EvalTaskConfig("MATH500", 0, generation=True),
    # AIME24 is NOT here; it runs as a dedicated 10-seed mean/stddev set (EVAL_TASK_SET=aime24_seeds).
    # These Plus benchmarks construct OpenAI chat-message requests.  Unlike the legacy raw-infill
    # ``humaneval`` task, sending them to /v1/completions makes vLLM reject every request as a
    # non-string prompt, so they must retain the normal Tier-2 chat route.
    EvalTaskConfig("HumanEvalPlus", 0, generation=True, unsafe_code=True),
    EvalTaskConfig("MBPPPlus", 0, generation=True, unsafe_code=True),
    # MMLUPro DEFERRED — fork construction load_dataset fails (not a clean pip dep); N/A in RESULTS.
    EvalTaskConfig("GPQADiamond", 0, generation=True),
    # CruxEval DEFERRED — the fork's CruxEval does `from execution import ...` (local-module import,
    # not a pip dep) → registration fails; not a clean install. Marked N/A in RESULTS.
    EvalTaskConfig("IFEval", 0, generation=True),
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
        with open(hf_hub_download(model, "config.json")) as config_file:
            cfg = json.load(config_file)
    except Exception as e:
        print(f"[warn] config.json fetch failed for {model}: {e!r}; using request defaults")
    txt = json.dumps(cfg).lower()
    arch = " ".join(cfg.get("architectures") or []).lower()
    tcfg = cfg.get("text_config", cfg) if isinstance(cfg.get("text_config", cfg), dict) else cfg
    if (
        "gated_delta_net" in txt
        or "linear_attn" in txt
        or "qwen3next" in arch
        or "qwen3_5" in arch
        or "qwen3.5" in model.lower()
        or "qwen3-next" in model.lower()
    ):
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


# ── Baked per-model eval defaults: the Delphi grug THINKING checkpoint ─────────────────────────────
# A no-override launch (just EVAL_MODEL=<grug>) MUST produce valid MATH500/AIME: correct context/gen
# split (the gen budget must NOT consume the whole context, or a long prompt overflows → vLLM 400s →
# EMPTY results — the bug this bakes out), the atomic <|start_think|>/<|end_think|> delimiters intact,
# the Delphi chat_template (NOT the generic `main` /think template), and the answer-channel loop curbed.
# Every value below is a DEFAULT — the matching EVAL_* env var still overrides it.
PROMPT_HEADROOM = 4096  # Min context reserved for the prompt; default gen budget = max_model_len - this.
GRUG_THINKING_MODELS = {"penfever/grug-67b-a2b-sft-s2-thinking-step630"}
CANONICAL_AIME_SEEDS = (42, 43, 44)


def _aime_seeds(seed_spec: str | None) -> tuple[int, ...]:
    """Return canonical AIME seeds unless a caller deliberately requests a sensitivity set."""
    if not seed_spec:
        return CANONICAL_AIME_SEEDS
    return tuple(int(seed) for seed in seed_spec.split(",") if seed.strip())


def _is_grug_thinking(model: str) -> bool:
    m = model.lower()
    return model in GRUG_THINKING_MODELS or ("grug" in m and "thinking" in m)


def _grug_profile(model: str) -> dict:
    """Baked eval defaults for the Delphi grug thinking checkpoint ({} for any other model)."""
    if not _is_grug_thinking(model):
        return {}
    return {
        "tp": 1,  # 256-expert MoE: experts shard via data + expert parallelism, NOT tensor parallelism
        # ``main`` was repaired at this exact commit to carry the byte-identical
        # Delphi thinking template.  Pinning the immutable model *and tokenizer*
        # revision lets vLLM consume the checkpoint's own metadata; do not keep a
        # separately fetched template as a second, drift-prone source of truth.
        "revision": "c8e0e4ae6a892bced2263a6894cd61be8aa3a93b",
        # Keep the tokenizer in the model repository.  The tokenizer-only
        # companion repo has a different commit history, so a model metadata
        # revision is not a valid --tokenizer-revision there.
        "tokenizer": model,
        # Grug MoE expert-parallel serve.  vLLM loads both model and tokenizer
        # metadata at the same repaired immutable revision.
        # ``distributed`` was a RunAI-streamer loader option. The canonical local-weight path
        # uses vLLM's ``auto`` loader, which rejects it as an unknown loader extra config.
        "vllm_extra_args": (
            "--data-parallel-size",
            "8",
            "--enable-expert-parallel",
            "--revision",
            "c8e0e4ae6a892bced2263a6894cd61be8aa3a93b",
            "--tokenizer-revision",
            "c8e0e4ae6a892bced2263a6894cd61be8aa3a93b",
        ),
        # skip_special_tokens=false → PRESERVE the atomic 128002/128003 delimiters (default True strips
        # them → model looks like it emits no CoT). repetition_penalty=1.1 → curb the answer-channel
        # loops (marin #7321) so long-CoT MATH500/GPQA samples terminate + box within the gen budget.
        "extra_gen_kwargs": {"skip_special_tokens": "false", "repetition_penalty": "1.1"},
    }


def build_config(model: str, tokenizer: str, tier: int) -> EvalchemyEvalConfig:
    prof = _grug_profile(model)  # baked defaults for the Delphi grug thinking checkpoint ({} otherwise)
    tp = int(os.environ.get("EVAL_TP", str(prof.get("tp", 8))))
    max_model_len = int(os.environ.get("EVAL_MAX_MODEL_LEN", "32768"))
    # ⚠ max_gen_toks MUST be < max_model_len (prompt + max_tokens ≤ context, else vLLM 400s every
    # request). Default 20480 leaves ~12k for even 25-shot prompts. lm-harness loglikelihood/MC tasks
    # ignore it (max_tokens=0); it only bounds the 4 generative tasks (gsm8k/triviaqa/nq_open/drop).
    # Tier-2 budget is MODEL-CLASS-dependent: thinking/GDN models need ~30720 (near-full 32k) for
    # their CoT + boxed answer under the chat template; instruct models finish at 20480. Tier-1 uses
    # 20480 (completion-style, shorter). AIME already forces 30720 via EVAL_MAX_GEN_TOKS in the launcher.
    # grug thinking: size the gen budget to LEAVE prompt headroom (max_model_len - PROMPT_HEADROOM =
    # 28672 @ 32k) so the CoT budget can't starve a long prompt into a context overflow → EMPTY result
    # (marin #7321). Other thinking/GDN models keep the historical near-full 30720; instruct 20480.
    _default_gen = max_model_len - PROMPT_HEADROOM if prof else 30720 if tier == 2 and _is_thinking(model) else 20480
    max_gen_toks = int(os.environ.get("EVAL_MAX_GEN_TOKS", str(_default_gen)))
    num_concurrent = int(os.environ.get("EVAL_NUM_CONCURRENT", "16"))
    # Per-model serve mitigations (GDN triton / limit-mm / native-context cap) derived automatically
    # from config.json — durable + identical across tiers, no hand-passed flags.
    # grug thinking: default the serve args to the expert-parallel + delphi-v0-think-revision set (env wins).
    env_extra = (
        tuple(os.environ["EVAL_VLLM_EXTRA_ARGS"].split())
        if os.environ.get("EVAL_VLLM_EXTRA_ARGS")
        else tuple(prof.get("vllm_extra_args", ()))
    )
    auto_extra, max_model_len = _auto_serve_overrides(model, max_model_len)
    # Merge: env-passed flags win; append an auto flag only if its flag-name isn't already present.
    merged = list(env_extra)
    i = 0
    while i < len(auto_extra):
        flag = auto_extra[i]
        pair = (
            auto_extra[i : i + 2]
            if (i + 1 < len(auto_extra) and not auto_extra[i + 1].startswith("--"))
            else auto_extra[i : i + 1]
        )
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
    # Subset override (serial tier-2 rerun of only the un-banked benchmarks): EVAL_TIER2_TASKS as a
    # comma list of benchmark names (e.g. "MATH500,MBPPPlus,GPQADiamond,IFEval" to skip an already-
    # harvested HumanEvalPlus). Applies on the tier-2 path; num_fewshot 0 (tier-2 is all 0-shot chat).
    if tier == 2 and os.environ.get("EVAL_TIER2_TASKS"):
        selected = {task.name: task for task in TIER2_TASKS}
        tasks = tuple(
            selected.get(name.strip(), EvalTaskConfig(name.strip(), 0, generation=True))
            for name in os.environ["EVAL_TIER2_TASKS"].split(",")
            if name.strip()
        )
    # Fast diagnostic smoke: a generative task (gsm8k) + an MC/loglikelihood task (hellaswag), small
    # --limit → proves BOTH request types score non-empty over local-completions before the full suite.
    if os.environ.get("EVAL_SMOKE") == "1":
        tasks = (EvalTaskConfig("gsm8k", 0), EvalTaskConfig("hellaswag", 10))
    # AIME24 policy: one process per canonical seed 42..44, with distinct task aliases so results
    # do not overwrite. Three valid paired samples are sufficient for the campaign statistic.
    if os.environ.get("EVAL_TASK_SET") == "math500":  # one-benchmark chat smoke (reasoning-parser test)
        tasks = (EvalTaskConfig("MATH500", 0),)
        apply_chat_template = True
    if os.environ.get("EVAL_TASK_SET") == "aime24_seeds":
        # EVAL_AIME_SEEDS may override the canonical three seeds for a deliberate sensitivity study.
        _seeds = _aime_seeds(os.environ.get("EVAL_AIME_SEEDS"))
        tasks = tuple(
            EvalTaskConfig("AIME24", 0, task_alias=f"AIME24_seed{s}", task_kwargs={"seed": s}, generation=True)
            for s in _seeds
        )
        apply_chat_template = True
    max_eval_instances = int(os.environ["EVAL_LIMIT"]) if os.environ.get("EVAL_LIMIT") else None
    # An operator may explicitly override the server's chat template for an
    # experiment.  Grug's default deliberately remains ``None``: its immutable
    # checkpoint revision above is the authoritative source for the Delphi
    # start-think/end-think protocol.  Passing a copied template would create a
    # second source of truth and reintroduce the historical main-template drift.
    chat_template_content = None
    ctf = os.environ.get("EVAL_CHAT_TEMPLATE_FILE")
    if ctf:
        chat_template_content = Path(ctf).read_text()
    # LOCAL (grug thinking model): extra --gen_kwargs (key=value, comma-separated) folded into every
    # lm-eval request. THE crux for Delphi thinking models: EVAL_EXTRA_GEN_KWARGS="skip_special_tokens=false"
    # PRESERVES the atomic 128002/128003 delimiters (vLLM's skip_special_tokens=True default STRIPS them,
    # which made the model look like it emitted no CoT — the original serving bug).
    # grug thinking default: {skip_special_tokens:false, repetition_penalty:1.1}; EVAL_EXTRA_GEN_KWARGS wins.
    extra_gen_kwargs: dict[str, str] = dict(prof.get("extra_gen_kwargs", {}))
    egk = os.environ.get("EVAL_EXTRA_GEN_KWARGS")
    if egk:
        extra_gen_kwargs = {}
        for pair in egk.split(","):
            pair = pair.strip()
            if not pair:
                continue
            k, _, v = pair.partition("=")
            extra_gen_kwargs[k.strip()] = v.strip()
    # Durable CoreWeave LOTA object store (marin-us-east-02a): the eval child writes each task's
    # results_*.json here (run_evalchemy_client passes the LOTA virtual-addressing storage_options).
    # Unique per-run sub-path so re-runs don't collide. Mac can't read LOTA directly (in-cluster only) —
    # harvest via the EVALCHEMY_RESULT log-print; the s3 JSONs are the durable deliverable/provenance.
    run_id = os.environ.get("EVAL_RUN_ID", uuid.uuid4().hex[:6])
    default_out = f"s3://marin-us-east-02a/iris/marinbase-eval/{_slug(model)}/tier{tier}-{run_id}"
    out_path = os.environ.get("EVAL_OUT_PATH", default_out)
    # extra_gen_kwargs only reaches EvalchemyEvalConfig if that field exists in the installed marin lib;
    # pass it ONLY when non-empty so the common (empty) path stays compatible with libs predating the
    # field. Non-empty (grug thinking skip_special_tokens=false) requires the field to be present.
    _egk = {"extra_gen_kwargs": extra_gen_kwargs} if extra_gen_kwargs else {}
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
            chat_template_content=chat_template_content,
        ),
        apply_chat_template=apply_chat_template,
        max_gen_toks=max_gen_toks,
        max_eval_instances=max_eval_instances,
        num_concurrent=num_concurrent,
        eval_image=EVAL_IMAGE,
        eval_cpu=8.0,
        eval_memory="32g",
        eval_disk="50g",
        **_egk,
    )


def main() -> None:
    model = os.environ.get("EVAL_MODEL", "Qwen/Qwen3-30B-A3B-Thinking-2507")
    # Grug defaults to the model repository's tokenizer so its immutable revision
    # covers the template, generation config, and tokenizer together.
    tokenizer = os.environ.get("EVAL_TOKENIZER") or _grug_profile(model).get("tokenizer") or model
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
    print(
        f"MODEL={model}  TIER={tier}  TP={config.serve.tensor_parallel_size}  "
        f"max_model_len={config.serve.max_model_len}  out_path={config.out_path}"
    )
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
        print(
            f"LOG_CMD: KUBECONFIG=~/.kube/coreweave-iris-gpu "
            f"/Users/benjaminfeuer/miniconda3/envs/otagent/bin/iris --cluster={CLUSTER} "
            f"job logs {job_id} --since-ms {submit_ms} --max-lines 400 --no-tail"
        )


if __name__ == "__main__":
    main()
