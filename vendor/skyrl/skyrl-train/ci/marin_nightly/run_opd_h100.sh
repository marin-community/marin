#!/usr/bin/env bash
# One real synchronous OPD step with a separate fixed vLLM teacher. This runs inside
# one H100x4 Iris task: policy, rollout, and teacher each own one GPU.
set -euo pipefail

REPOSITORY_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
NIGHTLY_RL_ENV="${NIGHTLY_RL_ENV:-$REPOSITORY_ROOT/.iris-nightly-opd-env}"
POLICY_MODEL="${POLICY_MODEL:-Qwen/Qwen3-0.6B}"
TEACHER_MODEL="${TEACHER_MODEL:-Qwen/Qwen3-1.7B}"
TEACHER_REVISION="${TEACHER_REVISION:-70d244cc86ccca08cf5af4e1e306ecf908b1ad5e}"
TEACHER_SOURCE="${TEACHER_SOURCE:-local_inference}"
DATA_DIR="${DATA_DIR:-$HOME/data/gsm8k_opd_nightly}"
LOG="${LOG:-$PWD/opd-nightly-run.log}"
SPEC="${SPEC:-ci/marin_nightly/specs/opd-qwen3-sync.json}"
# The nightly takes one synchronous step. An asynchronous smoke sets MAX_STALENESS_STEPS >= 1 and MAX_STEPS >= 2,
# so some trained groups come from an older policy than the one they train. The 8-row slice allows 4 steps.
MAX_STEPS="${MAX_STEPS:-1}"
MAX_STALENESS_STEPS="${MAX_STALENESS_STEPS:-0}"
# Positive staleness trains on rollouts from older policies, which needs an off-policy correction.
if (( MAX_STALENESS_STEPS > 0 )); then POLICY_LOSS_TYPE=behavior_clip; else POLICY_LOSS_TYPE=regular; fi
source "$REPOSITORY_ROOT/skyrl-train/ci/marin_nightly/resolve_runtime.sh" \
  "$REPOSITORY_ROOT" "$NIGHTLY_RL_ENV" production

cd "$REPOSITORY_ROOT/skyrl-train"
rm -rf skyrl-gym
cp -R ../skyrl-gym skyrl-gym
export PYTHONPATH="$PWD/skyrl-gym:$PWD:$REPOSITORY_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export VLLM_USE_DEEP_GEMM=0

TEACHER_ARGS=()
teacher_pid=""
cleanup_teacher() {
  if [[ -n "$teacher_pid" ]]; then
    kill "$teacher_pid" >/dev/null 2>&1 || true
    wait "$teacher_pid" >/dev/null 2>&1 || true
  fi
}
trap cleanup_teacher EXIT

case "$TEACHER_SOURCE" in
  local_inference)
    TEACHER_ARGS=(
      ++teachers.primary.source=local_inference
      ++teachers.primary.placement=pinned
      ++teachers.primary.model.path="$TEACHER_MODEL"
      ++teachers.primary.model.revision="$TEACHER_REVISION"
      ++teachers.primary.backend=vllm
      ++teachers.primary.evidence=chosen_token
      ++teachers.primary.resources.num_nodes=1
      ++teachers.primary.resources.gpus_per_node=1
      ++teachers.primary.resources.tensor_parallel_size=1
      ++teachers.primary.resources.colocation_group=teacher
    )
    ;;
  openai_compatible)
    TOKENIZER_FINGERPRINT=$("$PYTHON" -c \
      'import sys; from skyrl_train.inference_engines.vllm_teacher_oracle import tokenizer_vocabulary_fingerprint; from skyrl_train.tokenizer import create_tokenizer; print(tokenizer_vocabulary_fingerprint(create_tokenizer(sys.argv[1], disable_fast_tokenizer=False)))' \
      "$POLICY_MODEL")
    "$PYTHON" tests/fixtures/opd_http_teacher.py --port 18080 &
    teacher_pid=$!
    "$PYTHON" - <<'PY'
import time
import urllib.error
import urllib.request

for _ in range(100):
    try:
        urllib.request.urlopen("http://127.0.0.1:18080", timeout=1)
    except urllib.error.HTTPError:
        break
    except OSError:
        time.sleep(0.1)
else:
    raise RuntimeError("remote teacher fixture did not start")
PY
    TEACHER_ARGS=(
      ++teachers.primary.source=openai_compatible
      ++teachers.primary.placement=external
      ++teachers.primary.model.path="$TEACHER_MODEL"
      ++teachers.primary.model.revision="$TEACHER_REVISION"
      ++teachers.primary.endpoints="[{url:http://127.0.0.1:18080/v1,max_concurrency:8}]"
      ++teachers.primary.tokenizer_fingerprint="$TOKENIZER_FINGERPRINT"
      ++teachers.primary.max_sequence_length=32768
      ++teachers.primary.request_timeout_seconds=120
      ++teachers.primary.evidence=chosen_token
    )
    ;;
  *)
    echo "unsupported TEACHER_SOURCE: $TEACHER_SOURCE" >&2
    exit 2
    ;;
esac

echo "::: GPU and driver"
nvidia-smi --query-gpu=name,driver_version --format=csv
test -x "$PYTHON"
"$PYTHON" -c "import torch, vllm; print(f'torch {torch.__version__} | vllm {vllm.__version__}')"

echo "::: preparing a small GSM8K slice"
"$PYTHON" examples/gsm8k/gsm8k_dataset.py --output_dir "$DATA_DIR"
DATA_DIR="$DATA_DIR" "$PYTHON" - <<'PY'
import os
import pathlib

import polars as pl

data_dir = pathlib.Path(os.environ["DATA_DIR"])
for name, rows in (("train", 8), ("validation", 2)):
    path = data_dir / f"{name}.parquet"
    pl.read_parquet(path).head(rows).write_parquet(path)
PY

echo "::: training ${POLICY_MODEL} for ${MAX_STEPS} teacher-sensitive step(s) with ${TEACHER_MODEL}@${TEACHER_REVISION}"
echo "::: rollouts: max_staleness_steps=${MAX_STALENESS_STEPS}"
START=$(date +%s)
"$PYTHON" -m cloud.iris.telemetry_env -- \
  "$PYTHON" -m skyrl_train.entrypoints.main_base \
  data.train_data="['$DATA_DIR/train.parquet']" \
  data.val_data="['$DATA_DIR/validation.parquet']" \
  trainer.algorithm.advantage_estimator=grpo \
  trainer.algorithm.use_kl_loss=false \
  ++trainer.algorithm.distillation.objective=sampled_reverse_kl \
  ++trainer.algorithm.distillation.routing_plan=opd \
  ++trainer.algorithm.distillation.coefficient=0.5 \
  ++trainer.algorithm.distillation.reward_mode=add \
  "${TEACHER_ARGS[@]}" \
  ++teacher_routing.opd.revision=qwen3-sync-v1 \
  ++teacher_routing.opd.routes.default.teacher=primary \
  ++teacher_routing.opd.routes.default.weight=1.0 \
  trainer.policy.model.path="$POLICY_MODEL" \
  trainer.strategy=megatron \
  trainer.flash_attn=false \
  trainer.use_sample_packing=false \
  trainer.placement.colocate_all=false \
  trainer.placement.policy_num_gpus_per_node=1 \
  trainer.placement.critic_num_gpus_per_node=1 \
  trainer.placement.ref_num_gpus_per_node=1 \
  trainer.epochs=1 \
  trainer.max_steps="$MAX_STEPS" \
  trainer.rollout_buffer.max_staleness_steps="$MAX_STALENESS_STEPS" \
  trainer.algorithm.policy_loss_type="$POLICY_LOSS_TYPE" \
  trainer.train_batch_size=2 \
  trainer.policy_mini_batch_size=2 \
  trainer.micro_train_batch_size_per_gpu=1 \
  trainer.micro_forward_batch_size_per_gpu=1 \
  trainer.update_epochs_per_batch=1 \
  trainer.max_prompt_length=256 \
  trainer.eval_before_train=false \
  trainer.eval_interval=-1 \
  trainer.ckpt_interval=-1 \
  trainer.hf_save_interval=-1 \
  trainer.resume_mode=null \
  trainer.dump_eval_results=false \
  trainer.logger=console \
  trainer.project_name=marin_nightly \
  trainer.run_name=opd_qwen3_sync \
  generator.backend=vllm \
  generator.num_inference_engines=1 \
  generator.inference_engine_tensor_parallel_size=1 \
  generator.n_samples_per_prompt=2 \
  generator.sampling_params.max_generate_length=64 \
  generator.gpu_memory_utilization=0.7 \
  generator.run_engines_locally=true \
  generator.weight_sync_backend=nccl \
  environment.env_class=gsm8k \
  2>&1 | tee "$LOG"
ELAPSED=$(( $(date +%s) - START ))

echo "::: gating (run took ${ELAPSED}s)"
"$PYTHON" -m ci.marin_nightly.gate \
  --log "$LOG" --spec "$SPEC" --wall-clock-seconds "$ELAPSED"
