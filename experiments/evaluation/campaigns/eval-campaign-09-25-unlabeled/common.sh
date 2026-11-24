#!/usr/bin/env bash
# Shared helpers for the unlabeled 2026-09-25 evaluation campaign. Source this file.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MARIN_DIR="${MARIN_DIR:-$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)}"
CANONICAL_CONFIG_ROOT="$SCRIPT_DIR"
CAMPAIGN_CONFIG="$CANONICAL_CONFIG_ROOT/campaign.yaml"
MODEL_CONFIG_DIR="$CANONICAL_CONFIG_ROOT/model-configs"
CHAT_TEMPLATE_DIR="$CANONICAL_CONFIG_ROOT/chat-templates"
EVALUATION_CONFIG="$CANONICAL_CONFIG_ROOT/evaluations.yaml"
CONFIG_SNAPSHOT_DIR="$MARIN_DIR/experiments/evaluation/campaigns/.launch-snapshots/eval-campaign-09-25-unlabeled"
STAGING_ROOT="$MARIN_DIR/experiments/evaluation/campaigns/.launch-staging/eval-campaign-09-25-unlabeled"
EVAL_CAMPAIGN_SECRETS_ENV="${EVAL_CAMPAIGN_SECRETS_ENV:-}"
TOGETHER_SECRET_PROJECT="${TOGETHER_SECRET_PROJECT:-hai-gcp-models}"
COREWEAVE_SECRET_PROJECT="${COREWEAVE_SECRET_PROJECT:-hai-gcp-models}"

FEDERATED_CLUSTER="${FEDERATED_CLUSTER:-cw-rno2a}"
PRIORITY="${PRIORITY:-interactive}"

CAMPAIGN_SHA_MARIN=$(git -C "$MARIN_DIR" rev-parse HEAD)
eval "$(cd "$MARIN_DIR" && uv run python - <<'PY'
import shlex

from rigging.provenance import launch_provenance

print(f"CAMPAIGN_SOURCE_TREE={shlex.quote(launch_provenance().dedup_key)}")
PY
)"
export CAMPAIGN_SHA_MARIN CAMPAIGN_SOURCE_TREE

die() {
  echo "error: $*" >&2
  exit 1
}

reset_staging_root() {
  local expected="$MARIN_DIR/experiments/evaluation/campaigns/.launch-staging/eval-campaign-09-25-unlabeled"
  [ "$STAGING_ROOT" = "$expected" ] || die "refusing to clear unexpected staging root: $STAGING_ROOT"
  rm -rf -- "$STAGING_ROOT"
}

validate_marin_checkout() {
  git -C "$MARIN_DIR" rev-parse --is-inside-work-tree >/dev/null 2>&1 \
    || die "MARIN_DIR is not a git worktree: $MARIN_DIR"

  local unexpected
  unexpected=$(git -C "$MARIN_DIR" status --porcelain \
    | grep -vF '?? experiments/evaluation/campaigns/' || true)
  [ -z "$unexpected" ] || die "unexpected changes in campaign worktree:\n$unexpected"
}

resolve_together_api_key() {
  [ -n "${TOGETHER_API_KEY:-}" ] && return

  local resolved=""
  if [ -n "$EVAL_CAMPAIGN_SECRETS_ENV" ] && [ -r "$EVAL_CAMPAIGN_SECRETS_ENV" ]; then
    resolved=$(
      set +u
      source "$EVAL_CAMPAIGN_SECRETS_ENV" >/dev/null 2>&1
      printf '%s' "${TOGETHER_API_KEY:-}"
    )
  fi
  if [ -z "$resolved" ] && command -v gcloud >/dev/null 2>&1; then
    resolved=$(gcloud secrets versions access latest \
      --secret=TOGETHER_API_KEY \
      --project="$TOGETHER_SECRET_PROJECT" 2>/dev/null || true)
  fi
  [ -n "$resolved" ] || die "TOGETHER_API_KEY is unavailable"
  export TOGETHER_API_KEY="$resolved"
}

resolve_hf_token() {
  [ -n "${HF_TOKEN:-}" ] && return

  local resolved=""
  if [ -n "$EVAL_CAMPAIGN_SECRETS_ENV" ] && [ -r "$EVAL_CAMPAIGN_SECRETS_ENV" ]; then
    resolved=$(
      set +u
      source "$EVAL_CAMPAIGN_SECRETS_ENV" >/dev/null 2>&1
      printf '%s' "${HF_TOKEN:-${HUGGING_FACE_HUB_TOKEN:-}}"
    )
  fi

  local cached_token_file
  cached_token_file="$(python3 - <<'PY'
from pathlib import Path

print(Path.home() / ".cache" / "huggingface" / "token")
PY
)"
  if [ -z "$resolved" ] && [ -r "$cached_token_file" ]; then
    resolved=$(<"$cached_token_file")
  fi

  [ -n "$resolved" ] || die "HF_TOKEN is unavailable; export it or run hf auth login"
  export HF_TOKEN="$resolved"
}

resolve_coreweave_credentials() {
  if [ -z "${CW_KEY_ID:-}" ] || [ -z "${CW_KEY_SECRET:-}" ]; then
    command -v gcloud >/dev/null 2>&1 || die "gcloud is required to resolve CoreWeave object-storage credentials"
    CW_KEY_ID=$(gcloud secrets versions access latest \
      --secret=cw-object-storage-key-id \
      --project="$COREWEAVE_SECRET_PROJECT")
    CW_KEY_SECRET=$(gcloud secrets versions access latest \
      --secret=cw-object-storage-key-secret \
      --project="$COREWEAVE_SECRET_PROJECT")
  fi
  [ -n "$CW_KEY_ID" ] && [ -n "$CW_KEY_SECRET" ] \
    || die "CoreWeave object-storage credentials are unavailable"
  export CW_KEY_ID CW_KEY_SECRET

  # Local fsutil subprocesses use fsspec's generic S3 environment rather than
  # Iris's namespaced task injection. Configure both forms so reads and writes
  # reach CoreWeave's endpoint instead of AWS S3.
  export AWS_ACCESS_KEY_ID="${AWS_ACCESS_KEY_ID:-$CW_KEY_ID}"
  export AWS_SECRET_ACCESS_KEY="${AWS_SECRET_ACCESS_KEY:-$CW_KEY_SECRET}"
  export CW_S3_ENDPOINT="${CW_S3_ENDPOINT:-https://cwobject.com}"
  export AWS_ENDPOINT_URL="${AWS_ENDPOINT_URL:-$CW_S3_ENDPOINT}"
  export AWS_REGION="${AWS_REGION:-auto}"
  export AWS_DEFAULT_REGION="${AWS_DEFAULT_REGION:-auto}"
  if [ -z "${FSSPEC_S3:-}" ]; then
    FSSPEC_S3=$(uv run --project "$MARIN_DIR" python -c \
      'import json, os; from rigging.filesystem.s3_compat import fsspec_s3_conf; print(json.dumps(fsspec_s3_conf(os.environ["CW_S3_ENDPOINT"])))')
    export FSSPEC_S3
  fi
}

prepare_credentials() {
  resolve_hf_token
  resolve_together_api_key
}

# Select each evaluation's protocol for a model whose window cannot hold the default one.
# The resulting effective config is snapshotted at launch.
stage_evaluation_config() {
  local model=$1
  STAGED_EVALUATION_CONFIG="$STAGING_ROOT/$model/evaluations.yaml"
  export STAGED_EVALUATION_CONFIG
  mkdir -p "$(dirname "$STAGED_EVALUATION_CONFIG")"
  CAMPAIGN_CONFIG="$CAMPAIGN_CONFIG" EVALUATION_CONFIG="$EVALUATION_CONFIG" MODEL_NAME="$model" \
  STAGED_EVALUATION_CONFIG="$STAGED_EVALUATION_CONFIG" \
    uv run --project "$MARIN_DIR" python - <<'PY'
import os
from pathlib import Path

import yaml

campaign = yaml.safe_load(Path(os.environ["CAMPAIGN_CONFIG"]).read_text())
document = yaml.safe_load(Path(os.environ["EVALUATION_CONFIG"]).read_text())
protocols = (campaign.get("model_evaluation_protocols") or {}).get(os.environ["MODEL_NAME"], {})
for name, protocol in protocols.items():
    document["evaluations"][name]["protocol"] = protocol
Path(os.environ["STAGED_EVALUATION_CONFIG"]).write_text(yaml.safe_dump(document, sort_keys=False))
PY
}

# A model-specific canonical chat template may be composed into the effective model config.
# All other serving and agent settings are copied unchanged from the canonical model YAML.
stage_model_config() {
  local model=$1
  local source=${2:-$MODEL_CONFIG_DIR/$model.yaml}
  STAGED_MODEL_CONFIG="$STAGING_ROOT/$model/model.yaml"
  export STAGED_MODEL_CONFIG

  (cd "$MARIN_DIR" && \
  SOURCE_MODEL_CONFIG="$source" \
  STAGED_MODEL_CONFIG="$STAGED_MODEL_CONFIG" \
  CHAT_TEMPLATE_DIR="$CHAT_TEMPLATE_DIR" \
  MODEL_NAME="$model" \
  uv run python - <<'PY'
import os
from pathlib import Path

import yaml

source = Path(os.environ["SOURCE_MODEL_CONFIG"])
destination = Path(os.environ["STAGED_MODEL_CONFIG"])
document = yaml.safe_load(source.read_text())
chat_template = Path(os.environ["CHAT_TEMPLATE_DIR"]) / f"{os.environ['MODEL_NAME']}.jinja"
if chat_template.exists():
    document.setdefault("serve", {})["chat_template"] = chat_template.read_text()
destination.parent.mkdir(parents=True, exist_ok=True)
destination.write_text(yaml.safe_dump(document, sort_keys=False))
PY
  )
}

validate_campaign_configs() {
  (cd "$MARIN_DIR" && \
  MODEL_CONFIG_DIR="$MODEL_CONFIG_DIR" \
  CHAT_TEMPLATE_DIR="$CHAT_TEMPLATE_DIR" \
  EVALUATION_CONFIG="$EVALUATION_CONFIG" \
  CAMPAIGN_CONFIG="$CAMPAIGN_CONFIG" \
  uv run python - <<'PY'
import os
import re
from pathlib import Path

import yaml

from experiments.evaluation.curated import load_evaluations, corpus_steps, resolve_selection
from experiments.post_training.task_curation.campaign import CampaignRuntime
from marin.evaluation.model_config import load_model_config

models = Path(os.environ["MODEL_CONFIG_DIR"])
chat_templates = Path(os.environ["CHAT_TEMPLATE_DIR"])
evaluations = Path(os.environ["EVALUATION_CONFIG"])
campaign = yaml.safe_load(Path(os.environ["CAMPAIGN_CONFIG"]).read_text())

model_documents = {
    path.stem: yaml.safe_load(path.read_text())
    for path in models.glob("*.yaml")
}
native_32k_models = {
    name
    for name, document in model_documents.items()
    if document.get("serve", {}).get("max_model_len") == 32768
}
model_evaluation_protocols = campaign.get("model_evaluation_protocols") or {}
assert model_evaluation_protocols == {name: {"bixbench-pi": "pi-32k"} for name in native_32k_models}
assert {path.stem for path in chat_templates.glob("*.jinja")} <= {path.stem for path in models.glob("*.yaml")}
for path in models.glob("*.yaml"):
    document = yaml.safe_load(path.read_text())
    expected_thinking_format = (
        "qwen-chat-template"
        if path.stem in {"Qwen-Qwen3.6-35B-A3B", "Qwen-Qwen3-Next-80B-A3B-Instruct"}
        else "chat-template"
    )
    actual_thinking_format = (
        document.get("agent", {}).get("agent_kwargs", {}).get("thinking_format")
    )
    assert actual_thinking_format == expected_thinking_format, (
        path,
        actual_thinking_format,
        expected_thinking_format,
    )
selections = {selection.name: selection for selection in load_evaluations(evaluations)}
for selection in selections.values():
    corpus_steps(resolve_selection(selection, None), CampaignRuntime())
for protocols in model_evaluation_protocols.values():
    for name, protocol in protocols.items():
        corpus_steps(
            resolve_selection(selections[name].model_copy(update={"protocol": protocol}), None),
            CampaignRuntime(),
        )

fallbacks = {
    "openai/gpt-oss-20b": {"reasoning_effort": "low"},
    "IFM/K2-Horizon-MoVA-36B-A4B": {"reasoning_effort": "low"},
    "Aleph-Alpha/Kolibri-1": {"reasoning_effort": "none"},
    "zai-org/GLM-4.7-Flash": {"enable_thinking": False},
}
for model_path in models.glob("*.yaml"):
    model = load_model_config(model_path)
    assert dict(model.generation.thinking_off_template_kwargs) == fallbacks.get(model.location, {}), model.location
for path in models.glob("*.yaml"):
    revision = yaml.safe_load(path.read_text()).get("revision", "")
    assert re.fullmatch(r"[0-9a-f]{40}", revision), f"unpinned model revision in {path}"

nupa = yaml.safe_load(evaluations.read_text())["evaluations"]["nupa"]
assert nupa["benchmark"] == "nupa5k-loose"
assert "max_tokens" not in nupa.get("run", {})
PY
  )

}

snapshot_launch_configs() {
  mkdir -p "$CONFIG_SNAPSHOT_DIR"
  CAMPAIGN_SHA_MARIN="$CAMPAIGN_SHA_MARIN" \
  CAMPAIGN_SOURCE_TREE="$CAMPAIGN_SOURCE_TREE" \
  SNAPSHOT_DIR="$CONFIG_SNAPSHOT_DIR" \
  uv run --project "$MARIN_DIR" python - "$@" <<'PY'
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

import yaml

config_flags = {"--model-config", "--evaluation-config", "--policy-finalization-config"}
arguments = sys.argv[1:]
files = []
for index, argument in enumerate(arguments[:-1]):
    if argument in config_flags:
        files.append((argument.removeprefix("--"), Path(arguments[index + 1]).resolve()))
        if argument == "--policy-finalization-config":
            path = Path(arguments[index + 1]).resolve()
            config = yaml.safe_load(path.read_text())
            files.extend((kind, path.parent / config[kind]) for kind in ("judge_model_config",))

payload = {
    "argv": arguments,
    "pins": {
        "marin": os.environ["CAMPAIGN_SHA_MARIN"],
        "source_tree": os.environ["CAMPAIGN_SOURCE_TREE"],
    },
    "files": [
        {"kind": kind, "source": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        for kind, path in files
    ],
}
snapshot_id = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
root = Path(os.environ["SNAPSHOT_DIR"]) / snapshot_id
root.mkdir(parents=True, exist_ok=True)
(root / "launch.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
for index, (kind, path) in enumerate(files):
    shutil.copyfile(path, root / f"{index:02d}-{kind}-{path.name}")
print(snapshot_id)
PY
}

run_campaign_launch() {
  local submit=$1
  shift
  local snapshot_id
  snapshot_id=$(snapshot_launch_configs "$@")
  echo "config snapshot: $snapshot_id" >&2

  local flags=()
  if [ "$submit" != "1" ]; then
    flags+=(--dry-run)
  elif [ "${WAIT_FOR_RESULTS:-0}" != "1" ]; then
    flags+=(--no-wait)
  fi

  if [ "${#flags[@]}" -gt 0 ]; then
    (cd "$MARIN_DIR" && uv run python -m experiments.evaluation.cli launch "$@" "${flags[@]}")
  else
    (cd "$MARIN_DIR" && uv run python -m experiments.evaluation.cli launch "$@")
  fi
}
