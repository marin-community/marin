#!/usr/bin/env bash
# Shared helpers for the unlabeled 2026-09-25 evaluation campaign. Source this file.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MARIN_DIR="${MARIN_DIR:-$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)}"
CANONICAL_CONFIG_ROOT="$SCRIPT_DIR"
CAMPAIGN_CONFIG="$CANONICAL_CONFIG_ROOT/campaign.yaml"
MODEL_CONFIG_DIR="$CANONICAL_CONFIG_ROOT/model-configs"
CHAT_TEMPLATE_DIR="$CANONICAL_CONFIG_ROOT/chat-templates"
EVALCHEMY_CONFIG_DIR="$CANONICAL_CONFIG_ROOT/evalchemy-configs"
HARBOR_CONFIG_DIR="$CANONICAL_CONFIG_ROOT/harbor-configs"
JUDGE_MODEL_CONFIG_DIR="$CANONICAL_CONFIG_ROOT/judge-model-configs"
JUDGE_MODEL_CONFIG="$JUDGE_MODEL_CONFIG_DIR/MiniMaxAI-MiniMax-M3-MXFP8.yaml"
CONFIG_SNAPSHOT_DIR="$MARIN_DIR/experiments/evaluation/campaigns/.launch-snapshots/eval-campaign-09-25-unlabeled"
STAGING_ROOT="$MARIN_DIR/experiments/evaluation/campaigns/.launch-staging/eval-campaign-09-25-unlabeled"
EVAL_CAMPAIGN_SECRETS_ENV="${EVAL_CAMPAIGN_SECRETS_ENV:-}"
TOGETHER_SECRET_PROJECT="${TOGETHER_SECRET_PROJECT:-hai-gcp-models}"
COREWEAVE_SECRET_PROJECT="${COREWEAVE_SECRET_PROJECT:-hai-gcp-models}"

FEDERATED_CLUSTER="${FEDERATED_CLUSTER:-cw-rno2a}"
PRIORITY="${PRIORITY:-interactive}"
JUDGE_MODEL="${JUDGE_MODEL:-openai/gpt-oss-120b}"
JUDGE_BASE_URL="${JUDGE_BASE_URL:-https://api.together.xyz/v1}"
TAU2_JUDGE_MODEL="${TAU2_JUDGE_MODEL:-openai/openai/gpt-oss-120b}"

CAMPAIGN_SHA_MARIN=$(git -C "$MARIN_DIR" rev-parse HEAD)
eval "$(python3 - "$MARIN_DIR/lib/marin/src/marin/external_dependencies.py" <<'PY'
import re
import shlex
import sys
from pathlib import Path

text = Path(sys.argv[1]).read_text()
for name in ("EVALCHEMY", "HARBOR"):
    block = re.search(rf"{name} = ExternalDependency\((.*?)\n\)", text, re.DOTALL)
    if block is None:
        raise SystemExit(f"missing {name} dependency block")
    commit = re.search(r'commit="([0-9a-f]{40})"', block.group(1))
    if commit is None:
        raise SystemExit(f"missing {name} commit")
    print(f"CAMPAIGN_SHA_{name}={shlex.quote(commit.group(1))}")
PY
)"
export CAMPAIGN_SHA_MARIN CAMPAIGN_SHA_HARBOR CAMPAIGN_SHA_EVALCHEMY

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

  local pins
  pins=$(python3 - \
    "$MARIN_DIR/config/external/harbor/uv.lock" <<'PY'
import re
import sys
from pathlib import Path

lock = Path(sys.argv[1]).read_text()
for name in ("harbor", "harbor-tau3-bench-adapter"):
    block = re.search(rf'\[\[package\]\]\nname = "{re.escape(name)}"\n(.*?)(?=\n\[\[package\]\]|\Z)', lock, re.DOTALL)
    if block is None:
        raise SystemExit(f"missing Harbor runtime package in lock: {name}")
    match = re.search(r'source = \{ git = "[^"]*#([0-9a-f]{40})" \}', block.group(1))
    if match is None:
        raise SystemExit(f"missing immutable Harbor runtime commit for {name}")
    print(f"HARBOR_LOCK={match.group(1)}")
PY
  )
  [ "$(grep -cx "HARBOR_LOCK=$CAMPAIGN_SHA_HARBOR" <<<"$pins")" = 2 ] \
    || die "locked Harbor runtimes differ from $CAMPAIGN_SHA_HARBOR"

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

prepare_judge_environment() {
  resolve_hf_token
  resolve_together_api_key
  export JUDGE_API_KEY="$TOGETHER_API_KEY"
  export JUDGE_BASE_URL JUDGE_MODEL
  export OPENAI_API_KEY="$TOGETHER_API_KEY" OPENAI_BASE_URL="$JUDGE_BASE_URL"
  export TAU2_USER_MODEL="$TAU2_JUDGE_MODEL"
  export TAU2_NL_ASSERTIONS_MODEL="$TAU2_JUDGE_MODEL"
}

# Stage canonical Harbor configs, applying only model-specific policy values from
# the campaign config. The resulting effective configs are snapshotted at launch.
stage_harbor_configs() {
  local model=$1
  STAGED_HARBOR_DIR="$STAGING_ROOT/$model/harbor"
  export STAGED_HARBOR_DIR
  mkdir -p "$STAGED_HARBOR_DIR"
  cp -R "$HARBOR_CONFIG_DIR/." "$STAGED_HARBOR_DIR/"
  CAMPAIGN_CONFIG="$CAMPAIGN_CONFIG" MODEL_NAME="$model" STAGED_HARBOR_DIR="$STAGED_HARBOR_DIR" \
    uv run --project "$MARIN_DIR" python - <<'PY'
import os
from pathlib import Path

import yaml

campaign = yaml.safe_load(Path(os.environ["CAMPAIGN_CONFIG"]).read_text())
model_info = (campaign.get("harbor_model_info") or {}).get(os.environ["MODEL_NAME"])
profiles = campaign.get("harbor_harness_profiles") or {}
profile_by_eval = campaign.get("harbor_eval_harness_profiles") or {}
for path in Path(os.environ["STAGED_HARBOR_DIR"]).glob("*.yaml"):
    document = yaml.safe_load(path.read_text())
    changed = False
    profile_name = profile_by_eval.get(path.stem)
    profile = profiles.get(profile_name) if profile_name else None
    for agent in document.get("agents") or []:
        configured_kwargs = agent.get("kwargs") or {}
        configured_model_info = configured_kwargs.get("model_info")
        if model_info and configured_model_info:
            configured_model_info.update(model_info)
            changed = True
        if profile and agent.get("name") == profile["agent"]:
            configured_kwargs.update(profile.get("kwargs") or {})
            configured_kwargs.update(
                (profile.get("model_kwargs") or {}).get(os.environ["MODEL_NAME"], {})
            )
            agent["kwargs"] = configured_kwargs
            changed = True
    if changed:
        path.write_text(yaml.safe_dump(document, sort_keys=False))
PY
}

# A model-specific canonical chat template may be composed into the effective model config.
# All other serving and agent settings are copied unchanged from the canonical model YAML.
stage_model_config() {
  local model=$1
  local source="$MODEL_CONFIG_DIR/$model.yaml"
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
  EVALCHEMY_CONFIG_DIR="$EVALCHEMY_CONFIG_DIR" \
  CANONICAL_EVALCHEMY_CONFIG_DIR="$MARIN_DIR/experiments/evaluation/configs/evalchemy" \
  HARBOR_CONFIG_DIR="$HARBOR_CONFIG_DIR" \
  CAMPAIGN_CONFIG="$CAMPAIGN_CONFIG" \
  JUDGE_MODEL_CONFIG="$JUDGE_MODEL_CONFIG" \
  CAMPAIGN_SHA_HARBOR="$CAMPAIGN_SHA_HARBOR" \
  uv run python - <<'PY'
import os
import re
from pathlib import Path

import yaml

from experiments.evaluation.evals import EvalchemyDefinition
from marin.evaluation.evalchemy.config import load_evalchemy_config
from marin.evaluation.model_config import load_model_config

models = Path(os.environ["MODEL_CONFIG_DIR"])
chat_templates = Path(os.environ["CHAT_TEMPLATE_DIR"])
evalchemy = Path(os.environ["EVALCHEMY_CONFIG_DIR"])
canonical_evalchemy = Path(os.environ["CANONICAL_EVALCHEMY_CONFIG_DIR"])
harbor = Path(os.environ["HARBOR_CONFIG_DIR"])
campaign_config = Path(os.environ["CAMPAIGN_CONFIG"])

model_documents = {
    path.stem: yaml.safe_load(path.read_text())
    for path in models.glob("*.yaml")
}
campaign = yaml.safe_load(campaign_config.read_text())
harbor_model_info = campaign.get("harbor_model_info") or {}
harbor_harness_profiles = campaign.get("harbor_harness_profiles") or {}
harbor_eval_harness_profiles = campaign.get("harbor_eval_harness_profiles") or {}
native_32k_models = {
    name
    for name, document in model_documents.items()
    if document.get("serve", {}).get("max_model_len") == 32768
}
assert set(harbor_model_info) == native_32k_models
for name in native_32k_models:
    assert harbor_model_info[name] == {"max_input_tokens": 32768}
assert harbor_eval_harness_profiles == {"bixbench-pi": "pi-65k-16k"}
expected_model_kwargs = {
    name: {"compaction_reserve_tokens": 16384}
    for name in native_32k_models
}
assert harbor_harness_profiles == {
    "pi-65k-16k": {
        "agent": "pi",
        "kwargs": {"compaction_reserve_tokens": 32768},
        "model_kwargs": expected_model_kwargs,
    }
}
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
mrcr_variants = {"mrcr-32k": 32704, "mrcr-65k": 65472}
evalchemy_names = {path.stem for path in evalchemy.glob("*.yaml")}
assert len(evalchemy_names) == 20
assert len(list(harbor.glob("*.yaml"))) == 8
sotopia = yaml.safe_load((harbor / "sotopia-hard.yaml").read_text())
assert sotopia["datasets"] == [
    {
        "name": "hf://open-athena/sotopia-hard-harbor",
        "ref": "65be6de928ab289e43a2730182b5b75cf2f75d41",
    }
]

thinking_on = {"aime24", "math500", "olympiadbench", "mmlu-pro", "gpqa-diamond"}
thinking_off = {
    "humanevalplus",
    "mbppplus",
    "gsm8k-0shot",
    "triviaqa",
    "cruxeval",
    "financebench",
    "ifbench",
    "mrcr",
}
model_default = {"nupa"}
not_applicable = {"piqa", "winogrande", "boolq", "truthfulqa"}
assert thinking_on | thinking_off | model_default | not_applicable == evalchemy_names - set(mrcr_variants)
mrcr = yaml.safe_load((evalchemy / "mrcr.yaml").read_text())
for name, max_length in mrcr_variants.items():
    expected = dict(mrcr, max_length=max_length)
    assert yaml.safe_load((evalchemy / f"{name}.yaml").read_text()) == expected
for name in thinking_on | thinking_off:
    expected = {"enable_thinking": name in thinking_on}
    artifact_document = yaml.safe_load((evalchemy / f"{name}.yaml").read_text())
    canonical_document = yaml.safe_load((canonical_evalchemy / f"{name}.yaml").read_text())
    assert artifact_document.get("chat_template_kwargs") == expected, (name, artifact_document)
    assert canonical_document.get("chat_template_kwargs") == expected, (name, canonical_document)

# Exercise the same model/config merge used by the launcher. The benchmark setting must win over
# model defaults for every tracked model, not merely be present in the source YAML.
for model_path in models.glob("*.yaml"):
    model = load_model_config(model_path)
    if model.location == "open-athena/Snowball-67B-A2B-5.7T-Mixed-RLVR-Step38":
        assert model.generation.max_gen_toks == 16384, model.generation
    for name in thinking_on | thinking_off:
        config_path = evalchemy / f"{name}.yaml"
        source = load_evalchemy_config(config_path)
        resolved = EvalchemyDefinition(name=name, config_path=config_path).config_for(source, model, None)
        expected = {"enable_thinking": name in thinking_on}
        if name in thinking_off and model.generation.thinking_off_template_kwargs:
            expected = dict(model.generation.thinking_off_template_kwargs)
        assert resolved.chat_template_kwargs == expected, (
            model.location,
            name,
            resolved.chat_template_kwargs,
        )
    nupa_path = evalchemy / "nupa.yaml"
    nupa_source = load_evalchemy_config(nupa_path)
    nupa = EvalchemyDefinition(name="nupa", config_path=nupa_path).config_for(nupa_source, model, None)
    assert "enable_thinking" not in nupa.chat_template_kwargs, (model.location, nupa.chat_template_kwargs)
fallbacks = {
    "openai/gpt-oss-20b": {"reasoning_effort": "low"},
    "IFM/K2-Horizon-MoVA-36B-A4B": {"reasoning_effort": "low"},
}
for model_path in models.glob("*.yaml"):
    model = load_model_config(model_path)
    assert dict(model.generation.thinking_off_template_kwargs) == fallbacks.get(model.location, {}), model.location
for name in model_default | not_applicable:
    artifact_document = yaml.safe_load((evalchemy / f"{name}.yaml").read_text())
    assert "chat_template_kwargs" not in artifact_document, (name, artifact_document)
    canonical_path = canonical_evalchemy / f"{name}.yaml"
    if canonical_path.exists():
        canonical_document = yaml.safe_load(canonical_path.read_text())
        assert "chat_template_kwargs" not in canonical_document, (name, canonical_document)

for path in models.glob("*.yaml"):
    revision = yaml.safe_load(path.read_text()).get("revision", "")
    assert re.fullmatch(r"[0-9a-f]{40}", revision), f"unpinned model revision in {path}"

judge = load_model_config(Path(os.environ["JUDGE_MODEL_CONFIG"]))
assert judge.location == "MiniMaxAI/MiniMax-M3-MXFP8"
assert re.fullmatch(r"[0-9a-f]{40}", judge.revision or "")
assert judge.serve.tensor_parallel_size == 8
assert judge.serve.object_store_load_mode.value == "stage_local"
assert "--enforce-eager" not in judge.serve.vllm_extra_args
assert "--model-loader-extra-config" not in judge.serve.vllm_extra_args

for name in ("aime24", "olympiadbench", "gpqa-diamond"):
    document = yaml.safe_load((evalchemy / f"{name}.yaml").read_text())
    assert document.get("seed") == 42, f"{name} must use explicit seed 42"
    assert document.get("gen_kwargs") == "do_sample=true,temperature=0.7", (
        f"{name} must use seeded temperature sampling"
    )
assert yaml.safe_load((evalchemy / "nupa.yaml").read_text())["tasks"] == ["NUPA5K-Loose"]
nupa = yaml.safe_load((evalchemy / "nupa.yaml").read_text())
assert nupa["max_length"] == 73664
assert "max_tokens" not in nupa

terminal_bench = yaml.safe_load((harbor / "tb2-recovery.yaml").read_text())
assert terminal_bench["n_attempts"] == 3
assert terminal_bench["datasets"] == [{"name": "terminal-bench/terminal-bench-2-1", "ref": "6"}]
assert terminal_bench["agents"][0]["kwargs"]["model_info"] == {
    "max_input_tokens": 65536,
    "max_output_tokens": 16384,
}
bixbench = yaml.safe_load((harbor / "bixbench-pi.yaml").read_text())
assert "compaction_reserve_tokens" not in bixbench["agents"][0]["kwargs"]
assert bixbench["agents"][0]["kwargs"]["model_info"] == {
    "max_input_tokens": 65536,
    "max_output_tokens": 16384,
}
bfcl = yaml.safe_load((harbor / "bfclparity-pi.yaml").read_text())
assert bfcl["agents"][0]["version"] == "0.87.0"
openthoughts = yaml.safe_load((harbor / "ot-tblite-recovery.yaml").read_text())
assert openthoughts["n_attempts"] == 3
swebench = yaml.safe_load((harbor / "swebench-verified.yaml").read_text())
assert swebench["n_attempts"] == 1
assert swebench["agents"][0]["name"] == "mini-swe-agent"
assert swebench["agents"][0]["kwargs"]["version"] == "2.1.0"
assert swebench["datasets"] == [
    {
        "name": "swebench-verified",
        "version": "1.0",
        "registry_url": f"https://raw.githubusercontent.com/marin-community/harbor/{os.environ['CAMPAIGN_SHA_HARBOR']}/registry.json",
    }
]
for name in (
    "swebench-verified",
    "ot-tblite-recovery",
    "ds-1000-local",
    "bfclparity-pi",
    "tau3-pi",
    "sotopia-hard",
):
    document = yaml.safe_load((harbor / f"{name}.yaml").read_text())
    assert document["agents"][0]["kwargs"]["model_info"]["max_input_tokens"] == 32768

registry_prefix = f"https://raw.githubusercontent.com/marin-community/harbor/{os.environ['CAMPAIGN_SHA_HARBOR']}/"
for path in harbor.glob("*.yaml"):
    document = yaml.safe_load(path.read_text())
    for agent in document.get("agents") or []:
        assert agent.get("override_setup_timeout_sec") == 360, path
        assert agent.get("override_timeout_sec") == 1800, path
        assert "thinking_format" not in (agent.get("kwargs") or {}), path
    assert (document.get("verifier") or {}).get("override_timeout_sec") == 300, path
    retry = document.get("retry") or {}
    assert retry.get("exclude_exceptions") == [], path
    included = set(retry.get("include_exceptions") or [])
    assert {"AgentSetupTimeoutError", "VerifierTimeoutError"} <= included, path
    for dataset in document.get("datasets") or []:
        registry_url = dataset.get("registry_url")
        assert not registry_url or registry_url.startswith(registry_prefix), path
PY
  )

  grep -q '_REQUEST_TIMEOUT = 1800' "$MARIN_DIR/lib/marin/src/marin/evaluation/evalchemy/client.py" \
    || die "pinned Marin does not enforce the 30-minute Evalchemy request timeout"
  grep -q '_TRANSPORT_RETRY_BUDGET = 1800' "$MARIN_DIR/lib/marin/src/marin/evaluation/evalchemy/client.py" \
    || die "pinned Marin does not bound Evalchemy transport retries to 30 minutes"
  grep -q '_TRANSPORT_ATTEMPT_TIMEOUT = 300' "$MARIN_DIR/lib/marin/src/marin/evaluation/evalchemy/client.py" \
    || die "pinned Marin does not retry stalled Evalchemy requests within the policy window"
}

snapshot_launch_configs() {
  mkdir -p "$CONFIG_SNAPSHOT_DIR"
  CAMPAIGN_SHA_MARIN="$CAMPAIGN_SHA_MARIN" \
  CAMPAIGN_SHA_HARBOR="$CAMPAIGN_SHA_HARBOR" \
  CAMPAIGN_SHA_EVALCHEMY="$CAMPAIGN_SHA_EVALCHEMY" \
  SNAPSHOT_DIR="$CONFIG_SNAPSHOT_DIR" \
  python3 - "$@" <<'PY'
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

config_flags = {"--model-config", "--judge-model-config", "--harbor-config", "--evalchemy-config"}
arguments = sys.argv[1:]
files = []
for index, argument in enumerate(arguments[:-1]):
    if argument in config_flags:
        files.append((argument.removeprefix("--"), Path(arguments[index + 1]).resolve()))

payload = {
    "argv": arguments,
    "pins": {
        "marin": os.environ["CAMPAIGN_SHA_MARIN"],
        "harbor": os.environ["CAMPAIGN_SHA_HARBOR"],
        "evalchemy": os.environ["CAMPAIGN_SHA_EVALCHEMY"],
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
