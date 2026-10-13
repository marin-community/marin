"""Prepare and launch isolated Shared100 chat-template ablations."""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import yaml
from huggingface_hub import hf_hub_download


ROOT = Path(__file__).resolve().parent
CAMPAIGN = ROOT.parent.parent
ARTIFACTS = Path("/Users/benfeuer/Documents/experiments/active/eval-policy-09-25/artifacts")
OUTPUT = ARTIFACTS / "shared100-chat-template-ablation-20261013"
SHARED_MODEL = "open-athena/Grug-67B-A2B-Merge-NUPA-BFCL-Shared100-2026.10.09"


def pinned_file(repository: str, revision: str, filename: str, digest: str) -> bytes:
    path = Path(hf_hub_download(repository, filename, revision=revision))
    data = path.read_bytes()
    actual = hashlib.sha256(data).hexdigest()
    if actual != digest:
        raise ValueError(f"{repository}@{revision}/{filename}: SHA256 {actual} != {digest}")
    return data


def reasoning_only_template(template: str) -> str:
    think = (
        "{%- if message.get('reasoning_content') -%}"
        "{{- '<|start_think|>' ~ message.reasoning_content ~ '<|end_think|>' -}}"
        "{%- endif -%}"
    )
    plain = "{% generation %}{{- content | trim }}<|eot_id|>{% endgeneration %}"
    tool = "{% generation %}\n{%- if message.get('content') is string"
    if template.count(plain) != 1 or template.count(tool) != 1:
        raise ValueError("Shared100 template does not match the pinned reasoning-only patch anchors")
    template = template.replace(plain, "{% generation %}" + think + "{{- content | trim }}<|eot_id|>{% endgeneration %}")
    return template.replace(tool, "{% generation %}\n" + think + "\n{%- if message.get('content') is string")


def prepare(arm: str) -> Path:
    manifest = yaml.safe_load((ROOT / "variants.yaml").read_text())
    source = (ROOT / manifest["base_model_config"]).resolve()
    model = yaml.safe_load(source.read_text())
    if model["location"] != SHARED_MODEL or model["revision"] != manifest["base_model_revision"]:
        raise ValueError("base model config differs from pinned Shared100 revision")
    shared = pinned_file(SHARED_MODEL, model["revision"], "chat_template.jinja", manifest["shared_template_sha256"])
    pinned_file(SHARED_MODEL, model["revision"], "tokenizer.json", manifest["shared_tokenizer_sha256"])
    antidoom = pinned_file(
        manifest["antidoom_model"],
        manifest["antidoom_revision"],
        "chat_template.jinja",
        manifest["antidoom_template_sha256"],
    )
    pinned_file(
        manifest["antidoom_model"],
        manifest["antidoom_revision"],
        "tokenizer.json",
        manifest["antidoom_tokenizer_sha256"],
    )
    for repository, revision in ((SHARED_MODEL, model["revision"]), (manifest["antidoom_model"], manifest["antidoom_revision"])):
        pinned_file(
            repository,
            revision,
            "tokenizer_config.json",
            manifest["shared_and_antidoom_tokenizer_config_sha256"],
        )

    if arm == "reasoning_only":
        template = reasoning_only_template(shared.decode())
    else:
        template = antidoom.decode()
    model["name"] = f"shared100-chat-template-{arm}"
    model["serve"]["chat_template"] = template
    if arm == "template_and_tokenizer":
        model["tokenizer"] = manifest["antidoom_model"]
        model["tokenizer_revision"] = manifest["antidoom_revision"]

    output = OUTPUT / arm
    output.mkdir(parents=True, exist_ok=True)
    config = output / "effective-model.yaml"
    config.write_text(yaml.safe_dump(model, sort_keys=False))
    (output / "effective-chat-template.jinja").write_text(template)
    provenance = {
        "arm": arm,
        "base_model_config": str(source),
        "base_model_config_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "template_sha256": hashlib.sha256(template.encode()).hexdigest(),
        "effective_model_config_sha256": hashlib.sha256(config.read_bytes()).hexdigest(),
        "sources": manifest,
    }
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n")
    return config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("arm", choices=("template_only", "template_and_tokenizer", "reasoning_only"))
    parser.add_argument("--submit", action="store_true")
    args = parser.parse_args()

    config = prepare(args.arm)
    command = [
        "uv",
        "run",
        "python",
        "-m",
        "experiments.evaluation.cli",
        "launch",
        "--model-config",
        str(config),
        "--evalchemy-config",
        str(CAMPAIGN / "evalchemy-configs/math500.yaml"),
        "--evalchemy-config",
        str(ROOT / "aime24-one-repeat.yaml"),
        "--harbor-config",
        str(CAMPAIGN / "harbor-configs/bfclparity-pi.yaml"),
        "--federated_cluster",
        "cw-rno2a",
        "--priority",
        "interactive",
        "--description",
        f"Unlabeled Shared100 chat-template ablation: {args.arm}",
        "--no-wait" if args.submit else "--dry-run",
    ]
    subprocess.run(command, check=True, cwd=Path(__file__).resolve().parents[6])


if __name__ == "__main__":
    main()
