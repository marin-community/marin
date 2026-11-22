# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Prepare regional Mochi prompts and an embedding-free EAGLE3 seed draft."""

import argparse
import gzip
import io
import json
import logging
import random
import tempfile
import zipfile
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

import pyarrow.parquet as pq
import torch
from rigging.filesystem.buckets import filesystem_for
from rigging.filesystem.storage_path import StoragePath
from rolloutengine.harness.assistant_prefix import assistant_prefix_prompt
from skyrl_gym.task_records import episode_prompt
from skyrl_train.io import io as skyrl_io
from skyrl_train.io.remote_safetensors import RemoteSafetensorsTensorStore
from speculators.config import SpeculatorsConfig, VerifierConfig
from speculators.models.eagle3 import Eagle3DraftModel, Eagle3SpeculatorConfig
from speculators.proposals.greedy import GreedyTokenProposalConfig
from taskcompendium.identity import canonical_sha256
from taskcompendium.models import TaskSpec
from transformers import AutoTokenizer, LlamaConfig

logger = logging.getLogger(__name__)
TARGET_URI = "s3://marin-us-east-08a/marin/models/mochi-sft/" "hero-sft-fullab-step393k-v2/step-400407/2af3f828aa-cpu-a1"
TARGET_IDENTITY = "mochi-long-loop/adopted-sft@2026.11.19.110:b8dde345"
TOKENIZER = "marin-community/marin-tokenizer-tools"
TOKENIZER_REVISION = "3f6879acdaf3dae7874b2d48c7d11b9d8c6faf09"
TRAIN_POOL = (
    "s3://marin-us-east-08a/marin/users/benfeuer/long-loop-exp4/long-loop-exp4/"
    "rl-pool-nonagentic/2026.11.18.71/train.parquet"
)
BOOTSTRAP_ARCHIVES = (
    "s3://marin-us-east-08a/tmp/ttl=14d/skyrl/marin/users/benfeuer/mochi-long-loop/"
    "long-loop/mochi-throughput-train-30m/2026.11.21.120/attempts/trajectories/"
    "schema_v7/archives/phase=train/step=00000001"
)
DRAFT_VOCAB_SIZE = 32000
MAX_PROMPT_TOKENS = 90112
AUXILIARY_LAYERS = (3, 24, 45)
AGENTIC_SOURCE = "long-loop-smartie-swesmith"


@dataclass(frozen=True)
class DraftPrompt:
    identity: str
    group: str
    route: str
    prompt_token_ids: list[int]


@dataclass(frozen=True)
class PromptPreparation:
    prompts: list[DraftPrompt]
    frequencies: Counter[int]
    receipt: dict


def bootstrap_records() -> list[dict]:
    filesystem, prefix = filesystem_for(BOOTSTRAP_ARCHIVES)
    records = []
    for archive in sorted(filesystem.glob(prefix + "/*.zip")):
        with filesystem.open(archive, "rb") as stream, zipfile.ZipFile(io.BytesIO(stream.read())) as bundle:
            for name in bundle.namelist():
                if not name.endswith(".json.gz"):
                    continue
                record = json.loads(gzip.decompress(bundle.read(name)))
                provenance = record["provenance"]
                if provenance["model_source_identity"] != TARGET_IDENTITY or provenance["model_version_step"] != 0:
                    continue
                if record["trajectory"]["environment_extras"]["data_source"] != AGENTIC_SOURCE:
                    continue
                records.append(record)
    if not records:
        raise ValueError("No exact-token continuations from the adopted Mochi checkpoint")
    return records


def prepare_bootstrap_prompts() -> PromptPreparation:
    """Separate exact Mochi vocabulary bootstrap tokens from agentic generation prompts."""
    records = bootstrap_records()
    grouped_records: dict[str, list[dict]] = {}
    for record in records:
        group = canonical_sha256({"messages": record["prompt"]["messages"]})
        grouped_records.setdefault(group, []).append(record)
    frequency_groups = {group for group in grouped_records if int(group[:8], 16) % 2 == 0}
    frequencies: Counter[int] = Counter()
    prompts: dict[str, DraftPrompt] = {}
    for group, group_records in grouped_records.items():
        for record in group_records:
            if group in frequency_groups:
                response = record["response"]
                frequencies.update(
                    token
                    for token, supervised in zip(response["token_ids"], response["loss_mask"], strict=True)
                    if supervised
                )
                continue
            tokens = record["prompt"]["token_ids"]
            if not tokens or len(tokens) > MAX_PROMPT_TOKENS:
                continue
            prompts.setdefault(group, DraftPrompt(group, group, "agentic", tokens))
    if not frequencies:
        raise ValueError("Bootstrap training groups have no supervised target tokens")
    return PromptPreparation(
        list(prompts.values()),
        frequencies,
        {
            "bootstrap_archives": BOOTSTRAP_ARCHIVES,
            "bootstrap_frequency_groups": len(frequency_groups),
            "bootstrap_supervised_tokens": frequencies.total(),
        },
    )


def prepare_prompts(nonagentic_count: int, seed: int) -> PromptPreparation:
    """Sample rendered generation prompts and retain the separate bootstrap provenance."""
    bootstrap = prepare_bootstrap_prompts()
    prompts = {row.group: row for row in bootstrap.prompts}
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER, revision=TOKENIZER_REVISION)
    filesystem, path = filesystem_for(TRAIN_POOL)
    generator = random.Random(seed)
    nonagentic = 0
    with filesystem.open(path, "rb") as stream:
        parquet = pq.ParquetFile(stream)
        groups = list(range(parquet.num_row_groups))
        generator.shuffle(groups)
        for row_group in groups:
            rows = parquet.read_row_group(row_group, columns=["task_spec", "teacher_route"]).to_pylist()
            generator.shuffle(rows)
            for row in rows:
                task = TaskSpec.model_validate_json(row["task_spec"])
                prompt = episode_prompt(task)
                if prompt.messages[-1]["role"] == "assistant":
                    prompt = assistant_prefix_prompt(task)
                request = {"messages": list(prompt.messages), **prompt.options}
                group_digest = canonical_sha256({"messages": request["messages"], "tools": request.get("tools")})
                if group_digest in prompts:
                    continue
                tokens = tokenizer.apply_chat_template(
                    request["messages"],
                    tools=request.get("tools"),
                    tokenize=True,
                    return_dict=False,
                    **request.get("extra_body", {"add_generation_prompt": True}),
                    enable_thinking=True,
                )
                if not tokens or len(tokens) > MAX_PROMPT_TOKENS:
                    continue
                prompts[group_digest] = DraftPrompt(group_digest, group_digest, row["teacher_route"], tokens)
                nonagentic += 1
                if nonagentic == nonagentic_count:
                    break
            if nonagentic == nonagentic_count:
                break
    if nonagentic != nonagentic_count:
        raise ValueError(f"Only {nonagentic} distinct nonagentic prompts available")
    rows = list(prompts.values())
    generator.shuffle(rows)
    receipt = {
        **bootstrap.receipt,
        "target": TARGET_URI,
        "target_identity": TARGET_IDENTITY,
        "tokenizer": TOKENIZER,
        "tokenizer_revision": TOKENIZER_REVISION,
        "generation_prompts": len(rows),
        "routes": dict(Counter(row.route for row in rows)),
        "nonagentic_pool": TRAIN_POOL,
        "seed": seed,
    }
    return PromptPreparation(rows, bootstrap.frequencies, receipt)


def worker_spec(description: str | None) -> dict:
    """Read an immutable regional worker specification with its bundled source revision."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--spec", required=True)
    parser.add_argument("--source-revision", required=True)
    args = parser.parse_args()
    spec = json.loads(StoragePath(args.spec).read_text())
    spec["source_revision"] = args.source_revision
    return spec


def initialize_draft(directory: Path, frequencies: Counter[int], seed: int) -> dict:
    torch.manual_seed(seed)
    metadata_dir = directory / "target-metadata"
    metadata_dir.mkdir()
    index_name = "model.safetensors.index.json"
    (metadata_dir / index_name).write_bytes((StoragePath(TARGET_URI) / index_name).read_bytes())
    target_config = json.loads((StoragePath(TARGET_URI) / "config.json").read_text())
    if (target_config["hidden_size"], target_config["num_hidden_layers"], target_config["vocab_size"]) != (
        6144,
        48,
        128256,
    ):
        raise ValueError("Mochi target geometry changed")
    layer = LlamaConfig(
        hidden_size=6144,
        intermediate_size=8192,
        num_hidden_layers=1,
        num_attention_heads=48,
        num_key_value_heads=12,
        head_dim=128,
        vocab_size=128256,
        max_position_embeddings=262144,
        rms_norm_eps=target_config["rms_norm_eps"],
        rope_parameters={"rope_type": "default", "rope_theta": target_config["rope_theta"]},
        sliding_window=2048,
        use_sliding_window=True,
        layer_types=["sliding_attention"],
    )
    layer._attn_implementation = "sdpa"
    config = Eagle3SpeculatorConfig(
        transformer_layer_config=layer,
        draft_vocab_size=DRAFT_VOCAB_SIZE,
        eagle_aux_hidden_state_layer_ids=list(AUXILIARY_LAYERS),
        norm_before_fc=True,
        norm_before_residual=True,
        norm_output=True,
        embed_requires_grad=False,
        dtype="bfloat16",
        speculators_config=SpeculatorsConfig(
            algorithm="eagle3",
            default_proposal_method="greedy",
            proposal_methods=[GreedyTokenProposalConfig(speculative_tokens=3)],
            verifier=VerifierConfig(architectures=["GrugMoeForCausalLM"], name_or_path=None),
        ),
    )
    previous_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        model = Eagle3DraftModel(config)
    finally:
        torch.set_default_dtype(previous_dtype)
    ranked = sorted(range(128256), key=lambda token: (-frequencies[token], token))[:DRAFT_VOCAB_SIZE]
    selected = torch.tensor(sorted(ranked), dtype=torch.long)
    target_to_draft = torch.zeros(128256, dtype=torch.bool)
    target_to_draft[selected] = True
    model.load_vocab_mappings(target_to_draft, selected - torch.arange(DRAFT_VOCAB_SIZE))
    store = RemoteSafetensorsTensorStore(TARGET_URI, metadata_dir)
    try:
        target_head = store.load_tensors(["lm_head.weight"])["lm_head.weight"]
        with torch.no_grad():
            model.lm_head.weight.copy_(target_head[selected])
    finally:
        store.close()
    for name, parameter in model.named_parameters():
        if parameter.requires_grad or name == "lm_head.weight":
            if not torch.isfinite(parameter).all():
                raise FloatingPointError(f"Non-finite seed tensor: {name}")
    model.save_pretrained(directory / "seed-draft", safe_serialization=True)
    return {
        "trainable_parameters": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
        "auxiliary_layers": AUXILIARY_LAYERS,
        "draft_vocab_size": DRAFT_VOCAB_SIZE,
        "bootstrap_vocabulary_coverage": sum(frequencies[token] for token in selected.tolist()) / frequencies.total(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--nonagentic-count", required=True, type=int)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--source-revision", required=True)
    args = parser.parse_args()
    if not args.output.startswith("s3://marin-us-east-08a/"):
        raise ValueError("Mochi draft preparation must publish into regional 08a S3")
    output = StoragePath(args.output)
    if (output / "prepared.json").exists():
        raise FileExistsError(f"Prepared output already exists: {output}")
    preparation = prepare_prompts(args.nonagentic_count, args.seed)
    receipt = preparation.receipt
    receipt["source_revision"] = args.source_revision
    with tempfile.TemporaryDirectory(prefix="mochi-eagle3-prepare-") as scratch:
        directory = Path(scratch)
        receipt.update(initialize_draft(directory, preparation.frequencies, args.seed))
        (output / "prompts.jsonl").write_text(
            "".join(json.dumps(asdict(prompt)) + "\n" for prompt in preparation.prompts)
        )
        skyrl_io.upload_directory(str(directory / "seed-draft"), str(output / "seed-draft"))
        (output / "prepared.json").write_text(json.dumps(receipt, indent=2))
    logger.info("Prepared %s", json.dumps(receipt, sort_keys=True))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
