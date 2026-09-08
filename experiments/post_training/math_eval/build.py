# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Build the existing-source MVP in an east-region CPU task."""

import argparse
import hashlib
import json
import tempfile
from dataclasses import dataclass
from pathlib import Path

from datasets import concatenate_datasets, load_dataset
from fray.types import ResourceConfig
from marin.execution.artifact import Artifact
from marin.execution.lazy import ArtifactStep
from marin.execution.remote import remote
from rigging.filesystem.storage_path import StoragePath, prefix_join
from transformers import AutoTokenizer

from experiments.post_training.curriculum_rl import pool as curriculum
from experiments.post_training.math_eval.contract import QWEN, SNOWBALL, template_source
from experiments.post_training.math_eval.pool import SourceRows, build_pool, write_pool

GSM8K_REVISION = "e53f048856ff4f594e959d75785d2c2d37b678ee"
MATH_REVISION = "21a5633873b6a120296cce3e2df9d5550074f4a3"
MATH500_REVISION = "6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be"
PLATINUM_REVISION = "e762492455a1cf7967de89f05b6bef72fc713b66"
QWEN_REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"
SNOWBALL_TOKENIZER_SHA256 = "881c9c36c359e1617afef6f7583403567931b7b4f43f6552d2b2155a131650a2"


def mvp_sources() -> list[SourceRows]:
    """Reuse the curriculum record builders with pinned, separately locked benchmarks."""
    train = curriculum._gsm8k_records("train", 7473, revision=GSM8K_REVISION)
    sources = [SourceRows("openai/gsm8k", GSM8K_REVISION, "MIT", train, "hash")]
    sources.append(SourceRows("EleutherAI/hendrycks_math", MATH_REVISION, "MIT", curriculum._math_records(), "hash"))
    for rg_bin in curriculum.REASONING_GYM_BINS:
        sources.append(
            SourceRows(
                f"reasoning_gym/{rg_bin.task}/{rg_bin.seed}",
                "0.1.25",
                "Apache-2.0",
                curriculum._reasoning_gym_records(rg_bin, "train"),
                "hash",
                generator_seed=rg_bin.seed,
                generator_config=dict(rg_bin.knobs),
            )
        )
    platinum = load_dataset("madrylab/gsm8k-platinum", "main", split="test", revision=PLATINUM_REVISION)
    platinum_rows = []
    for index, row in enumerate(platinum):
        match = curriculum._GSM8K_FINAL_ANSWER.search(row["answer"])
        if match is None:
            raise ValueError(f"Platinum reference {index} has no numeric gold")
        platinum_rows.append(
            curriculum._pool_record(
                question=row["question"],
                answer=match.group(1).replace(",", ""),
                pool_bin=curriculum.GSM8K_BIN,
                split="test",
                index=index,
                data_source="heldout-gsm8k-platinum",
            )
        )
    sources.append(
        SourceRows("madrylab/gsm8k-platinum", PLATINUM_REVISION, "MIT AND CC-BY-SA-4.0", platinum_rows, "heldout")
    )
    math_test = concatenate_datasets(
        [
            load_dataset(curriculum.MATH_DATASET, subject, split="test", revision=MATH_REVISION)
            for subject in curriculum.MATH_SUBJECTS
        ]
    )
    math_rows = []
    for index, row in enumerate(math_test):
        answer = curriculum.boxed_answer(row["solution"])
        if answer is None:
            raise ValueError(f"MATH test reference {index} has no boxed gold")
        math_rows.append(
            curriculum._pool_record(
                question=row["problem"],
                answer=answer,
                pool_bin=curriculum.MATH_BINS_BY_LEVEL[row["level"]],
                split="test",
                index=index,
                data_source=f"heldout-math-{row['level'].lower().replace(' ', '-')}",
            )
        )
    sources.append(SourceRows("EleutherAI/hendrycks_math/test", MATH_REVISION, "MIT", math_rows, "heldout"))
    sources.append(SourceRows("HuggingFaceH4/MATH-500", MATH500_REVISION, "MIT", curriculum._math500_records(), "ood"))
    return sources


@dataclass(frozen=True)
class MathPoolConfig:
    output_path: str
    snowball_model_path: str
    version: str
    code_sha: str
    max_prompt_tokens: int = 1024


def write_math_pool(config: MathPoolConfig) -> None:
    """Materialize both model views without copying model weights."""
    if not config.output_path.startswith("s3://marin-us-east-02a/"):
        raise ValueError("The math pool is built in east-02a")
    if not config.snowball_model_path.startswith("s3://marin-us-east-02a/"):
        raise ValueError("Snowball tokenizer metadata must stay in east-02a")
    if StoragePath(prefix_join(config.output_path, "selection.json")).exists():
        raise ValueError("A completed pool artifact is immutable; choose a new version")
    qwen = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B", revision=QWEN_REVISION)
    if qwen.chat_template != template_source(QWEN):
        raise ValueError("Pinned Qwen template does not match KE0")
    with tempfile.TemporaryDirectory() as temporary:
        directory = Path(temporary)
        tokenizer_dir = directory / "tokenizer"
        tokenizer_dir.mkdir()
        for filename in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja"):
            content = StoragePath(prefix_join(config.snowball_model_path, filename)).read_bytes()
            (tokenizer_dir / filename).write_bytes(content)
        if hashlib.sha256((tokenizer_dir / "tokenizer.json").read_bytes()).hexdigest() != SNOWBALL_TOKENIZER_SHA256:
            raise ValueError("Snowball tokenizer changed")
        snowball = AutoTokenizer.from_pretrained(tokenizer_dir, local_files_only=True)
        if snowball.chat_template != template_source(SNOWBALL):
            raise ValueError("Snowball export template does not match KE0")
        hashes = {
            "qwen": hashlib.sha256(qwen.backend_tokenizer.to_str().encode()).hexdigest(),
            "snowball": SNOWBALL_TOKENIZER_SHA256,
        }
        built = build_pool(
            mvp_sources(),
            {
                "qwen": lambda text: qwen.encode(text, add_special_tokens=False),
                "snowball": lambda text: snowball.encode(text, add_special_tokens=False),
            },
            version=config.version,
            code_sha=config.code_sha,
            tokenizer_hashes=hashes,
            max_prompt_tokens=config.max_prompt_tokens,
        )
        output = directory / "pool"
        write_pool(built, output)
        for path in sorted(output.rglob("*")):
            if path.is_file():
                StoragePath(prefix_join(config.output_path, str(path.relative_to(output)))).write_bytes(
                    path.read_bytes()
                )
        print(
            "KE1_POOL_BUILD_PASS "
            + json.dumps(
                {
                    "manifest_sha256": built.selection["manifest_sha256"],
                    "rows": {split: len(ids) for split, ids in built.selection["rows"].items()},
                    "dropped": len(built.selection["dropped"]),
                }
            )
        )


def math_pool_step(name: str, version: str, *, snowball_model_path: str, code_sha: str) -> ArtifactStep[Artifact]:
    return ArtifactStep(
        name=name,
        version=version,
        artifact_type=Artifact,
        run=remote(
            write_math_pool,
            resources=ResourceConfig.with_cpu(cpu=4, ram="16g", disk="32g"),
            pip_packages=["reasoning-gym==0.1.25"],
        ),
        build_config=lambda ctx: MathPoolConfig(ctx.output_path, snowball_model_path, version, code_sha),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--snowball-model-path", required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument("--code-sha", required=True)
    arguments = parser.parse_args()
    write_math_pool(MathPoolConfig(**vars(arguments)))
