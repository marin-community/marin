# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Build and qualify one MLX recipe through a shared artifact graph."""

import hashlib
import json
import subprocess
import tomllib
from pathlib import Path
from uuid import uuid4

import click
from marin.execution.lazy import OUT, ArtifactStep, apply
from marin.experiment.cli import build_options
from rigging.filesystem.cluster_config import marin_prefix
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.storage_path import StoragePath

from experiments.mlx_export.golden_vllm import VLLM_GPU_RELEASE, submit_golden
from experiments.mlx_export.provenance import Recipe, file_digest, validate_revision

PROJECT = Path(__file__).resolve().parent


def source_digest() -> str:
    files = [*sorted(PROJECT.glob("*.py")), PROJECT / "pyproject.toml", PROJECT / "uv.lock"]
    return hashlib.sha256(json.dumps({p.name: file_digest(p) for p in files}, sort_keys=True).encode()).hexdigest()


def mac_command(script: str, arguments: list[str], code_sha256: str) -> None:
    if source_digest() != code_sha256:
        raise ValueError("MLX project changed after graph construction")
    subprocess.run(
        ["uv", "run", "--project", str(PROJECT), "--frozen", "python", str(PROJECT / script), *arguments],
        check=True,
    )


def prompt_stage(repo: str, revision: str, code_sha256: str, output: str) -> None:
    Path(output).mkdir(parents=True, exist_ok=True)
    mac_command(
        "prompts.py", ["--repo", repo, "--revision", revision, "--out", str(Path(output) / "prompts.json")], code_sha256
    )


def model_stage(repo: str, revision: str, recipe: str, code_sha256: str, output: str) -> None:
    mac_command(
        "convert.py", ["--repo", repo, "--revision", revision, "--recipe", recipe, "--work-dir", output], code_sha256
    )


def golden_stage(
    repo: str,
    revision: str,
    prompts: str,
    release_tag: str,
    gpus: int,
    remote_prefix: str,
    code_sha256: str,
    output: str,
) -> None:
    if source_digest() != code_sha256:
        raise ValueError("MLX project changed after graph construction")
    if tomllib.loads(VLLM_GPU_RELEASE.read_text())["release_tag"] != release_tag:
        raise ValueError("vLLM release changed after graph construction")
    remote_output = str(StoragePath(remote_prefix) / uuid4().hex / "golden.json")
    submit_golden(repo, revision, Path(prompts) / "prompts.json", gpus, remote_output, release_tag)
    configure_coreweave_s3()
    Path(output).mkdir(parents=True, exist_ok=True)
    (Path(output) / "golden.json").write_bytes(StoragePath(remote_output).read_bytes())


def dump_stage(model: str, recipe: str, prompts: str, golden: str, code_sha256: str, output: str) -> None:
    Path(output).mkdir(parents=True, exist_ok=True)
    mac_command(
        "dump_logprobs.py",
        [
            "--model",
            str(Path(model) / f"mlx-{recipe}"),
            "--prompts",
            str(Path(prompts) / "prompts.json"),
            "--golden",
            str(Path(golden) / "golden.json"),
            "--out",
            str(Path(output) / "dump.json"),
        ],
        code_sha256,
    )


def parity_stage(model: str, recipe: str, dump: str, golden: str, code_sha256: str, output: str) -> None:
    Path(output).mkdir(parents=True, exist_ok=True)
    mac_command(
        "compare_parity.py",
        [
            "--model",
            str(Path(model) / f"mlx-{recipe}"),
            "--recipe",
            recipe,
            "--dump",
            str(Path(dump) / "dump.json"),
            "--golden",
            str(Path(golden) / "golden.json"),
            "--report",
            str(Path(output) / "parity.md"),
        ],
        code_sha256,
    )


def build(repo: str, revision: str, recipe: Recipe, gpus: int, remote_prefix: str) -> ArtifactStep:
    validate_revision(revision)
    if not Path(marin_prefix()).is_absolute():
        raise ValueError("Set MARIN_PREFIX to an absolute local directory on the Mac")
    code = source_digest()
    release = tomllib.loads(VLLM_GPU_RELEASE.read_text())["release_tag"]
    name = f"mlx-export/{repo}/{revision}/{recipe}"
    prompts = apply(f"{name}/prompts", prompt_stage, repo=repo, revision=revision, code_sha256=code, output=OUT)
    model = apply(
        f"{name}/model", model_stage, repo=repo, revision=revision, recipe=recipe.value, code_sha256=code, output=OUT
    )
    golden = apply(
        f"{name}/golden",
        golden_stage,
        repo=repo,
        revision=revision,
        prompts=prompts,
        release_tag=release,
        gpus=gpus,
        remote_prefix=remote_prefix,
        code_sha256=code,
        output=OUT,
    )
    dump = apply(
        f"{name}/dump",
        dump_stage,
        model=model,
        recipe=recipe.value,
        prompts=prompts,
        golden=golden,
        code_sha256=code,
        output=OUT,
    )
    return apply(
        f"{name}/parity",
        parity_stage,
        model=model,
        recipe=recipe.value,
        dump=dump,
        golden=golden,
        code_sha256=code,
        output=OUT,
    )


@click.command()
@click.option("--repo", required=True)
@click.option("--revision", required=True)
@click.option("--recipe", type=click.Choice([r.value for r in Recipe]), required=True)
@click.option("--gpus", type=click.IntRange(min=1), required=True)
@click.option("--remote-prefix", required=True, help="S3 directory in the GPU cluster's bucket for reference JSON.")
@build_options
def main(repo: str, revision: str, recipe: str, gpus: int, remote_prefix: str) -> ArtifactStep:
    return build(repo, revision, Recipe(recipe), gpus, remote_prefix)


if __name__ == "__main__":
    main()
