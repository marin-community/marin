# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Convert a Grug MoE Hugging Face export into a quantized MLX build.

Downloads a pinned revision, adds ``grug_moe_mlx.py`` as the repo's
``model_file``, quantizes it with one of the recipes below, enables mlx-lm's
tool-call parser, and checks that the result loads.

Recipes:
  8bit    every quantizable linear at 8 bits.
  mixed4  routed experts at 4 bits, everything else at 8 bits. Uniform 4-bit
          failed parity on chat and tool prompts for Snowball 67B-A2B.

Gated norms and the attention gate stay unquantized in both recipes.

Usage (Mac only):
  uv run python convert.py --repo open-athena/Marin-Snowball-67B-A2B-v0.1 \
      --revision e4aa78c7c9729dea7238c2e7d1ac7fa57492e0a6 --recipe mixed4 --work-dir out
"""

import argparse
import hashlib
import json
import shutil
from enum import StrEnum
from pathlib import Path

from huggingface_hub import snapshot_download
from mlx_lm import load
from mlx_lm.convert import convert

MODEL_FILE = Path(__file__).resolve().parent / "grug_moe_mlx.py"
GROUP_SIZE = 64
NON_EXPERT_BITS = 8
TOOL_PARSER = "json_tools"


class Recipe(StrEnum):
    EIGHT_BIT = "8bit"
    MIXED_4BIT = "mixed4"


EXPERT_BITS = {Recipe.EIGHT_BIT: 8, Recipe.MIXED_4BIT: 4}


def quant_params(path: str, _module) -> bool | dict:
    """Per-module quantization: experts take the recipe's default bits, the rest 8 bits."""
    if "gated_norm" in path or path.endswith("attn_gate"):
        return False
    if ".mlp.experts." in path:
        return True
    return {"group_size": GROUP_SIZE, "bits": NON_EXPERT_BITS}


def prepare_source(repo: str, revision: str, src: Path) -> None:
    snapshot_download(repo, revision=revision, local_dir=src)
    shutil.copyfile(MODEL_FILE, src / MODEL_FILE.name)
    config_path = src / "config.json"
    config = json.loads(config_path.read_text())
    assert config["model_type"] == "grug_moe", f"{repo}@{revision} is not a Grug MoE export"
    config["model_file"] = MODEL_FILE.name
    config_path.write_text(json.dumps(config, indent=2) + "\n")


def finish_build(dst: Path) -> None:
    tokenizer_config_path = dst / "tokenizer_config.json"
    tokenizer_config = json.loads(tokenizer_config_path.read_text())
    tokenizer_config["tool_parser_type"] = TOOL_PARSER
    tokenizer_config_path.write_text(json.dumps(tokenizer_config, indent=2) + "\n")


def check_build(dst: Path, recipe: Recipe) -> None:
    config = json.loads((dst / "config.json").read_text())
    assert config["model_file"] == MODEL_FILE.name
    quantization = config["quantization"]
    assert quantization["bits"] == EXPERT_BITS[recipe], quantization["bits"]
    overrides = {path: params for path, params in quantization.items() if isinstance(params, dict)}
    assert overrides, "no non-expert overrides; did the module names in grug_moe_mlx.py change?"
    leaked = [path for path in overrides if ".mlp.experts." in path or "gated_norm" in path]
    assert not leaked, f"unexpected overrides: {leaked[:3]}"
    assert all(params["bits"] == NON_EXPERT_BITS for params in overrides.values())
    # A strict, lazy load checks every weight name and shape without reading the weights.
    load(str(dst), lazy=True, trust_remote_code=True)
    digest = hashlib.sha256((dst / MODEL_FILE.name).read_bytes()).hexdigest()
    print(f"{dst}: {len(overrides)} modules at {NON_EXPERT_BITS} bits, experts at {quantization['bits']} bits")
    print(f"{MODEL_FILE.name} sha256 {digest}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo", required=True, help="Hugging Face repo of the Grug MoE export")
    parser.add_argument("--revision", required=True, help="commit SHA to convert")
    parser.add_argument("--recipe", required=True, type=Recipe, choices=list(Recipe))
    parser.add_argument("--work-dir", required=True, type=Path, help="holds src/ and mlx-<recipe>/")
    args = parser.parse_args()

    src = args.work_dir / "src"
    dst = args.work_dir / f"mlx-{args.recipe}"
    assert not dst.exists(), f"{dst} exists; remove it to rebuild"
    prepare_source(args.repo, args.revision, src)
    convert(
        hf_path=str(src),
        mlx_path=str(dst),
        quantize=True,
        q_bits=EXPERT_BITS[args.recipe],
        q_group_size=GROUP_SIZE,
        quant_predicate=quant_params,
        trust_remote_code=True,
    )
    finish_build(dst)
    check_build(dst, args.recipe)
    # Loading the build imports the model file; don't publish the bytecode.
    shutil.rmtree(dst / "__pycache__", ignore_errors=True)


if __name__ == "__main__":
    main()
