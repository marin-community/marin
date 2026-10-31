# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Write a bf16 copy of an Axolotl Stage 0 export (pod-side, Axolotl image, CPU node).

Axolotl saves the fp32 master weights. Serving and downstream training read the bf16 copy;
the fp32 export stays beside it for weight analysis. Tensors are cast shard by shard, so
the job needs memory for one shard, not the model. Every non-weight file (config, tokenizer
with its embedded chat template, generation config, run manifest) is copied unchanged, and
the config's ``torch_dtype``/``dtype`` is set to bfloat16.

    python experiments/post_training/axolotl_sft/export_bf16.py SRC_DIR DST_DIR
"""

import json
import shutil
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file


def export_bf16(source: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    for path in sorted(source.iterdir()):
        if path.is_dir():
            continue
        if path.suffix == ".safetensors":
            tensors = load_file(path)
            cast = {
                name: tensor.to(torch.bfloat16) if tensor.is_floating_point() else tensor
                for name, tensor in tensors.items()
            }
            save_file(cast, destination / path.name, metadata={"format": "pt"})
        elif path.name == "config.json":
            config = json.loads(path.read_text())
            for key in ("torch_dtype", "dtype"):
                config[key] = "bfloat16"
            if isinstance(config.get("text_config"), dict):
                config["text_config"]["dtype"] = "bfloat16"
            (destination / path.name).write_text(json.dumps(config, indent=2))
        elif path.name == "model.safetensors.index.json":
            index = json.loads(path.read_text())
            index.get("metadata", {}).pop("total_size", None)
            (destination / path.name).write_text(json.dumps(index, indent=2))
        else:
            shutil.copy2(path, destination / path.name)


if __name__ == "__main__":
    export_bf16(Path(sys.argv[1]), Path(sys.argv[2]))
