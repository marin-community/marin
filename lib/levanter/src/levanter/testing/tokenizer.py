# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import json
import shutil
from pathlib import Path


def stage_gpt2_tokenizer(source_dir: Path, output_dir: Path) -> Path:
    """Stage Levanter's checked-in GPT-2 tokenizer in a loadable directory."""
    shutil.copy(source_dir / "gpt2_tokenizer.json", output_dir / "tokenizer.json")
    shutil.copy(source_dir / "gpt2_tokenizer_config.json", output_dir / "tokenizer_config.json")
    (output_dir / "config.json").write_text(json.dumps({"model_type": "gpt2", "vocab_size": 5027}))
    return output_dir
