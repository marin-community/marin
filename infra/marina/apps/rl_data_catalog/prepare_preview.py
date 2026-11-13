# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Stage the Atlas preview with its exact catalog models inside the application package."""

import argparse
import hashlib
import json
import shutil
import tomllib
from pathlib import Path

from rigging.provenance import git_output

APP_SOURCE = Path("infra/marina/apps/rl_data_catalog")
PRODUCTION_APP_ID = "fb11c931-5861-4878-8bb5-a964d652b45f"
PREVIEW_APP_ID = "c9742a15-fb46-4d56-bc88-aa916351ebd3"
MODEL_SOURCES = {
    "source_identity.py": "experiments/post_training/task_curation/source_identity.py",
    "curation.py": "lib/finestore/src/finestore/rl/curation.py",
    "source_fingerprints.py": "lib/taskcompendium/src/taskcompendium/source_fingerprints.py",
}
IMPORT_RELOCATIONS = {
    "from experiments.post_training.task_curation.source_identity import ": "from .bundled.source_identity import ",
    "from finestore.rl.curation import ": "from .bundled.curation import ",
    "from taskcompendium.source_fingerprints import ": "from .bundled.source_fingerprints import ",
}


def prepare_preview(checkout: Path, output: Path) -> None:
    """Write a new preview-only package and record hashes of its canonical model inputs."""
    source = checkout / APP_SOURCE
    if output.resolve().is_relative_to(checkout.resolve()):
        raise ValueError("Stage the preview outside the source checkout")
    manifest_text = (source / "applet.toml").read_text()
    if tomllib.loads(manifest_text)["applet_id"] != PRODUCTION_APP_ID:
        raise ValueError("The source manifest does not identify the maintained Atlas")
    shutil.copytree(source, output, ignore=shutil.ignore_patterns("__pycache__", "tests"))
    (output / "applet.toml").write_text(manifest_text.replace(PRODUCTION_APP_ID, PREVIEW_APP_ID))
    bundled = output / "server/bundled"
    bundled.mkdir()
    (bundled / "__init__.py").write_text("# Copyright The Marin Authors\n# SPDX-License-Identifier: Apache-2.0\n")
    modules = []
    for name, relative in MODEL_SOURCES.items():
        raw = (checkout / relative).read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        (bundled / name).write_bytes(raw)
        modules.append(
            {
                "source": relative,
                "source_sha256": digest,
                "packaged": f"server/bundled/{name}",
                "packaged_sha256": digest,
                "transformation": "Unchanged",
            }
        )
    for relative in ("server/app.py", "server/curriculum.py", "server/quality.py", "server/contamination.py"):
        path = output / relative
        content = path.read_text()
        for before, after in IMPORT_RELOCATIONS.items():
            content = content.replace(before, after)
        path.write_text(content)
    metadata = {
        "source_revision": git_output(["rev-parse", "HEAD"], checkout),
        "modules": modules,
        "preview_only": True,
    }
    (output / "dist/bundled-model-provenance.json").write_text(json.dumps(metadata, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkout", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    prepare_preview(args.checkout, args.output)


if __name__ == "__main__":
    main()
