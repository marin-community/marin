# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Stage every accepted conversion release as its lexical population and emit the scan manifest.

Runs the validated release-population staging per accepted release (the local
population write, then the archive copy to the lexical text root), rewrites the
staged records' paths to their regional locations, and assembles the
populations.json + source-ids.json pair the lexical scan consumes.
"""

import argparse
import json
import logging
import subprocess
from pathlib import Path

LOGGER = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True, help="The cohort build plan JSON (source_id, release_repo).")
    parser.add_argument("--artifact-version", required=True, help="The release version the populations stage from.")
    parser.add_argument("--workdir", type=Path, required=True, help="Local scratch for the staged populations.")
    parser.add_argument(
        "--text-root", required=True, help="The lexical text root the staged populations copy to (object storage)."
    )
    parser.add_argument("--output", type=Path, required=True, help="The scan manifest to write.")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    from huggingface_hub import HfApi, hf_hub_download  # noqa: PLC0415
    from marin.execution.artifact import (  # noqa: PLC0415
        ArtifactRecord,
        result_type_name,
    )

    from experiments.post_training.task_curation.harbor_cohort_release import (  # noqa: PLC0415
        release_repo as release_repo_name,
    )
    from experiments.post_training.task_curation.lexical_artifacts import (  # noqa: PLC0415
        SourceTextPopulation,
    )
    from experiments.post_training.task_curation.release_population import (  # noqa: PLC0415
        stage_release_population,
    )

    api = HfApi()
    populations = []
    for entry in json.loads(args.plan.read_text()):
        source_id = entry["source_id"]
        slug = source_id.partition(":")[2].replace("/", "-").replace("@", "-").lower()[:48].rstrip("-")
        repo = release_repo_name(source_id, args.artifact_version)
        head = api.repo_info(repo, repo_type="dataset").sha
        manifest = json.loads(
            Path(hf_hub_download(repo, "manifest.json", repo_type="dataset", revision=head)).read_bytes()
        )
        local_out = args.workdir / slug
        LOGGER.info("staging %s (%s rows)", source_id, manifest["converted_rows"])
        staged = stage_release_population(source_id, repo, head, manifest=manifest, output_path=str(local_out))
        text_uri = f"{args.text_root.rstrip('/')}/{slug}/{args.artifact_version}"
        subprocess.run(
            ["uv", "run", "--no-sync", "fsutil", "cp", "-r", str(local_out) + "/", text_uri + "/"],
            check=True,
            stdout=subprocess.DEVNULL,
        )
        record = staged.model_dump(mode="json")
        record.update(
            path=f"{text_uri}/",
            dataset_id=repo,
            dataset_revision=head,
            main_output_dir=f"{text_uri}/outputs/main",
            specs_output_dir=f"{text_uri}/specs",
        )
        # The scan's raw_load requires the artifact record sidecar at the staged
        # path; write it through fsutil (the local s3fs has no credentials). The
        # record carries the payload minus its path, matching what a step runner
        # persists, so raw_load rebuilds the full model at the staged location.
        population = SourceTextPopulation(**record)
        payload_fields = {name for name in SourceTextPopulation.model_fields if name != "path"}
        sidecar = args.workdir / f"{slug}.artifact.json"
        sidecar.write_text(
            ArtifactRecord(
                output_path=text_uri,
                result_type=result_type_name(SourceTextPopulation),
                result=population.model_dump(mode="json", include=payload_fields),
            ).model_dump_json(indent=2)
        )
        subprocess.run(
            ["uv", "run", "--no-sync", "fsutil", "cp", str(sidecar), text_uri + "/.artifact.json"],
            check=True,
            stdout=subprocess.DEVNULL,
        )
        populations.append(record)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(populations, indent=1) + "\n")
    source_ids = [p["source_id"] for p in populations]
    ids_path = args.output.with_name("source-ids.json")
    ids_path.write_text(json.dumps(source_ids, indent=1) + "\n")
    print(json.dumps({"staged": len(populations), "manifest": str(args.output), "source_ids": str(ids_path)}, indent=1))


if __name__ == "__main__":
    main()
