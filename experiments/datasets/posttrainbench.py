# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""PostTrainBench open-weights agent traces as a lazy Dataset handle.

198 ten-hour post-training trajectories from aisa-group/PostTrainBench-Trajectories,
covering the runs driven by an open-weights agent model (GLM, Kimi, MiniMax),
rendered into role-tagged turns, normalized, and tokenized. Each document opens with
the grader's verdict on the attempt, so training conditions on the outcome. See
:mod:`marin.datakit.download.posttrainbench` for the run selection, the outcome
prefix, and the anti-cheating filter.
"""

from marin.datakit.download.posttrainbench import HF_DATASET_ID, HF_REVISION, OPEN_WEIGHT_GLOBS, transform
from marin.datakit.normalize import normalize_to_parquet
from marin.execution.lazy import ArtifactStep
from marin.experiment.data import dataset_main, hf_download, tokenized
from marin.processing.tokenize.tokenize import TokenizedCache

from experiments.marin_tokenizer import marin_tokenizer


def _run_transform(cfg: dict) -> None:
    transform(input_path=cfg["input_path"], output_path=cfg["output_path"])


def _run_normalize(cfg: dict) -> None:
    normalize_to_parquet(input_path=cfg["input_path"], output_path=cfg["output_path"])


def posttrainbench_open_weights_dataset(*, tokenizer: str = marin_tokenizer) -> ArtifactStep[TokenizedCache]:
    """PostTrainBench open-weights agent traces as a tokenized Dataset handle."""
    dl = hf_download(
        "raw/posttrainbench-trajectories",
        hf_id=HF_DATASET_ID,
        revision=HF_REVISION,
        urls_glob=OPEN_WEIGHT_GLOBS,
        version="2026.07.29",
    )
    processed = ArtifactStep(
        name="processed/posttrainbench-open-weights",
        version="2026.07.29",
        artifact_type=TokenizedCache,
        run=_run_transform,
        build_config=lambda ctx: {
            "input_path": ctx.artifact_path(dl),
            "output_path": ctx.output_path,
            "schema_version": "v2",
        },
        deps=(dl,),
    )
    norm = ArtifactStep(
        name="normalized/posttrainbench-open-weights",
        version="2026.07.29",
        artifact_type=TokenizedCache,
        run=_run_normalize,
        build_config=lambda ctx: {"input_path": ctx.artifact_path(processed), "output_path": ctx.output_path},
        deps=(processed,),
    )
    return tokenized(
        "posttrainbench-open-weights",
        tokenizer=tokenizer,
        raw=norm,
        glob="outputs/main/*.parquet",
        version="2026.07.29",
    )


if __name__ == "__main__":
    dataset_main({"posttrainbench-open-weights": posttrainbench_open_weights_dataset()})
