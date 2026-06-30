# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""StarCoder2-Extras dataset tokenization as lazy Dataset handles.

Subsets: ir_cpp, ir_python, ir_rust, ir_low_resource, documentation, kaggle.
Each subset is downloaded separately from bigcode/starcoder2data-extras,
normalized (reading from the ``content`` column), and tokenized.
"""

from fray.types import ResourceConfig
from marin.datakit.download.starcoder2_extras import HF_DATASET_ID, HF_REVISION, SUBSETS
from marin.datakit.normalize import normalize_to_parquet
from marin.execution.lazy import ArtifactStep
from marin.experiment.data import hf_download, tokenized
from marin.processing.tokenize.tokenize import TokenizedCache

from experiments.marin_tokenizer import marin_tokenizer

# The documentation subset contains a single 64MB OpenJDK record that peaks at
# ~9GB RSS during tokenization; pass extra RAM to the tokenize Fray worker.
_DOC_TOKENIZE_RESOURCES = ResourceConfig(ram="32g", disk="10g")


def _run_normalize(cfg: dict) -> None:
    normalize_to_parquet(
        input_path=cfg["input_path"],
        output_path=cfg["output_path"],
        text_field=cfg["text_field"],
        id_field=cfg["id_field"],
        file_extensions=tuple(cfg["file_extensions"]),
    )


def starcoder2_extras_datasets(*, tokenizer: str = marin_tokenizer) -> list[ArtifactStep[TokenizedCache]]:
    """One tokenized Dataset handle per starcoder2data-extras subset."""
    datasets = []
    for subset in SUBSETS:
        dl = hf_download(
            f"raw/starcoder2_extras/{subset}",
            hf_id=HF_DATASET_ID,
            revision=HF_REVISION,
            urls_glob=[f"{subset}/*.parquet"],
            pin=f"raw/starcoder2_extras-{HF_REVISION}/{subset}",
            version="2026.06.28",
        )
        norm = ArtifactStep(
            name=f"normalized/starcoder2_extras/{subset}",
            version="2026.06.28",
            artifact_type=TokenizedCache,
            run=_run_normalize,
            build_config=lambda ctx, _dl=dl: {
                "input_path": ctx.artifact_path(_dl),
                "output_path": ctx.output_path,
                "text_field": "content",
                "id_field": "id",
                "file_extensions": [".parquet"],
            },
            deps=(dl,),
        )
        doc_resources = _DOC_TOKENIZE_RESOURCES if subset == "documentation" else None
        datasets.append(
            tokenized(
                f"starcoder2_extras/{subset}",
                tokenizer=tokenizer,
                raw=norm,
                glob="outputs/main/*.parquet",
                resources=doc_resources,
                version="2026.06.28",
            )
        )
    return datasets
