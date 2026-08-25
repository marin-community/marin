# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""trillionlabs/TheBioCollection download + normalize helpers.

Two synthetic biology/chemistry pretraining streams built from molecular and
reaction data:

- ``free_text_stream``: free-text renderings of molecules (SMILES/IUPAC/InChI
  plus physicochemical and graph descriptors).
- ``instruction_stream``: instruction/response pairs (e.g. reaction-product
  prediction).

Each stream is a directory of zstd-compressed JSONL shards
(``data/<stream>/*.jsonl.zst``) whose ``text`` field already holds the document
content (the only other column, ``record_type``, is dropped by normalize). The
two streams share one HF repo, so each is staged under its own
``raw/biocollection/<stream>`` path and downloaded via a per-stream glob.
"""

from marin.datakit.download.huggingface import download_hf_step
from marin.datakit.normalize import normalize_step
from marin.execution.step_spec import StepSpec

HF_DATASET_ID = "trillionlabs/TheBioCollection"
HF_REVISION = "c4593c2"

STREAMS = ("free_text_stream", "instruction_stream")


def download_biocollection_step(stream: str) -> StepSpec:
    """Download a single TheBioCollection stream's zstd-JSONL shards."""
    return download_hf_step(
        f"raw/biocollection/{stream}",
        hf_dataset_id=HF_DATASET_ID,
        revision=HF_REVISION,
        hf_urls_glob=[f"data/{stream}/*.jsonl.zst"],
    )


def biocollection_normalize_steps() -> dict[str, tuple[StepSpec, ...]]:
    """Return ``(download, normalize)`` chains for both TheBioCollection streams.

    Keyed by the registry name convention ``"biocollection/<stream>"``.
    """
    chains: dict[str, tuple[StepSpec, ...]] = {}
    for stream in STREAMS:
        marin_name = f"biocollection/{stream}"
        download = download_biocollection_step(stream)
        normalize = normalize_step(
            name=f"normalized/{marin_name}",
            download=download,
            text_field="text",
            file_extensions=(".jsonl.zst",),
        )
        chains[marin_name] = (download, normalize)
    return chains
