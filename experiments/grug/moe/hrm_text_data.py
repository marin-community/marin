# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""HRM-Text training mixture: per-source cleaners + tokenize + mixture.

Mirrors the cleaning pipeline in `sapientinc/data_io` (see
`pipe/clean_*.py` and `pipe_clustered/clean_*.py`) but emits parquet under
marin's executor with the canonical ``(instruction, response, condition)``
schema that ``SupervisedLmDatasetFormat`` reads.

Each source has:
1. A ``clean_<src>`` function — runs as an ExecutorStep, downloads the upstream
   HF dataset, applies the same filter/transform as the data_io cleaner, and
   writes parquet to ``<output_path>/all.parquet`` (or split files).
2. A tokenize step — runs ``default_tokenize`` on the cleaned parquet with
   ``SupervisedLmDatasetFormat(input_key="instruction", target_key="response")``.
3. An entry in ``HRM_TEXT_MIXTURE`` with the upstream weight (derived from
   `data_io/prefix_config.yaml` max_per_file / repeat).

Sources covered (HF-available; transforms ported 1:1 from data_io):
  - openmathinstruct2  (nvidia/OpenMathInstruct-2)         cot + direct
  - acereason          (nvidia/AceReason-1.1-SFT)          math, strip <think>
  - openthoughts2      (open-thoughts/OpenThoughts2-1M)    filter code sources
  - sudoku_extreme     (sapientinc/sudoku-extreme)         "Solve the Sudoku" prompt
  - dmmath             (sapientinc/HRM-Text-data-io-cleaned-20260515)
  - textbookreasoning  (MegaScience/TextbookReasoning)     cot + direct
  - gsm8k_train        (openai/gsm8k)                       extract final answer
  - math_train         (EleutherAI/hendrycks_math)          all subsets
  - omnimath           (KbsdJames/Omni-MATH)                cot + direct
  - numinamath         (AI-MO/NuminaMath-1.5)               filter synthetic
  - natural_reasoning  (facebook/natural_reasoning)         filter proofs
  - principia          (facebook/principia-collection)
  - webinstruct_verified (TIGER-Lab/WebInstruct-verified)
  - no_robots          (HuggingFaceH4/no_robots)            concat system+user
  - flan               (Open-Orca/FLAN)                     fsopt + zsopt 6 subsets each

Skipped (need local files / heavier ports — log as follow-ups):
  - amps_khan, ampsmathematica (need amps.tar.gz from Google Drive)
  - Platypus/scibench         (needs git clone)
  - SYNTH                     (PleIAs/SYNTH raw, polars-heavy filter)
  - tasksource                (297-line subset-allowlist transform)
  - Platypus/ARB, openbookqa, reclor  (small; skipped only to keep this PR
                                       focused — would map cleanly)
"""

from __future__ import annotations

import os
import re
from collections.abc import Callable
from dataclasses import dataclass

from fray.types import ResourceConfig
from levanter.data.text import SupervisedLmDatasetFormat
from marin.execution.executor import ExecutorStep, this_output_path, versioned
from marin.execution.remote import remote
from marin.processing.tokenize import TokenizeConfig, lm_mixture_data_config, tokenize
from marin.processing.tokenize.data_configs import TokenizerStep
from marin.processing.tokenize.tokenize import HfDatasetSpec

from experiments.defaults import default_tokenize
from experiments.llama import llama3_tokenizer

# ---------------------------------------------------------------------------
# Common
# ---------------------------------------------------------------------------

_CLEAN_RESOURCES = ResourceConfig(cpu=2, ram="32g", disk="64g")
_FORMAT = SupervisedLmDatasetFormat(input_key="instruction", target_key="response")


@dataclass(frozen=True)
class _Source:
    name: str  # short slug for path/component name
    weight: float  # mixture weight (derived from prefix_config.yaml)
    cleaner_step: ExecutorStep | None  # None if the source is read directly from HF (no cleaning needed)
    tokenize_step: ExecutorStep


_CHUNK_SIZE = 10_000


def _stream_records_to_parquet(
    output_path: str,
    filename: str,
    records,  # Iterable[dict[str, str]]
) -> int:
    """Stream ``records`` to ``<output_path>/<filename>`` in chunks of 10k rows.

    Returns the total number of rows written. Avoids OOM on multi-million-row
    sources (openmathinstruct2 / numinamath / etc.) by writing each chunk as a
    record batch and never materializing the full dataset in Python lists.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq
    from rigging.filesystem import open_url

    schema = pa.schema(
        [
            ("instruction", pa.string()),
            ("response", pa.string()),
            ("condition", pa.string()),
        ]
    )

    target = os.path.join(output_path, filename)
    total = 0
    buffer = {"instruction": [], "response": [], "condition": []}

    with open_url(target, mode="wb") as fh:
        with pq.ParquetWriter(fh, schema) as writer:  # type: ignore[arg-type]

            def flush() -> None:
                nonlocal total
                if not buffer["instruction"]:
                    return
                batch = pa.RecordBatch.from_pydict(buffer, schema=schema)
                writer.write_batch(batch)
                total += len(buffer["instruction"])
                buffer["instruction"].clear()
                buffer["response"].clear()
                buffer["condition"].clear()

            for r in records:
                buffer["instruction"].append(r["instruction"])
                buffer["response"].append(r["response"])
                buffer["condition"].append(r["condition"])
                if len(buffer["instruction"]) >= _CHUNK_SIZE:
                    flush()

            flush()
    return total


# ---------------------------------------------------------------------------
# Per-source cleaners (ported 1:1 from data_io)
# ---------------------------------------------------------------------------


def _strip_think(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)


def _clean_openmathinstruct2(output_path: str) -> None:
    """openmathinstruct2: single-pass over the stream emits both cot and direct rows.

    Tags rows with "synth,cot" + "synth,direct" inline using ``__split__`` (we
    don't want to iterate the 14M-row stream twice).
    """
    import pyarrow as pa
    import pyarrow.parquet as pq
    from datasets import load_dataset
    from rigging.filesystem import open_url

    dataset = load_dataset("nvidia/OpenMathInstruct-2", split="train", streaming=True)
    original_sources = {"math", "gsm8k"}

    schema = pa.schema(
        [
            ("instruction", pa.string()),
            ("response", pa.string()),
            ("condition", pa.string()),
        ]
    )

    cot_buf = {"instruction": [], "response": [], "condition": []}
    direct_buf = {"instruction": [], "response": [], "condition": []}

    cot_path = os.path.join(output_path, "cot.parquet")
    direct_path = os.path.join(output_path, "direct.parquet")

    with open_url(cot_path, mode="wb") as cot_fh, open_url(direct_path, mode="wb") as direct_fh:
        with pq.ParquetWriter(cot_fh, schema) as cot_writer, pq.ParquetWriter(direct_fh, schema) as direct_writer:

            def flush_buf(buf, writer):
                if not buf["instruction"]:
                    return
                writer.write_batch(pa.RecordBatch.from_pydict(buf, schema=schema))
                buf["instruction"].clear()
                buf["response"].clear()
                buf["condition"].clear()

            for row in dataset:
                cot_buf["instruction"].append(row["problem"])
                cot_buf["response"].append(row["generated_solution"])
                cot_buf["condition"].append("synth,cot")

                if row["problem_source"] not in original_sources:
                    direct_buf["instruction"].append(row["problem"])
                    direct_buf["response"].append(row["expected_answer"])
                    direct_buf["condition"].append("synth,direct")

                if len(cot_buf["instruction"]) >= _CHUNK_SIZE:
                    flush_buf(cot_buf, cot_writer)
                if len(direct_buf["instruction"]) >= _CHUNK_SIZE:
                    flush_buf(direct_buf, direct_writer)

            flush_buf(cot_buf, cot_writer)
            flush_buf(direct_buf, direct_writer)


def _clean_acereason(output_path: str) -> None:
    # source: pipe_clustered/clean_acereason.py
    from datasets import load_dataset

    dataset = load_dataset("nvidia/AceReason-1.1-SFT", split="train", streaming=True)

    def it():
        for row in dataset:
            if row["category"] != "math":
                continue
            yield {
                "condition": "synth,cot",
                "instruction": row["input"],
                "response": _strip_think(row["output"]),
            }

    _stream_records_to_parquet(output_path, "all.parquet", it())


def _clean_openthoughts2(output_path: str) -> None:
    # source: pipe_clustered/clean_openthoughts2.py
    from datasets import load_dataset

    remove_sources = {
        "dolphin",
        "evolcodegolf",
        "glaive",
        "magicoder",
        "sharegpt",
        "codefeedback",
        "nvidia_math",
    }
    dataset = load_dataset("open-thoughts/OpenThoughts2-1M", split="train", streaming=True)

    def it():
        for row in dataset:
            if row["source"] in remove_sources:
                continue
            convos = row["conversations"]
            if len(convos) != 2 or convos[0]["from"] != "user" or convos[1]["from"] != "assistant":
                continue
            user, assistant = convos[0]["value"], convos[1]["value"]
            lower_out = assistant.lower()
            if "python" in user.lower() or "python" in lower_out or "```" in lower_out:
                continue
            yield {
                "condition": "synth,cot",
                "instruction": user,
                "response": _strip_think(assistant),
            }

    _stream_records_to_parquet(output_path, "all.parquet", it())


def _clean_sudoku_extreme(output_path: str) -> None:
    # source: pipe_clustered/clean_sudoku.py
    import csv

    from huggingface_hub import hf_hub_download

    local_csv = hf_hub_download("sapientinc/sudoku-extreme", "train.csv", repo_type="dataset")

    def it():
        with open(local_csv, newline="") as f:
            reader = csv.reader(f)
            next(reader)  # header
            for _src, q, a, _rating in reader:
                if len(q) != 81 or len(a) != 81:
                    continue
                yield {
                    "condition": "direct",
                    "instruction": f"Solve the Sudoku\n\n{q.replace('.', '0')}",
                    "response": a,
                }

    _stream_records_to_parquet(output_path, "all.parquet", it())


def _clean_textbookreasoning(output_path: str) -> None:
    """textbookreasoning: single-pass emits cot + (filtered) direct rows."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    from datasets import load_dataset
    from rigging.filesystem import open_url

    dataset = load_dataset("MegaScience/TextbookReasoning", split="train", streaming=True)

    schema = pa.schema([("instruction", pa.string()), ("response", pa.string()), ("condition", pa.string())])
    cot_buf = {"instruction": [], "response": [], "condition": []}
    direct_buf = {"instruction": [], "response": [], "condition": []}

    with (
        open_url(os.path.join(output_path, "cot.parquet"), mode="wb") as cot_fh,
        open_url(os.path.join(output_path, "direct.parquet"), mode="wb") as direct_fh,
    ):
        with pq.ParquetWriter(cot_fh, schema) as cot_writer, pq.ParquetWriter(direct_fh, schema) as direct_writer:

            def flush(buf, writer):
                if not buf["instruction"]:
                    return
                writer.write_batch(pa.RecordBatch.from_pydict(buf, schema=schema))
                buf["instruction"].clear()
                buf["response"].clear()
                buf["condition"].clear()

            for row in dataset:
                cot_buf["instruction"].append(row["question"])
                cot_buf["response"].append(row["answer"])
                cot_buf["condition"].append("synth,cot")
                lower_q = row["question"].lower()
                if "prove" not in lower_q and "show that" not in lower_q:
                    direct_buf["instruction"].append(row["question"])
                    direct_buf["response"].append(row["reference_answer"])
                    direct_buf["condition"].append("noisy,direct")
                if len(cot_buf["instruction"]) >= _CHUNK_SIZE:
                    flush(cot_buf, cot_writer)
                if len(direct_buf["instruction"]) >= _CHUNK_SIZE:
                    flush(direct_buf, direct_writer)

            flush(cot_buf, cot_writer)
            flush(direct_buf, direct_writer)


def _clean_gsm8k_train(output_path: str) -> None:
    # source: pipe/clean_gsm8k_train.py
    from datasets import load_dataset

    dataset = load_dataset("openai/gsm8k", "main", split="train", streaming=True)

    def it():
        for row in dataset:
            parts = row["answer"].split("#### ")
            if len(parts) != 2:
                continue
            yield {
                "condition": "direct",
                "instruction": row["question"].strip(),
                "response": parts[-1].strip(),
            }

    _stream_records_to_parquet(output_path, "all.parquet", it())


def _clean_math_train(output_path: str) -> None:
    # source: pipe/clean_math_train.py
    from datasets import get_dataset_config_names, load_dataset

    subsets = get_dataset_config_names("EleutherAI/hendrycks_math")

    def it():
        for subset in subsets:
            dataset = load_dataset("EleutherAI/hendrycks_math", subset, split="train", streaming=True)
            for row in dataset:
                yield {
                    "condition": "cot",
                    "instruction": row["problem"],
                    "response": row["solution"].strip(),
                }

    _stream_records_to_parquet(output_path, "all.parquet", it())


def _clean_omnimath(output_path: str) -> None:
    # source: pipe/clean_omnimath.py (note: cleaner uses 'test' split, the only one published)
    from datasets import load_dataset

    dataset = load_dataset("KbsdJames/Omni-MATH", split="test", streaming=True)

    def it():
        for row in dataset:
            yield {
                "condition": "cot",
                "instruction": row["problem"],
                "response": row["solution"].strip(),
            }
            yield {
                "condition": "direct",
                "instruction": row["problem"],
                "response": row["answer"].strip(),
            }

    _stream_records_to_parquet(output_path, "all.parquet", it())


def _clean_numinamath(output_path: str) -> None:
    # source: pipe/clean_numinamath.py
    from datasets import load_dataset

    dataset = load_dataset("AI-MO/NuminaMath-1.5", split="train", streaming=True)

    def it():
        for row in dataset:
            if (
                row.get("synthetic")
                or row.get("problem_is_valid") != "Yes"
                or row.get("solution_is_valid") != "Yes"
                or row.get("solution") is None
                or row.get("answer") is None
            ):
                continue
            problem = row["problem"]
            solution = row["solution"]
            if "http" in problem or "http" in solution or "Translate the text above into English" in solution:
                continue
            problem, solution = problem.strip(), solution.strip()
            if not problem or not solution:
                continue
            yield {"condition": "noisy,cot", "instruction": problem, "response": solution}
            if row["question_type"] != "proof" and row["answer"] != "proof":
                answer = row["answer"].strip()
                if answer:
                    yield {"condition": "noisy,direct", "instruction": problem, "response": answer}

    _stream_records_to_parquet(output_path, "all.parquet", it())


def _clean_natural_reasoning(output_path: str) -> None:
    # source: pipe/clean_natural_reasoning.py
    from datasets import load_dataset

    dataset = load_dataset("facebook/natural_reasoning", split="train", streaming=True)

    def it():
        for row in dataset:
            ref = row["reference_answer"].strip()
            if not ref:
                continue
            lower_q = row["question"].lower()
            if "prove" in lower_q or "show that" in lower_q:
                continue
            yield {"condition": "noisy,direct", "instruction": row["question"], "response": ref}

    _stream_records_to_parquet(output_path, "all.parquet", it())


def _clean_principia(output_path: str) -> None:
    # source: pipe/clean_principia_collection.py
    from datasets import load_dataset

    dataset_dict = load_dataset("facebook/principia-collection", streaming=True)

    def it():
        for split_dataset in dataset_dict.values():
            for row in split_dataset:
                yield {
                    "condition": "synth,direct",
                    "instruction": row["problem_statement"],
                    "response": row["answer"],
                }

    _stream_records_to_parquet(output_path, "all.parquet", it())


def _clean_webinstruct_verified(output_path: str) -> None:
    # source: pipe/clean_webinstruct_verified.py
    from datasets import load_dataset

    dataset = load_dataset("TIGER-Lab/WebInstruct-verified", split="train", streaming=True)

    def it():
        for row in dataset:
            yield {
                "condition": "direct",
                "instruction": row["question"],
                "response": row["answer"],
            }

    _stream_records_to_parquet(output_path, "all.parquet", it())


def _clean_no_robots(output_path: str) -> None:
    # source: pipe/clean_no_robots.py
    from datasets import load_dataset

    dataset_dict = load_dataset("HuggingFaceH4/no_robots", streaming=True)

    def it():
        for split_dataset in dataset_dict.values():
            for row in split_dataset:
                msgs = row["messages"]
                if len(msgs) < 2:
                    continue
                system_content = ""
                start = 0
                if msgs[0]["role"] == "system":
                    system_content = msgs[0]["content"] + "\n\n"
                    start = 1
                if len(msgs) > start + 1 and msgs[start]["role"] == "user" and msgs[start + 1]["role"] == "assistant":
                    yield {
                        "condition": "cot",
                        "instruction": system_content + msgs[start]["content"],
                        "response": msgs[start + 1]["content"],
                    }

    _stream_records_to_parquet(output_path, "all.parquet", it())


# ---------------------------------------------------------------------------
# Step wiring
# ---------------------------------------------------------------------------


def _make_clean_step(
    name: str,
    cleaner: Callable[[str], None],
    *,
    resources: ResourceConfig = _CLEAN_RESOURCES,
) -> ExecutorStep:
    fn = remote(cleaner, resources=resources)
    return ExecutorStep(
        name=os.path.join("hrm_text_cleaned", name),
        fn=fn,
        config=this_output_path(),
    )


_TOKENIZE_RESOURCES = ResourceConfig(cpu=4, ram="16g", disk="10g")
_TOKENIZE_ENV = {
    "TRANSFORMERS_NO_TORCH": "1",
    "TRANSFORMERS_NO_TORCHVISION": "1",
    "USE_TORCH": "0",
    "TORCH_DISABLE_GLOBAL_DEPS": "1",
}


def _make_tokenize_from_clean(name: str, clean_step: ExecutorStep) -> ExecutorStep:
    """Wrap tokenize with @remote so it runs as its own Iris job, like
    ``experiments.defaults.default_tokenize`` does — without this the tokenize
    function runs inline in the entrypoint (cpu=0.1, ram=1GB) and stalls."""
    return ExecutorStep(
        name=os.path.join("tokenized", "hrm_text", name),
        fn=remote(
            tokenize,
            resources=_TOKENIZE_RESOURCES,
            pip_dependency_groups=["cpu"],
            env_vars=_TOKENIZE_ENV,
        ),
        config=TokenizeConfig(
            train_paths=[clean_step.cd("*.parquet")],
            validation_paths=versioned([]),
            cache_path=this_output_path(),
            tokenizer=versioned(llama3_tokenizer),
            format=_FORMAT,
            tags=["hrm_text", name],
            # num_shards=1 bundles all input parquet files into a single tokenize
            # worker. Trades parallelism for a much smaller scheduling footprint
            # (one 0.1-core worker per source instead of N) — useful when the
            # iris cluster is heavily contended.
            num_shards=1,
        ),
    )


# Read dmmath directly from the HF-published cleaned dataset (it is the only
# subset published under data_clustered/ on sapientinc/HRM-Text-data-io-cleaned-20260515).
_dmmath_tokenize: ExecutorStep = default_tokenize(
    name="hrm_text/dmmath",
    dataset=HfDatasetSpec(id="sapientinc/HRM-Text-data-io-cleaned-20260515"),
    tokenizer=llama3_tokenizer,
    format=_FORMAT,
    tags=("hrm_text", "dmmath"),
)


def _build_source(name: str, cleaner: Callable[[str], None], weight: float) -> _Source:
    clean = _make_clean_step(name, cleaner)
    tok = _make_tokenize_from_clean(name, clean)
    return _Source(name=name, weight=weight, cleaner_step=clean, tokenize_step=tok)


# ---------------------------------------------------------------------------
# Mixture
# ---------------------------------------------------------------------------

# Weights derived from data_io/prefix_config.yaml:
#  - The "per-epoch row cap" (max_per_file x repeat) approximates each source's
#    contribution to one HRM-Text epoch.
#  - We pass these directly to lm_mixture_data_config; Levanter normalizes them.
# Caps that are >= the full source size effectively read the whole source once.
_SOURCES: tuple[_Source, ...] = (
    # Clustered, large-cap sources (multi-file in upstream)
    _build_source("openmathinstruct2", _clean_openmathinstruct2, 2_000_000.0),
    _build_source("acereason", _clean_acereason, 2_000_000.0),
    _build_source("openthoughts2", _clean_openthoughts2, 500_000.0),
    _build_source("sudoku_extreme", _clean_sudoku_extreme, 1_000_000.0),
    _build_source("textbookreasoning", _clean_textbookreasoning, 100_000.0),
    _build_source("numinamath", _clean_numinamath, 100_000.0),
    _build_source("natural_reasoning", _clean_natural_reasoning, 100_000.0),
    _build_source("principia", _clean_principia, 100_000.0),
    # Small, repeated sources (x10 in prefix_config)
    _build_source("omnimath", _clean_omnimath, 10.0 * 4_400.0),  # ~4.4k rows x 10
    _build_source("gsm8k_train", _clean_gsm8k_train, 10.0 * 7_473.0),
    _build_source("math_train", _clean_math_train, 10.0 * 7_500.0),
    _build_source("webinstruct_verified", _clean_webinstruct_verified, 10.0 * 200_000.0),
    _build_source("no_robots", _clean_no_robots, 10.0 * 9_500.0),
)


_DMMATH_SOURCE: _Source = _Source(
    name="dmmath",
    weight=100_000.0 * 168,  # ~168 files x 100k cap
    cleaner_step=None,
    tokenize_step=_dmmath_tokenize,
)


def _all_sources() -> tuple[_Source, ...]:
    return (*_SOURCES, _DMMATH_SOURCE)


def hrm_text_mixture():
    """Build the HRM-Text training LmDataConfig."""
    components: dict[str, TokenizerStep] = {}
    weights: dict[str, float] = {}
    for src in _all_sources():
        components[src.name] = src.tokenize_step
        weights[src.name] = src.weight
    return lm_mixture_data_config(components=components, weights=weights)


def hrm_text_clean_steps() -> list[ExecutorStep]:
    return [s.cleaner_step for s in _SOURCES if s.cleaner_step is not None]


def hrm_text_tokenize_steps() -> list[ExecutorStep]:
    return [s.tokenize_step for s in _all_sources()]


__all__ = [
    "hrm_text_clean_steps",
    "hrm_text_mixture",
    "hrm_text_tokenize_steps",
]
