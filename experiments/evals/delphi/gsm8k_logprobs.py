# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Per-token logprobs for the 10 Delphi scaling-ladder checkpoints on GSM8K.

For each checkpoint, scores ``prompt + solution`` (the lm-eval-harness 5-shot
context concatenated with the gold chain-of-thought from the dataset) and
saves per-token losses (and optionally top-k logprobs) to GCS via
``default_save_logprobs``.

Reuses ``download_gsm8k_step`` and ``N_PROBLEMS`` from ``gsm8k.py`` so the
problem subset is bit-identical to the rollout/grade pipeline.

# Output

Per checkpoint, written to
``analysis/logprobs/delphi/gsm8k/<slug>/gsm8k/outputs.jsonl.gz`` — one record
per gsm8k problem in input order:

    {"token_ids": [...], "losses": [...],
     "top_k_token_ids": [[...]], "top_k_logprobs": [[...]]}

# Aligning logprobs with the per-problem accuracy from grade.py

Output is 1:1 with the input gsm8k JSONL by row index, because:
- ``save_logprobs._force_pack_data`` forces ``pack=True`` on every component,
  routing data through ``GreedyPrepackedDataset`` (preserves doc boundaries —
  one input doc → one segment → one output record).
- ``slice_strategy="left"`` truncation only fires for single-doc-exceeds-Pos,
  which can't happen at GSM8K sizes vs. ``MAX_EVAL_LENGTH=4096``.
- ``TPU_TYPE="v5p-8"`` is single-host, so the eval data loader is
  single-process and ``multihost_utils.process_allgather(..., tiled=True)``
  doesn't interleave.

To attach ``problem_id`` and join with ``grades/delphi/gsm8k/<slug>/graded.jsonl.gz``:

    1. Read ``raw/gsm8k/gsm8k-00000.jsonl.gz`` once to get ``problem_id`` per row.
    2. For each output row N, optionally re-tokenize input row N with the exact
       ``BatchTokenizer`` recipe and assert byte-equality:

           text_plus_eos = row["prompt"] + row["solution"] + " " + tokenizer.eos_token
           expected = [tokenizer.bos_token_id, *tokenizer.encode(text_plus_eos,
                                                                 add_special_tokens=False)]
           assert tuple(out_records[N]["token_ids"]) == tuple(expected)

    3. Pre-flight asserts: ``len(out_records) == len(input_rows)`` and
       ``max(len(expected_ids)) < MAX_EVAL_LENGTH``.

    4. Compute the prompt/solution boundary (longest common prefix between
       ``[bos] + tokenizer.encode(row["prompt"])`` and ``expected``) so you can
       slice ``losses`` into prompt-prediction vs. solution-prediction parts.
       Note: ``next_token_loss[t]`` is ``-log P(token_ids[t+1] | token_ids[≤t])``,
       so the solution contribution is ``losses[prompt_token_len-1 : -1]``.

# Usage

    iris job run -- python experiments/evals/delphi/gsm8k_logprobs.py
"""

import json
import logging
import os
from dataclasses import dataclass

import fsspec
from levanter.data.text import TextLmDatasetFormat
from levanter.models.qwen import Qwen3Config

from experiments.defaults import default_tokenize
from experiments.evals.delphi.delphi_checkpoints import DELPHI_CHECKPOINTS
from experiments.evals.delphi.gsm8k import N_PROBLEMS, download_gsm8k_step
from fray.cluster import ResourceConfig
from rigging.filesystem import marin_prefix
from marin.evaluation.save_logprobs import default_save_logprobs
from marin.evaluation.utils import discover_hf_checkpoints
from marin.execution.executor import (
    ExecutorStep,
    InputName,
    executor_main,
    output_path_of,
    this_output_path,
    versioned,
)
from marin.execution.remote import remote
from marin.processing.tokenize.data_configs import mixture_for_evaluation
from marin.utils import fsspec_glob

logger = logging.getLogger(__name__)


# === Configuration ===
TPU_TYPE = "v5p-8"
LOGPROBS_TOP_K: int | None = 10
MAX_EVAL_LENGTH = 4096
# All 10 Delphi checkpoints share a Llama-3-equivalent vocab (``nemotron_mix``
# tokenizes with ``meta-llama/Meta-Llama-3.1-8B``; both are in
# ``_EQUIVALENT_TOKENIZERS``). Hardcoded as a HF repo ID because
# ``load_tokenizer`` can't resolve GCS step paths.
TOKENIZER = "marin-community/marin-tokenizer"


# === Build a JSONL with text = prompt + solution (one row per gsm8k problem, in order) ===


@dataclass(frozen=True)
class BuildTextConfig:
    input_path: str
    output_path: str
    n_problems: int | None = None


def build_text_jsonl(config: BuildTextConfig) -> None:
    """Read gsm8k JSONL.gz and emit one row per problem with ``text = prompt + solution``.

    Order is preserved (input row N → output row N), and rows are subsetted to
    ``n_problems`` to match the rollout/grade pipeline. Only the ``text`` field
    is emitted because ``TextLmDatasetFormat`` only consumes ``text``; downstream
    alignment with grading uses row-correspondence and re-joins through
    ``problem_id`` from the original gsm8k JSONL.
    """
    paths = sorted(fsspec_glob(os.path.join(config.input_path, "*.jsonl.gz")))
    if not paths:
        raise FileNotFoundError(f"No gsm8k shards found under {config.input_path}")

    out_path = os.path.join(config.output_path, "text-00000.jsonl.gz")
    written = 0
    with fsspec.open(out_path, "wt", compression="gzip") as fout:
        for path in paths:
            with fsspec.open(path, "rt", compression="gzip") as fin:
                for line in fin:
                    if config.n_problems is not None and written >= config.n_problems:
                        break
                    rec = json.loads(line)
                    fout.write(json.dumps({"text": rec["prompt"] + rec["solution"]}) + "\n")
                    written += 1
            if config.n_problems is not None and written >= config.n_problems:
                break

    if config.n_problems is not None and written < config.n_problems:
        raise RuntimeError(
            f"Requested {config.n_problems} problems but only {written} available under {config.input_path}"
        )
    logger.info(f"Wrote {written} text rows to {out_path}")


build_text_step = ExecutorStep(
    name="raw/gsm8k_logprobs/text",
    fn=remote(build_text_jsonl),
    config=BuildTextConfig(
        input_path=output_path_of(download_gsm8k_step),
        output_path=this_output_path(),
        n_problems=versioned(N_PROBLEMS),
    ),
)


# === Per-checkpoint logprobs ===


def _resolve_step_path(ckpt_rel: str) -> str:
    """Resolve a Delphi ``/hf`` rel path → latest ``/hf/step-N`` snapshot full path.

    Same call ``rollout.py`` makes inside the remote fn (``discover_hf_checkpoints``
    sorts by mtime and returns checkpoints with both ``config.json`` and
    ``tokenizer_config.json`` present); we resolve at graph-build time so the
    model config and tokenizer can be read from the snapshot directly.
    """
    full = os.path.join(marin_prefix(), ckpt_rel)
    snapshots = discover_hf_checkpoints(full)
    if not snapshots:
        raise FileNotFoundError(f"No HF step snapshots under {full}")
    return snapshots[-1]


def build_steps() -> list[ExecutorStep]:
    resolved: dict[str, str] = {slug: _resolve_step_path(rel) for slug, rel in DELPHI_CHECKPOINTS.items()}

    tokenized_step = default_tokenize(
        name="gsm8k_logprobs",
        dataset=build_text_step,
        tokenizer=TOKENIZER,
        format=TextLmDatasetFormat(),
        is_validation=True,
    )
    eval_data = mixture_for_evaluation({"gsm8k": tokenized_step})

    steps: list[ExecutorStep] = [build_text_step, tokenized_step]
    for slug, snap in resolved.items():
        # Read the actual arch/dims back out of the snapshot's config.json.
        # Delphi models are Qwen3-arch with Llama3 RoPE (see
        # ``experiments/scaling_law_sweeps/completed_adamh.py:236``), so Qwen3Config
        # is the right converter template — same pattern as
        # ``experiments/agent_scaling/ot_trace_logprobs.py:75``.
        model_config = Qwen3Config().hf_checkpoint_converter(ref_checkpoint=snap).config_from_hf_checkpoint(snap)
        logprobs_step = default_save_logprobs(
            checkpoint=InputName.hardcoded(snap),
            model=model_config,
            data=eval_data,
            resource_config=ResourceConfig.with_tpu(TPU_TYPE),
            checkpoint_is_hf=True,
            top_k=versioned(LOGPROBS_TOP_K),
            max_eval_length=versioned(MAX_EVAL_LENGTH),
            name=f"delphi/gsm8k/{slug}",
        )
        steps.append(logprobs_step)

    return steps


if __name__ == "__main__":
    executor_main(
        steps=build_steps(),
        description=(
            "Per-token logprobs for the 10 Delphi scaling-ladder checkpoints on "
            "GSM8K (prompt + gold solution, test split)."
        ),
    )
