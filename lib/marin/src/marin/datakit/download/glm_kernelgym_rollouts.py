# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""marin-community/glm-5.2-kernelgym-rollouts dataset download and transform.

3,200 GPU kernel-optimization trajectories from ``zai-org/GLM-5.2-FP8`` on KernelGym
(KernelBench-derived) tasks, split evenly between inline CUDA and Triton. Each row is a
repair loop: the model writes a kernel, KernelGym evaluates it, the verdict comes back as
a user message, and the model tries again. Every row ends on a kernel that passed the
correctness tests, so the corpus is entirely successful trajectories.

Two properties of the raw rows shape this transform.

``messages`` carries sampler bookkeeping that ``turns`` does not. Every row ends with a
bare ``KERNELGYM_FINAL`` assistant message — the stop sentinel, not model output — and
across the corpus 122 further assistant messages are rejected answers to the interleaved
"is another attempt worthwhile?" prompt. ``turns`` holds only the real generations, so we
keep an assistant message only when it appears there.

Reasoning is inline in assistant content with a closing ``</think>`` and no opener, because
the opener comes from the GLM chat template and was never echoed back. Like
``superior_reasoning`` and ``synthetic1``, the other sources whose reasoning arrives inline
in a text field, we leave the reasoning inline rather than introduce markup the plain-text
tokenizer would not read; the tag itself becomes a paragraph break (see
:func:`join_reasoning_and_answer`). A turn missing the tag entirely is one that spent its
token budget mid-reasoning: 3,711 of 3,724 untruncated turns close exactly once, while
2,181 of 2,459 truncated turns never close.
"""

from enum import StrEnum

from fray.types import ResourceConfig
from zephyr import counters
from zephyr.dataset import Dataset
from zephyr.execution import ZephyrContext
from zephyr.readers import load_jsonl

from marin.datakit.download.huggingface import download_hf_step
from marin.datakit.download.rollout_transforms import render_role_message, text_document
from marin.datakit.normalize import normalize_step
from marin.execution.step_spec import StepSpec

HF_DATASET_ID = "marin-community/glm-5.2-kernelgym-rollouts"
HF_REVISION = "6dbe98d"
DATA_GLOB = "data/*.jsonl.gz"

REASONING_CLOSE_TAG = "</think>"


class TruncationFilter(StrEnum):
    """Which rows to drop when a generation turn spent its whole sampling budget.

    A turn that spent the budget was cut off mid-generation. ``stop_reason`` reads
    ``model_final`` either way, so the only signal is ``usage.completion_tokens`` reaching
    the row's own ``max_tokens`` (16,384 across this corpus, but read per row rather than
    pinned here). 2,459 of 6,183 turns are truncated.
    """

    FINAL_TURN = "final_turn"
    """Drop only rows whose last turn was truncated (365 of 3,200 rows). Truncation
    earlier in a trajectory is signal, not damage: KernelGym rejects the cut-off kernel
    and the next turn recovers from it."""

    ANY_TURN = "any_turn"
    """Drop rows with any truncated turn (1,946 of 3,200 rows). Use when no unclosed
    reasoning block may appear anywhere in the context window."""


def join_reasoning_and_answer(content: str) -> str:
    """Replace GLM's dangling ``</think>`` with a paragraph break.

    The shared ``strip_think_tags`` deletes the tag outright, which works for sources that
    wrap it in newlines. GLM does not: responses read ``...write the final code.</think>Looking
    at the profile...``, so deleting the tag alone would run the last reasoning sentence into
    the first word of the answer.
    """
    return "\n\n".join(part.strip() for part in content.split(REASONING_CLOSE_TAG) if part.strip())


def render_message(message: dict) -> str:
    """Render one message as a role-tagged block, resolving GLM's dangling reasoning tag."""
    if message["role"] != "assistant":
        return render_role_message(message)
    return render_role_message({**message, "content": join_reasoning_and_answer(message["content"])})


def render_conversation(messages: list[dict], turns: list[dict]) -> str:
    """Render a trajectory as a tagged transcript of its recorded generation turns.

    Assistant messages absent from ``turns`` are sampler bookkeeping — the
    ``KERNELGYM_FINAL`` stop sentinel and rejected replies to the "keep optimizing?"
    prompt — and are dropped along with the trailing prompts they leave unanswered, so the
    transcript ends on the kernel that passed.
    """
    generations = {turn["response"] for turn in turns}
    kept = [m for m in messages if m["role"] != "assistant" or m["content"] in generations]
    while kept and kept[-1]["role"] != "assistant":
        kept.pop()

    return "\n\n".join(render_message(m) for m in kept)


def row_to_doc(row: dict, truncation_filter: TruncationFilter) -> list[dict]:
    messages = row["messages"]
    turns = row["turns"]
    if not messages or not turns:
        counters.pipeline.update_counter("glm_kernelgym_rollouts/dropped_empty", 1)
        return []

    truncated = [turn["usage"]["completion_tokens"] >= row["max_tokens"] for turn in turns]
    if truncated[-1] or (truncation_filter is TruncationFilter.ANY_TURN and any(truncated)):
        counters.pipeline.update_counter("glm_kernelgym_rollouts/dropped_truncated", 1)
        return []

    text = render_conversation(messages, turns)
    if not text:
        counters.pipeline.update_counter("glm_kernelgym_rollouts/dropped_empty", 1)
        return []

    counters.pipeline.update_counter("glm_kernelgym_rollouts/kept", 1)
    return [text_document(text, HF_DATASET_ID)]


def transform(input_path: str, output_path: str, truncation_filter: TruncationFilter) -> None:
    pipeline = (
        Dataset.from_files(f"{input_path}/**/*.jsonl.gz")
        .flat_map(load_jsonl)
        .flat_map(lambda row: row_to_doc(row, truncation_filter))
        .write_parquet(f"{output_path}/data-{{shard:05d}}-of-{{total:05d}}.parquet", skip_existing=True)
    )
    ctx = ZephyrContext(name="glm-kernelgym-rollouts-transform", resources=ResourceConfig(cpu=1, ram="8g"))
    ctx.execute(pipeline)


def download_glm_kernelgym_rollouts_step(
    truncation_filter: TruncationFilter = TruncationFilter.FINAL_TURN,
) -> StepSpec:
    """Download and transform GLM-5.2 KernelGym rollouts into tagged transcripts.

    ``truncation_filter`` is a parameter rather than a hardcoded choice because it enters
    the step hash: each setting lands at its own output path, so producing the strict
    variant is a rerun rather than an edit, and it cannot silently serve the lenient
    variant's cache.
    """
    dl = download_hf_step(
        "raw/glm-5.2-kernelgym-rollouts",
        hf_dataset_id=HF_DATASET_ID,
        revision=HF_REVISION,
        hf_urls_glob=[DATA_GLOB],
    )

    return StepSpec(
        name="processed/glm-5.2-kernelgym-rollouts",
        deps=[dl],
        fn=lambda output_path: transform(
            input_path=dl.output_path,
            output_path=output_path,
            truncation_filter=truncation_filter,
        ),
        hash_attrs={"version": "v1", "truncation_filter": truncation_filter.value},
    )


def glm_kernelgym_rollouts_normalize_steps() -> tuple[StepSpec, ...]:
    """Return the full ``(download+transform, normalize)`` chain for glm-5.2-kernelgym-rollouts."""
    processed = download_glm_kernelgym_rollouts_step()
    return (
        processed,
        normalize_step(name="normalized/glm-5.2-kernelgym-rollouts", download=processed),
    )
