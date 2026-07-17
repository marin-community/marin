# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Generate representative vLLM E2E prompts from pinned row selectors."""

import argparse
import hashlib
import json
import logging
from pathlib import Path

from datasets import load_dataset
from huggingface_hub import hf_hub_download, snapshot_download
from levanter.tokenizers import load_tokenizer
from marin.transform.evaluation.raw_lm_eval import GSM8K_COT_FEWSHOT_EXAMPLES, _render_mmlu_description
from rigging.filesystem import StoragePath

TOKENIZER_NAME = "marin-community/marin-tokenizer"
TOKENIZER_REVISION = "a5ca45f2feb6c959bd87b81689aa7279b5bdcaa2"
PROMPT_FIXTURE_PREFIX = "gs://marin-public/test-data/vllm/e2e/representative-eval-prompts"
SELECTORS_PATH = Path(__file__).parent / "resources" / "representative_eval_prompt_selectors.json"

logger = logging.getLogger(__name__)


def _render_mmlu_row(row: dict, include_answer: bool) -> str:
    answer = f" {'ABCD'[row['answer']]}" if include_answer else ""
    return "\n".join(
        [
            f"Question: {row['question']}",
            *(f"{label}. {choice}" for label, choice in zip("ABCD", row["choices"], strict=True)),
            f"Answer:{answer}",
        ]
    )


def _render_mmmlu_row(row: dict, include_answer: bool) -> str:
    answer = f" {row['Answer']}" if include_answer else ""
    return "\n".join(
        [
            f"Question: {row['Question']}",
            *(f"{label}. {row[label]}" for label in "ABCD"),
            f"Answer:{answer}",
        ]
    )


def build_prompt_fixture() -> bytes:
    selectors = json.loads(SELECTORS_PATH.read_text())
    prompts = {}

    source = selectors["humaneval"]
    rows = load_dataset(source["dataset"], split="test", revision=source["revision"])
    for case_id, row_index in source["cases"]:
        prompts[case_id] = rows[row_index]["prompt"].strip()

    source = selectors["ifeval"]
    rows = load_dataset(source["dataset"], split="train", revision=source["revision"])
    for case_id, row_index in source["cases"]:
        prompts[case_id] = rows[row_index]["prompt"].strip()

    source = selectors["mmlu"]
    validation = load_dataset(source["dataset"], source["config"], split="validation", revision=source["revision"])
    dev = load_dataset(source["dataset"], source["config"], split="dev", revision=source["revision"])
    for case_id, query_index, support_indices in source["cases"]:
        query = validation[query_index]
        prompts[case_id] = "\n\n".join(
            [
                _render_mmlu_description(query["subject"]),
                *(_render_mmlu_row(dev[index], True) for index in support_indices),
                _render_mmlu_row(query, False),
            ]
        )

    source = selectors["gsm8k"]
    test = load_dataset(source["dataset"], source["config"], split="test", revision=source["revision"])
    train = load_dataset(source["dataset"], source["config"], split="train", revision=source["revision"])
    fixed_supports = [f"Q: {question}\nA: {answer}" for question, answer in GSM8K_COT_FEWSHOT_EXAMPLES[:4]]
    for case_id, query_index in source["fixed_cases"]:
        prompts[case_id] = "\n\n".join([*fixed_supports, f"Q: {test[query_index]['question']}\nA:"])
    for case_id, query_index, support_indices in source["train_cases"]:
        supports = [f"Q: {train[index]['question']}\nA: {train[index]['answer']}" for index in support_indices]
        prompts[case_id] = "\n\n".join([*supports, f"Q: {test[query_index]['question']}\nA:"])

    source = selectors["structeval"]
    rows = load_dataset(source["dataset"], split="test", revision=source["revision"])
    for case_id, row_index in source["cases"]:
        prompts[case_id] = rows[row_index]["query"].strip()

    source = selectors["atlas"]
    rows = load_dataset(source["dataset"], split="train", revision=source["revision"])
    for case_id, row_index in source["direct_cases"]:
        prompts[case_id] = rows[row_index]["prompt"].strip()
    for case_id, row_indices in source["catalog_cases"]:
        selected = [rows[index] for index in row_indices]
        tools = [
            {
                "name": row["tool"],
                "description": row["description"],
                "input_schema": json.loads(row["input_schema"]),
            }
            for row in selected
        ]
        prompts[case_id] = (
            "Select the single best tool for the user request. Respond with one JSON tool call and no prose.\n\n"
            f"Available tools:\n{json.dumps(tools, ensure_ascii=False, indent=2)}\n\n"
            f"User request: {selected[-1]['scenario']}\n\nTool call:"
        )

    source = selectors["mmmlu"]
    for case_id, config, topic, query_index, support_indices in source["cases"]:
        rows = load_dataset(source["dataset"], config, split="test", revision=source["revision"])
        prompts[case_id] = "\n\n".join(
            [
                f"The following are multiple choice questions (with answers) about {topic}.",
                *(_render_mmmlu_row(rows[index], True) for index in support_indices),
                _render_mmmlu_row(rows[query_index], False),
            ]
        )

    source = selectors["longbench"]
    path = hf_hub_download(source["dataset"], source["file"], repo_type="dataset", revision=source["revision"])
    rows = json.loads(Path(path).read_text())
    for case_id, row_index in source["cases"]:
        row = rows[row_index]
        prompts[case_id] = (
            "Read the following context and answer the multiple-choice question.\n\n"
            f"Context:\n{row['context']}\n\n"
            f"Question: {row['question']}\n"
            f"A. {row['choice_A']}\nB. {row['choice_B']}\nC. {row['choice_C']}\nD. {row['choice_D']}\nAnswer:"
        )

    tokenizer = load_tokenizer(
        snapshot_download(
            TOKENIZER_NAME,
            revision=TOKENIZER_REVISION,
            allow_patterns=["tokenizer*", "special_tokens*", "added_tokens*", "chat_template*"],
        )
    )
    cases = [
        {"id": case_id, "prompt": prompt, "prompt_token_ids": tokenizer.encode(prompt, add_special_tokens=False)}
        for case_id, prompt in prompts.items()
    ]
    return (
        json.dumps(
            {
                "tokenizer": TOKENIZER_NAME,
                "tokenizer_revision": TOKENIZER_REVISION,
                "cases": sorted(cases, key=lambda case: case["id"]),
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode()


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    fixture_bytes = build_prompt_fixture()
    fixture_sha256 = hashlib.sha256(fixture_bytes).hexdigest()
    fixture_uri = f"{PROMPT_FIXTURE_PREFIX}/{fixture_sha256}.json"
    if args.stage:
        StoragePath(fixture_uri).write_bytes(fixture_bytes)
    if args.output:
        args.output.write_bytes(fixture_bytes)
    logger.info(
        "Prompt fixture: uri=%s sha256=%s byte_size=%d",
        fixture_uri,
        fixture_sha256,
        len(fixture_bytes),
    )


if __name__ == "__main__":
    main()
