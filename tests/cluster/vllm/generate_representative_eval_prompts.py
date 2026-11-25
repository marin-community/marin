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
from rigging.filesystem.storage_path import StoragePath

MMLU_DESCRIPTION_TEMPLATE = "The following are multiple choice questions (with answers) about {subject}."
# lm-eval's gsm8k-cot few-shot exemplars.
GSM8K_COT_FEWSHOT_EXAMPLES: tuple[tuple[str, str], ...] = (
    (
        "There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, "
        "there will be 21 trees. How many trees did the grove workers plant today?",
        "There are 15 trees originally. Then there were 21 trees after some more were planted. "
        "So there must have been 21 - 15 = 6. The answer is 6.",
    ),
    (
        "If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?",
        "There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5. The answer is 5.",
    ),
    (
        "Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?",
        "Originally, Leah had 32 chocolates. Her sister had 42. So in total they had 32 + 42 = 74. "
        "After eating 35, they had 74 - 35 = 39. The answer is 39.",
    ),
    (
        "Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. "
        "How many lollipops did Jason give to Denny?",
        "Jason started with 20 lollipops. Then he had 12 after giving some to Denny. "
        "So he gave Denny 20 - 12 = 8. The answer is 8.",
    ),
    (
        "Shawn has five toys. For Christmas, he got two toys each from his mom and dad. "
        "How many toys does he have now?",
        "Shawn started with 5 toys. If he got 2 toys each from his mom and dad, then that is 4 more toys. "
        "5 + 4 = 9. The answer is 9.",
    ),
    (
        "There were nine computers in the server room. Five more computers were installed each day, "
        "from monday to thursday. How many computers are now in the server room?",
        "There were originally 9 computers. For each of 4 days, 5 more computers were added. "
        "So 5 * 4 = 20 computers were added. 9 + 20 is 29. The answer is 29.",
    ),
    (
        "Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. "
        "How many golf balls did he have at the end of wednesday?",
        "Michael started with 58 golf balls. After losing 23 on tuesday, he had 58 - 23 = 35. "
        "After losing 2 more, he had 35 - 2 = 33 golf balls. The answer is 33.",
    ),
    (
        "Olivia has $23. She bought five bagels for $3 each. How much money does she have left?",
        "Olivia had 23 dollars. 5 bagels for 3 dollars each will be 5 x 3 = 15 dollars. "
        "So she has 23 - 15 dollars left. 23 - 15 is 8. The answer is 8.",
    ),
)


def _render_mmlu_description(subject: str) -> str:
    if not subject:
        raise ValueError("MMLU examples must include a subject")
    return MMLU_DESCRIPTION_TEMPLATE.format(subject=subject.replace("_", " "))


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
