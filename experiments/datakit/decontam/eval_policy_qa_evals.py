# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Decontamination records for the non-agentic evals of the post-training eval policy (#9566).

Each eval is read from the exact data the policy evaluates: the HF revision or
vendored file pinned by the Evalchemy loader under
``vendor/evalchemy/eval/chat_benchmarks``, or, for the lm-eval tasks, the split
lm-eval v0.4.12 scores at the HF revision pinned here. GSM8K, MMLU-Pro, HumanEval
and GPQA Diamond are indexed by the other suites.

Each item yields one record ``<eval>-<item_id>`` plus view records for separately
meaningful parts (``<eval>-<item_id>-<view>``). Paragraphs are separated by a blank
line. Prompt templates and few-shot preambles that every item shares are not
indexed. NUPA5K-Loose is not indexed: once its per-task instruction template is
removed, an item holds only numbers, which produce no 13-gram of words.
"""

import dataclasses
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import HfFileSystem, hf_hub_download

from experiments.datakit.decontam.pinned_evals import MIN_LETTER_WORDS, PinnedEval, letter_words

REPO_ROOT = Path(__file__).resolve().parents[3]
CHAT_BENCHMARKS = REPO_ROOT / "vendor/evalchemy/eval/chat_benchmarks"

MATH500 = ("HuggingFaceH4/MATH-500", "6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be")
HUMANEVALPLUS = ("evalplus/humanevalplus", "d32357cf319e50e9c8d8dab5ea876c72b0fd321b")
MBPPPLUS = ("evalplus/mbppplus", "b2d74c91837c3f2a20c1299ae98133cbe7cfa077")
MBPPPLUS_FILE = "data/test-00000-of-00001-d5781c9c51e02795.parquet"
OLYMPIADBENCH = ("lmms-lab/OlympiadBench", "c24a1391397fcfe50afaea9210d2c29066494b69")
CRUXEVAL = ("cruxeval-org/cruxeval", "b96af0450242eb4da433032b90998f25588a5d0f")
MRCR = ("openai/mrcr", "f4c69fae7cf81f7ca26b9fee34b392a50f6b8a1d")
# The loaders below do not pin a revision; these are the revisions read on 2026-11-03.
IFBENCH = ("allenai/IFBench_test", "ed7c12452594b7028b15807b86d3938150b663ae")
PIQA = ("baber/piqa", "142f6d7367fd9877f0fb3b5734ea6a545f54cdd1")
WINOGRANDE = ("allenai/winogrande", "01e74176c63542e6b0bcb004dcdea22d94fb67b5")
BOOLQ = ("aps/super_glue", "3de24cf8022e94f4ee4b9d55a6f539891524d646")
TRUTHFULQA = ("truthfulqa/truthful_qa", "741b8276f2d1982aa3d5b832d3ee81ed3b896490")
TRIVIAQA = ("mandarjoshi/trivia_qa", "0f7faf33a3908546c6fd5b73a660e0f8ff173c2f")
OLYMPIADBENCH_SUBSET = "OlympiadBench/data/olympiadbench.jsonl"
AIME24_FILE = "AIME24/data/aime24.json"
FINANCEBENCH_FILE = "FinanceBench/data/financebench.jsonl"

# A part gets its own view record only when it has this many words with a letter.
MIN_VIEW_WORDS = 30
# A test-harness line shared by this many items is boilerplate, not item content.
SHARED_LINE_ITEMS = 3
# The policy runs MRCR at max_length 73,728 tokens, so only rows of at most 65,536
# o200k tokens run. A token holds far fewer than MRCR_MAX_CHARS_PER_TOKEN characters,
# so every such row has at most MRCR_MAX_CHARS characters; all of them are indexed,
# a superset that needs no tokenizer.
MRCR_MAX_TOKENS = 65_536
MRCR_MAX_CHARS_PER_TOKEN = 10
MRCR_MAX_CHARS = MRCR_MAX_TOKENS * MRCR_MAX_CHARS_PER_TOKEN
MRCR_FILES = tuple(f"{n}needle/{n}needle_{part}.parquet" for n in (2, 4, 8) for part in (0, 1))


@dataclasses.dataclass
class EvalRecords:
    """Records of one eval: a full record per item plus views long enough to stand alone."""

    name: str
    rows: list[dict[str, str]] = dataclasses.field(default_factory=list)

    def add(self, item_id: str, parts: list[str], views: dict[str, str] | None = None) -> None:
        text = paragraphs(parts)
        if letter_words(text) < MIN_LETTER_WORDS:
            return
        self.rows.append({"id": f"{self.name}-{item_id}", "text": text})
        for view, view_text in (views or {}).items():
            view_text = view_text.strip()
            if view_text == text or letter_words(view_text) < MIN_VIEW_WORDS:
                continue
            self.rows.append({"id": f"{self.name}-{item_id}-{view}", "text": view_text})


def paragraphs(parts: list[str]) -> str:
    return "\n\n".join(part.strip() for part in parts if part and part.strip())


def hf_file(repo_rev: tuple[str, str], filename: str, cache_dir: Path) -> str:
    repo, revision = repo_rev
    return hf_hub_download(repo, filename, repo_type="dataset", revision=revision, cache_dir=str(cache_dir / "hf"))


def read_jsonl(path: str | Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def read_parquet(
    repo_rev: tuple[str, str], filename: str, cache_dir: Path, columns: list[str] | None = None
) -> list[dict]:
    return pq.read_table(hf_file(repo_rev, filename, cache_dir), columns=columns).to_pylist()


def strip_shared_lines(texts: list[str]) -> list[str]:
    """Drop lines that appear in at least SHARED_LINE_ITEMS of the texts."""
    line_items = Counter(line for text in texts for line in {line.strip() for line in text.splitlines()})
    shared = {line for line, n in line_items.items() if n >= SHARED_LINE_ITEMS}
    return ["\n".join(line for line in text.splitlines() if line.strip() not in shared) for text in texts]


def math500(cache_dir: Path) -> list[dict[str, str]]:
    """Problem and solution (the boxed answer is in the solution); the prompt template is not indexed."""
    out = EvalRecords("math500")
    for row in read_jsonl(hf_file(MATH500, "test.jsonl", cache_dir)):
        item_id = row["unique_id"].removeprefix("test/").removesuffix(".json")
        out.add(item_id, [row["problem"], row["solution"]], {"prompt": row["problem"], "solution": row["solution"]})
    return out.rows


def humanevalplus(cache_dir: Path) -> list[dict[str, str]]:
    """Prompt plus canonical solution; the tests view holds the plus test inputs and results.

    The completion instruction and test-harness lines shared by SHARED_LINE_ITEMS items
    (is_floats, assertion, check loop) are not indexed.
    """
    out = EvalRecords("humanevalplus")
    rows = read_jsonl(hf_file(HUMANEVALPLUS, "test.jsonl", cache_dir))
    tests = strip_shared_lines([row["test"] for row in rows])
    for row, test in zip(rows, tests, strict=True):
        out.add(
            row["task_id"].removeprefix("HumanEval/"),
            [row["prompt"] + row["canonical_solution"]],
            {"prompt": row["prompt"], "solution": row["canonical_solution"], "tests": test},
        )
    return out.rows


def mbppplus(cache_dir: Path) -> list[dict[str, str]]:
    """Problem, the three shown asserts and the reference code; the tests view holds the plus tests.

    The 3-shot preamble the loader prepends to every prompt is not indexed (its three
    items are indexed once as their own records), nor are shared test-harness lines.
    """
    out = EvalRecords("mbppplus")
    rows = read_parquet(MBPPPLUS, MBPPPLUS_FILE, cache_dir)
    tests = strip_shared_lines([row["test"] for row in rows])
    for row, test in zip(rows, tests, strict=True):
        prompt = paragraphs([row["prompt"], "\n".join(row["test_list"])])
        out.add(str(row["task_id"]), [prompt, row["code"]], {"prompt": prompt, "solution": row["code"], "tests": test})
    return out.rows


def olympiadbench(cache_dir: Path) -> list[dict[str, str]]:
    """The whole test_en pool the policy's vendored subset was drawn from.

    Records hold context, question and final answer; the pool has no reference
    solutions, and images are not indexed. Every subset problem must match its pool row.
    """
    subset = read_jsonl(CHAT_BENCHMARKS / OLYMPIADBENCH_SUBSET)
    out = EvalRecords("olympiadbench")
    columns = ["question_id", "context", "question", "final_answer"]
    by_id = {}
    for row in read_parquet(OLYMPIADBENCH, "data/test_en-00000-of-00001.parquet", cache_dir, columns):
        problem = (row["context"] + "\n\n" if row["context"] else "") + row["question"]
        by_id[row["question_id"]] = problem
        out.add(row["question_id"], [problem, "\n".join(row["final_answer"] or [])])
    mismatched = [row["id"] for row in subset if by_id.get(row["id"]) != row["problem"]]
    if mismatched:
        raise ValueError(f"OlympiadBench subset items differ from the pinned pool: {mismatched}")
    return out.rows


def aime24(cache_dir: Path) -> list[dict[str, str]]:
    """Problem, reference solution and answer; the prompt template is not indexed."""
    out = EvalRecords("aime24")
    for row in read_jsonl(CHAT_BENCHMARKS / AIME24_FILE):
        out.add(
            row["id"].removeprefix("aime24-"),
            [row["problem"], row["reference_solution"], str(row["expected_answer"])],
            {"prompt": row["problem"], "solution": row["reference_solution"]},
        )
    return out.rows


def cruxeval(cache_dir: Path) -> list[dict[str, str]]:
    """The function and its completed assertion, which serve both the input and output directions."""
    out = EvalRecords("cruxeval")
    for row in read_jsonl(hf_file(CRUXEVAL, "test.jsonl", cache_dir)):
        out.add(row["id"], [row["code"], f"assert f({row['input']}) == {row['output']}"], {"prompt": row["code"]})
    return out.rows


def financebench(cache_dir: Path) -> list[dict[str, str]]:
    """The vendored 50-item sample: evidence page, question and answer. Filing PDFs are not indexed."""
    out = EvalRecords("financebench")
    for row in read_jsonl(CHAT_BENCHMARKS / FINANCEBENCH_FILE):
        out.add(
            row["id"].removeprefix("financebench_id_"),
            [row["evidence_text"], row["question"], row["answer"]],
            {"prompt": row["question"], "answer": row["answer"]},
        )
    return out.rows


def ifbench(cache_dir: Path) -> list[dict[str, str]]:
    """The prompt with its inline constraint; there is no reference answer."""
    out = EvalRecords("ifbench")
    for row in read_parquet(IFBENCH, "data/train-00000-of-00001.parquet", cache_dir, ["key", "prompt"]):
        out.add(str(row["key"]), [row["prompt"]])
    return out.rows


def mrcr(cache_dir: Path) -> list[dict[str, str]]:
    """The needle request, the final question, every needle response and the expected answer.

    The synthetic filler turns and the leading few-shot example are not indexed. Item id
    is ``<file stem>_<row index in that file>``.
    """
    out = EvalRecords("mrcr")
    fs = HfFileSystem()
    for filename in MRCR_FILES:
        with fs.open(f"datasets/{MRCR[0]}@{MRCR[1]}/{filename}") as f:
            rows = pq.read_table(f).to_pylist()
        for index, row in enumerate(rows):
            if row["n_chars"] > MRCR_MAX_CHARS:
                continue
            messages = json.loads(row["prompt"])
            target = row["answer"].removeprefix(row["random_string_to_prepend"])
            (position,) = [i for i, m in enumerate(messages) if m["role"] == "assistant" and m["content"] == target]
            request = messages[position - 1]["content"]
            needles = [
                messages[i + 1]["content"]
                for i, m in enumerate(messages[:-1])
                if m["role"] == "user" and m["content"] == request and messages[i + 1]["role"] == "assistant"
            ]
            if len(needles) != row["n_needles"]:
                raise ValueError(f"MRCR {filename} row {index}: {len(needles)} needles, expected {row['n_needles']}")
            item_id = f"{Path(filename).stem}_{index}"
            out.add(item_id, [request, messages[-1]["content"], *needles, row["answer"]], {"answer": row["answer"]})
    return out.rows


def piqa(cache_dir: Path) -> list[dict[str, str]]:
    """lm-eval piqa scores validation: goal and both solutions; id is the row index."""
    out = EvalRecords("piqa")
    for index, row in enumerate(read_parquet(PIQA, "piqa_validation.parquet", cache_dir)):
        out.add(str(index), [row["goal"], row["sol1"], row["sol2"]])
    return out.rows


def winogrande(cache_dir: Path) -> list[dict[str, str]]:
    """lm-eval scores winogrande_xl validation: the sentence filled with the correct option, then raw."""
    out = EvalRecords("winogrande")
    rows = read_parquet(WINOGRANDE, "winogrande_xl/validation-00000-of-00001.parquet", cache_dir)
    for index, row in enumerate(rows):
        answer = row["option1"] if row["answer"] == "1" else row["option2"]
        out.add(str(index), [row["sentence"].replace("_", answer, 1), row["sentence"]])
    return out.rows


def boolq(cache_dir: Path) -> list[dict[str, str]]:
    """lm-eval boolq scores validation: passage, question and answer."""
    out = EvalRecords("boolq")
    for row in read_parquet(BOOLQ, "boolq/validation-00000-of-00001.parquet", cache_dir):
        out.add(str(row["idx"]), [row["passage"], row["question"] + "?", "yes" if row["label"] == 1 else "no"])
    return out.rows


def truthfulqa(cache_dir: Path) -> list[dict[str, str]]:
    """lm-eval truthfulqa_mc2 scores validation: question and all MC2 choices; id is the row index."""
    out = EvalRecords("truthfulqa")
    rows = read_parquet(TRUTHFULQA, "multiple_choice/validation-00000-of-00001.parquet", cache_dir)
    for index, row in enumerate(rows):
        out.add(str(index), [row["question"], "\n".join(row["mc2_targets"]["choices"])])
    return out.rows


def triviaqa(cache_dir: Path) -> list[dict[str, str]]:
    """lm-eval triviaqa scores rc.nocontext validation: question and answer value.

    Rows repeating a question id with the same question and answer are indexed once;
    an id with a different question or answer gets a suffixed id. Aliases and the
    5-shot train examples are not indexed.
    """
    rows = read_parquet(
        TRIVIAQA, "rc.nocontext/validation-00000-of-00001.parquet", cache_dir, ["question_id", "question", "answer"]
    )
    out = EvalRecords("triviaqa")
    texts_by_id: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        text = paragraphs([row["question"], row["answer"]["value"]])
        if text not in texts_by_id[row["question_id"]]:
            texts_by_id[row["question_id"]].append(text)
    for question_id, texts in texts_by_id.items():
        for n, text in enumerate(texts):
            out.add(question_id if n == 0 else f"{question_id}_{n}", [text])
    return out.rows


def _hf(repo_rev: tuple[str, str]) -> tuple[str, str]:
    return f"https://huggingface.co/datasets/{repo_rev[0]}", repo_rev[1]


def _vendored(relative: str) -> tuple[str, str]:
    """Source and revision of a vendored Evalchemy data file: its repository path and content digest."""
    path = CHAT_BENCHMARKS / relative
    return str(path.relative_to(REPO_ROOT)), f"sha256:{hashlib.sha256(path.read_bytes()).hexdigest()}"


EVAL_POLICY_QA_EVALS: tuple[PinnedEval, ...] = (
    PinnedEval("MATH-500", "math500", *_hf(MATH500), 1037, math500),
    PinnedEval("HumanEval+", "humanevalplus", *_hf(HUMANEVALPLUS), 395, humanevalplus),
    PinnedEval("MBPP+", "mbppplus", *_hf(MBPPPLUS), 675, mbppplus),
    PinnedEval("OlympiadBench", "olympiadbench", *_hf(OLYMPIADBENCH), 2078, olympiadbench),
    PinnedEval("PIQA", "piqa", *_hf(PIQA), 1684, piqa),
    PinnedEval("WinoGrande", "winogrande", *_hf(WINOGRANDE), 1267, winogrande),
    PinnedEval("BoolQ", "boolq", *_hf(BOOLQ), 3270, boolq),
    PinnedEval("TruthfulQA", "truthfulqa", *_hf(TRUTHFULQA), 817, truthfulqa),
    PinnedEval("TriviaQA", "triviaqa", *_hf(TRIVIAQA), 5793, triviaqa),
    PinnedEval("AIME24", "aime24", *_vendored(AIME24_FILE), 83, aime24),
    PinnedEval("CRUXEval", "cruxeval", *_hf(CRUXEVAL), 635, cruxeval),
    PinnedEval("FinanceBench", "financebench", *_vendored(FINANCEBENCH_FILE), 79, financebench),
    PinnedEval("IFBench", "ifbench", *_hf(IFBENCH), 297, ifbench),
    PinnedEval("MRCR", "mrcr", *_hf(MRCR), 3012, mrcr),
)
