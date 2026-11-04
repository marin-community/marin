# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Build contamination-scan targets for the non-agentic post-training policy evals.

Covers the evals of the Marin post-training eval policy (issue #9566) that do not
run in an agent harness. Each eval is read from the exact data the campaign
evaluates: the HF revision or vendored file pinned by the Evalchemy loader under
``vendor/evalchemy/eval/chat_benchmarks``, or, for the lm-eval tasks, the split
lm-eval v0.4.12 scores at the HF revision pinned here.

Writes ``OUT_DIR/<eval>/eval.parquet`` with string columns ``id`` and ``text``, one
record per item (id ``<eval>-<item_id>``) plus view records for separately
meaningful parts (id ``<eval>-<item_id>-<view>``), and ``OUT_DIR/.manifest.json``.
Paragraphs are separated by a blank line. Prompt templates and few-shot preambles
that every item shares are not indexed. ``OUT_DIR/.verbatim_items.json`` holds
each item's question and answer, keyed by its full record id, for
`verbatim_copies.py`: ``free`` items with long answers, ``near`` items whose short
answer counts only after the question (``distractors`` are the wrong options), and
``question`` items without a reference answer. MRCR and NUPA5K have none.

Usage: uv run --with tiktoken python experiments/contam_report/targets_policy_qa.py OUT_DIR
"""

import hashlib
import json
import os
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import tiktoken
from huggingface_hub import HfFileSystem, hf_hub_download

from experiments.contam_report.prepare_targets import write

REPO_ROOT = Path(__file__).resolve().parents[2]
CHAT_BENCHMARKS = REPO_ROOT / "vendor/evalchemy/eval/chat_benchmarks"

MATH500 = ("HuggingFaceH4/MATH-500", "6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be")
HUMANEVALPLUS = ("evalplus/humanevalplus", "d32357cf319e50e9c8d8dab5ea876c72b0fd321b")
MBPPPLUS = ("evalplus/mbppplus", "b2d74c91837c3f2a20c1299ae98133cbe7cfa077")
MBPPPLUS_FILE = "data/test-00000-of-00001-d5781c9c51e02795.parquet"
OLYMPIADBENCH = ("lmms-lab/OlympiadBench", "c24a1391397fcfe50afaea9210d2c29066494b69")
CRUXEVAL = ("cruxeval-org/cruxeval", "b96af0450242eb4da433032b90998f25588a5d0f")
MRCR = ("openai/mrcr", "f4c69fae7cf81f7ca26b9fee34b392a50f6b8a1d")
NUPA = ("HaotongYang/NUPA_text", "01e3831ec00dfd618a77d9f6fe7fc0d327ad16d7")
# The loaders below do not pin a revision; these are the revisions read on 2026-11-03.
IFBENCH = ("allenai/IFBench_test", "ed7c12452594b7028b15807b86d3938150b663ae")
PIQA = ("baber/piqa", "142f6d7367fd9877f0fb3b5734ea6a545f54cdd1")
WINOGRANDE = ("allenai/winogrande", "01e74176c63542e6b0bcb004dcdea22d94fb67b5")
BOOLQ = ("aps/super_glue", "3de24cf8022e94f4ee4b9d55a6f539891524d646")
TRUTHFULQA = ("truthfulqa/truthful_qa", "741b8276f2d1982aa3d5b832d3ee81ed3b896490")
TRIVIAQA = ("mandarjoshi/trivia_qa", "0f7faf33a3908546c6fd5b73a660e0f8ff173c2f")

# A record needs this many words with a letter to produce a 13-gram.
MIN_RECORD_WORDS = 13
# A part gets its own view record only when it has this many words with a letter.
MIN_VIEW_WORDS = 30
# A test-harness line shared by this many items is boilerplate, not item content.
SHARED_LINE_ITEMS = 3
# MBPPPlus prefixes every prompt with examples[1..3] as few-shot examples.
MBPPPLUS_SHOT_INDICES = (1, 2, 3)
# The policy runs MRCR at max_length 73728, so only bins with upper bound <= 65536 run.
MRCR_BIN_UPPER_BOUNDS = (8_192, 16_384, 32_768, 65_536, 131_072, 262_144, 524_288, 1_048_576)
MRCR_MAX_LENGTH = 73_728
# o200k tokens hold far fewer than this many characters, so longer rows cannot fit a bin.
MRCR_MAX_CHARS_PER_TOKEN = 10
MRCR_FILES = tuple(f"{n}needle/{n}needle_{part}.parquet" for n in (2, 4, 8) for part in (0, 1))


@dataclass
class EvalTargets:
    name: str
    source: str
    revision: str
    split: str
    items: int
    rows: list[dict] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    skipped: int = 0
    view_counts: Counter = field(default_factory=Counter)
    extra: dict = field(default_factory=dict)
    verbatim_items: dict[str, dict] = field(default_factory=dict)

    def add(self, item_id: str, parts: list[str], views: dict[str, str] | None = None, item: dict | None = None) -> None:
        """Add the full item record and the views long enough to stand alone, and its verbatim ``item``."""
        text = paragraphs(parts)
        if letter_words(text) < MIN_RECORD_WORDS:
            self.skipped += 1
            return
        self.rows.append({"id": f"{self.name}-{item_id}", "text": text})
        if item is not None:
            self.verbatim_items[f"{self.name}-{item_id}"] = {"eval": self.name, **item}
        for view, view_text in (views or {}).items():
            view_text = view_text.strip()
            if view_text == text or letter_words(view_text) < MIN_VIEW_WORDS:
                continue
            self.rows.append({"id": f"{self.name}-{item_id}-{view}", "text": view_text})
            self.view_counts[view] += 1

    def manifest(self) -> dict:
        notes = list(self.notes)
        if self.skipped:
            notes.append(f"{self.skipped} items skipped: fewer than {MIN_RECORD_WORDS} words with a letter.")
        if self.view_counts:
            notes.append(f"Views with fewer than {MIN_VIEW_WORDS} words with a letter are omitted.")
        return {
            "name": self.name,
            "source": self.source,
            "revision": self.revision,
            "split": self.split,
            "items": self.items,
            "records": len(self.rows) - sum(self.view_counts.values()),
            "views": dict(sorted(self.view_counts.items())),
            "notes": " ".join(notes),
            **self.extra,
        }


def free(question: str, answer: str) -> dict:
    return {"kind": "free", "question": question, "answers": [answer]}


def near(question: str, answers: list[str], distractors: list[str] = ()) -> dict:
    return {"kind": "near", "question": question, "answers": answers, "distractors": list(distractors)}


def letter_words(text: str) -> int:
    return sum(1 for word in text.split() if any(c.isalpha() for c in word))


def paragraphs(parts: list[str]) -> str:
    return "\n\n".join(part.strip() for part in parts if part and part.strip())


def hf_file(repo_rev: tuple[str, str], filename: str) -> str:
    repo, revision = repo_rev
    return hf_hub_download(repo, filename, repo_type="dataset", revision=revision)


def read_jsonl(path: str | Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def read_parquet(repo_rev: tuple[str, str], filename: str, columns: list[str] | None = None) -> list[dict]:
    return pq.read_table(hf_file(repo_rev, filename), columns=columns).to_pylist()


def vendored(relative: str) -> tuple[Path, str]:
    """Return a vendored Evalchemy data file and its content digest."""
    path = CHAT_BENCHMARKS / relative
    return path, f"sha256:{hashlib.sha256(path.read_bytes()).hexdigest()}"


def strip_shared_lines(texts: list[str]) -> tuple[list[str], int]:
    """Drop lines that appear in at least SHARED_LINE_ITEMS of the texts."""
    line_items = Counter(line for text in texts for line in {line.strip() for line in text.splitlines()})
    shared = {line for line, n in line_items.items() if n >= SHARED_LINE_ITEMS}
    stripped = ["\n".join(line for line in text.splitlines() if line.strip() not in shared) for text in texts]
    return stripped, len(shared)


def math500() -> EvalTargets:
    out = EvalTargets("math500", MATH500[0], MATH500[1], "test", 0)
    rows = read_jsonl(hf_file(MATH500, "test.jsonl"))
    out.items = len(rows)
    for row in rows:
        item_id = row["unique_id"].removeprefix("test/").removesuffix(".json")
        out.add(
            item_id,
            [row["problem"], row["solution"]],
            {"prompt": row["problem"], "solution": row["solution"]},
            free(row["problem"], row["solution"]),
        )
    out.notes.append(
        "Records hold problem and solution (the boxed answer is in the solution); the loader's 'Mark your solution "
        "with \\boxed' template is not indexed."
    )
    return out


def humanevalplus() -> EvalTargets:
    out = EvalTargets("humanevalplus", HUMANEVALPLUS[0], HUMANEVALPLUS[1], "test.jsonl", 0)
    rows = read_jsonl(hf_file(HUMANEVALPLUS, "test.jsonl"))
    out.items = len(rows)
    tests, shared = strip_shared_lines([row["test"] for row in rows])
    for row, test in zip(rows, tests, strict=True):
        item_id = row["task_id"].removeprefix("HumanEval/")
        out.add(
            item_id,
            [row["prompt"] + row["canonical_solution"]],
            {"prompt": row["prompt"], "solution": row["canonical_solution"], "tests": test},
            free(row["prompt"], row["canonical_solution"]),
        )
    out.notes.append(
        "Records hold prompt plus canonical solution; the tests view holds the plus test inputs and results. "
        f"Not indexed: the loader's completion instruction and {shared} test-harness lines (is_floats, assertion, "
        f"check loop) shared by >= {SHARED_LINE_ITEMS} items."
    )
    return out


def mbppplus() -> EvalTargets:
    out = EvalTargets("mbppplus", MBPPPLUS[0], MBPPPLUS[1], "test", 0)
    rows = read_parquet(MBPPPLUS, MBPPPLUS_FILE)
    out.items = len(rows)
    tests, shared = strip_shared_lines([row["test"] for row in rows])
    for row, test in zip(rows, tests, strict=True):
        prompt = paragraphs([row["prompt"], "\n".join(row["test_list"])])
        out.add(
            str(row["task_id"]),
            [prompt, row["code"]],
            {"prompt": prompt, "solution": row["code"], "tests": test},
            free(row["prompt"], row["code"]),
        )
    shots = ", ".join(str(rows[i]["task_id"]) for i in MBPPPLUS_SHOT_INDICES)
    out.notes.append(
        "Records hold problem, the three shown asserts and the reference code; the tests view holds the plus test "
        f"inputs and results. Not indexed: the 3-shot preamble (task ids {shots}) the loader prepends to every "
        f"prompt (those items are indexed once as their own records), and {shared} test-harness lines shared by "
        f">= {SHARED_LINE_ITEMS} items."
    )
    return out


def olympiadbench() -> EvalTargets:
    subset_path, subset_digest = vendored("OlympiadBench/data/olympiadbench.jsonl")
    subset = read_jsonl(subset_path)
    out = EvalTargets("olympiadbench", OLYMPIADBENCH[0], OLYMPIADBENCH[1], "test_en", 0)
    columns = ["question_id", "context", "question", "final_answer", "source"]
    pool = read_parquet(OLYMPIADBENCH, "data/test_en-00000-of-00001.parquet", columns)
    out.items = len(pool)
    by_id = {}
    for row in pool:
        problem = (row["context"] + "\n\n" if row["context"] else "") + row["question"]
        by_id[row["question_id"]] = problem
        answers = [a for a in row["final_answer"] or [] if a.strip()]
        item = near(problem, answers) if answers else {"kind": "question", "question": problem}
        out.add(row["question_id"], [problem, "\n".join(row["final_answer"] or [])], item=item)
    subset_ids = [row["id"] for row in subset]
    mismatched = [row["id"] for row in subset if by_id.get(row["id"]) != row["problem"]]
    if mismatched:
        raise ValueError(f"OlympiadBench subset items differ from the pinned pool: {mismatched}")
    out.extra["policy_subset"] = {
        "source": str(subset_path.relative_to(REPO_ROOT)),
        "revision": subset_digest,
        "items": len(subset),
        "ids": subset_ids,
    }
    out.notes.append(
        f"The policy evaluates the vendored {len(subset)}-question subset (policy_subset.ids); the whole test_en pool "
        "it was drawn from is indexed, and every subset problem matches its pool row exactly. Records hold context, "
        "question and final answer; the pool has no reference solutions. Images are not indexed."
    )
    return out


def aime24() -> EvalTargets:
    path, digest = vendored("AIME24/data/aime24.json")
    rows = read_jsonl(path)
    out = EvalTargets("aime24", str(path.relative_to(REPO_ROOT)), digest, "test", len(rows))
    for row in rows:
        out.add(
            row["id"].removeprefix("aime24-"),
            [row["problem"], row["reference_solution"], str(row["expected_answer"])],
            {"prompt": row["problem"], "solution": row["reference_solution"]},
            free(row["problem"], row["reference_solution"]),
        )
    out.notes.append("Records hold problem, reference solution and answer; the prompt template is not indexed.")
    return out


def cruxeval() -> EvalTargets:
    out = EvalTargets("cruxeval", CRUXEVAL[0], CRUXEVAL[1], "test", 0)
    rows = read_jsonl(hf_file(CRUXEVAL, "test.jsonl"))
    out.items = len(rows)
    for row in rows:
        out.add(
            row["id"],
            [row["code"], f"assert f({row['input']}) == {row['output']}"],
            {"prompt": row["code"]},
            near(row["code"], [row["output"]]),
        )
    out.notes.append(
        "Records hold the function and its completed assertion, which serve both the input and output directions. "
        "Not indexed: the shared CoT instruction and one-shot example."
    )
    return out


def financebench() -> EvalTargets:
    path, digest = vendored("FinanceBench/data/financebench.jsonl")
    rows = read_jsonl(path)
    out = EvalTargets("financebench", str(path.relative_to(REPO_ROOT)), digest, "test", len(rows))
    evidence_items = Counter(row["evidence_text"] for row in rows)
    for row in rows:
        out.add(
            row["id"].removeprefix("financebench_id_"),
            [row["evidence_text"], row["question"], row["answer"]],
            {"prompt": row["question"], "answer": row["answer"]},
            near(row["question"], [row["answer"]]),
        )
    out.notes.append(
        "Vendored 50-item sample of PatronusAI/financebench. Records hold evidence page, question and answer; "
        "question and answer get their own views when long enough. Not indexed: the filing PDFs and the prompt "
        f"template. {sum(n for n in evidence_items.values() if n > 1)} items share an evidence page with another item."
    )
    return out


def ifbench() -> EvalTargets:
    out = EvalTargets("ifbench", IFBENCH[0], IFBENCH[1], "train", 0)
    rows = read_parquet(IFBENCH, "data/train-00000-of-00001.parquet", ["key", "prompt"])
    out.items = len(rows)
    for row in rows:
        out.add(str(row["key"]), [row["prompt"]], item={"kind": "question", "question": row["prompt"]})
    out.notes.append("Records hold the prompt with its inline constraint; there is no reference answer.")
    return out


def mrcr_bin(total_tokens: int) -> int | None:
    lower = 4_096
    for upper in MRCR_BIN_UPPER_BOUNDS:
        if lower <= total_tokens <= upper:
            return upper
        lower = upper + 1
    return None


def mrcr() -> EvalTargets:
    out = EvalTargets("mrcr", MRCR[0], MRCR[1], "train", 0)
    encoding = tiktoken.get_encoding("o200k_base")
    max_upper = max(upper for upper in MRCR_BIN_UPPER_BOUNDS if upper <= MRCR_MAX_LENGTH)
    fs = HfFileSystem()
    cells = Counter()
    for filename in MRCR_FILES:
        with fs.open(f"datasets/{MRCR[0]}@{MRCR[1]}/{filename}") as f:
            rows = pq.read_table(f).to_pylist()
        for index, row in enumerate(rows):
            if row["n_chars"] > max_upper * MRCR_MAX_CHARS_PER_TOKEN:
                continue
            messages = json.loads(row["prompt"])
            tokens = sum(len(encoding.encode(m["content"])) for m in messages) + len(encoding.encode(row["answer"]))
            upper = mrcr_bin(tokens)
            if upper is None or upper > max_upper:
                continue
            cells[(upper, row["n_needles"])] += 1
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
    out.items = sum(cells.values())
    out.extra["cells"] = {f"{upper}/{needles}": n for (upper, needles), n in sorted(cells.items())}
    out.notes.append(
        f"Items are the rows the policy runs at max_length {MRCR_MAX_LENGTH}: o200k token count (message contents "
        f"plus answer) in a bin with upper bound <= {max_upper}, all needle counts. Records hold the needle request, "
        "the final question, every needle response and the expected answer. Not indexed: the synthetic filler turns "
        "and the leading few-shot example message. Item id is <file stem>_<row index in that file>."
    )
    return out


def nupa5k_loose() -> EvalTargets:
    manifest_path, _ = vendored("NUPA5K-Loose/data/nupa5k_manifest.jsonl")
    identities = read_jsonl(manifest_path)
    out = EvalTargets("nupa5k_loose", NUPA[0], NUPA[1], "test", len(identities))
    with open(hf_file(NUPA, "test.json")) as f:
        source = json.load(f)
    wanted = defaultdict(set)
    for identity in identities:
        wanted[(identity["task_name"], str(identity["digit"]))].add(identity["sha256"])
    found = 0
    for task_name, by_digit in source.items():
        template = os.path.commonprefix([text for texts in by_digit.values() for text in texts])
        template = template[: template.rfind(" ") + 1]
        for digit, texts in by_digit.items():
            digests = wanted.get((task_name, digit), set())
            selected = {}
            for text in texts:
                digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
                if digest in digests:
                    selected[digest] = text
            for digest, text in sorted(selected.items()):
                found += 1
                out.add(f"{task_name}_{digit}_{digest[:16]}", [text.removeprefix(template)])
    if found != len(identities):
        raise ValueError(f"NUPA5K: resolved {found} of {len(identities)} manifest identities")
    out.notes.append(
        "Each item is a per-task instruction template shared by every item of its task plus the operands and answer. "
        "With the template removed only numbers remain, so no item can produce a 13-gram of words and none is indexed."
    )
    return out


def piqa() -> EvalTargets:
    rows = read_parquet(PIQA, "piqa_validation.parquet")
    out = EvalTargets("piqa", PIQA[0], PIQA[1], "validation", len(rows))
    for index, row in enumerate(rows):
        correct, wrong = (row["sol1"], row["sol2"]) if row["label"] == "0" else (row["sol2"], row["sol1"])
        out.add(str(index), [row["goal"], row["sol1"], row["sol2"]], item=near(row["goal"], [correct], [wrong]))
    out.notes.append("lm-eval piqa scores validation. Records hold goal and both solutions; id is the row index.")
    return out


def winogrande() -> EvalTargets:
    rows = read_parquet(WINOGRANDE, "winogrande_xl/validation-00000-of-00001.parquet")
    out = EvalTargets("winogrande", WINOGRANDE[0], WINOGRANDE[1], "winogrande_xl/validation", len(rows))
    for index, row in enumerate(rows):
        answer, wrong = (row["option1"], row["option2"]) if row["answer"] == "1" else (row["option2"], row["option1"])
        filled = row["sentence"].replace("_", answer, 1)
        item = near(row["sentence"], [answer], [wrong])
        out.add(str(index), [filled, row["sentence"]], item=item)
    out.notes.append(
        "lm-eval winogrande scores winogrande_xl validation (5-shot from train, not indexed). Records hold the "
        "sentence with the blank filled by the correct option and the raw sentence; id is the row index."
    )
    return out


def boolq() -> EvalTargets:
    rows = read_parquet(BOOLQ, "boolq/validation-00000-of-00001.parquet")
    out = EvalTargets("boolq", BOOLQ[0], BOOLQ[1], "boolq/validation", len(rows))
    for row in rows:
        answers, wrong = (["yes", "true"], ["no", "false"]) if row["label"] == 1 else (["no", "false"], ["yes", "true"])
        out.add(
            str(row["idx"]),
            [row["passage"], row["question"] + "?", answers[0]],
            item=near(row["question"], answers, wrong),
        )
    passages = Counter(row["passage"] for row in rows)
    out.notes.append(
        "lm-eval boolq scores validation. Records hold passage, question and answer. "
        f"{sum(n for n in passages.values() if n > 1)} items share a passage with another item."
    )
    return out


def truthfulqa() -> EvalTargets:
    rows = read_parquet(TRUTHFULQA, "multiple_choice/validation-00000-of-00001.parquet")
    out = EvalTargets("truthfulqa", TRUTHFULQA[0], TRUTHFULQA[1], "multiple_choice/validation", len(rows))
    for index, row in enumerate(rows):
        mc1 = row["mc1_targets"]
        best = [c for c, label in zip(mc1["choices"], mc1["labels"], strict=True) if label == 1]
        wrong = [c for c, label in zip(mc1["choices"], mc1["labels"], strict=True) if label == 0]
        item = near(row["question"], best, wrong)
        out.add(str(index), [row["question"], "\n".join(row["mc2_targets"]["choices"])], item=item)
    out.notes.append(
        "lm-eval truthfulqa_mc2 scores validation. Records hold question and all MC2 choices; id is the row index. "
        "Not indexed: the shared 6-question QA preamble."
    )
    return out


def triviaqa() -> EvalTargets:
    rows = read_parquet(
        TRIVIAQA, "rc.nocontext/validation-00000-of-00001.parquet", ["question_id", "question", "answer"]
    )
    out = EvalTargets("triviaqa", TRIVIAQA[0], TRIVIAQA[1], "rc.nocontext/validation", len(rows))
    texts_by_id = defaultdict(dict)
    for row in rows:
        text = paragraphs([row["question"], row["answer"]["value"]])
        texts_by_id[row["question_id"]].setdefault(text, near(row["question"], [row["answer"]["value"]]))
    variants = 0
    for question_id, texts in texts_by_id.items():
        for n, (text, item) in enumerate(texts.items()):
            variants += n > 0
            out.add(question_id if n == 0 else f"{question_id}_{n}", [text], item=item)
    out.notes.append(
        f"lm-eval triviaqa scores rc.nocontext validation: {len(rows)} rows, {len(texts_by_id)} question ids. Rows "
        f"repeating a question id with the same question and answer are indexed once; {variants} ids with a "
        "different question or answer get a suffixed id. Records hold question and answer value; aliases and the "
        "5-shot train examples are not indexed."
    )
    return out


EVALS = (
    math500,
    humanevalplus,
    mbppplus,
    olympiadbench,
    piqa,
    winogrande,
    boolq,
    truthfulqa,
    triviaqa,
    aime24,
    cruxeval,
    financebench,
    ifbench,
    mrcr,
    nupa5k_loose,
)
SCHEMA = pa.schema([("id", pa.string()), ("text", pa.string())])


def main() -> None:
    out_dir = Path(sys.argv[1])
    manifest = []
    verbatim_items = {}
    for build in EVALS:
        targets = build()
        ids = [row["id"] for row in targets.rows]
        if len(set(ids)) != len(ids):
            raise ValueError(f"{targets.name}: duplicate record ids")
        if targets.rows:
            write(out_dir, targets.name, targets.rows)
        else:
            path = out_dir / targets.name / "eval.parquet"
            path.parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(SCHEMA.empty_table(), path)
            print(f"{targets.name}: 0 records -> {path}")
        manifest.append(targets.manifest())
        verbatim_items.update(targets.verbatim_items)
    (out_dir / ".manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (out_dir / ".verbatim_items.json").write_text(json.dumps(verbatim_items))


if __name__ == "__main__":
    main()
