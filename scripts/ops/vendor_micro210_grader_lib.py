# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Vendor the Micro210 embedded grading library from the pinned runtime-v4 wheel.

The self-contained Micro210 policy version (#12060) embeds every grader's code in each
row's TaskSpec, so grading no longer imports the ambient ``runtime-v4`` evalchemy wheel.
This script extracts the modules the Micro210 graders actually execute from the
hash-pinned wheel at ``open-athena/Marin-Mini-Eval-Policy-Index@d5dd315d...`` (the same
pin the frozen table carries in ``grader.environment.package_spec``), applies the surgical
fixes documented in the manifest, and writes them to
``experiments/evaluation/micro210_grader`` with a provenance manifest. Re-running it is
idempotent; the manifest records each file's origin and SHA-256 so tests can verify the
committed tree without network access.

Not vendored, by design:

- ``eval.graders.specs`` and the unrelated graders it registers (their third-party
  dependencies are not Micro210 grading dependencies). ``micro210_registry.py`` maps the
  grader names Micro210 rows use directly to vendored functions.
- ``eval.graders.programs`` and ``eval.chat_benchmarks.CruxEval``: the CRUXEval grader is
  replaced by main's canonical implementation (#11922/#11928) in ``micro210_crux.py``.
- ``eval.sources.lm_eval`` and the ``lm_eval`` machinery: gsm8k and triviaqa grading is
  reproduced by ``micro210_lm_eval_min.py`` (filters and metrics only), differential-tested
  against the wheel.
- Modules reachable only through lazy imports inside graders no Micro210 row selects
  (HMMT matharena, JEEBench, MixEval, LiveBench, ...): their imports never execute for a
  Micro210 row, and the isolated gate fails closed if one ever does.
"""

import argparse
import hashlib
import io
import json
import shutil
import zipfile
from pathlib import Path

from huggingface_hub import hf_hub_download

REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT = REPO_ROOT / "experiments/evaluation/micro210_grader"

WHEEL_REPO = "open-athena/Marin-Mini-Eval-Policy-Index"
WHEEL_REVISION = "d5dd315d58f0f47514dd0a2208e90e2a48541328"
WHEEL_PATH = "runtime-v4/wheels/evalchemy-0.1.0-py3-none-any.whl"
WHEEL_SHA256 = "9970bac60757b1da49d473519a8cec40ee15fa2fbd08f807607c119f66e36257"

# The Micro210 rows never import the ``lm_eval`` package machinery; the vendored tree
# carries only the boxed-answer helpers ``eval.graders.answer_extraction`` imports, from
# the hash-pinned lm_eval wheel the frozen table also names in ``package_spec``.
LM_EVAL_WHEEL_PATH = "runtime-v4/wheels/lm_eval-0.4.12-py3-none-any.whl"
LM_EVAL_WHEEL_SHA256 = "ab969c0afc9d9bb3ab50e8be36ce092071bb5f81ed64af3aa57e70c727ef464e"
LM_EVAL_UTILS_MEMBER = "lm_eval/tasks/hendrycks_math/utils.py"
LM_EVAL_STUB_MEMBERS = ("lm_eval/__init__.py", "lm_eval/tasks/__init__.py", "lm_eval/tasks/hendrycks_math/__init__.py")

# The NLTK data the Micro210 IFBench instruction checkers reach: sent_tokenize
# (punkt_tab/english plus the legacy punkt pickle) and the stopwords list.
NLTK_DATA_MEMBERS = (
    "eval/chat_benchmarks/IFBench/.nltk_data/tokenizers/punkt/english.pickle",
    "eval/chat_benchmarks/IFBench/.nltk_data/tokenizers/punkt_tab/english/abbrev_types.txt",
    "eval/chat_benchmarks/IFBench/.nltk_data/tokenizers/punkt_tab/english/collocations.tab",
    "eval/chat_benchmarks/IFBench/.nltk_data/tokenizers/punkt_tab/english/ortho_context.tab",
    "eval/chat_benchmarks/IFBench/.nltk_data/tokenizers/punkt_tab/english/sent_starters.txt",
    "eval/chat_benchmarks/IFBench/.nltk_data/corpora/stopwords/english",
)

# Vendored verbatim. A trailing "/" names a directory whose own ``__init__.py`` is
# included (from the wheel, or as a stub when the wheel ships none).
VERBATIM = (
    "eval/",
    "eval/contracts/",
    "eval/contracts/eval_spec.py",
    "eval/contracts/failures.py",
    "eval/contracts/sample_results.py",
    "eval/limits.py",
    "eval/generation_stops.py",
    "eval/completion_response.py",
    "eval/execution/",
    "eval/execution/judge.py",
    "eval/execution/backend.py",
    "eval/runtime/",
    "eval/runtime/pool.py",
    "eval/runtime/resources.py",
    "eval/runtime/program_process.py",
    "eval/graders/",
    "eval/graders/answer_extraction.py",
    "eval/graders/answer_equivalence.py",
    "eval/graders/gsm8k.py",
    "eval/graders/judged.py",
    "eval/graders/simpleqa.py",
    "eval/graders/mmlu.py",
    "eval/graders/minerva_math.py",
    "eval/graders/math_specs.py",
    "eval/graders/structured.py",
    "eval/graders/instructions.py",
    "eval/graders/program_suites.py",
    "eval/lm_eval_tasks/",
    "eval/lm_eval_tasks/gsm8k/",
    "eval/lm_eval_tasks/gsm8k/utils.py",
    "eval/lm_eval_tasks/short_answer_extraction.py",
    "eval/chat_benchmarks/",
    "eval/chat_benchmarks/GPQADiamond/",
    "eval/chat_benchmarks/GPQADiamond/testing_utils.py",
    "eval/chat_benchmarks/HLE/",
    "eval/chat_benchmarks/HLE/testing_utils.py",
    "eval/chat_benchmarks/NUPA-Loose/",
    "eval/chat_benchmarks/NUPA-Loose/scorer.py",
    "eval/chat_benchmarks/IFBench/",
    "eval/chat_benchmarks/IFBench/grader.py",
    "eval/chat_benchmarks/IFBench/instructions.py",
    "eval/chat_benchmarks/IFBench/instructions_registry.py",
    "eval/chat_benchmarks/IFBench/instructions_util.py",
    "eval/chat_benchmarks/HumanEval/",
    "eval/chat_benchmarks/HumanEval/human_eval/",
    "eval/chat_benchmarks/HumanEval/human_eval/evaluation.py",
    "eval/chat_benchmarks/HumanEval/human_eval/execution.py",
    "eval/chat_benchmarks/HumanEval/utils/",
    "eval/chat_benchmarks/HumanEval/utils/utils.py",
    "eval/chat_benchmarks/HumanEvalPlus/",
    "eval/chat_benchmarks/HumanEvalPlus/human_eval_plus/",
    "eval/chat_benchmarks/HumanEvalPlus/human_eval_plus/evaluation.py",
    "eval/chat_benchmarks/HumanEvalPlus/human_eval_plus/execution.py",
    "eval/chat_benchmarks/HumanEvalPlus/utils/",
    "eval/chat_benchmarks/HumanEvalPlus/utils/utils.py",
    "eval/chat_benchmarks/MBPPPlus/",
    "eval/chat_benchmarks/MBPPPlus/mbpp_plus/",
    "eval/chat_benchmarks/MBPPPlus/mbpp_plus/evaluation.py",
    "eval/chat_benchmarks/MBPPPlus/mbpp_plus/execution.py",
    "eval/chat_benchmarks/BigCodeBench/",
    "eval/chat_benchmarks/BigCodeBench/sanitize.py",
    "eval/chat_benchmarks/RepoBench/",
    "eval/chat_benchmarks/RepoBench/archive_data/",
    "eval/chat_benchmarks/RepoBench/archive_data/utils.py",
    "eval/chat_benchmarks/zeroeval/",
    "eval/chat_benchmarks/zeroeval/src/",
    "eval/chat_benchmarks/zeroeval/src/evaluation/",
    "eval/chat_benchmarks/zeroeval/src/evaluation/eval_utils.py",
    "evalchemy_config/",
    "evalchemy_config/limits.py",
    "evalchemy_config/config.py",
)

IFBENCH_INSTRUCTIONS = "eval/chat_benchmarks/IFBench/instructions.py"
IFBENCH_INSTRUCTIONS_UTIL = "eval/chat_benchmarks/IFBench/instructions_util.py"
BACKEND = "eval/execution/backend.py"
MATH_SPECS = "eval/graders/math_specs.py"

# The European-capitals checker must accept Kyiv between Brussels and Prague (#11926);
# main's taskcompendium copy carries the same one-line order fix.
KYIV_FIX_BEFORE = '"Brussels", "Prague", "Luxembourg",'
KYIV_FIX_AFTER = '"Brussels", "Kyiv", "Prague", "Luxembourg",'

# The vendored tree bundles exactly the NLTK data the Micro210 IFBench checkers reach
# (punkt_tab for sent_tokenize; the punkt pickle and stopwords for the shared helpers).
# Upstream's ``download_nltk_resources`` also probes the POS tagger and would reach the
# network in the isolated verifier; the bundled lookup replaces the downloads.
NLTK_FIX_BEFORE = '''def download_nltk_resources():
    """Download 'punkt' and 'stopwords' if not already installed"""
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt", quiet=True)
    try:
        nltk.data.find("tokenizers/punkt_tab")
    except LookupError:
        nltk.download("punkt_tab", quiet=True)
    try:
        nltk.data.find("corpora/stopwords")
    except LookupError:
        nltk.download("stopwords", quiet=True)
    try:
        nltk.data.find("taggers/averaged_perceptron_tagger_eng")
    except LookupError:
        nltk.download("averaged_perceptron_tagger_eng", quiet=True)'''

NLTK_FIX_AFTER = '''def download_nltk_resources():
    """Verify the bundled tokenizers and stopwords; the vendored grader never downloads."""
    for resource in ("tokenizers/punkt", "tokenizers/punkt_tab", "corpora/stopwords"):
        nltk.data.find(resource)'''

BACKEND_TRIM_NOTE = (
    "trim: keep RETRYABLE_STATUSES, retryable_transport and transport_retry_delay only; the judge"
    " transport needs no lm_eval model APIs, transformers, or the local serving stack"
)

MATH_SPECS_TRIM_NOTE = (
    "trim: drop graders no Micro210 row selects (gsm8k, matharena, answer_number, jee) and the"
    " gsm8k-utils import only they used"
)

LM_EVAL_UTILS = "lm_eval/tasks/hendrycks_math/utils.py"
LM_EVAL_UTILS_TRIM_NOTE = (
    "trim: drop process_docs/process_results and the datasets import only they use; grading needs the"
    " boxed-answer helpers and is_equiv"
)

STUB_INIT = '"""Vendored package marker for the Micro210 embedded grader."""\n'

# Marin-authored modules that ship inside the embedded grading library. They are hashed
# into the manifest (never transformed) so the library carries its own provenance.
MARIN_AUTHORED_MODULES = (
    "micro210_crux.py",
    "micro210_lm_eval_min.py",
    "micro210_registry.py",
)


def wheel_bytes(cache: Path, *, repository: str, revision: str, path: str, sha256: str) -> bytes:
    """Fetch (or reuse) a pinned wheel and verify its SHA-256."""
    cached = cache / f"{Path(path).name}-{sha256[:12]}.whl"
    if cached.exists():
        raw = cached.read_bytes()
        if hashlib.sha256(raw).hexdigest() != sha256:
            raise ValueError(f"Cached wheel at {cached} does not match the pinned digest")
        return raw
    fetched = hf_hub_download(repo_id=repository, repo_type="dataset", revision=revision, filename=path)
    raw = Path(fetched).read_bytes()
    if hashlib.sha256(raw).hexdigest() != sha256:
        raise ValueError(f"Downloaded {path} does not match the pinned digest")
    cache.mkdir(parents=True, exist_ok=True)
    shutil.copy(fetched, cached)
    return raw


def trimmed_backend(source: str) -> str:
    lines = source.splitlines()
    anchor = lines.index("RETRYABLE_STATUSES = frozenset({408, 429, 500, 502, 503, 504})")
    retryable = lines.index("def retryable_transport(error: Exception) -> bool:")
    delay = lines.index("def transport_retry_delay(error: Exception) -> float:")
    end = next(index for index in range(delay + 1, len(lines)) if lines[index].startswith(("def ", "class ")))
    header = "\n".join(lines[:3]) + "\n"
    imports = (
        "from aiohttp import ClientConnectionError, ClientPayloadError, ClientResponseError\n"
        "from datetime import datetime, timezone\n"
        "from email.utils import parsedate_to_datetime\n"
    )
    body = "\n".join(lines[anchor : anchor + 1] + lines[retryable:end]).rstrip("\n") + "\n"
    return header + "\n" + imports + "\n" + body


def trimmed_math_specs(source: str) -> str:
    lines = source.splitlines(keepends=True)
    dropped_starts = ("def gsm8k(", "def matharena(", "def answer_number(", "def jee(")
    kept, skipping = [], False
    for line in lines:
        if any(line.startswith(start) for start in dropped_starts):
            skipping = True
            continue
        if skipping and line.strip() and not line[:1].isspace() and not line.startswith(")"):
            skipping = False
        if not skipping:
            kept.append(line)
    text = "".join(kept)
    removed_import = "from eval.lm_eval_tasks.gsm8k.utils import extract_gsm8k_flexible_answer\n"
    if text.count(removed_import) != 1:
        raise ValueError("math_specs.py does not carry the expected gsm8k-utils import")
    return text.replace(removed_import, "")


def trimmed_lm_eval_utils(source: str) -> str:
    lines = source.splitlines(keepends=True)
    text = "".join(lines)
    if text.count("import datasets\n") != 1:
        raise ValueError("hendrycks_math utils.py does not carry the expected datasets import")
    text = text.replace("import datasets\n", "")
    dropped_starts = ("def process_docs(", "def process_results(")
    kept, skipping = [], False
    for line in text.splitlines(keepends=True):
        if any(line.startswith(start) for start in dropped_starts):
            skipping = True
            continue
        if skipping and line.strip() and not line[:1].isspace() and not line.startswith(")"):
            skipping = False
        if not skipping:
            kept.append(line)
    return "".join(kept)


def transform(name: str, source: str) -> tuple[str, dict]:
    """Return the vendored text for one wheel member plus its provenance note."""
    if name == IFBENCH_INSTRUCTIONS:
        if source.count(KYIV_FIX_BEFORE) != 1:
            raise ValueError("IFBench instructions.py does not match the expected pre-Kyiv checker source")
        return source.replace(KYIV_FIX_BEFORE, KYIV_FIX_AFTER), {
            "patch": "ifbench-kyiv-11926",
            "note": "insert Kyiv between Brussels and Prague in EuropeanCapitalsSortChecker (from main #11926)",
        }
    if name == IFBENCH_INSTRUCTIONS_UTIL:
        if source.count(NLTK_FIX_BEFORE) != 1:
            raise ValueError("IFBench instructions_util.py does not match the expected download_nltk_resources source")
        return source.replace(NLTK_FIX_BEFORE, NLTK_FIX_AFTER), {
            "patch": "ifbench-bundled-nltk",
            "note": "download_nltk_resources verifies only the bundled resources; no network in the verifier",
        }
    if name == BACKEND:
        return trimmed_backend(source), {"patch": "backend-transport-trim", "note": BACKEND_TRIM_NOTE}
    if name == MATH_SPECS:
        return trimmed_math_specs(source), {"patch": "math-specs-trim", "note": MATH_SPECS_TRIM_NOTE}
    if name == LM_EVAL_UTILS:
        return trimmed_lm_eval_utils(source), {"patch": "lm-eval-utils-trim", "note": LM_EVAL_UTILS_TRIM_NOTE}
    return source, {}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wheel-cache", type=Path, default=REPO_ROOT / ".cache" / "micro210-grader-lib")
    arguments = parser.parse_args()

    cache = arguments.wheel_cache
    evalchemy = zipfile.ZipFile(
        io.BytesIO(
            wheel_bytes(cache, repository=WHEEL_REPO, revision=WHEEL_REVISION, path=WHEEL_PATH, sha256=WHEEL_SHA256)
        )
    )
    lm_eval = zipfile.ZipFile(
        io.BytesIO(
            wheel_bytes(
                cache,
                repository=WHEEL_REPO,
                revision=WHEEL_REVISION,
                path=LM_EVAL_WHEEL_PATH,
                sha256=LM_EVAL_WHEEL_SHA256,
            )
        )
    )
    members = {info.filename for info in evalchemy.infolist()}

    planned: dict[str, tuple[zipfile.ZipFile, str]] = {}  # vendored path -> (archive, wheel member | "" stub)
    for entry in VERBATIM:
        if entry.endswith(".py"):
            planned[entry] = (evalchemy, entry)
            continue
        init = f"{entry}__init__.py"
        planned[init] = (evalchemy, init) if init in members else (evalchemy, "")
    planned[LM_EVAL_UTILS_MEMBER] = (lm_eval, LM_EVAL_UTILS_MEMBER)
    for stub in LM_EVAL_STUB_MEMBERS:
        planned[stub] = (lm_eval, "")
    for archive, name in planned.values():
        if name and name not in {info.filename for info in archive.infolist()}:
            raise ValueError(f"Pinned wheel has no member {name}")

    manifest_entries = []
    OUTPUT.mkdir(parents=True, exist_ok=True)
    previous = OUTPUT / "MANIFEST.json"
    if previous.exists():
        # Remove exactly the vendored files this script owns (listed in the previous
        # manifest, excluding Marin-authored modules it does not regenerate).
        for entry in json.loads(previous.read_text())["files"]:
            if entry.get("patch") == "marin-authored":
                continue
            (OUTPUT / entry["path"]).unlink(missing_ok=True)
    for vendored_path, (archive, name) in sorted(planned.items()):
        if not name:
            source, note = STUB_INIT, {"patch": "stub-init", "note": "package marker absent from the wheel"}
        else:
            source, note = transform(name, archive.read(name).decode())
        destination = OUTPUT / vendored_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(source)
        manifest_entries.append(
            {
                "path": vendored_path,
                "wheel_member": name,
                "sha256": hashlib.sha256(source.encode()).hexdigest(),
                "bytes": len(source.encode()),
                **note,
            }
        )

    # Bundled NLTK data for the IFBench rows, extracted verbatim as binary resources.
    for member in NLTK_DATA_MEMBERS:
        payload = evalchemy.read(member)
        vendored_path = member
        destination = OUTPUT / vendored_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(payload)
        manifest_entries.append(
            {
                "path": vendored_path,
                "wheel_member": member,
                "sha256": hashlib.sha256(payload).hexdigest(),
                "bytes": len(payload),
                "patch": "nltk-data",
                "note": "bundled tokenizer/stopword data the selected IFBench checkers reach",
            }
        )

    # Marin-authored modules of the embedded library, hashed where they sit in the tree.
    for module in MARIN_AUTHORED_MODULES:
        payload = (OUTPUT / module).read_bytes()
        manifest_entries.append(
            {
                "path": module,
                "wheel_member": "",
                "sha256": hashlib.sha256(payload).hexdigest(),
                "bytes": len(payload),
                "patch": "marin-authored",
                "note": "Marin module of the embedded grader (not from the wheel)",
            }
        )

    manifest = {
        "source_wheels": [
            {
                "repository": WHEEL_REPO,
                "revision": WHEEL_REVISION,
                "path": WHEEL_PATH,
                "sha256": WHEEL_SHA256,
            },
            {
                "repository": WHEEL_REPO,
                "revision": WHEEL_REVISION,
                "path": LM_EVAL_WHEEL_PATH,
                "sha256": LM_EVAL_WHEEL_SHA256,
            },
        ],
        "files": manifest_entries,
    }
    (OUTPUT / "MANIFEST.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    total = sum(entry["bytes"] for entry in manifest_entries)
    print(f"vendored {len(manifest_entries)} files, {total} bytes, manifest at {OUTPUT / 'MANIFEST.json'}")


if __name__ == "__main__":
    main()
