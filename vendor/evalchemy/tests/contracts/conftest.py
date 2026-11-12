"""Keep source contracts offline at the Hugging Face file boundary."""

import csv
import gzip
import hashlib
import json
from pathlib import Path

import pytest

from eval.chat_benchmarks.IFEval import eval_instruct as ifeval
from eval.chat_benchmarks.MATH500 import eval_instruct as math500
from eval.chat_benchmarks.MixEval import eval_instruct as mixeval
from eval.chat_benchmarks.SimpleQA import eval_instruct as simpleqa
from eval.chat_benchmarks.PDBThink import data as pdbthink_data


@pytest.fixture(scope="session")
def simpleqa_csv(tmp_path_factory):
    path = tmp_path_factory.mktemp("simpleqa") / "simpleqa.csv"
    with path.open("w", newline="", encoding="utf-8") as data:
        writer = csv.DictWriter(data, fieldnames=["problem", "answer", "metadata"])
        writer.writeheader()
        for index in range(simpleqa.DATASET_SIZE):
            writer.writerow({"problem": f"question {index}", "answer": f"answer {index}", "metadata": "{'topic': 'fixture'}"})
    return path


@pytest.fixture(scope="session")
def math500_jsonl(tmp_path_factory):
    path = tmp_path_factory.mktemp("math500") / "math500.jsonl"
    path.write_text("".join(json.dumps({"problem": f"Compute {index} + 0.", "answer": str(index)}) + "\n" for index in range(500)))
    return path


@pytest.fixture(autouse=True)
def offline_other_hf_files(monkeypatch, simpleqa_csv, math500_jsonl, request):
    if request.node.get_closest_marker("data_integration"):
        return
    fixtures = Path(__file__).parent / "fixtures" / "hf-other"

    def fetch_fixture(repo_id, filename, *, revision, cache_prefix=None):
        if repo_id == "basicv8vc/SimpleQA":
            return str(simpleqa_csv)
        if repo_id == "HuggingFaceH4/MATH-500":
            return str(math500_jsonl)
        if repo_id == "MixEval/MixEval":
            return str(fixtures / Path(filename).relative_to("data/mixeval-jsonl"))
        return str(fixtures / filename)

    for module in (simpleqa, mixeval, ifeval, math500):
        monkeypatch.setattr(module, "fetch_hf", fetch_fixture)


@pytest.fixture
def pdbthink_bundle(tmp_path, monkeypatch):
    """Serve invented benchmark records through the real checksum-verified loader."""
    manifest_path = tmp_path / "manifest.json"
    monkeypatch.setattr(pdbthink_data, "MANIFEST_PATH", manifest_path)
    pdbthink_data.manifest.cache_clear()

    def write(records):
        payload = gzip.compress(json.dumps({"renders": records}).encode(), mtime=0)
        path = tmp_path / pdbthink_data.BUNDLE_PATH
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        manifest_path.write_text(json.dumps({"bundle_sha256": hashlib.sha256(payload).hexdigest(),
                                             "n_renders": len(records)}))
        pdbthink_data.manifest.cache_clear()
        return str(tmp_path)

    yield write
    pdbthink_data.manifest.cache_clear()
