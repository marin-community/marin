import importlib.machinery
import importlib.util
import sys
import types
from pathlib import Path

MODULE_PATH = Path(__file__).resolve().parents[1] / "src" / "marin" / "download" / "uncheatable_eval" / "download.py"
SPEC = importlib.util.spec_from_file_location("marin.download.uncheatable_eval.download", MODULE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("Failed to load uncheatable eval download module for testing")

# Stub heavy optional dependencies before loading the module under test.
torch_stub = types.ModuleType("torch")
torch_stub.__dict__.update(
    {
        "cuda": types.SimpleNamespace(is_available=lambda: False),
        "device": lambda *_, **__: None,
        "__version__": "0.0.0",
        "Tensor": type("_TensorStub", (), {}),
        "nn": types.SimpleNamespace(),
    }
)
torch_stub.__spec__ = importlib.machinery.ModuleSpec("torch", loader=None)
sys.modules.setdefault("torch", torch_stub)

execution_stub = types.ModuleType("marin.execution")
execution_stub.THIS_OUTPUT_PATH = object()
execution_stub.ExecutorStep = type("ExecutorStep", (), {})
execution_stub.VersionedValue = type("VersionedValue", (), {})
execution_stub.ensure_versioned = staticmethod(lambda value: value)
execution_stub.this_output_path = staticmethod(lambda: "output")
sys.modules.setdefault("marin.execution", execution_stub)

download_mod = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = download_mod
SPEC.loader.exec_module(download_mod)

UncheatableEvalDataset = download_mod.UncheatableEvalDataset
_normalize_record = download_mod._normalize_record
_parse_available_dumps = download_mod._parse_available_dumps
_select_latest_dumps = download_mod._select_latest_dumps


def test_parse_available_dumps_filters_unexpected_entries():
    entries = [
        {
            "name": "ao3_english_20240701to20240714.json",
            "download_url": "https://example.com/ao3.json",
            "sha": "sha-1",
            "size": 10,
        },
        {"name": "dev.json", "download_url": "https://example.com/dev.json"},
        {"name": "invalid_name", "download_url": "https://example.com/invalid"},
    ]

    datasets = _parse_available_dumps(entries)

    assert len(datasets) == 1
    dataset = datasets[0]
    assert dataset.benchmark == "ao3_english"
    assert dataset.start_date == "20240701"
    assert dataset.end_date == "20240714"
    assert dataset.download_url == "https://example.com/ao3.json"


def test_select_latest_dumps_picks_highest_end_date():
    datasets = [
        UncheatableEvalDataset(
            benchmark="ao3_english",
            start_date="20240601",
            end_date="20240615",
            name="ao3_english_20240601to20240615.json",
            download_url="https://example.com/old.json",
        ),
        UncheatableEvalDataset(
            benchmark="ao3_english",
            start_date="20240701",
            end_date="20240714",
            name="ao3_english_20240701to20240714.json",
            download_url="https://example.com/new.json",
        ),
        UncheatableEvalDataset(
            benchmark="wikipedia_english",
            start_date="20240701",
            end_date="20240714",
            name="wikipedia_english_20240701to20240714.json",
            download_url="https://example.com/wiki.json",
        ),
    ]

    latest = _select_latest_dumps(datasets)

    assert [dataset.download_url for dataset in latest] == [
        "https://example.com/new.json",
        "https://example.com/wiki.json",
    ]


def test_normalize_record_from_string_uses_generated_id():
    dataset = UncheatableEvalDataset(
        benchmark="ao3_english",
        start_date="20240701",
        end_date="20240714",
        name="ao3_english_20240701to20240714.json",
        download_url="https://example.com/new.json",
    )

    record = _normalize_record("hello world", dataset, 5)

    assert record["id"].endswith("_000005")
    assert record["text"] == "hello world"
    assert record["source"] == dataset.source_label


def test_normalize_record_prefers_existing_id_and_text_fields():
    dataset = UncheatableEvalDataset(
        benchmark="wikipedia_english",
        start_date="20240701",
        end_date="20240714",
        name="wikipedia_english_20240701to20240714.json",
        download_url="https://example.com/wiki.json",
    )

    raw = {"id": "abc123", "text": "An article"}
    record = _normalize_record(raw, dataset, 0)

    assert record["id"] == "abc123"
    assert record["text"] == "An article"
    assert record["source"] == dataset.source_label


def test_normalize_record_accepts_list_text_field():
    dataset = UncheatableEvalDataset(
        benchmark="github_python",
        start_date="20250701",
        end_date="20250715",
        name="github_python_20250701to20250715.json",
        download_url="https://example.com/github.json",
    )

    raw = {"metadata": {"doc_id": "xyz"}, "paragraphs": ["line 1", "line 2"]}
    record = _normalize_record(raw, dataset, 3)

    assert record["id"] == "xyz"
    assert record["text"] == "line 1\nline 2"


def test_normalize_record_falls_back_to_json_string():
    dataset = UncheatableEvalDataset(
        benchmark="github_python",
        start_date="20250701",
        end_date="20250715",
        name="github_python_20250701to20250715.json",
        download_url="https://example.com/github.json",
    )

    record = _normalize_record({}, dataset, 0)
    assert record["text"] == "{}"
    assert record["id"].endswith("_000000")
