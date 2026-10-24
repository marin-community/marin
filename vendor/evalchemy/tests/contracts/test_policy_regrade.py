"""Policy finalization preserves generation and publishes only complete regrades."""

import json
import threading
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import aiohttp
import pytest
from finestore.eval import EvalSample, EvaluationStore, Grading, SampleKind
from finestore.reader import ReadView

from eval.policy_regrade import CONFIG_ENV, RAW_SAMPLES, VERDICTS, main


@contextmanager
def judge_server(response):
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            requests.append(request)
            status, content = response(request)
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"choices": [{"message": {"content": content}}]}).encode())

        def log_message(self, *_args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", requests
    finally:
        server.shutdown()
        thread.join()
        server.server_close()


def archive(root, benchmark, output):
    doc = {"question": "Revenue?", "answer": "10"} if benchmark == "financebench" else {"problem": "2 + 2?", "answer": ["4"]}
    with EvaluationStore.open(str(root), writer_id="candidate") as store:
        store.add_sample(EvalSample(task=benchmark, doc_id="0", kind=SampleKind.GENERATION,
                                   output=output, doc=json.dumps(doc), metrics={"accuracy": 0.0},
                                   grading=Grading(method="old", metric="accuracy", score=0.0)), trial_id="0")
        store.seal()


def configure(monkeypatch, root, benchmark, endpoint):
    monkeypatch.setenv(CONFIG_ENV, json.dumps({"results_path": str(root), "benchmark": benchmark,
        "endpoint": endpoint, "model": "minimax", "judge_config": {"revision": "frozen"}}))


def test_finance_regrade_escalates_empty_labels_and_resumes_without_regeneration(tmp_path, monkeypatch):
    archive(tmp_path, "financebench", "10")
    with judge_server(lambda request: (200, "" if request["max_tokens"] == 128 else "correct")) as (endpoint, requests):
        configure(monkeypatch, tmp_path, "financebench", endpoint)
        main()
        first = ReadView(str(tmp_path))
        rows = first.scan("samples").to_pylist(maps_as_pydicts="strict")
        assert rows[0]["output"] == "10"
        assert rows[0]["metrics"]["accuracy"] == 1.0
        assert rows[0]["metrics"]["policy_raw_accuracy"] == 0.0
        assert [r["max_tokens"] for r in requests] == [128, 512]
        assert requests[-1]["structured_outputs"] == {"choice": ["correct", "incorrect", "not_attempted"]}
        assert json.loads(first.read_blob(RAW_SAMPLES))[0]["grading"]["score"] == 0.0
        main()
        assert len(requests) == 2
        assert json.loads(ReadView(str(tmp_path)).read_blob(VERDICTS))["trials"][0]["score"] == 1.0


def test_failed_judge_does_not_publish_final_verdicts_or_replace_source_grade(tmp_path, monkeypatch):
    archive(tmp_path, "financebench", "10")
    with judge_server(lambda request: (400, "bad request")) as (endpoint, _requests):
        configure(monkeypatch, tmp_path, "financebench", endpoint)
        with pytest.raises(aiohttp.ClientResponseError, match="400"):
            main()
    reader = ReadView(str(tmp_path))
    assert reader.read_blob(VERDICTS) is None
    assert reader.scan("samples").to_pylist(maps_as_pydicts="strict")[0]["grading"]["score"] == 0.0


@pytest.mark.parametrize("output,score", [(r"\\boxed{4}", 1.0), ("no boxed answer", 0.0)])
def test_olympiad_native_deterministic_grades_do_not_call_judge(tmp_path, monkeypatch, output, score):
    archive(tmp_path, "olympiadbench", output)
    with judge_server(lambda request: (400, "unexpected")) as (endpoint, requests):
        configure(monkeypatch, tmp_path, "olympiadbench", endpoint)
        main()
        assert requests == []
    verdicts = json.loads(ReadView(str(tmp_path)).read_blob(VERDICTS))["trials"]
    assert verdicts[0]["score"] == score


def test_resume_rejects_changed_candidate_answers(tmp_path, monkeypatch):
    archive(tmp_path, "financebench", "10")
    with judge_server(lambda request: (200, "correct")) as (endpoint, requests):
        configure(monkeypatch, tmp_path, "financebench", endpoint)
        main()
        archive(tmp_path, "financebench", "20")
        with pytest.raises(ValueError, match="different source answers"):
            main()
        assert len(requests) == 1
