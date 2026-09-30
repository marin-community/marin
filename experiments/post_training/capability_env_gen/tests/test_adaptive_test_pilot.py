import ast
import importlib.util
import json
from pathlib import Path


def module():
    path = Path(__file__).parents[1] / "scripts" / "run_adaptive_test_pilot.py"
    spec = importlib.util.spec_from_file_location("adaptive_pilot", path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def test_frozen_mutants_are_distinct_and_have_cases():
    m = module()
    assert len(m.INTERFACES) == 4 and len(m.MUTATIONS) == 6
    assert len(m.ALL_CASE_IDS) == 9
    for interface, source in m.INTERFACES.items():
        for name in m.MUTATIONS:
            assert m.mutant_source(interface, name) != source
            ast.parse(m.instrument_source(interface, m.mutant_source(interface, name)))


def test_prompts_separate_rewrite_and_adapter_contracts():
    m = module()
    assert "complete standalone" in m.prompt("tuple_function", "rewrite")
    assert "defining invoke" in m.prompt("tuple_function", "adapter")
    assert "Candidate source is untrusted" in m.SYSTEM
    assert "CANONICAL TEST SOURCE" in m.prompt("tuple_function", "rewrite")


def test_rewrite_integrity_rejects_known_weakening_controls():
    m = module()
    assert not m.rewrite_integrity_issues(m.CANONICAL_TEST)
    assert "removed_assertion" in m.rewrite_integrity_issues(
        m.CANONICAL_TEST.replace("assert invoke([]) == 0", "invoke([]) == 0")
    )
    assert any(
        x.startswith("missing_case:")
        for x in m.rewrite_integrity_issues(
            m.CANONICAL_TEST.replace("# empty", "# omitted")
        )
    )
    assert "skip" in m.rewrite_integrity_issues(
        m.CANONICAL_TEST + "\nimport pytest; pytest.skip()"
    )
    assert "bare_catchall" in m.rewrite_integrity_issues(
        m.CANONICAL_TEST + "\ntry: pass\nexcept: pass"
    )


def test_replay_rejects_tampered_generated_code(tmp_path):
    m = module()
    response = {"content": json.dumps({"code": "value = 1\n"}), "finish_reason": "stop"}
    (tmp_path / "request.json").write_text(json.dumps({"request": {"model": m.MODEL}}))
    (tmp_path / "response.json").write_text(json.dumps(response))
    (tmp_path / "model-metadata.json").write_text(
        json.dumps({"completion_sha256": m.sha(response["content"])})
    )
    (tmp_path / "generated.py").write_text("value = 2\n")
    try:
        m.load_replay(tmp_path)
    except ValueError as error:
        assert "not derived" in str(error)
    else:
        raise AssertionError("tampered generated.py was accepted")


def test_missing_usage_on_length_uses_server_context_and_retains_failed_attempt(
    tmp_path,
):
    m = module()

    class Client:
        def __init__(self):
            self.requests = []

        def complete(self, request, events):
            self.requests.append(request)
            if len(self.requests) == 1:
                return {"finish_reason": "length", "content": "incomplete"}
            return {"finish_reason": "stop", "content": '{"code":"value = 1"}'}

    client = Client()
    result, params = m.complete_adaptation(
        client, {**m.MODEL_PARAMS, "messages": []}, tmp_path
    )
    assert len(client.requests) == 2
    assert client.requests[0]["max_tokens"] == 131072
    assert client.requests[1]["max_tokens"] is None
    assert params["max_tokens"] is None
    assert m.parse_code(result) == "value = 1"
    assert (
        json.loads((tmp_path / "response.length.json").read_text())["content"]
        == "incomplete"
    )
    assert json.loads((tmp_path / "response.json").read_text()) == result
