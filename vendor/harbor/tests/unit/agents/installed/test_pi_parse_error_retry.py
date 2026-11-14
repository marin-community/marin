"""Protocol recovery must expose feedback without executing malformed tool arguments."""

import json
import subprocess
from pathlib import Path

import pytest

from harbor.agents.installed.pi import Pi


@pytest.mark.parametrize("recorded_call", json.loads(Path(__file__).with_name("pi_malformed_43604.json").read_text()))
@pytest.mark.parametrize("retry_limit", [0, 2])
def test_parse_error_feedback_is_bounded_and_preserves_original_response(tmp_path, retry_limit, recorded_call):
    source = Path(__file__).parents[4] / "src/harbor/agents/installed/pi_parse_error_retry.js"
    script = source.read_text().replace("__PARSE_ERROR_RETRIES__", str(retry_limit))
    extension = tmp_path / "retry.mjs"
    extension.write_text(script)
    malformed = recorded_call["raw_message"]
    runner = """
const module = await import(process.argv[1]);
const hooks = new Map();
module.parseErrorRetry({ on: (event, handler) => hooks.set(event, handler) });
const raw = JSON.parse(process.argv[2]);
const original = { role: "assistant", content: [{ type: "text", text: raw }], stopReason: "stop" };
const before = JSON.stringify(original);
const decisions = [];
let projected;
for (let retry = 0; retry < 3; retry++) {
  const result = hooks.get("turn_end")({ outcome: "completed", message: original, messageEntryId: "source" });
  decisions.push(result?.continue ?? false);
  if (result) {
    const entry = result.entries[0];
    projected = hooks.get("context")({ messages: [original, {
      role: "custom", customType: entry.customType, content: entry.content,
      details: entry.details, timestamp: 0,
    }] }).messages;
  }
}
const valid = { role: "assistant", content: [{ type: "text", text: '<tool_call>{"name":"read","arguments":{"path":"x"}}</tool_call>' }] };
process.stdout.write(JSON.stringify({ decisions, projected, unchanged: before === JSON.stringify(original), validRejected: module.hasInvalidToolJSON(valid) }));
"""
    completed = subprocess.run(
        ["node", "--input-type=module", "--eval", runner, extension.as_uri(), json.dumps(malformed)],
        check=True, capture_output=True, text=True,
    )
    result = json.loads(completed.stdout)
    assert result["decisions"] == ([True, True, False] if retry_limit else [False, False, False])
    assert result["unchanged"] and not result["validRejected"]
    if retry_limit:
        assistant, feedback = result["projected"]
        assert assistant["content"][0]["text"] == malformed
        assert assistant["content"][1]["name"] == "__parse_error_feedback__"
        assert assistant["content"][1]["arguments"] == {}
        assert feedback["role"] == "toolResult" and feedback["isError"]
        assert feedback["content"][0]["text"] == (
            "Tool call JSON could not be parsed. No tool was executed. Return a new tool call "
            "using valid JSON with name and arguments. Do not repeat the malformed text."
        )


@pytest.mark.parametrize("retries,api_base", [(True, "http://fixture/v1"), (False, "http://fixture/v1"), (2, None)])
def test_parse_error_retry_rejects_silent_control_or_invalid_javascript(tmp_path, retries, api_base):
    with pytest.raises(ValueError, match="parse_error_retries"):
        Pi(logs_dir=tmp_path, model_name="fixture", api_base=api_base,
           thinking_format="chat-template", parse_error_retries=retries)


def test_parse_error_retry_negative_controls_leave_context_and_termination_unchanged(tmp_path):
    source = Path(__file__).parents[4] / "src/harbor/agents/installed/pi_parse_error_retry.js"
    extension = tmp_path / "retry.mjs"
    extension.write_text(source.read_text().replace("__PARSE_ERROR_RETRIES__", "2"))
    runner = r"""
const module = await import(process.argv[1]);
const hooks = new Map();
module.parseErrorRetry({ on: (event, handler) => hooks.set(event, handler) });
const raw = '<tool_call>{"name":"edit","arguments":{BROKEN}}</tool_call>';
const assistant = text => ({ role: "assistant", content: [{type:"text",text}] });
const messages = [assistant("Done."), assistant('<tool_call>{"name":"exit","code":0}</tool_call>'),
  {...assistant(raw), content:[{type:"toolCall",id:"fresh",name:"read",arguments:{path:"x"}},{type:"text",text:raw}]}];
const results = messages.map(message => hooks.get("turn_end")({outcome:"completed",message}));
for (const outcome of ["aborted","error"]) results.push(hooks.get("turn_end")({outcome,message:assistant(raw)}));
const original = assistant(raw);
const first = hooks.get("turn_end")({outcome:"completed",message:original,messageEntryId:"one"});
const valid = {role:"assistant",content:[{type:"toolCall",id:"valid",name:"read",arguments:{path:"x"}}]};
results.push(hooks.get("turn_end")({outcome:"completed",message:valid}));
const second = hooks.get("turn_end")({outcome:"completed",message:original,messageEntryId:"two"});
const custom = decision => ({role:"custom",customType:decision.entries[0].customType,details:decision.entries[0].details,content:decision.entries[0].content});
const projection = hooks.get("context")({messages:[original,custom(first),valid,original,custom(second)]}).messages;
const exhausted = hooks.get("turn_end")({outcome:"completed",message:original});
process.stdout.write(JSON.stringify({unmodified:results.every(result => result === undefined), exhausted:exhausted===undefined,
  errorResults:projection.filter(message => message.role==="toolResult"), original}));
"""
    result = json.loads(subprocess.run(
        ["node", "--input-type=module", "--eval", runner, extension.as_uri()],
        check=True, capture_output=True, text=True,
    ).stdout)
    assert result["unmodified"] and result["exhausted"]
    assert len(result["errorResults"]) == 2
    assert result["errorResults"][0]["toolCallId"] != result["errorResults"][1]["toolCallId"]
    assert all(message["isError"] for message in result["errorResults"])
