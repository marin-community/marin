from harbor.agents.installed.claude_code import ClaudeCode as InstalledClaudeCode
from harbor.agents.local.claude_code import ClaudeCode as LocalClaudeCode
from harbor.models.agent.context import AgentContext


def test_local_trajectory_converter_receives_model_alias(monkeypatch, tmp_path) -> None:
    captured_aliases: list[str | None] = []

    def capture_alias(self, context: AgentContext) -> None:
        captured_aliases.append(self.model_alias)

    monkeypatch.setattr(
        InstalledClaudeCode,
        "populate_context_post_run",
        capture_alias,
    )
    agent = LocalClaudeCode(
        logs_dir=tmp_path,
        model_name="hosted_vllm/org/canonical-model",
        model_alias="hosted_vllm/resumed-endpoint",
    )

    agent.populate_context_post_run(AgentContext())

    assert captured_aliases == ["hosted_vllm/resumed-endpoint"]
