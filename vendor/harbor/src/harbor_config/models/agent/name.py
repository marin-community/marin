from enum import Enum


class AgentName(str, Enum):
    NOP = "nop"
    ORACLE = "oracle"
    TERMINUS_2 = "terminus-2"
    TERMINUS_KIRA = "terminus-kira"
    CODEX = "codex"
    MINI_SWE_AGENT = "mini-swe-agent"
    OPENCODE = "opencode"
    PI = "pi"
    TASK_SPEC_CHAT = "task-spec-chat"
    TASK_SPEC_RESPONSE = "task-spec-response"

    @classmethod
    def values(cls) -> set[str]:
        return {member.value for member in cls}
