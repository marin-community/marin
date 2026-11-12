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

    @classmethod
    def values(cls) -> set[str]:
        return {member.value for member in cls}
