"""
GBNFAdapter — Grammar-constrained output adapter using GBNF format.

GBNF (GGML BNF) is the grammar format used by llama.cpp compatible inference
servers to constrain model output to a specific structure. Because the model
physically cannot produce tokens outside the grammar, format errors drop to
zero — at the cost of requiring a compatible backend.

How it works
------------
1. ``format()``   — builds ToonAdapter-style prompts (### Field: markers).
2. ``__call__()`` — generates a GBNF grammar from the signature's output
                    fields and injects it into ``lm_kwargs["extra_body"]``
                    before the LM call.
3. ``parse()``    — same section-marker parsing as ToonAdapter.

Compatible backends
-------------------
- llama.cpp server  (``--grammar`` / ``extra_body={"grammar": ...}``)
- vLLM with ``--guided-decoding-backend outlines`` or ``xgrammar`` does NOT
  use this format; use XGrammarAdapter instead.

Usage
-----
::

    import dspy
    from experiments.dspy.adapters.gbnf import GBNFAdapter

    lm = dspy.LM("openai/...", base_url="http://localhost:8080", api_key="none")
    dspy.configure(lm=lm, adapter=GBNFAdapter())
"""

import enum
import re
from types import UnionType
from typing import Any, Union, get_args, get_origin

from dspy import Adapter
from dspy.signatures.signature import Signature


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _format_field_name(name: str) -> str:
    return name.replace("_", " ").title()


def _escape_gbnf(s: str) -> str:
    return s.replace("\\", "\\\\").replace('"', '\\"')


# ---------------------------------------------------------------------------
# GBNF grammar generation
# ---------------------------------------------------------------------------

def _enum_rule(annotation: type[enum.Enum]) -> tuple[str, str]:
    """Return (rule_name, rule_definition) for an Enum type."""
    rule_name = re.sub(r"(?<!^)(?=[A-Z])", "-", annotation.__name__).lower()
    alternatives = " | ".join(f'"{_escape_gbnf(str(m.value))}"' for m in annotation)
    return rule_name, f"{rule_name} ::= {alternatives}"


def _type_to_rule_ref(annotation: Any, rules: dict[str, str]) -> str:
    """Recursively map a Python type annotation to a GBNF rule name.

    Side-effects: may add new rules to *rules*.
    """
    # Unwrap Optional / Union — use the first non-None type
    origin = get_origin(annotation)
    if origin in (Union, UnionType):
        non_none = [a for a in get_args(annotation) if a is not type(None)]
        if non_none:
            return _type_to_rule_ref(non_none[0], rules)

    if annotation is None or annotation is str:
        rules.setdefault("string", r'string ::= [^\n]+')
        return "string"

    if annotation is int:
        rules.setdefault("integer", r'integer ::= "-"? [0-9]+')
        return "integer"

    if annotation is float:
        rules.setdefault("number", r'number ::= "-"? ([0-9]+ ("." [0-9]*)? | "." [0-9]+)')
        return "number"

    if annotation is bool:
        rules.setdefault("boolean", 'boolean ::= "true" | "false"')
        return "boolean"

    if isinstance(annotation, type) and issubclass(annotation, enum.Enum):
        rule_name, rule_def = _enum_rule(annotation)
        rules.setdefault(rule_name, rule_def)
        return rule_name

    origin = get_origin(annotation)
    if origin is list or isinstance(annotation, type) and issubclass(annotation, list):
        args = get_args(annotation)
        inner = _type_to_rule_ref(args[0], rules) if args else "string"
        rule_name = f"list-of-{inner}"
        if rule_name not in rules:
            rules[rule_name] = f'{rule_name} ::= "- " {inner} ("\\n- " {inner})* "\\n"?'
        return rule_name

    # fallback
    rules.setdefault("string", r'string ::= [^\n]+')
    return "string"


def build_gbnf_grammar(signature: type[Signature]) -> str:
    """Build a GBNF grammar string from a DSPy signature's output fields.

    Args:
        signature: A DSPy Signature class.

    Returns:
        A multi-line GBNF grammar string ready to pass to a llama.cpp server.
    """
    if not signature.output_fields:
        return r'root ::= [^\x00]*'

    rules: dict[str, str] = {}
    field_parts: list[str] = []

    for name, field in signature.output_fields.items():
        ref = _type_to_rule_ref(field.annotation, rules)
        f_name = _format_field_name(name)
        field_rule_name = f"{name.replace('_', '-')}-section"
        rules[field_rule_name] = f'{field_rule_name} ::= "### {f_name}:\\n" {ref} "\\n"'
        field_parts.append(field_rule_name)

    if len(field_parts) == 1:
        root_rule = f'root ::= {field_parts[0]} "\\n"*'
    else:
        root_rule = f"root ::= {field_parts[0]}" + "".join(
            f' ("\\n"+ {f})' for f in field_parts[1:]
        ) + ' "\\n"*'

    return "\n".join([root_rule, *list(rules.values())])


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------

class GBNFAdapter(Adapter):
    """Grammar-constrained DSPy adapter for llama.cpp compatible servers.

    Passes a GBNF grammar derived from the signature's output fields to the
    inference server via ``extra_body["grammar"]``, eliminating format errors.
    Prompt formatting follows the same ToonAdapter ``### Field:`` convention.
    """

    def format(
        self,
        signature: type[Signature],
        demos: list[dict[str, Any]],
        inputs: dict[str, Any],
    ) -> list[dict[str, str]]:
        messages: list[dict[str, str]] = []

        system_content = [signature.instructions.strip(), "\nExpected Output Fields:"]
        for name, field in signature.output_fields.items():
            type_name = getattr(field.annotation, "__name__", str(field.annotation))
            system_content.append(f"- {name} ({type_name}): {field.description or ''}")
        messages.append({"role": "system", "content": "\n".join(system_content)})

        user_content: list[str] = []
        for name, _field in signature.input_fields.items():
            val = inputs.get(name, "")
            if isinstance(val, list):
                val = "\n".join(f"- {v}" for v in val)
            user_content.append(f"### {_format_field_name(name)}:\n{val}\n")

        user_content.append("Respond strictly using this format:")
        for name in signature.output_fields:
            user_content.append(f"### {_format_field_name(name)}:")

        messages.append({"role": "user", "content": "\n".join(user_content)})
        return messages

    def __call__(
        self,
        lm,
        lm_kwargs: dict[str, Any],
        signature: type[Signature],
        demos: list[dict[str, Any]],
        inputs: dict[str, Any],
    ) -> list[dict[str, Any]]:
        extra_body = dict(lm_kwargs.get("extra_body", {}) or {})
        if "grammar" in extra_body:
            raise ValueError("GBNFAdapter: 'grammar' already provided in extra_body.")
        grammar = build_gbnf_grammar(signature)
        new_lm_kwargs = lm_kwargs.copy()
        new_lm_kwargs["extra_body"] = {**extra_body, "grammar": grammar}
        return super().__call__(lm, new_lm_kwargs, signature, demos, inputs)

    def parse(
        self,
        signature: type[Signature],
        completion: str,
    ) -> dict[str, Any]:
        result: dict[str, Any] = {name: None for name in signature.output_fields}

        if not completion or not completion.strip():
            return result

        header_re = re.compile(r"###\s+([^\n:]+)\s*:\s*", re.IGNORECASE)
        parts = header_re.split(completion)

        i = 1
        while i + 1 < len(parts):
            raw_name = parts[i].strip().lower()
            raw_value = parts[i + 1].strip()
            i += 2

            matched = next(
                (
                    n for n in signature.output_fields
                    if n.lower() == raw_name.replace(" ", "_")
                    or _format_field_name(n).lower() == raw_name
                ),
                None,
            )
            if matched is None:
                continue

            result[matched] = self._coerce(raw_value, signature.output_fields[matched].annotation)

        return result

    @staticmethod
    def _coerce(value: str, annotation: Any) -> Any:
        # Unwrap Optional / Union — use the first non-None type
        origin = get_origin(annotation)
        if origin in (Union, UnionType):
            non_none = [a for a in get_args(annotation) if a is not type(None)]
            if non_none:
                return GBNFAdapter._coerce(value, non_none[0])

        if annotation is bool:
            v_lower = value.lower().strip()
            if v_lower == "true":
                return True
            if v_lower == "false":
                return False
            return None

        if isinstance(annotation, type) and issubclass(annotation, enum.Enum):
            v = value.strip().lower()
            for member in annotation:
                if str(member.value).lower() == v:
                    return member
            for member in annotation:
                if member.name.lower() == v:
                    return member
            return None

        origin = get_origin(annotation)
        if annotation is list or origin is list:
            return [line.lstrip("-•* ").strip() for line in value.splitlines() if line.strip()]

        try:
            if annotation is int:
                return int(value)
            if annotation is float:
                return float(value)
        except (ValueError, TypeError):
            return None

        return value
