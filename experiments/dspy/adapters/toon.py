"""
ToonAdapter — Alpaca-style structured output adapter.

Uses "### {Field}:" section markers instead of the JSON-based (BAML) or
delimiter-based (Chat) formats.  Because the prompt style is fundamentally
different from the adapters used during training, this adapter is primarily
useful as an **out-of-distribution (OOD) evaluation target**: we want to
know whether a model trained on BAML/Chat data can still follow instructions
when they arrive in TOON format.

Prompt structure
----------------
System message:
    <signature instructions>
    Output fields and their types.

User message:
    ### Claim:
    <value>

    ### Evidence:
    <value>

    Respond with the following fields:
    ### Label:
    ### Reasoning:

Assistant response (expected):
    ### Label:
    Supports

    ### Reasoning:
    <reasoning text>
"""

import re
from typing import Any

from dspy import Adapter
from dspy.signatures.signature import Signature


class ToonAdapter(Adapter):
    """Alpaca-style adapter that formats prompts using '### Field:' markers.

    Extends :class:`dspy.Adapter` and overrides ``format`` / ``parse`` so
    that the full prompt pipeline (system + user message construction,
    response parsing) uses TOON-style markers rather than JSON or the
    ``[[ ## field ## ]]`` delimiters used by ChatAdapter.
    """

    # ------------------------------------------------------------------
    # format() — called by dspy.Adapter.__call__ to build the messages
    # ------------------------------------------------------------------

    def format(
        self,
        signature: type[Signature],
        demos: list[dict[str, Any]],
        inputs: dict[str, Any],
    ) -> list[dict[str, str]]:
        """Return a list of chat messages in TOON format.

        Args:
            signature: The DSPy signature describing inputs and outputs.
            inputs: A dict mapping input field names to their values.
            demos: Few-shot demonstrations (currently ignored).

        Returns:
            A list of ``{"role": ..., "content": ...}`` dicts.
        """
        messages: list[dict[str, str]] = []

        # ---- System message ----
        system_lines = [signature.instructions.strip(), ""]
        system_lines.append("Output fields and their expected types:")
        for name, field in signature.output_fields.items():
            type_name = getattr(field.annotation, "__name__", str(field.annotation))
            desc = f"  - {name} ({type_name})"
            if field.description:
                desc += f": {field.description}"
            system_lines.append(desc)

        messages.append({"role": "system", "content": "\n".join(system_lines)})

        # ---- User message ----
        user_lines: list[str] = []
        for name, _field in signature.input_fields.items():
            value = inputs.get(name, "")
            if isinstance(value, list):
                value = "\n".join(f"- {v}" for v in value)
            user_lines.append(f"### {name.capitalize()}:")
            user_lines.append(str(value))
            user_lines.append("")

        user_lines.append("Respond using the following fields:")
        for name in signature.output_fields:
            user_lines.append(f"### {name.capitalize()}:")

        messages.append({"role": "user", "content": "\n".join(user_lines)})

        return messages

    # ------------------------------------------------------------------
    # parse() — called by dspy.Adapter.__call__ to parse the LM response
    # ------------------------------------------------------------------

    def parse(
        self,
        signature: type[Signature],
        completion: str,
    ) -> dict[str, Any]:
        """Extract output fields from a TOON-formatted completion.

        Looks for ``### FieldName:`` markers and collects the text that
        follows each one up to the next marker (or end of string).

        Args:
            signature: The DSPy signature (used to know which fields to
                extract and what types to coerce them to).
            completion: The raw text returned by the language model.

        Returns:
            A dict mapping output field names to their parsed values.
            Fields that cannot be found are set to ``None``.
        """
        output_fields = signature.output_fields
        result: dict[str, Any] = {name: None for name in output_fields}

        if not completion or not completion.strip():
            return result

        field_names = list(output_fields.keys())

        # Split on ALL "### Something:" markers so values don't bleed into
        # the next section even when the next field is not an output field.
        any_header_pattern = re.compile(r"###\s+(\w+)\s*:[ \t]*", re.IGNORECASE)
        parts = any_header_pattern.split(completion)
        # parts layout: [pre_text, name1, value1, name2, value2, ...]
        i = 1
        while i + 1 < len(parts):
            raw_name  = parts[i].strip().lower()
            raw_value = parts[i + 1].strip()
            i += 2

            # Keep only known output fields
            matched_name = next(
                (n for n in field_names if n.lower() == raw_name), None
            )
            if matched_name is None:
                continue

            field_annotation = output_fields[matched_name].annotation
            result[matched_name] = self._coerce(raw_value, field_annotation)

        return result

    # ------------------------------------------------------------------
    # Internal helper
    # ------------------------------------------------------------------

    @staticmethod
    def _coerce(value: str, annotation: Any) -> Any:
        """Attempt to coerce *value* to the field's annotated type.

        Falls back to the raw string if coercion is not straightforward.
        """
        if annotation is None or annotation is str:
            return value

        # Handle list types (e.g. list[str] or bare list) — parse bullet lines
        try:
            from typing import get_origin
            if annotation is list or get_origin(annotation) is list:
                lines = [line.lstrip("-•* ").strip() for line in value.splitlines()]
                return [line for line in lines if line]
        except Exception:
            pass

        # Handle Enum types (e.g. ClaimVerificationLabel)
        try:
            import enum
            if isinstance(annotation, type) and issubclass(annotation, enum.Enum):
                # Try matching by value (case-insensitive)
                value_lower = value.strip().lower()
                for member in annotation:
                    if member.value.lower() == value_lower:
                        return member
                # Try matching by name
                for member in annotation:
                    if member.name.lower() == value_lower:
                        return member
        except Exception:
            pass

        # int / float
        if annotation is int:
            try:
                return int(value)
            except ValueError:
                return None
        if annotation is float:
            try:
                return float(value)
            except ValueError:
                return None

        # bool
        if annotation is bool:
            return value.strip().lower() in ("true", "yes", "1")

        return value
