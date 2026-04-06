"""
XGrammarAdapter — JSON-schema constrained output adapter for vLLM.

vLLM's OpenAI-compatible API accepts a ``guided_json`` parameter in
``extra_body`` that forces the model to produce output matching a JSON
schema. When the vLLM server is started with XGrammar as the structured-
output backend (``--structured-outputs-config.backend=xgrammar``), this
schema is compiled into an efficient grammar at request time, making format
errors impossible.

How it works
------------
1. ``format()``   — builds a JSON-oriented system prompt that describes the
                    expected output schema clearly to the model.
2. ``__call__()`` — derives a JSON schema from the signature's output fields
                    and injects it into ``lm_kwargs["extra_body"]["guided_json"]``
                    before the LM call.
3. ``parse()``    — parses the constrained JSON output back into a dict of
                    typed Python values.

Compatible backends
-------------------
- vLLM  (``--guided-decoding-backend xgrammar`` or ``outlines``)
- Any OpenAI-compatible server that supports ``guided_json`` in extra_body.
- Does NOT work with llama.cpp; use GBNFAdapter for that.

Usage
-----
::

    import dspy
    from experiments.dspy.adapters.xgrammar import XGrammarAdapter

    lm = dspy.LM("openai/...", base_url="http://localhost:8000", api_key="none")
    dspy.configure(lm=lm, adapter=XGrammarAdapter())
"""

import enum
import json
import re
from types import UnionType
from typing import Any, Union, get_args, get_origin

from dspy import Adapter
from dspy.signatures.signature import Signature
from dspy.utils.exceptions import AdapterParseError


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _format_field_name(name: str) -> str:
    return name.replace("_", " ").title()


def _is_optional(annotation: Any) -> bool:
    origin = get_origin(annotation)
    if origin in (Union, UnionType):
        return type(None) in get_args(annotation)
    return False


# ---------------------------------------------------------------------------
# JSON schema generation
# ---------------------------------------------------------------------------

def _annotation_to_json_schema(annotation: Any) -> dict[str, Any]:
    """Convert a Python type annotation to a JSON Schema dict."""
    origin = get_origin(annotation)
    args = get_args(annotation)

    # Union / Optional
    if origin in (Union, UnionType):
        non_none = [a for a in args if a is not type(None)]
        if len(non_none) == 1:
            return _annotation_to_json_schema(non_none[0])
        return {"anyOf": [_annotation_to_json_schema(a) for a in non_none]}

    if annotation is None or annotation is str:
        return {"type": "string"}

    if annotation is int:
        return {"type": "integer"}

    if annotation is float:
        return {"type": "number"}

    if annotation is bool:
        return {"type": "boolean"}

    if isinstance(annotation, type) and issubclass(annotation, enum.Enum):
        return {"type": "string", "enum": [str(m.value) for m in annotation]}

    if origin is list or annotation is list:
        item_type = args[0] if args else str
        return {"type": "array", "items": _annotation_to_json_schema(item_type)}

    # fallback
    return {"type": "string"}


def build_json_schema(signature: type[Signature]) -> dict[str, Any]:
    """Build a JSON Schema object from a DSPy signature's output fields.

    Args:
        signature: A DSPy Signature class.

    Returns:
        A JSON Schema dict ready to pass as ``guided_json`` to vLLM.
    """
    properties: dict[str, Any] = {}
    required: list[str] = []

    for name, field in signature.output_fields.items():
        schema = _annotation_to_json_schema(field.annotation)
        if field.description:
            schema["description"] = field.description
        properties[name] = schema
        if not _is_optional(field.annotation):
            required.append(name)

    return {
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": False,
    }


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------

class XGrammarAdapter(Adapter):
    """JSON-schema constrained DSPy adapter for vLLM (XGrammar backend).

    Derives a JSON schema from the DSPy signature's output fields and passes
    it to the vLLM server via ``extra_body["guided_json"]``. The model is
    forced to produce valid JSON, eliminating format errors entirely.
    """

    def format(
        self,
        signature: type[Signature],
        demos: list[dict[str, Any]],
        inputs: dict[str, Any],
    ) -> list[dict[str, str]]:
        messages: list[dict[str, str]] = []

        schema = build_json_schema(signature)
        system_lines = [
            signature.instructions.strip(),
            "",
            "Respond with a single JSON object matching this schema:",
            json.dumps(schema, indent=2),
        ]
        messages.append({"role": "system", "content": "\n".join(system_lines)})

        user_lines: list[str] = []
        for name, _field in signature.input_fields.items():
            val = inputs.get(name, "")
            if isinstance(val, list):
                val = "\n".join(f"- {v}" for v in val)
            user_lines.append(f"### {_format_field_name(name)}:")
            user_lines.append(str(val))
            user_lines.append("")

        user_lines.append("Return ONLY the JSON object.")
        messages.append({"role": "user", "content": "\n".join(user_lines)})

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
        if "guided_json" in extra_body:
            raise ValueError("XGrammarAdapter: 'guided_json' already provided in extra_body.")
        schema = build_json_schema(signature)
        new_lm_kwargs = lm_kwargs.copy()
        new_lm_kwargs["extra_body"] = {**extra_body, "guided_json": schema}
        return super().__call__(lm, new_lm_kwargs, signature, demos, inputs)

    def parse(
        self,
        signature: type[Signature],
        completion: str,
    ) -> dict[str, Any]:
        result: dict[str, Any] = {name: None for name in signature.output_fields}

        if not completion or not completion.strip():
            raise AdapterParseError(
                adapter_name="XGrammarAdapter",
                signature=signature,
                lm_response=completion or "",
            )

        text = completion.strip()

        # 1. Markdown fence
        fence_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
        if fence_match:
            text = fence_match.group(1).strip()
        else:
            # 2. Fallback: first {...}
            obj_match = re.search(r"\{[\s\S]*\}", text)
            if obj_match:
                text = obj_match.group(0)

        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            raise AdapterParseError(
                adapter_name="XGrammarAdapter",
                signature=signature,
                lm_response=completion,
            )

        if not isinstance(parsed, dict):
            raise AdapterParseError(
                adapter_name="XGrammarAdapter",
                signature=signature,
                lm_response=completion,
            )

        for name, field in signature.output_fields.items():
            if name in parsed:
                result[name] = self._coerce(parsed[name], field.annotation)

        return result

    @staticmethod
    def _coerce(value: Any, annotation: Any) -> Any:
        """Coerce a parsed JSON value to the field's annotated Python type."""
        # Unwrap Optional / Union — use the first non-None type
        origin = get_origin(annotation)
        if origin in (Union, UnionType):
            non_none = [a for a in get_args(annotation) if a is not type(None)]
            if non_none:
                return XGrammarAdapter._coerce(value, non_none[0])

        if annotation is bool:
            return value if isinstance(value, bool) else None

        if isinstance(annotation, type) and issubclass(annotation, enum.Enum):
            v = str(value).strip().lower()
            for member in annotation:
                if str(member.value).lower() == v:
                    return member
            for member in annotation:
                if member.name.lower() == v:
                    return member
            return None

        origin = get_origin(annotation)
        if annotation is list or origin is list:
            if isinstance(value, list):
                args = get_args(annotation)
                if args:
                    return [XGrammarAdapter._coerce(item, args[0]) for item in value]
                return value

            # tolerant fallback for sloppy model output
            if isinstance(value, str):
                lines = [line.strip() for line in value.splitlines() if line.strip()]
                if len(lines) > 1:
                    return [line.lstrip("-•* ").strip() for line in lines]
                if "," in value:
                    return [v.strip() for v in value.split(",") if v.strip()]

            return None

        if annotation is int:
            try:
                return int(value)
            except (ValueError, TypeError):
                return None

        if annotation is float:
            try:
                return float(value)
            except (ValueError, TypeError):
                return None

        if annotation is str:
            return str(value) if value is not None else None

        return value
