"""Public document operations shared by generated recipe roots."""

from collections.abc import Mapping, Sequence
import json
from typing import Any, Self

from .documents import combine_parts, set_path
from .model import Section, freeze, merge_mappings, thaw
from .settings import parse_setting


class RecipeDocument(Section):
    """Validate complete sparse documents after combining or overriding contributions."""

    @classmethod
    def from_document(cls, document: Mapping[str, Any]) -> Self:
        return cls.model_validate_json(json.dumps(thaw(freeze(document)), allow_nan=False))

    @classmethod
    def combine(cls, *, base: Section | None = None, **parts: Section) -> Self:
        document = combine_parts(
            base.to_skyrl() if base is not None else {},
            {name: part.to_skyrl() for name, part in parts.items()},
        )
        return cls.from_document(document)

    def merge(self, *patches: Section) -> Self:
        document = self.to_skyrl()
        for patch in patches:
            document = merge_mappings(document, patch.to_skyrl())
        return type(self).from_document(document)

    def with_settings(self, settings: Sequence[str]) -> Self:
        document = self.to_skyrl()
        for setting in settings:
            key, separator, raw = setting.partition("=")
            if not separator or not key:
                raise ValueError(f"a setting must have dotted.key=value form: {setting!r}")
            patch = {}
            set_path(patch, key, parse_setting(type(self), key, raw))
            document = merge_mappings(document, patch)
        return type(self).from_document(document)
