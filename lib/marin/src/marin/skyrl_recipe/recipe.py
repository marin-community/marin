"""Sparse author parts and complete recipes for Hydra launch composition."""

from collections.abc import Mapping, Sequence
from typing import Any, Literal, Self
from urllib.parse import urlsplit

from pydantic import Field, PositiveInt, ValidationError, model_validator

from .budget import ContextBudget
from .documents import MISSING, get_path
from .model import FrozenMap, OpenMap, Section, unset_field
from .ownership import OWNER_MESSAGES, REMOVED
from .rules import RLEntrypoint, validate_engine_init_kwargs, validate_tp_divides_heads
from .sections import RecipeSections


class ConfigGroups(Section):
    """Hydra group selections supported by Iris recipes."""

    terminal_bench_config: Literal["terminal_bench"] = unset_field()
    algorithm_recipe: Literal["cispo", "dapo", "dr_grpo", "ftpo", "grpo", "gspo", "mopd", "opd"] = unset_field()


class RecipePatch(RecipeSections):
    """An authored contribution with optional context and sparse section values."""

    entrypoint: RLEntrypoint = RLEntrypoint.STANDARD
    context_budget: ContextBudget | None = None
    config_groups: ConfigGroups = Field(default_factory=ConfigGroups)
    terminal_bench: OpenMap | None = None
    extra_env: OpenMap = Field(default_factory=FrozenMap)
    model_num_attention_heads: PositiveInt | None = None

    @classmethod
    def from_document(cls, document: Mapping[str, Any]) -> Self:
        removed = [f"{path}: {message}" for path, message in REMOVED.items() if get_path(document, path) is not MISSING]
        if removed:
            raise ValueError("; ".join(removed))
        try:
            return super().from_document(document)
        except ValidationError as error:
            owned = []
            for problem in error.errors():
                path = ".".join(str(part) for part in problem["loc"])
                if problem["type"] == "extra_forbidden" and path in OWNER_MESSAGES:
                    owned.append(f"{path}: {OWNER_MESSAGES[path]}")
            if owned:
                raise ValueError("; ".join(owned)) from error
            raise

    @model_validator(mode="after")
    def _open_mapping_owners(self) -> Self:
        document = self.to_skyrl()
        removed = [f"{path}: {message}" for path, message in REMOVED.items() if get_path(document, path) is not MISSING]
        if removed:
            raise ValueError("; ".join(removed))
        # Closed fields are rejected by their generated classes; only open-map owners reach this stage.
        owned = [
            f"{path}: {message}" for path, message in OWNER_MESSAGES.items() if get_path(document, path) is not MISSING
        ]
        if owned:
            raise ValueError("; ".join(owned))
        validate_engine_init_kwargs(self.generator.engine_init_kwargs)
        return self

    def with_settings(self, settings: Sequence[str]) -> Self:
        for setting in settings:
            path = setting.partition("=")[0]
            if path in REMOVED:
                raise ValueError(f"{path}: {REMOVED[path]}")
            if path in OWNER_MESSAGES:
                raise ValueError(f"{path}: {OWNER_MESSAGES[path]}")
        return super().with_settings(settings)


class SkyRLRecipe(RecipePatch):
    """A complete sparse recipe with a required rollout context declaration."""

    context_budget: ContextBudget

    @model_validator(mode="after")
    def _complete_author_rules(self) -> Self:
        validate_tp_divides_heads(self.generator.inference_engine_tensor_parallel_size, self.model_num_attention_heads)
        model = get_path(self.to_skyrl(), "generator.speculative_decoding.model", {})
        if isinstance(model, Mapping) and urlsplit(model.get("source_uri", "")).scheme in ("s3", "gs", "gcs"):
            identity = model.get("source_identity", "")
            digest = identity.removeprefix("sha256:")
            if (
                not identity.startswith("sha256:")
                or len(digest) != 64
                or any(c not in "0123456789abcdef" for c in digest)
            ):
                raise ValueError(
                    "generator.speculative_decoding.model.source_identity: artifact sources require sha256:<64 lowercase hex digits>"
                )
        return self
