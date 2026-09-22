# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, field, fields

from draccus.choice_types import ChoiceRegistry

DEFAULT_KEEP_INLINE_IMAGES_IN = ["li", "p", "td", "th", "h1", "h2", "h3", "h4", "h5", "h6", "a"]


class ExtractionConfig(ChoiceRegistry):
    pass


@dataclass(frozen=True)
@ExtractionConfig.register_subclass("markdownify")
class HtmlToMarkdownConfig(ExtractionConfig):
    include_images: bool = True
    include_links: bool = True

    heading_style: str = "ATX"
    keep_inline_images_in: list = field(default_factory=lambda: DEFAULT_KEEP_INLINE_IMAGES_IN.copy())

    @property
    def markdownify_kwargs(self) -> dict:
        """The markdownify options this config overrides; every other option keeps markdownify's default."""
        exclude = {"include_images", "include_links"}
        return {f.name: getattr(self, f.name) for f in fields(self) if f.name not in exclude}


@dataclass(frozen=True)
@ExtractionConfig.register_subclass("resiliparse")
class ResiliparseConfig(ExtractionConfig):
    links: bool = False
    prepend_title: bool = True

    list_bullets: bool = True
    alt_texts: bool = False
    form_fields: bool = False
    noscript: bool = False
    comments: bool | None = None
    skip_elements: list | None = None

    markdownify_config: HtmlToMarkdownConfig = field(default_factory=HtmlToMarkdownConfig)

    @property
    def resiliparse_kwargs(self) -> dict:
        exclude = {"markdownify_config", "prepend_title"}
        return {f.name: getattr(self, f.name) for f in fields(self) if f.name not in exclude}
