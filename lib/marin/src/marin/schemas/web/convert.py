# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, field, fields

from draccus.choice_types import ChoiceRegistry

ASTERISK = "*"
SPACES = "spaces"


DEFAULT_KEEP_INLINE_IMAGES_IN = ["li", "p", "td", "th", "h1", "h2", "h3", "h4", "h5", "h6", "a"]


class ExtractionConfig(ChoiceRegistry):
    pass


@dataclass(frozen=True)
@ExtractionConfig.register_subclass("markdownify")
class HtmlToMarkdownConfig:
    include_images: bool = True
    include_links: bool = True

    heading_style: str = "ATX"
    keep_inline_images_in: list = field(default_factory=lambda: DEFAULT_KEEP_INLINE_IMAGES_IN.copy())
    autolinks = True
    bullets = "*+-"  # An iterable of bullet types.
    code_language = ""
    code_language_callback = None
    convert = None
    default_title = False
    escape_asterisks = True
    escape_underscores = True
    newline_style = SPACES
    strip = None
    strong_em_symbol = ASTERISK
    sub_symbol = ""
    sup_symbol = ""
    wrap = False
    wrap_width = 80

    @property
    def markdownify_kwargs(self) -> dict:
        exclude = {"include_images", "include_links"}
        return {f.name: getattr(self, f.name) for f in fields(self) if f.name not in exclude}


@dataclass(frozen=True)
@ExtractionConfig.register_subclass("resiliparse")
class ResiliparseConfig:
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

    @property
    def markdownify_kwargs(self) -> dict:
        exclude = {*list(self.resiliparse_kwargs.keys()), "prepend_title"}
        return {f.name: getattr(self, f.name) for f in fields(self.markdownify_config) if f.name not in exclude}
