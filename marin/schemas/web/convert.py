from dataclasses import dataclass, field, fields

from draccus.choice_types import ChoiceRegistry

# from markdownify import ASTERISK, SPACES
# TODO(Herumb): Can we remove this import. We don't want any imports in the configs. We want configs inline.
ASTERISK = "*"
SPACES = "spaces"


@dataclass(frozen=True)
class ExtractionConfig(ChoiceRegistry):
    pass


@dataclass(frozen=True)
@ExtractionConfig.register_subclass("trafilatura")
class TrafilaturaConfig(ExtractionConfig):
    favor_precision: bool = False
    favor_recall: bool = True
    include_comments: bool = False
    deduplicate: bool = False

    # Default config for Trafilatura
    include_images: bool = False
    include_tables: bool = True
    include_links: bool = False
    include_formatting: bool = False
    output_format: str = "txt"

    @classmethod
    def fineweb_config(cls) -> "TrafilaturaConfig":
        """
        Using the configuration as per the `fineweb` method in datatrove
        codebase: https://github.com/huggingface/datatrove/blob/c7f6f516abc1349e4995451ff4017790d00d2d68/examples/fineweb.py#L42

        Returns:
            TrafilaturaConfig: Configuration object for Trafilatura.
        """
        return cls(
            favor_precision=True,
            favor_recall=False,
            include_comments=False,
            deduplicate=True,
        )

    @classmethod
    def default_config(cls) -> "TrafilaturaConfig":
        return cls(
            favor_precision=False,
            favor_recall=True,
            include_comments=False,
            deduplicate=False,
        )

    @classmethod
    def get_preset_config(cls, config: str) -> "TrafilaturaConfig":
        if config == "fineweb":
            return cls.fineweb_config()
        elif config == "default":
            return cls.default_config()
        else:
            raise Exception(f"Invalid preset config: {config}. Please use 'fineweb' or 'default'.")


DEFAULT_KEEP_INLINE_IMAGES_IN = ["li", "p", "td", "th", "h1", "h2", "h3", "h4", "h5", "h6", "a"]


@dataclass(frozen=True)
@ExtractionConfig.register_subclass("markdownify")
class HtmlToMarkdownConfig(ExtractionConfig):

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

    @classmethod
    def default_config(cls) -> "HtmlToMarkdownConfig":
        return cls()

    @classmethod
    def get_preset_config(cls, config: str) -> "HtmlToMarkdownConfig":
        if config == "default":
            return cls.default_config()
        else:
            raise Exception(f"Invalid preset config: {config}. Please use 'default'.")

    @property
    def markdownify_kwargs(self) -> dict:
        exclude = {"include_images", "include_links"}
        return {f.name: getattr(self, f.name) for f in fields(self) if f.name not in exclude}


@dataclass(frozen=True)
@ExtractionConfig.register_subclass("resiliparse")
class ResiliparseConfig(ExtractionConfig):
    preserve_formatting: bool = False
    main_content: bool = True
    links: bool = False
    prepend_title: bool = True

    list_bullets: bool = True
    alt_texts: bool = False
    form_fields: bool = False
    noscript: bool = False
    comments: bool | None = None
    skip_elements: list | None = None

    use_custom_variant: bool = False  # If True, it'll use our custom fork of Resiliparse for extraction.
    markdownify_config: HtmlToMarkdownConfig = field(default_factory=HtmlToMarkdownConfig.default_config)

    @classmethod
    def default_config(cls) -> "ResiliparseConfig":
        return cls()

    @classmethod
    def get_preset_config(cls, config: str) -> "ResiliparseConfig":
        if config == "default":
            return cls.default_config()
        else:
            raise Exception(f"Invalid preset config: {config}. Please use 'default'.")

    @property
    def resiliparse_kwargs(self) -> dict:
        exclude = {"use_custom_variant", "markdownify_config", "prepend_title"}
        return {
            f.name: getattr(self, f.name)
            for f in fields(self)
            if f.name not in exclude and (self.use_custom_variant and f.name != "preserve_formatting")
        }

    @property
    def markdownify_kwargs(self) -> dict:
        exclude = {"use_custom_variant", *list(self.resiliparse_kwargs.keys()), "prepend_title"}
        return {f.name: getattr(self, f.name) for f in fields(self.markdownify_config) if f.name not in exclude}
