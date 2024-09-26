from dataclasses import dataclass


@dataclass(frozen=True)
class TrafilaturaConfig:
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
