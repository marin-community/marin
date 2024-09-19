from dataclasses import dataclass


@dataclass
class TrafilaturaConfig:
    favor_precision: bool
    favor_recall: bool
    include_comments: bool
    deduplicate: bool
    include_comments: bool

    # Default config for Trafilatura
    include_images: bool = False
    include_tables: bool = True
    include_links: bool = False
    include_formatting: bool = False
    output_format: str = "txt"


    @classmethod
    def fineweb_config(cls) -> 'TrafilaturaConfig':
        return cls(
            favor_precision=True,
            favor_recall=False,
            include_comments=False,
            deduplicate=True,
        )

    @classmethod
    def default_config(cls) -> 'TrafilaturaConfig':
        return cls(
            favor_precision=False,
            favor_recall=True,
            include_comments=False,
            deduplicate=False,
        )
