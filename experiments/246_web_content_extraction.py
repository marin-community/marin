############################################################
# Transform HTML to text

from marin.execution.executor import ExecutorStep, executor_main, this_output_path, versioned
from marin.schemas.web.convert import HtmlToMarkdownConfig, ResiliparseConfig, TrafilaturaConfig
from scripts.fineweb.process_parquet_fw import ParquetFWConfig, process_fw_dump

raw_data = "gs://marin-us-central2/raw/fineweb/cd85054/"

transform_trafilatura_default_step = ExecutorStep(
    name="documents/fineweb-small-trafilatura",
    fn=process_fw_dump,
    config=ParquetFWConfig(
        input_path=raw_data,
        cc_dumps=versioned(["CC-MAIN-2024-18"]),
        output_path_md=this_output_path("md"),
        output_path_text=this_output_path("text"),
        extract_method=versioned("trafilatura"),
        config=TrafilaturaConfig(
            favor_precision=versioned(False),
            favor_recall=versioned(True),
            include_comments=versioned(False),
            deduplicate=versioned(False),
        ),
    ),
)

transform_trafilatura_favor_precision_step = ExecutorStep(
    name="documents/fineweb-small-trafilatura-favor-precision",
    fn=process_fw_dump,
    config=ParquetFWConfig(
        input_path=raw_data,
        cc_dumps=versioned(["CC-MAIN-2024-18"]),
        output_path_md=this_output_path("md"),
        output_path_text=this_output_path("text"),
        extract_method=versioned("trafilatura"),
        config=TrafilaturaConfig(
            favor_precision=versioned(True),
            favor_recall=versioned(False),
            include_comments=versioned(False),
            deduplicate=versioned(False),
        ),
    ),
)

transform_resiliparse_default_step = ExecutorStep(
    name="documents/fineweb-small-resiliparse-default",
    fn=process_fw_dump,
    config=ParquetFWConfig(
        input_path=raw_data,
        cc_dumps=versioned(["CC-MAIN-2024-18"]),
        output_path_md=this_output_path("md"),
        output_path_text=this_output_path("text"),
        extract_method=versioned("resiliparse"),
        config=ResiliparseConfig(
            preserve_formatting=versioned(False),
            main_content=versioned(True),
            links=versioned(False),
        ),
    ),
)

transform_resiliparse_preserve_formatting_step = ExecutorStep(
    name="documents/fineweb-small-resiliparse-preserve-formatting",
    fn=process_fw_dump,
    config=ParquetFWConfig(
        input_path=raw_data,
        cc_dumps=versioned(["CC-MAIN-2024-18"]),
        output_path_md=this_output_path("md"),
        output_path_text=this_output_path("text"),
        extract_method=versioned("resiliparse"),
        config=ResiliparseConfig(
            preserve_formatting=versioned(True),
            main_content=versioned(True),
            links=versioned(False),
        ),
    ),
)

transform_readability_step = ExecutorStep(
    name="documents/fineweb-small-readability",
    fn=process_fw_dump,
    config=ParquetFWConfig(
        input_path=raw_data,
        cc_dumps=versioned(["CC-MAIN-2024-18"]),
        output_path_md=this_output_path("md"),
        output_path_text=this_output_path("text"),
        extract_method=versioned("readability"),
        config=HtmlToMarkdownConfig(
            include_images=versioned(False),
            include_links=versioned(False),
        ),
    ),
)

############################################################

if __name__ == "__main__":
    executor_main(
        steps=[
            transform_trafilatura_default_step,
            transform_trafilatura_favor_precision_step,
            transform_resiliparse_default_step,
            transform_resiliparse_preserve_formatting_step,
            transform_readability_step,
        ]
    )
