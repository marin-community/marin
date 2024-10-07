############################################################
# Transform HTML to text

from marin.execution.executor import ExecutorStep, executor_main, this_output_path, versioned
from marin.schemas.web.convert import HtmlToMarkdownConfig, TrafilaturaConfig
from scripts.fineweb.process_parquet_fw import ParquetFWConfig, process_fw_dump

raw_data = "gs://marin-us-central2/raw/fineweb/cd85054/CC-MAIN-2024-18"

transform_trafilatura_step = ExecutorStep(
    name="documents/helloworld_fw-herumb-cc0934/fineweb_last_dump_trafilatura",
    fn=process_fw_dump,
    config=ParquetFWConfig(
        input_path=raw_data,
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

transform_resiliparse_step = ExecutorStep(
    name="documents/helloworld_fw-herumb-cc0934/fineweb_last_dump_resiliparse",
    fn=process_fw_dump,
    config=ParquetFWConfig(
        input_path=raw_data,
        output_path_md=this_output_path("md"),
        output_path_text=this_output_path("text"),
        extract_method=versioned("resiliparse"),
    ),
)

transform_readability_step = ExecutorStep(
    name="documents/helloworld_fw-herumb-cc0934/fineweb_last_dump_readability",
    fn=process_fw_dump,
    config=ParquetFWConfig(
        input_path=raw_data,
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
            transform_trafilatura_step,
            transform_resiliparse_step,
            transform_readability_step,
        ]
    )
