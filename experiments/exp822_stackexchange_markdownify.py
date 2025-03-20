"""
This experiment converts Stack Exchange HTML to markdown using Resiliparse's custom fork. We introduce
a template for the markdownified Stack Exchange data in threaded format.

The template for threaded format is:
```
Question:
<question>
Answer:
> <votes>
<answer_1>

Answer 2:
> <votes>
<answer_2>

Tags: <tags>
```

Reference Issue: https://github.com/stanford-crfm/marin/issues/822
"""

from marin.execution.executor import ExecutorStep, executor_main, this_output_path, versioned
from marin.schemas.web.convert import HtmlToMarkdownConfig, ResiliparseConfig
from operations.transform.stackexchange.transform_stackexchange import (
    StackExchangeExtractionConfig,
    process_stackexchange_dump,
)

stackexchange_text_resiliparse_custom_fork = ExecutorStep(
    name="documents/stackexchange-resiliparse-custom-fork",
    fn=process_stackexchange_dump,
    config=StackExchangeExtractionConfig(
        input_path=versioned("gs://marin-us-central2/documents/stackexchange/v2024-04-02/md-complete"),
        output_path=this_output_path(),
        extract_method="resiliparse",
        extract_config=ResiliparseConfig(
            preserve_formatting=True,
            main_content=True,
            links=False,
            use_custom_variant=True,
            markdownify_config=HtmlToMarkdownConfig(
                include_images=False,
                include_links=False,
            ),
        ),
    ),
    pip_dependency_groups=["download_transform"],
)


if __name__ == "__main__":
    executor_main(
        steps=[
            stackexchange_text_resiliparse_custom_fork,
        ]
    )
