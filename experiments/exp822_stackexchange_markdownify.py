# Copyright 2025 The Marin Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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

Reference Issue: https://github.com/marin-community/marin/issues/822
"""

from marin.execution.executor import executor_main, versioned
from marin.execution import step, StepContext, StepRef, deferred, output
from marin.schemas.web.convert import HtmlToMarkdownConfig, ResiliparseConfig
from marin.transform.stackexchange.transform_stackexchange import StackExchangeExtractionConfig
from marin.transform.stackexchange.transform_stackexchange import process_stackexchange_dump as _process_stackexchange_dump

# Mark library functions as deferred
process_stackexchange_dump = deferred(_process_stackexchange_dump)

@step(name="documents/stackexchange-resiliparse-custom-fork")
def stackexchange_text_resiliparse_custom_fork_step():
    return process_stackexchange_dump(StackExchangeExtractionConfig(
        input_path=versioned("gs://marin-us-central2/documents/stackexchange/v2024-04-02/md-complete"),
        output_path=output(),
        extract_method="resiliparse",
        extract_config=ResiliparseConfig(
            links=False,
            markdownify_config=HtmlToMarkdownConfig(
                include_images=False,
                include_links=False,
            ),
        ),
    ))

stackexchange_text_resiliparse_custom_fork = stackexchange_text_resiliparse_custom_fork_step().with_output_path("documents/stackexchange-resiliparse-custom-fork-ab41ad")


if __name__ == "__main__":
    executor_main(
        steps=[
            stackexchange_text_resiliparse_custom_fork,
        ]
    )
