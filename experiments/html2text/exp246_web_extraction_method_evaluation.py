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

from experiments.evals.evals import default_eval
from experiments.html2text.exp246_web_extraction_method_training import (
    fineweb_readability_1_4b_model,
    fineweb_resiliparse_preserve_formatting_1_4b_model,
    fineweb_trafilatura_1_4b_model,
    fineweb_trafilatura_favor_precision_1_4b_model,
    transform_resiliparse_default,
)
from marin.execution.executor import executor_main

readability_eval = default_eval(fineweb_readability_1_4b_model)

trafilatura_default_eval = default_eval(fineweb_trafilatura_1_4b_model)

trafilatura_favor_precision_eval = default_eval(fineweb_trafilatura_favor_precision_1_4b_model)

resiliparse_default_eval = default_eval(transform_resiliparse_default)

resiliparse_preserve_formatting_eval = default_eval(fineweb_resiliparse_preserve_formatting_1_4b_model)


if __name__ == "__main__":
    executor_main(
        steps=[
            readability_eval,
            trafilatura_default_eval,
            trafilatura_favor_precision_eval,
            resiliparse_default_eval,
            resiliparse_preserve_formatting_eval,
        ]
    )
