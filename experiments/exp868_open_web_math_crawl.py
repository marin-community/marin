import json

from experiments.crawl.default import default_crawl
from marin.crawl.common.schemas import ParquetConfig
from marin.crawl.get_open_web_math_crawl_yield import main
from marin.execution.executor import executor_main, this_output_path

open_web_math_crawling_steps = default_crawl(
    config=ParquetConfig(
        input_path="gs://marin-us-central2/raw/open-web-math-fde8ef8/fde8ef8/huggingface.co/datasets/open-web-math/open-web-math/resolve/fde8ef8/data/",
        output_path=this_output_path(),
        source_name="open-web-math-test-2",
        columns=["url", "date", "metadata"],
        warc_path_extractor=lambda x: json.loads(x)["warc_path"],
        max_files=1,
    ),
    yield_fn=main,
)

if __name__ == "__main__":
    executor_main(
        steps=open_web_math_crawling_steps,
    )
