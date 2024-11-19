from marin.execution.executor import ExecutorStep, executor_main, this_output_path
from operations.download.wikipedia.download import DownloadConfig, download


transfer_wikipedia_dump = ExecutorStep(
    name="raw/wikipedia",
    fn=download,
    config=DownloadConfig(
        input_path="https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles.xml.bz2",
        output_path=this_output_path("enwiki-latest-pages-articles.xml"),
    ),
)

if __name__ == "__main__":
    executor_main(steps=[transfer_wikipedia_dump])