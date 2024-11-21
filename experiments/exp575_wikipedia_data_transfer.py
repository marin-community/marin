from marin.execution.executor import ExecutorStep, executor_main, this_output_path, versioned
from operations.download.wikipedia.download import DownloadConfig, download

raw_wikipedia_dump = ExecutorStep(
    name="raw/wikipedia",
    fn=download,
    config=DownloadConfig(
        input_path=versioned(
            "https://dumps.wikimedia.org/enwiki/20241101/enwiki-20241101-pages-articles-multistream.xml.bz2"
        ),
        output_path=this_output_path("enwiki-latest-pages-articles.xml"),
    ),
)

if __name__ == "__main__":
    executor_main(steps=[raw_wikipedia_dump])
