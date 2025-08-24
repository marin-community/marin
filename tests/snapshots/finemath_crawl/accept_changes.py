import os

from tqdm import tqdm

from marin.crawl.get_finemath_crawl_yield import selectors
from marin.schemas.web.convert import ResiliparseConfig

# from resiliparse.html.extract import extract_
from marin.web.convert import convert_page

BANNED_SELECTORS = [
    "body.footer",
]


def prepare_expected_output(html: str) -> str:
    content = convert_page(
        html,
        extract_method="resiliparse",
        config=ResiliparseConfig(
            main_content=True,
            skip_elements=selectors,
            prepend_title=False,
            preserve_formatting=True,
        ),
    )["content"]

    return content.strip()


if __name__ == "__main__":
    input_path = "tests/snapshots/finemath_crawl/inputs"
    expected_path = "tests/snapshots/finemath_crawl/expected"

    if not os.path.exists(expected_path):
        os.makedirs(expected_path)

    input_files = os.listdir(input_path)

    for input_file in tqdm(input_files):
        with open(os.path.join(input_path, input_file), "r") as f:
            html = f.read()
            expected = prepare_expected_output(html)

            with open(os.path.join(expected_path, input_file.replace("html", "md")), "w") as f:
                print(expected, file=f)
