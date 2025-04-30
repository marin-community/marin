import json

import fsspec
import ray
import datetime
import argparse

from marin import markdown
from bs4 import BeautifulSoup
import pathlib


def transform_abstract(html: BeautifulSoup):
    # Transform the abstract from h6 to h2
    abstract = html.findAll("h6", {"class": "ltx_title_abstract"})
    for ab in abstract:
        ab.name = "h2"
    return html


def remove_authors(html: BeautifulSoup):
    # Remove authors since we only care about information after first section
    authors = html.findAll("div", {"class": "ltx_authors"})
    for author in authors:
        author.decompose()
        section = author.previous_sibling
        while section:
            new_section = section.previous_sibling
            section.decompose()
            section = new_section
    return html


def remove_title_page(html: BeautifulSoup):
    # Remove title page since we only care about information after first section
    title_page = html.findAll("div", {"class": "ltx_titlepage"})
    for tp in title_page:
        tp.decompose()


def clean_li(html: BeautifulSoup):
    # Remove the li tags since they repeat the same information (eg 1. 1.)
    tags = html.findAll("span", {"class": "ltx_tag_item"})
    for tag in tags:
        tag.decompose()
    tags = html.findAll("span", {"class": "ltx_tag_listingline"})
    for tag in tags:
        tag.decompose()


def remove_biblio(html: BeautifulSoup):
    # Remove the biblio since there is a lot of noise
    biblio = html.findAll("section", {"id": "bib"})
    for bib in biblio:
        bib.decompose()


def remove_footnotes(html: BeautifulSoup):
    # Remove footnotes since they are plopped in the middle of the text
    footnotes = html.findAll("div", {"class": "ltx_role_footnote"})
    for fn in footnotes:
        fn.decompose()


def remove_biblinks(html: BeautifulSoup):
    # Remove the biblinks since we are removing the biblio
    biblinks = html.findAll("a", {"class": "ltx_ref"})
    for biblink in biblinks:
        # Removes reference links
        # biblink.decompose()
        # Removes linking but keeps text
        biblink.unwrap()


def remove_references(html: BeautifulSoup):
    # Remove the reference section
    references = html.findAll("section", {"id": "ltx_bibliography"})
    for ref in references:
        ref.decompose()

    # Remove the references section
    references = html.findAll("ul", {"class": "ltx_biblist"})
    for ref in references:
        ref.decompose()


def linelisting_to_newline(html: BeautifulSoup):
    # Turn new line listings into new lines
    linelisting = html.findAll("div", {"class": "ltx_listingline"})
    for fn in linelisting:
        fn.append(BeautifulSoup("<br>", "html.parser"))


def unwrap_eqn(html: BeautifulSoup) -> BeautifulSoup:
    """
    Extract alttext from math element and convert to LaTeX format.
    Returns BeautifulSoup object with the formatted equation.
    """
    math_elements = html.find_all("math")

    for math_elem in math_elements:
        if not math_elem or "alttext" not in math_elem.attrs:
            continue

        equation = math_elem["alttext"]
        is_display = math_elem.get("display") == "block"

        if is_display:
            formatted_eq = BeautifulSoup(f"<p><br><br>$${equation}$$<br><br></p>", "html.parser")
        else:
            formatted_eq = BeautifulSoup(f"${equation}$", "html.parser")

        math_elem.replace_with(formatted_eq)

    return html


def deconstruct_eqn(html: BeautifulSoup):
    # Unwrap equation tables to ensure math mode is not in a table
    eqntables = html.findAll("table", {"class": "ltx_eqn_table"})
    for eqn in eqntables:
        eqn.append(BeautifulSoup("<br>", "html.parser"))
        eqn.unwrap()
    eqnrows = html.findAll("tr", {"class": "ltx_eqn_row"})
    for eqn in eqnrows:
        eqn.append(BeautifulSoup("<br>", "html.parser"))
        eqn.unwrap()

    eqncell = html.findAll("td", {"class": "ltx_eqn_cell"})
    for eqn in eqncell:
        eqn.unwrap()


def remove_ar5iv_footer(html: BeautifulSoup):
    # This is the ar5iv footer generated on xyz date
    footer = html.findAll("footer")
    for fn in footer:
        fn.decompose()


def remove_before_section(html: BeautifulSoup):
    # We only care about information after the first section
    section = html.find("section")
    if section:
        section = section.previous_sibling
        while section:
            new_section = section.previous_sibling
            section.extract()
            section = new_section


def remove_title(html: BeautifulSoup):
    # Title is added by markdown parser
    title = html.find("title")
    if title:
        title.decompose()


def remove_figure_captions(html: BeautifulSoup):
    # Remove the figure captions since they are not needed
    captions = html.findAll("figcaption", {"class": "ltx_caption"})
    for caption in captions:
        caption.decompose()


def clean_html(html: BeautifulSoup | str) -> str:
    if isinstance(html, str):
        html = BeautifulSoup(html, "html.parser")
    remove_authors(html)
    remove_title_page(html)
    clean_li(html)
    remove_biblio(html)
    remove_footnotes(html)
    remove_biblinks(html)
    linelisting_to_newline(html)
    deconstruct_eqn(html)
    remove_ar5iv_footer(html)
    remove_before_section(html)
    remove_title(html)
    return str(html)


@ray.remote(memory=512 * 1024 * 1024)  # 512 MB
def clean_ar5iv_html(file, prefix_path, output_path, file_size):
    """
    Takes in the input file and processes it to get the html content.
    Args:
    input_file_path (str): The input file to process
    zip_path (str): The path to the zip file
    """

    try:
        outs = ""
        with fsspec.open(file, "rb", compression="gzip") as outputf:
            for _ in range(file_size):
                line = outputf.readline()
                if not line:
                    break
                html_blob = json.loads(line)
                content = clean_html(html_blob["text"])
                outs += (
                    json.dumps(
                        {
                            "id": html_blob["id"],  # MANDATORY: source-specific identifier
                            "text": content,  # MANDATORY: textual content of the document
                            "source": "ar5iv",  # MANDATORY: source of the data, such as peS2o, common-crawl, etc.
                            "added": datetime.datetime.now().isoformat(),  # OPTIONAL: timestamp ai2 acquired this data
                        }
                    )
                    + "\n"
                )
        _file_path = pathlib.Path(file.replace("gs://", ""))
        _output_path = pathlib.Path(output_path.replace("gs://", ""))
        if _file_path.is_relative_to(prefix_path.replace("gs://", "")):
            out_file = _output_path / "html_clean" / _file_path.relative_to(prefix_path.replace("gs://", ""))
            if output_path.startswith("gs://"):
                out_file = "gs://" + str(out_file)
        else:
            raise Exception(f"File {file} is not in the prefix path {prefix_path}")
        with fsspec.open(out_file, "wb", compression="gzip") as outputf:
            outputf.write(outs.encode("utf-8"))
        print(f"Wrote to file {out_file}")
    except FileNotFoundError as e:
        print(f"Error reading the zip file: {e}")
        return False

    return True


@ray.remote(memory=1024 * 1024 * 1024)  # 1 GB
def markdownify_ar5iv_html(file, prefix_path, output_path, file_size):
    """
    Takes in the input file and processes it to get the html content.
    Args:
    input_file_path (str): The input file to process
    zip_path (str): The path to the zip file
    """

    try:
        outs = ""
        print(f"Starting Processing for the ar5iv file: {file}")
        with fsspec.open(file, "rb", compression="gzip") as outputf:
            for _ in range(file_size):
                line = outputf.readline()
                if not line:
                    break
                html_blob = json.loads(line)
                content = BeautifulSoup(html_blob["text"], "html.parser")
                try:
                    content = markdown.MyMarkdownConverter().convert_soup(content)
                except Exception as e:
                    print(f"Error converting to markdown: {e}")
                    print("content: ", content)
                    raise e
                # cleanup: replace nbsp as space
                # this isn't quite right if we preserve html in places, but we currently are not doing that
                content = content.replace("\xa0", " ").strip()
                outs += (
                    json.dumps(
                        {
                            "id": html_blob["id"],  # MANDATORY: source-specific identifier
                            "text": content,  # MANDATORY: textual content of the document
                            "source": "ar5iv",  # MANDATORY: source of the data, such as peS2o, common-crawl, etc.
                            "added": datetime.datetime.now().isoformat(),  # OPTIONAL: timestamp ai2 acquired this data
                        }
                    )
                    + "\n"
                )
        _file_path = pathlib.Path(file.replace("gs://", ""))
        _output_path = pathlib.Path(output_path.replace("gs://", ""))
        if _file_path.is_relative_to(prefix_path.replace("gs://", "")):
            out_file = _output_path / "md" / _file_path.relative_to(prefix_path.replace("gs://", ""))
            if output_path.startswith("gs://"):
                out_file = "gs://" + str(out_file)
        else:
            raise Exception(f"File {file} is not in the prefix path {prefix_path}")
        with fsspec.open(out_file, "wb", compression="gzip") as outputf:
            outputf.write(outs.encode("utf-8"))
        print(f"Wrote to file {out_file}")
    except FileNotFoundError as e:
        print(f"Error reading the zip file: {e}")
        return False

    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert ar5iv to markdown.")
    parser.add_argument("--input_path", type=str, help="Path to the ar5iv jsonl folder", required=True)
    parser.add_argument("--output_path", type=str, help="Path to the ar5iv output folder", required=True)
    parser.add_argument("--file_size", type=int, help="Number of ar5iv documents in a file", required=False, default=256)

    args = parser.parse_args()
    if args.input_path.startswith("gs://"):
        fs = fsspec.filesystem("gcs")
    else:
        fs = fsspec.filesystem("file")
    html_folder = args.input_path
    output_path = args.output_path
    files = fs.find(html_folder)

    MAX_NUM_PENDING_TASKS = 600  # Max number of html files we want to process in pending state
    ray.init()
    result_refs = []

    for html in files:
        if len(result_refs) > MAX_NUM_PENDING_TASKS:
            # update result_refs to only
            # track the remaining tasks.
            ready_refs, result_refs = ray.wait(result_refs, num_returns=1)
            try:
                ray.get(ready_refs)
            except Exception as e:
                print(f"Error processing the group: {e}")
                continue
        if "gs://" in html_folder:
            html = "gs://" + html
        result_refs.append(clean_ar5iv_html.remote(html, html_folder, output_path, args.file_size))
    try:
        ray.get(result_refs)
    except Exception as e:
        print(f"Error processing the group: {e}")

    result_refs = []

    if args.input_path.startswith("gs://"):
        fs = fsspec.filesystem("gcs")
    else:
        fs = fsspec.filesystem("file")

    clean_html_folder = pathlib.Path(output_path.replace("gs://", "")) / "html_clean"
    files = fs.find(clean_html_folder)

    for html in files:
        if len(result_refs) > MAX_NUM_PENDING_TASKS:
            # update result_refs to only
            # track the remaining tasks.
            ready_refs, result_refs = ray.wait(result_refs, num_returns=1)
            try:
                ray.get(ready_refs)
            except Exception as e:
                print(f"Error processing the group: {e}")
                continue
        if "gs://" in html_folder:
            html = "gs://" + html
        print(f"Starting Processing for the ar5iv file: {html}")
        result_refs.append(markdownify_ar5iv_html.remote(html, output_path, output_path, args.file_size))

    # Wait for all the tasks to finish
    try:
        ray.get(result_refs)
    except Exception as e:
        print(f"Error processing the group: {e}")
