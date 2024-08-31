import re
import subprocess

from bs4 import BeautifulSoup


# remove citations
def clean_text(text):
    # This regex pattern matches any sequence of [number]
    # get rid of references
    pattern = "\^\([^)]+\)"
    cleaned_text = re.sub(pattern, "", text)
    # clean empty lines
    lines = cleaned_text.split("\n")
    clean_lines = []
    for line in lines:
        if not line.strip():
            clean_lines.append("\n")
        elif line.strip() == "[]":
            clean_lines.append("\n")
        else:
            clean_lines.append(line)
    cleaned_text = "\n".join(clean_lines)
    cleaned_text = re.sub("[\n]{2,}", "\n\n", cleaned_text)
    return cleaned_text


# convert html to md
def html2md(html):
    # get title
    soup = BeautifulSoup(html, "html.parser")
    doc_title = soup.title.string
    # get headers
    headers = [(tag.text, tag.name) for tag in soup.find_all(["h2", "h3", "h4"])]
    headers_dict = {}
    for header in headers:
        if header[0] not in headers_dict:
            headers_dict[header[0]] = []
            prefix = {"h2": "##", "h3": "###", "h4": "####"}[header[1]]
            headers_dict[header[0]].append(prefix)
    # generate plain text with pandoc
    cmd = "pandoc --from html --to plain --wrap=none"
    text = str(subprocess.check_output(cmd, input=html, text=True, shell=True))
    # clean text
    text = clean_text(text)
    # apply headers
    lines = text.split("\n")
    markdown_lines = []
    for line in lines:
        if line.strip() in headers_dict:
            title = line.strip()
            if headers_dict[title]:
                markdown_lines.append(headers_dict[title][0] + " " + title)
                headers_dict[title].pop(0)
        else:
            markdown_lines.append(line)
    # create final doc
    doc = f"# {doc_title}\n\n" + ("\n".join(markdown_lines))
    return doc
