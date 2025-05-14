#!/usr/bin/env python3
"""
This script:
- Reads all the GitHub issues with the `experiment` label.
- Prompts a language model to generate a summary.
- Writes the summary to `docs/reports/summary.md`.

Usage:
pip install PyGithub openai
python scripts/generate_experiment_summary.py
"""

import os
import re
from github import Github
from openai import OpenAI

MODEL = "gpt-4o-mini"
OUTPUT_PATH = "docs/reports/summary.md"

def get_github_issues():
    """Get all issues with the `experiment` label."""
    g = Github()
    repo = g.get_repo("marin-community/marin")
    return repo.get_issues(labels=["experiment"], state="all")

def convert_issue_to_xml(issue):
    return f"""<issue>
        <number>{issue.number}</number>
        <title>{issue.title}</title>
        <state>{issue.state}</state>
        <created_at>{issue.created_at}</created_at>
        <updated_at>{issue.updated_at}</updated_at>
        <labels>{','.join([label.name for label in issue.labels])}</labels>
        <url>{issue.html_url}</url>
        <body>\n{issue.body}</body>
        <comments>{issue.comments}</comments>
    </issue>\n"""

prompt_template = """
You are a helpful assistant that summarizes GitHub issues.

Given the following issues:
{issues}

Generate a summary of the issues.  Please keep the following in mind:
- The summary should be in markdown format.
- The summary should link to each of the issues (e.g., `[#123](https://github.com/marin-community/marin/issues/123)`).
- The summary should contain the following sections:
  1. Give a chronological summary of the issues.  Include the date and title of the issues.
  2. Give a summary grouped by the topic.  Summarize the findings.
"""

def main():
    print("Getting GitHub issues...")
    issues = list(get_github_issues())
    print(f"Read {len(issues)} issues.")
    xml = "\n".join([convert_issue_to_xml(issue) for issue in issues])

    prompt = prompt_template.format(issues=xml)
    print(f"Asking an LM to summarize (prompt is {len(prompt)} characters long)...")
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
    )
    summary = response.choices[0].message.content

    # Write the summary to a file
    print(f"Writing summary ({len(summary)} characters) to {OUTPUT_PATH}...")
    with open(OUTPUT_PATH, "w") as f:
        f.write(summary)


if __name__ == "__main__":
    main()
