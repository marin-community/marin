"""
This script converts a markdown file containing tasks into GitHub issues.

It uses the Gemini API to parse markdown into structured JSON, then creates GitHub issues with:
- Title and description from the markdown
- Labels (parsed from hashtags like #p0)
- Assignees (mapped from person names like "will" -> "Helw150")
- A "Release" milestone
- Timestamp showing it was auto-created

The script prompts for confirmation before creating each issue.

Usage: python gh_issues_from_markdown.py <markdown_file>

"""

import os
from datetime import datetime
import requests
import json
from typing import List

# Configuration
GITHUB_REPO = "stanford-crfm/marin"
GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]
GITHUB_TOKEN = os.environ["GITHUB_TOKEN"]
MILESTONE_NAME = "Release"

headers = {
    "Authorization": f"token {GITHUB_TOKEN}",
    "Accept": "application/vnd.github+json",
}

# I've got a gemini key so

MARKDOWN_TO_JSON_PROMPT = """
Convert the following markdown to a JSON object with the following fields:
- title: The title of the task
- description: The description of the task
- labels: The labels of the task (if any)
- assignee: The assignee of the task (if any)

For example, the output might look like
[
  {
    "title": "Finish marin.io landing page",
    "description": "Includes GitHub link, artifact links, and Discord form.",
    "labels": ["p0", "website"]
  },
  {
    "title": "Upload Spoonbill SFT to HF",
    "description": "This is a p0 release artifact. Should include README and model card.",
    "labels": ["p0", "artifacts"]
  }
]

For example, given this markdown:

```markdown
 * Documentation
    * Note: we’re mostly following the diataxis framework
    * Your goal is to go through existing docs
    * README \#p0
    * Tutorials
      * Set up Marin \#p1 \[May 1\]
      * Launch a basic hello world experiment \#p1 \[May 1\]
    * How To
      * Add an HF dataset \#p1 \[May 4\] \- Chris
      * Training a model \- Will
        * Replicate DCLM 1b/1x \#p1 \[May 4\] 72\_baselines.py Will
        * Replicate DCLM 7b/1x \#p2 \[May 4\] Will
```

The output should be:

```json
[
  {
    "title": "Update the README",
    "description": "Update the README for release."
    "labels": ["p0", "documentation"]
  },
  {
    "title": "Tutorial: Set up Marin",
    "description": "Add or improve a tutorial for setting up Marin. Add it to the documentation folder.",
    "labels": ["p1", "tutorials", "documentation"]
  },
  {
    "title": "Tutorial: Launch a basic hello world experiment",
    "description": "Add or improve a tutorial for launching a basic hello world experiment. Add it to the documentation folder.",
    "labels": ["p1", "tutorials", "documentation"]
  },
  {
    "title": "How To: Add an HF dataset",
    "description": "Add or improve a how-to for adding an HF dataset. Add it to the documentation folder.",
    "labels": ["p1", "how-to", "documentation"],
    "assignee": "BabyChouSr"
  },
  {
    "title": "How To: Training a model",
    "description": "Add or improve a how-to for training a model. Add it to the documentation folder.",
    "labels": ["p2", "how-to", "documentation"],
    "assignee": "Helw150"
  },
  {
    "title": "How To: Replicate DCLM 1b/1x",
    "description": "Add or improve a how-to for replicating DCLM 1b/1x. Add it to the documentation folder.",
    "labels": ["p1", "how-to", "documentation"],
    "assignee": "Helw150"
  }
]
```

Skip bullets that do not look like tasks.

Use the following map from names to assignees:

- "abhinav" or "abhi" -> abhinavg4
- "ahmed" -> ahmeda14960
- "chris" -> BabyChouSr
- "david" -> dlwh
- "herumb" -> krypticmouse
- "niki" -> nikil-ravi
- "rohith" -> RohithKuditipudi
- "will" -> Helw150
- "percy" -> percyliang

Below is the markdown to convert. RESPONSE WITH JSON AND NOTHING ELSE. DO NOT ESCAPE IT INSIDE A MARKDOWN CODE BLOCK.
"""


def convert_markdown_to_json_tasks(markdown_file_path: str):
    with open(markdown_file_path, "r") as f:
        markdown_text = f.read()

    response = requests.post(
        "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key=" + GEMINI_API_KEY,
        json={"contents": [{"parts": [{"text": MARKDOWN_TO_JSON_PROMPT + "\n\n" + markdown_text}]}]},
        headers={"Content-Type": "application/json"},
    )
    response.raise_for_status()
    response_json = response.json()
    print(response_json)

    # Extract the text content from the response
    text_content = response_json["candidates"][0]["content"]["parts"][0]["text"]

    # Remove the markdown code block markers if present
    if text_content.startswith("```json"):
        text_content = text_content[7:]
    if text_content.endswith("```"):
        text_content = text_content[:-3]

    # Parse the JSON content
    return json.loads(text_content.strip())


def get_milestone_id():
    """Return the milestone ID for 'Release'"""
    url = f"https://api.github.com/repos/{GITHUB_REPO}/milestones"
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    for m in response.json():
        if m["title"] == MILESTONE_NAME:
            return m["number"]
    raise ValueError(f"Milestone '{MILESTONE_NAME}' not found")


def create_issue(title: str, body: str, labels: List[str], milestone_id: int, assignee: str):
    url = f"https://api.github.com/repos/{GITHUB_REPO}/issues"
    data = {
        "title": title,
        "body": f"\[Created by Marin Auto-PM at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\]\n\n{body}",
        "labels": labels,
        "milestone": milestone_id,
        "assignee": assignee,
    }
    response = requests.post(url, headers=headers, json=data)
    response.raise_for_status()
    print(f"Created issue: {title}")


def create_issues_from_json(json_tasks: List[dict]):
    milestone_id = get_milestone_id()

    print(f"Found {len(json_tasks)} tasks.")

    for task in json_tasks:
        title = task["title"]
        body = task.get("description", "")
        labels = task.get("labels", [])
        assignee = task.get("assignee", None)
        # see if we want to create an issue for this task
        print(f"Title: {title}")
        print(f"Body: {body}")
        print(f"Labels: {labels}")
        print(f"Milestone ID: {milestone_id}")
        print(f"Assignee: {assignee}")
        print("-" * 80)
        while True:
            ok = input("Create issue? (Y/n)").strip().lower()
            if ok in ["y", "yes", "n", "no", "q", "quit", ""]:
                if ok == "q" or ok == "quit":
                    exit(0)
                ok = "y" if ok in ["y", "yes", ""] else "n"
                break
            else:
                print("Invalid input. Please enter 'y' or 'n'.")
        if ok == "n":
            continue

        create_issue(title, body, labels, milestone_id, assignee)


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: python create_issues.py <path_to_markdown_file>")
        exit(1)
    markdown_file_path = sys.argv[1]
    json_tasks = convert_markdown_to_json_tasks(markdown_file_path)
    create_issues_from_json(json_tasks)
