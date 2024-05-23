"""
extract.py

Defines utilities for interacting with and extracting data from StackExchange Posts, stored in a custom XML format.
Focuses on extracting *only* (question, answer) data for each post.

StackExchange Posts =>> XML Schema:
    - Id
    - PostTypeId (listed in the PostTypes table)
        + 1 = Question
        + 2 = Answer
        + 3 = Orphaned tag wiki
        + ...
    - AcceptedAnswerId (only present if PostTypeId = 1)
    - ParentId (only present if PostTypeId = 2)
    - CreationDate =>> All dates are UTC
    - Score (generally non-zero for only Questions, Answers, and Moderator Nominations)
    - ViewCount (nullable)
    - Body (as rendered HTML, not Markdown)
    - ...

Ref: https://meta.stackexchange.com/questions/2677/database-schema-documentation-for-the-public-data-dump-and-sede
"""

import io
import xml.etree.ElementTree as ET
from datetime import datetime
from enum import Enum
from typing import Iterator

from marin.markdown import to_markdown
from marin.overwatch import initialize_overwatch
from marin.schema.document import StackExchangeAnswer, StackExchangeThreadMetadata

# Initialize Overwatch =>> Wraps `logging.Logger`
overwatch = initialize_overwatch(__name__)


# Formatting Options for StackExchange Threads
class StackExchangeMarkdownFormat(Enum):
    # fmt: off
    ATOMIC = "atomic"           # Formats question and answer content as independent documents with no metadata
    QA_PAIR = "qa-pair"         # Formats each question/answer pair as an independent document with title metadata
    COMPLETE = "complete"       # Formats a full thread as a single document (answers sorted by vote count)
    # fmt: on


def markdownify_thread(
    thread_metadata: StackExchangeThreadMetadata,
    markdown_format: StackExchangeMarkdownFormat = StackExchangeMarkdownFormat.COMPLETE,
) -> list[tuple[str, str]]:
    """Convert the rendered HTML in a StackExchange thread to clean Markdown, returning (unique ID, markdown)."""
    title, question, answers = thread_metadata.title, thread_metadata.question, thread_metadata.answers
    question_md, answers_md = to_markdown(question), [to_markdown(a.body) for a in answers]

    match markdown_format:
        case StackExchangeMarkdownFormat.ATOMIC:
            question_doc = (f"q-{thread_metadata.id}", question_md)
            answer_docs = [(f"a-{answers[i].id}", answer_md) for i, answer_md in enumerate(answers_md)]

            return [question_doc, *answer_docs]

        case StackExchangeMarkdownFormat.QA_PAIR:
            all_docs = []
            for i, answer_md in enumerate(answers_md):
                qa_md = f"# {title}\n\n{question_md.strip()}\n\n## Answer\n\n{answer_md.strip()}"
                all_docs.append((f"qa-{thread_metadata.id}+{answers[i].id}", qa_md))

            return all_docs

        case StackExchangeMarkdownFormat.COMPLETE:
            all_answer_md = "## Answer\n\n".join(answers_md)
            thread_md = f"# {title}\n\n{question_md.strip()}\n\n## Answer\n\n{all_answer_md.strip()}"

            return [(f"thread-{thread_metadata.id}", thread_md)]

        case _:
            raise ValueError(f"Unsupported Markdown conversion format: {markdown_format = }")


def extract_stackexchange_threads(
    subdomain: str,
    post_xml_content: str | io.BytesIO,
    processed_ids: set[str],
    min_vote_threshold: int = -1024,
    max_answer_threshold: int = 512,
) -> Iterator[StackExchangeThreadMetadata]:
    """Parses StackExchange Post XML content, and marshals into a (raw) instance of StackExchangeThreadMetadata."""
    question_threads: dict[str, StackExchangeThreadMetadata] = {}
    expected_answer_counts: dict[str, int] = {}
    skipped_question_ids: set[str] = set()

    # Iterate over XML Content
    for _, element in ET.iterparse(post_xml_content):
        if element.tag != "row":
            continue

        # Switch on "PostType" in {"1" (Question), "2" (Answer)}
        if (post_type := element.get("PostTypeId")) == "1":
            # [Short-Circuit] If element ID in `processed_ids`, add to `skipped_question_ids` and continue
            if (question_id := element.get("Id")) in processed_ids:
                skipped_question_ids.add(question_id)
                continue

            # Extract top-level thread metadata
            title, body, tags = element.get("Title"), element.get("Body"), element.get("Tags", "").strip("|").split("|")
            accepted_answer_id = element.get("AcceptedAnswerId")

            # [CONTRACT] All dates/times are UTC
            creation_time_utc = datetime.fromisoformat(element.get("CreationDate"))

            # Filter on votes & answer counts =>> note that votes can be negative!
            question_votes, n_answers = int(element.get("Score")), int(element.get("AnswerCount", 0))
            if (question_votes < min_vote_threshold) or (n_answers == 0):
                skipped_question_ids.add(question_id)
                continue

            # Add to trackers
            expected_answer_counts[question_id] = n_answers
            question_threads[question_id] = StackExchangeThreadMetadata(
                id=question_id,
                subdomain=subdomain,
                url=f"https://{subdomain}.stackexchange.com/questions/{question_id}",
                title=title,
                question=body,
                tags=tags,
                creation_time_utc=creation_time_utc,
                votes=question_votes,
                accepted_answer_id=accepted_answer_id,
                answers=[],
            )

        elif post_type == "2":
            # [CONTRACT] In StackExchange XML, answers will always come *after* questions!
            answer_id, parent_question_id = element.get("Id"), element.get("ParentId")
            if parent_question_id in skipped_question_ids:
                continue

            elif parent_question_id not in question_threads:
                overwatch.warning(f"Found answer `{answer_id}` with unknown parent `{parent_question_id}`")
                continue

            # Add individual answers to `thread_metadata: StackExchangeThreadMetadata`
            thread_metadata = question_threads[parent_question_id]
            accepted_answer_id = thread_metadata.accepted_answer_id

            # Filter on votes (or acceptance)
            answer_votes = int(element.get("Score"))
            if (answer_votes >= min_vote_threshold) or (answer_id == accepted_answer_id):
                thread_metadata.answers.append(
                    StackExchangeAnswer(
                        id=answer_id,
                        body=element.get("Body"),
                        creation_time_utc=datetime.fromisoformat(element.get("CreationDate")),
                        votes=answer_votes,
                    )
                )

            # Decrement and check counter of expected number of answers --> tells us when thread is complete!
            expected_answer_counts[parent_question_id] -= 1
            if expected_answer_counts[parent_question_id] == 0:
                # Sort by vote count (descending order), keep `max_answers`
                thread_metadata.answers.sort(key=lambda a: (a.id == accepted_answer_id, a.votes), reverse=True)
                thread_metadata.answers = thread_metadata.answers[:max_answer_threshold]

                # Yield =>> Garbage Collect
                yield thread_metadata

                del question_threads[parent_question_id]
                del expected_answer_counts[parent_question_id]
