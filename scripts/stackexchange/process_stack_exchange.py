import tempfile
from dataclasses import dataclass

import json
import os
import xml.etree.ElementTree as ET
from typing import Optional

import fsspec
import tqdm
from fsspec.callbacks import TqdmCallback
from markweb.markdown import to_markdown

# StackExchange dumps come in two flavors: single file and multiple file.
# StackOverflow is special in that it is multiple file:
# stackoverflow.com-Badges.7z download
# stackoverflow.com-Comments.7z download
# stackoverflow.com-PostHistory.7z download
# stackoverflow.com-PostLinks.7z download
# stackoverflow.com-Posts.7z download
# stackoverflow.com-Tags.7z download
# stackoverflow.com-Users.7z download
# stackoverflow.com-Votes.7z

# We only use the Posts file.

# The file format for StackExchange is an XML dump of posts.
# Posts are stored in escaped HTML format, so we can use our markweb pipeline to handle them.
# <row Id="1" PostTypeId="1" AcceptedAnswerId="51" CreationDate="2016-01-12T18:45:19.963" Score="11" ViewCount="458" Body="&lt;p&gt;When I've printed an object I've had to choose between high resolution and quick prints.  What techniques or technologies can I use or deploy to speed up my high resolution prints?&lt;/p&gt;&#xA;" OwnerUserId="16" LastActivityDate="2017-10-31T02:31:08.560" Title="How to obtain high resolution prints in a shorter period of time?" Tags="|resolution|speed|quality|" AnswerCount="2" CommentCount="6" ContentLicense="CC BY-SA 3.0" />

# we want to extract question answer pairs. We keep the accepted answer for simplicity. We thought about keeping
# the highest voted answer, but that's more work

ANSWERS_TO_KEEP = 500
VOTE_THRESHOLD = -1000

@dataclass
class Answer:
    id: str
    body: str
    date: str
    votes: int

@dataclass
class AnsweredQuestion:
    id: str
    title: str
    question: str
    accepted_answer_id: Optional[str]
    date: str
    tags: list[str]
    votes: int
    answers: list[Answer]




def process_posts_xml(xml):
    questions: dict[str, AnsweredQuestion] = {}
    answer_counts : dict[str, int] = {}

    for event, elem in tqdm.tqdm(ET.iterparse(xml), desc='Processing StackExchange XML'):
        if elem.tag == 'row':
            post_type = elem.get('PostTypeId')
            if post_type == '1': # question
                question_id = elem.get('Id')
                tags = elem.get('Tags', '').strip('|').split('|')
                title = elem.get('Title')
                body = elem.get('Body')
                votes = int(elem.get('Score'))
                answer_count = int(elem.get('AnswerCount', 0))
                accepted_answer_id = elem.get('AcceptedAnswerId')

                if votes < VOTE_THRESHOLD:
                    continue

                if answer_count == 0:
                    continue

                aq = AnsweredQuestion(
                    id=question_id,
                    title=title,
                    question=body,
                    date=elem.get('CreationDate'),
                    tags=tags,
                    votes=votes,
                    accepted_answer_id=accepted_answer_id,
                    answers=[]
                )
                questions[question_id] = aq
                answer_counts[question_id] = answer_count

            elif post_type == '2':  # answer
                # answers can't come before questions, so safe to assume we have the question
                id = elem.get('Id')
                parent_id = elem.get('ParentId')
                if parent_id in questions:
                    aq = questions[parent_id]
                    answer_counts[parent_id] -= 1
                    accepted_answer_id = aq.accepted_answer_id

                    votes = int(elem.get('Score'))
                    if votes >= VOTE_THRESHOLD or id == accepted_answer_id:
                        aq.answers.append(
                            Answer(
                                id=id,
                                body=elem.get('Body'),
                                date=elem.get('CreationDate'),
                                votes=votes
                            )
                        )

                    if answer_counts[parent_id] == 0:
                        # we have all the answers
                        # sort and keep the top 5
                        aq.answers.sort(key=lambda a: (a.id == accepted_answer_id,  a.votes), reverse=True)
                        aq.answers = aq.answers[:ANSWERS_TO_KEEP]

                        yield aq
                        del questions[parent_id]
                        del answer_counts[parent_id]

        elem.clear()

    # if len(questions_with_accepted_answers) > 0:
    #     raise ValueError(f"Found {len(questions_with_accepted_answers)} questions without answers")



if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print('Usage: python process_stack_exchange.py <posts.xml>', file=sys.stderr)
        sys.exit(1)
    xml = sys.argv[1]
    if xml.endswith('.7z'):
        domain = os.path.basename(xml)[:-3]
    else:
        dirname = os.path.dirname(xml)
        domain = os.path.basename(dirname)

    print(f"Processing {xml} for domain {domain}", file=sys.stderr)

    out_path = f"{domain}.jsonl.gz"
    if os.path.exists(out_path):
        print(f"Output file {out_path} already exists, skipping", file=sys.stderr)
        sys.exit(0)

    with tempfile.TemporaryDirectory() as tmpdir:
        # the file might be in a blob store, so we need to download it
        if not os.path.exists(xml):
            print(f"Downloading {xml}", file=sys.stderr)
            cb = TqdmCallback()
            fs = fsspec.get_fs_token_paths(xml)[0]
            local_path = os.path.join(tmpdir, os.path.basename(xml))
            fs.get_file(xml, local_path, callback=cb)
            del cb
        else:
            local_path = xml

        # if it's a 7z file, we want to extract the posts.xml file
        if local_path.endswith('.7z'):
            import py7zr
            with py7zr.SevenZipFile(local_path, mode='r') as archive:
                archive.extract(tmpdir, targets=['Posts.xml'])
                xml = os.path.join(tmpdir, 'Posts.xml')

                if not os.path.exists(xml):
                    raise FileNotFoundError(f"Expected to find Posts.xml in {local_path}, but it was not found")

        with fsspec.open(out_path, 'wt', compression='gzip') as f:
            for aq in process_posts_xml(xml):
                question_markdown = to_markdown(aq.question)
                answers_markdown = [to_markdown(a.body) for a in aq.answers]
                answer_markdown = '\n\n## Answer\n\n'.join(answers_markdown)

                text = f"# {aq.title}\n\n{question_markdown.strip()}\n\n## Answer\n\n{answer_markdown.strip()}"

                out = {
                    'metadata': {
                        'source': 'stackexchange',
                        'domain': domain,
                        'url': f'https://{domain}/questions/{aq.id}',
                        'id': aq.id,
                        'date': aq.date,
                        'tags': aq.tags,
                        'votes': aq.votes,
                        'answer_ids': [a.id for a in aq.answers],
                        'answer_votes': [a.votes for a in aq.answers],
                        'answer_date': [a.date for a in aq.answers]
                    },
                    'text': text
                }

                print(json.dumps(out), file=f)




