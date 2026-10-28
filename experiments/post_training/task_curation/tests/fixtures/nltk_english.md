# English NLTK test resources

`nltk_english.zip` contains the English Punkt tables and stopword list used by
the offline instruction-grading fixtures. The original resource README files
are included in the archive. These resources come from
[`nltk/nltk_data`](https://github.com/nltk/nltk_data/tree/550b6625bcef1f2abff2ff770a5a0d272c9c6b2a).
`nltk_english.json` records the source archive hashes, each retained file hash,
and the repacked fixture hash. File contents are unchanged.

Release graders use the full, SHA-verified source archives declared in
`datasets/evalchemy/instruction_environments.py`; they do not use this fixture.
