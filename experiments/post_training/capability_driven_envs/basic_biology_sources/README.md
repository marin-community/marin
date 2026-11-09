# Government introductory biology sources (raw HTML only)

This is a **source archive, not an RL dataset or model-input corpus**. It
replaces the previously proposed OpenStax books: their current pages explicitly
require written permission for LLM training/ingestion, so those books were not
imported. The two government sources here are suitable candidates for a
**genetics-first** pilot; neither by itself covers a full biology curriculum.

- [MedlinePlus Genetics: *Help Me Understand Genetics*](https://medlineplus.gov/genetics/understanding/) — beginner chapters on cells, DNA, genes, inheritance, testing, and research. The [NLM content policy](https://medlineplus.gov/about/using/usingcontent/) explicitly lists the handbook as public domain, asks for acknowledgment, and distinguishes other **copyrighted MedlinePlus material**. Do not import the encyclopedia or third-party licensed material by assuming the whole site is public domain.
- [NHGRI fact sheets](https://www.genome.gov/about-genomics/fact-sheets) — short public explanations of genomics concepts. The [NHGRI copyright policy](https://www.genome.gov/about-nhgri/Policies-Guidance/Copyright) says NHGRI website content is public domain **unless otherwise indicated** and requests acknowledgment. Check individual pages for third-party credit lines before extracting text.

`fetch_sources.py` downloads **only HTML** from the official handbook's index,
sections and article links and the NHGRI fact-sheet index and direct links; it
does not fetch images, linked third-party pages, PDFs, or datasets. Each page
and both rights-policy pages are hashed separately. The first frozen archive
(2026-11-08) contains **111 MedlinePlus handbook pages + 28 NHGRI index/fact-sheet
pages**, approximately 9 MB of raw HTML. The manifest SHA-256 is
`9f880299f679ec033b6dd68b422266b7f04a9ecc9ca38ef748b8e6f855b784cd`.
Private snapshot:

```text
s3://marin-us-east-02a/marin4sc/taskforge/source-corpora/us-gov-biology/raw-html-v1-20261108/
```

Reproduce a new snapshot locally, then publish it with credentials provided via
`FSSPEC_S3` (never in git or command output):

```bash
uv run python experiments/post_training/capability_driven_envs/basic_biology_sources/fetch_sources.py \
  --output /tmp/us-gov-biology-new-snapshot
uv run python experiments/post_training/capability_driven_envs/basic_biology_sources/publish_snapshot.py \
  --input /tmp/us-gov-biology-new-snapshot \
  --destination s3://marin-us-east-02a/marin4sc/taskforge/source-corpora/us-gov-biology/raw-html-v2/
```

The publisher verifies every hash, refuses differing pre-existing objects and
writes `manifest.json` last. **Do not feed the raw HTML directly to an author
model**: first extract article body text, verify any page-level rights exceptions
and image/text attribution, exclude navigation/third-party snippets, preserve
source URL and source-family identity, and independently check factual accuracy
and answer provenance. Task design and RL training decisions are separate work.
