# StackExchange

Instructions for downloading and processing raw StackExchange dumps, along with descriptions of data formats and file
structure.

## Downloading Raw Data

Raw StackExchange dumps are available at https://archive.org/download/stackexchange. We use the dump from 2024-04-02.
We exclude "meta" sites and only use the main sites (i.e., we use "3dprinting.stackexchange.com.7z" but don't use
"3dprinting.meta.stackexchange.com.7z"). The full dump is approximately 100 GB.

**Downloading Data to GCS**: To get the raw data, we use the GCS Storage Transfer Service to perform the data transfer.
To kick off the job, create `stackexchange-urls.tsv` using the following instructions (per @dlwh):

- Go to `[https://archive.org/details/stackexchange](https://archive.org/details/stackexchange)`
- Expand the `7z` sidebar, copy all the names (w/ mouse)
- Paste into a text editor (i.e., VSCode)
- Run (sequence of find/replace commands - regex mode) 
  + Remove all " download" strings -- match on `download `
  + Remove all file sizes (e.g., 188M) -- match on `^\d.*?\d[KMG]`
  + Remove all `meta` sites -- match on `.*\.meta\..*\n`
  + Prepend URL Prefix `https://archive.org/download/stackexchange/` to each line
  + Insert `TsvHttpData-1.0` on the first line

Pass this file to the Storage Transfer Job CLI to kick off the transfer.

## Processing the Raw Data

We provide three different ways to "markdownify" StackExchange threads (questions and corresponding answers):
- "separate" (`gs://marin-data/processed/stackexchange/v2024-04-02/md-separate`): Each question and individual answer is
  formatted independently, as its own document (this is what Dolma does).
- "qa-pair" (`gs://marin-data/processed/stackexchange/v2024-04-02/md-qa-pair`): Each (question, answer) pair in a thread
  is formatted as its own document (note that this duplicates each question multiple times).
- "complete" (`gs://marin-data/processed/stackexchange/v2024-04-02/md-complete`): An entire thread (question, answer1, 
  answer2, ...) is formatted as a single document.

To launch a processing job via Ray, run (from root of this repository):

```bash
ray job submit --address=http://127.0.0.1:8265 --working-dir . --no-wait -- \ 
  python scripts/stackexchange/process.py --markdown_format="separate" 
```
