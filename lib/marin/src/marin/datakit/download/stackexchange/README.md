# Downloading Stackexchange Data

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
