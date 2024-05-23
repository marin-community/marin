# StackExchange

Instructions for downloading and processing raw StackExchange dumps, along with descriptions of data formats and file
structure.

## Downloading Raw Data

*TODO (dlwh, siddk) :: Provide instructions for generating `stackexchange-urls.tsv`?*

Data is available at https://archive.org/download/stackexchange. We use the dump from 2024-04-02.
We exclude "meta" sites and only use the main sites (i.e., we use "3dprinting.stackexchange.com.7z" but don't use
"3dprinting.meta.stackexchange.com.7z"). The full dump is approximately 100 GB.

**Using GCS**: Start a storage transfer service job on GCS to copy the data from the public dataset to your bucket.
You can upload the `stackexchange-urls.tsv` to an http server somewhere and use it.


## Processing Pipeline


---


### Process the data

There is a single script that can process one site at a time: `process_stack_exchange.py`.

```bash
python process_stack_exchange.py <path to 7z file or Posts.xml>
```

The file can be on GCS or local. The script will download the file if it is on GCS.

As a convenience, we provide a script that processes all sites in a loop: `process_all.bash`.

```bash
bash process_all.bash
```

This will read the `stack_exchange_urls.tsv` file and process all sites. If you filter our StackOverflow, this
runs in couple hours on a MacBook.
