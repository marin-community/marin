# StackExchange

## Steps to create the dataset


### Download the data

Data is available at https://archive.org/download/stackexchange. We used the dump from 2024-04-02.
We exclude "meta" sites and only use the main sites. So we don't use "3dprinting.meta.stackexchange.com.7z"
but we use "3dprinting.stackexchange.com.7z".

It is about 100 GB.

Using GCS: Start a storage transfer service job on GCS to copy the data from the public dataset to your bucket.
You can upload the `stack_exchange_urls.tsv` to an http server somewhere and use it.

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