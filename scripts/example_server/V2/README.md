# Example Server (V2)

## Links

1. Example server home: <url> : TODO
2. Each domain is available under <url>/{domain}/{version}

## How should your data be formatted

1. Your data should be in standard Dolma + Marin format.
2. Your processed files should be present in `marin-data/processed/{domain}/{version}/{format}/**/<filename>.jsonl.gz`.
   Eg:- see `marin-data/processed/fineweb/fw-v1.0`
3. `{format}` can be `html, md, text` right now.

## How does example server work

1. Every 5 min, example server runs `check-new` job which checks if there is a any new/modified `**/*.jsonl.gz` file
   inside  `{domain}/{version}/{format}`.
2. If there is a new/modified folder, it reads all the `**/jsonl.gz` files and samples `1000` lines from them using
   reservoir sampling.
3. It then saves the sampled lines in `marin-data/examples/{domain}/{version}/{format}/samples.jsonl`.
4. It renders them as per `{format}`.

## Important thing to keep in mind

1. The rendering is based on `{format}` in the file path and not content-type inside the jsonl.
2. It might take upto 10 min for the new samples to be available on the server. To run `check-new` job instantly please
   visit <url>/check-new.
3. If you want to re-render a {domain}/{version}/{format} folder, you can create/modify any `**/jsonl.gz` file in that
   folder.
   