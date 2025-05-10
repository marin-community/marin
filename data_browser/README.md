# Data Browser

Marin comes with a data browser that makes it easy to
view datasets (in various formats) and experiments produced by the executor.
After installing the necessary dependencies, run:

```bash
cd data_browser
python server.py --config conf/local.conf
```

## Development

To start the data browser:

    docker-compose up --build

And then open http://localhost:5000 in your browser.


For members of the core Marin dev team, see [README-internal.md](README-internal.md) for
information on deployment and other internal details.
