## Data browser

Marin comes with a [data browser](https://github.com/marin-community/marin-data-browser/blob/main/README.md) that makes it easy to
view datasets (in various formats) and experiments produced by the executor.
After installing the necessary dependencies, run:

```bash
cd data_browser
python server.py --config conf/local.conf
```

Once the server is started, go to
[http://localhost:5000](http://localhost:5000) and navigate around to the
experiment JSON file to get a nicer view of the experiment (the URL is also
printed out when you run the experiment).

See the [data browser README](https://github.com/marin-community/marin-data-browser/blob/main/README.md) for more details.
